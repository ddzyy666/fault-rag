"""Run from repository root; uses existing SQL chunks and Qdrant index."""

import argparse
import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from app.core.config import settings
from app.db.database import async_session_factory, engine
from app.repositories.document import get_knowledge_base_chunks_with_documents
from app.schemas.evaluation import RetrievalEvaluationRequest
from app.services.embedding import get_embedding_provider
from app.services.reranker import get_reranker_provider
from app.services.retrieval_evaluation import evaluate_retrieval
from app.services.sparse_embedding import get_sparse_embedding_provider
from app.services.vector_store import get_vector_store


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-base-id", required=True, type=UUID)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument(
        "--labels", type=Path, default=Path("sample_evaluations/air_compressor_labels.json")
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation_reports"))
    args = parser.parse_args()
    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    run_dir = args.output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    def save(name: str, value: object) -> None:
        (run_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    store = None
    try:
        async with async_session_factory() as session:
            records = await get_knowledge_base_chunks_with_documents(
                session, args.knowledge_base_id
            )
            if not records:
                raise ValueError("No indexed chunks found")
            snapshot = [
                {
                    "chunk_id": str(c.id),
                    "document_id": str(d.id),
                    "filename": d.filename,
                    "content": c.content,
                    "metadata": c.extra_metadata,
                }
                for c, d in records
            ]
            save("corpus.json", snapshot)
            resolved = []
            for case in labels["cases"]:
                matches = []
                for section in case["sections"]:
                    found = [
                        c
                        for c, d in records
                        if d.filename == "空压机维修知识库.md"
                        and c.extra_metadata.get("section_title") == section
                    ]
                    if len(found) != 1:
                        raise ValueError(f"Label needs review: {case['case_id']} {section}")
                    matches.extend(found)
                if not any(case["evidence"] in c.content.replace("\n", "") for c in matches):
                    raise ValueError(f"Evidence mismatch: {case['case_id']}")
                resolved.append({**case, "relevant_chunk_ids": [str(c.id) for c in matches]})
            save("labels.json", resolved)
            embedding = get_embedding_provider()
            sparse_embedding = get_sparse_embedding_provider()
            store = get_vector_store()
            # Exclude first model load from measured query latency.
            embedding.embed_query("空压机检索预热")
            save(
                "metadata.json",
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "knowledge_base_id": str(args.knowledge_base_id),
                    "embedding_model": embedding.model_name,
                    "sparse_embedding_model": sparse_embedding.model_name,
                    "reranker_model": settings.reranker_model_name,
                    "reranker_enabled": settings.reranker_enabled,
                    "candidate_multiplier": settings.retrieval_candidate_multiplier,
                    "rrf_k": settings.rrf_k,
                    "corpus_sha256": hashlib.sha256(
                        json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()
                    ).hexdigest(),
                    "annotation": labels["annotation"],
                    "latency_note": "Serial single run after embedding warmup; not load testing.",
                },
            )
            report = [
                "# 空压机检索基线实验",
                "",
                labels["annotation"],
                "",
                "Top K=5，向量阈值为空；固定现有索引与分块，未调参。",
                "",
                "| 分组 | 方案 | Hit@5 | Recall@5 | MRR | nDCG@5 | 平均ms | 重排成功 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            failures = []
            for split in ("development", "validation"):
                selected = [c for c in resolved if c["split"] == split]
                payload = RetrievalEvaluationRequest.model_validate(
                    {"cases": selected, "top_k": 5, "score_threshold": None}
                )
                if args.local_only:
                    payload.variants = [v for v in payload.variants if not v.rerank]
                save(f"{split}.request.json", payload.model_dump(mode="json"))
                result = await evaluate_retrieval(
                    session,
                    knowledge_base_id=args.knowledge_base_id,
                    payload=payload,
                    embedding_provider=embedding,
                    sparse_embedding_provider=sparse_embedding,
                    vector_store=store,
                    reranker_provider=get_reranker_provider(),
                )
                save(f"{split}.results.json", result.model_dump(mode="json"))
                for variant in result.variants:
                    m = variant.metrics
                    report.append(
                        f"| {split} | {variant.name} | {m.hit_rate_at_k:.3f} | "
                        f"{m.recall_at_k:.3f} | {m.mrr:.3f} | {m.ndcg_at_k:.3f} | "
                        f"{m.average_latency_ms} | {m.reranker_applied_count}/{m.case_count} |"
                    )
                    for case in variant.cases:
                        if case.recall_at_k < 1 or case.reciprocal_rank < 1:
                            failures.append(
                                f"- {split}/{variant.name}/{case.case_id}: "
                                f"{case.query}；相关排名 {case.relevant_ranks}"
                            )
                print(f"{split} completed", flush=True)
            report += [
                "",
                "## 非满召回或首位未命中用例",
                "",
                *failures,
                "",
                "## 实验边界",
                "",
                "- 单份演示资料、AI初标，需人工复核；不能声称工厂诊断准确率。",
                "- 重排成功数不足时，该方案含回退结果，不能作为完整重排性能结论。",
                "- 分组按问题固定，同一章节可能跨组；不是跨文档泛化测试。",
                "- 只评估检索，不评估答案正确性、拒答能力或多轮诊断。",
                "- 延迟来自串行单次执行，受缓存与网络影响；不代表并发性能。",
                "- 文档中推荐问答切片保留为干扰项，不标为答案证据。",
            ]
            (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
            print(str(run_dir.resolve()), flush=True)
    except Exception as exc:
        save(
            "status.json",
            {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        if store is not None:
            store.client.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
