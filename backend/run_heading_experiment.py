"""Isolated development-only ablation; defaults to offline/local retrieval."""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from app import models  # noqa: F401
from app.core.config import settings
from app.db.base import Base
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.schemas.evaluation import RetrievalEvaluationRequest
from app.services.document_indexing import index_document
from app.services.embedding import get_embedding_provider
from app.services.reranker import get_reranker_provider
from app.services.retrieval_evaluation import evaluate_retrieval
from app.services.text_splitter import ChunkingConfig, build_document_chunks
from app.services.vector_store import QdrantVectorStore
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path, default=Path("evaluation_reports/20260908T082307374071Z")
    )
    parser.add_argument("--with-rerank", action="store_true")
    args = parser.parse_args()
    folder = Path("evaluation_reports") / (
        "heading-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    folder.mkdir(parents=True)

    def save(name, value):
        (folder / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    corpus = json.loads((args.baseline / "corpus.json").read_text(encoding="utf-8"))
    labels = json.loads((args.baseline / "labels.json").read_text(encoding="utf-8"))
    source = Path("sample_documents/空压机维修知识库.md").read_text(encoding="utf-8")
    drafts = build_document_chunks(
        [SimpleNamespace(page_number=1, content=source)], "manual.md", ChunkingConfig()
    )
    # Freeze chunk identity and labels; reject ambiguous or changed body boundaries.
    if len(drafts) != len(corpus):
        raise ValueError("Chunk count changed; label review required")
    treatment = []
    for old, new in zip(corpus, drafts, strict=True):
        if old["metadata"]["section_title"] != new.metadata["section_title"]:
            raise ValueError("Section order changed")
        body = old["content"].split("\n", 1)[1]
        if not new.content.endswith(body):
            raise ValueError("Body boundaries changed; not a heading-only comparison")
        treatment.append({**old, "content": new.content, "metadata": new.metadata})
    save("baseline.corpus.json", corpus)
    save("heading.corpus.json", treatment)
    save(
        "metadata.json",
        {
            "baseline": str(args.baseline),
            "with_rerank": args.with_rerank,
            "embedding": settings.embedding_model_name,
            "reranker": settings.reranker_model_name,
            "rrf_k": settings.rrf_k,
            "candidate_multiplier": settings.retrieval_candidate_multiplier,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "development_only": True,
            "chunk_count": len(corpus),
            "body_boundaries_unchanged": True,
        },
    )
    payload = RetrievalEvaluationRequest.model_validate(
        {
            "cases": [c for c in labels if c["split"] == "development"],
            "top_k": 5,
            "score_threshold": None,
        }
    )
    if not args.with_rerank:
        payload.variants = [v for v in payload.variants if not v.rerank]
    save("request.json", payload.model_dump(mode="json"))
    embedding = get_embedding_provider()
    results = {}
    try:
        embedding.embed_query("预热")
        for name, records in [("baseline", corpus), ("heading", treatment)]:
            db_path = (folder / f"{name}.db").resolve().as_posix()
            engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
            client = QdrantClient(path=str(folder / f"{name}.qdrant"))
            store = QdrantVectorStore(client, "experiment")
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                    kb = KnowledgeBase(name="heading-experiment")
                    session.add(kb)
                    await session.flush()
                    doc = Document(
                        id=UUID(records[0]["document_id"]),
                        knowledge_base_id=kb.id,
                        filename="空压机维修知识库.md",
                        storage_path="snapshot",
                        status=DocumentStatus.CHUNKED,
                    )
                    session.add(doc)
                    await session.flush()
                    for i, c in enumerate(records):
                        session.add(
                            DocumentChunk(
                                id=UUID(c["chunk_id"]),
                                document_id=doc.id,
                                chunk_index=i,
                                content=c["content"],
                                page_number=1,
                                extra_metadata=c["metadata"],
                            )
                        )
                    await session.commit()
                    await index_document(session, doc, embedding, store)
                    result = await evaluate_retrieval(
                        session,
                        knowledge_base_id=kb.id,
                        payload=payload,
                        embedding_provider=embedding,
                        vector_store=store,
                        reranker_provider=get_reranker_provider(),
                    )
                    results[name] = result
                    save(f"{name}.results.json", result.model_dump(mode="json"))
                    print(name + " completed", flush=True)
            finally:
                client.close()
                await engine.dispose()
        report = [
            "# 父标题单变量实验",
            "",
            "仅20条开发集；26条切片正文与标签不变，保留推荐问题章节。",
            "",
            "| 语料 | 方案 | Recall@5 | MRR | nDCG@5 | 重排成功 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, result in results.items():
            for v in result.variants:
                m = v.metrics
                report.append(
                    f"| {name} | {v.name} | {m.recall_at_k:.3f} | {m.mrr:.3f} | "
                    f"{m.ndcg_at_k:.3f} | {m.reranker_applied_count}/20 |"
                )
        report += ["", "## 逐题排名变化（旧 → 新）", ""]
        for old, new in zip(results["baseline"].variants, results["heading"].variants, strict=True):
            for a, b in zip(old.cases, new.cases, strict=True):
                if a.relevant_ranks != b.relevant_ranks:
                    report.append(
                        f"- {old.name}/{a.case_id}: {a.relevant_ranks} → {b.relevant_ranks}"
                    )
        report += ["", "AI初标、单文档小样本；不代表生产效果。未运行重排时不能推断重排后的排名。"]
        (folder / "report.md").write_text("\n".join(report), encoding="utf-8")
        print(folder.resolve(), flush=True)
    except Exception as exc:
        save("status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    asyncio.run(main())
