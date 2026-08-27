import math
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.document import get_knowledge_base_chunks_with_documents
from app.schemas.evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluationCaseResult,
    RetrievalEvaluationMetrics,
    RetrievalEvaluationRequest,
    RetrievalEvaluationResult,
    RetrievalEvaluationVariantResult,
)
from app.services.embedding import EmbeddingProvider
from app.services.hybrid_search import retrieve_knowledge_base
from app.services.reranker import RerankerProvider
from app.services.vector_store import QdrantVectorStore


class EvaluationDatasetError(ValueError):
    """评估集包含不属于当前知识库或尚未索引的切片。"""


@dataclass(frozen=True, slots=True)
class CaseScores:
    relevant_ranks: list[int]
    hit: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def calculate_case_scores(
    retrieved_chunk_ids: list[UUID],
    relevant_chunk_ids: list[UUID],
    top_k: int,
) -> CaseScores:
    """使用二元相关性标注计算一条问题的检索指标。"""
    relevant = set(relevant_chunk_ids)
    top_results = retrieved_chunk_ids[:top_k]
    relevant_ranks = [
        rank for rank, chunk_id in enumerate(top_results, start=1) if chunk_id in relevant
    ]
    hit_count = len(relevant_ranks)
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_hits = min(len(relevant), top_k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return CaseScores(
        relevant_ranks=relevant_ranks,
        hit=bool(relevant_ranks),
        precision_at_k=hit_count / top_k,
        recall_at_k=hit_count / len(relevant),
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=dcg / ideal_dcg if ideal_dcg else 0.0,
    )


async def evaluate_retrieval(
    session: AsyncSession,
    *,
    knowledge_base_id: UUID,
    payload: RetrievalEvaluationRequest,
    embedding_provider: EmbeddingProvider,
    vector_store: QdrantVectorStore,
    reranker_provider: RerankerProvider,
) -> RetrievalEvaluationResult:
    """在同一评估集上运行多组检索配置并计算可比较指标。"""
    await _validate_ground_truth(session, knowledge_base_id, payload.cases)
    variant_results: list[RetrievalEvaluationVariantResult] = []

    for variant in payload.variants:
        case_results: list[RetrievalEvaluationCaseResult] = []
        for case in payload.cases:
            started_at = perf_counter()
            retrieval = await retrieve_knowledge_base(
                session,
                knowledge_base_id=knowledge_base_id,
                query=case.query,
                top_k=payload.top_k,
                score_threshold=payload.score_threshold,
                mode=variant.retrieval_mode,
                rerank=variant.rerank,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                reranker_provider=reranker_provider,
            )
            latency_ms = round((perf_counter() - started_at) * 1000)
            retrieved_ids = [item.chunk_id for item in retrieval.items]
            scores = calculate_case_scores(
                retrieved_ids,
                case.relevant_chunk_ids,
                payload.top_k,
            )
            case_results.append(
                RetrievalEvaluationCaseResult(
                    case_id=case.case_id,
                    query=case.query,
                    relevant_chunk_ids=case.relevant_chunk_ids,
                    retrieved_chunk_ids=retrieved_ids,
                    relevant_ranks=scores.relevant_ranks,
                    hit=scores.hit,
                    precision_at_k=round(scores.precision_at_k, 6),
                    recall_at_k=round(scores.recall_at_k, 6),
                    reciprocal_rank=round(scores.reciprocal_rank, 6),
                    ndcg_at_k=round(scores.ndcg_at_k, 6),
                    latency_ms=latency_ms,
                    reranker_applied=retrieval.reranker_applied,
                )
            )

        variant_results.append(
            RetrievalEvaluationVariantResult(
                name=variant.name,
                retrieval_mode=variant.retrieval_mode,
                rerank=variant.rerank,
                metrics=_aggregate_metrics(case_results),
                cases=case_results,
            )
        )

    best_variant = max(
        variant_results,
        key=lambda result: (
            result.metrics.mrr,
            result.metrics.recall_at_k,
            result.metrics.ndcg_at_k,
            -result.metrics.average_latency_ms,
        ),
    ).name
    return RetrievalEvaluationResult(
        knowledge_base_id=knowledge_base_id,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        best_variant=best_variant,
        variants=variant_results,
    )


async def _validate_ground_truth(
    session: AsyncSession,
    knowledge_base_id: UUID,
    cases: list[RetrievalEvaluationCase],
) -> None:
    indexed_records = await get_knowledge_base_chunks_with_documents(
        session,
        knowledge_base_id,
    )
    valid_chunk_ids = {chunk.id for chunk, _ in indexed_records}
    invalid_labels = [
        f"{case.case_id}: {chunk_id}"
        for case in cases
        for chunk_id in case.relevant_chunk_ids
        if chunk_id not in valid_chunk_ids
    ]
    if invalid_labels:
        details = "; ".join(invalid_labels[:5])
        raise EvaluationDatasetError(f"相关切片不属于当前知识库或尚未索引：{details}")


def _aggregate_metrics(
    cases: list[RetrievalEvaluationCaseResult],
) -> RetrievalEvaluationMetrics:
    latencies = sorted(case.latency_ms for case in cases)
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    return RetrievalEvaluationMetrics(
        case_count=len(cases),
        hit_rate_at_k=round(mean(case.hit for case in cases), 6),
        precision_at_k=round(mean(case.precision_at_k for case in cases), 6),
        recall_at_k=round(mean(case.recall_at_k for case in cases), 6),
        mrr=round(mean(case.reciprocal_rank for case in cases), 6),
        ndcg_at_k=round(mean(case.ndcg_at_k for case in cases), 6),
        average_latency_ms=round(mean(latencies), 2),
        p95_latency_ms=latencies[p95_index],
        reranker_applied_count=sum(case.reranker_applied for case in cases),
    )
