from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.embedding import EmbeddingProvider
from app.services.keyword_search import KeywordSearchHit
from app.services.reranker import RerankerError, RerankerProvider
from app.services.semantic_search import (
    SemanticSearchHit,
    search_hybrid_knowledge_base,
    search_knowledge_base,
)
from app.services.sparse_embedding import SparseEmbeddingProvider
from app.services.vector_store import QdrantVectorStore


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    items: list[SemanticSearchHit]
    mode: RetrievalMode
    reranker_applied: bool
    reranker_model: str | None


async def retrieve_knowledge_base(
    session: AsyncSession,
    *,
    knowledge_base_id: UUID,
    query: str,
    top_k: int,
    score_threshold: float | None,
    mode: RetrievalMode,
    rerank: bool,
    embedding_provider: EmbeddingProvider,
    sparse_embedding_provider: SparseEmbeddingProvider,
    vector_store: QdrantVectorStore,
    reranker_provider: RerankerProvider,
) -> RetrievalResult:
    """召回候选，按配置执行RRF融合与Cross-Encoder重排。"""
    candidate_limit = max(top_k, top_k * settings.retrieval_candidate_multiplier)
    if mode == RetrievalMode.HYBRID:
        candidates = await search_hybrid_knowledge_base(
            session,
            knowledge_base_id,
            query,
            candidate_limit,
            score_threshold,
            embedding_provider,
            sparse_embedding_provider,
            vector_store,
        )
    else:
        candidates = await search_knowledge_base(
            session,
            knowledge_base_id,
            query,
            candidate_limit,
            score_threshold,
            embedding_provider,
            vector_store,
        )

    reranker_applied = False
    reranker_requested = rerank and settings.reranker_enabled
    reranker_model = reranker_provider.model_name if reranker_requested else None
    if reranker_requested and candidates:
        try:
            reranked = await reranker_provider.rerank(
                query,
                [candidate.content for candidate in candidates],
                top_k,
            )
            if reranked:
                candidates = [
                    _with_rerank_score(candidates[result.index], result.score)
                    for result in reranked
                ]
                reranker_applied = True
        except RerankerError:
            reranker_applied = False

    return RetrievalResult(
        items=candidates[:top_k],
        mode=mode,
        reranker_applied=reranker_applied,
        reranker_model=reranker_model,
    )


def reciprocal_rank_fusion(
    vector_hits: list[SemanticSearchHit],
    keyword_hits: list[KeywordSearchHit],
    *,
    rrf_k: int = 60,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> list[SemanticSearchHit]:
    """融合两个排名列表，避免直接比较量纲不同的原始分数。"""
    candidates: dict[UUID, dict[str, object]] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        candidates[hit.chunk_id] = {
            "base": hit,
            "vector_score": hit.vector_score or hit.score,
            "keyword_score": None,
            "rrf": vector_weight / (rrf_k + rank),
            "sources": ["vector"],
        }

    for rank, hit in enumerate(keyword_hits, start=1):
        candidate = candidates.setdefault(
            hit.chunk_id,
            {
                "base": _from_keyword_hit(hit),
                "vector_score": None,
                "keyword_score": None,
                "rrf": 0.0,
                "sources": [],
            },
        )
        candidate["keyword_score"] = hit.score
        candidate["rrf"] = float(candidate["rrf"]) + keyword_weight / (rrf_k + rank)
        sources = candidate["sources"]
        if isinstance(sources, list) and "keyword" not in sources:
            sources.append("keyword")

    ranked = sorted(candidates.values(), key=lambda item: float(item["rrf"]), reverse=True)
    maximum_score = float(ranked[0]["rrf"]) if ranked else 1.0
    return [_with_fusion_metadata(candidate, maximum_score) for candidate in ranked]


def _from_keyword_hit(hit: KeywordSearchHit) -> SemanticSearchHit:
    return SemanticSearchHit(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        filename=hit.filename,
        content=hit.content,
        score=hit.score,
        page_number=hit.page_number,
        section_title=hit.section_title,
        keyword_score=hit.score,
        retrieval_sources=["keyword"],
    )


def _with_fusion_metadata(
    candidate: dict[str, object],
    maximum_score: float,
) -> SemanticSearchHit:
    base = candidate["base"]
    if not isinstance(base, SemanticSearchHit):
        raise TypeError("RRF候选数据不正确")
    fusion_score = float(candidate["rrf"]) / maximum_score if maximum_score else 0.0
    sources = candidate["sources"]
    return SemanticSearchHit(
        chunk_id=base.chunk_id,
        document_id=base.document_id,
        filename=base.filename,
        content=base.content,
        score=round(fusion_score, 6),
        page_number=base.page_number,
        section_title=base.section_title,
        vector_score=_optional_float(candidate["vector_score"]),
        keyword_score=_optional_float(candidate["keyword_score"]),
        fusion_score=round(fusion_score, 6),
        retrieval_sources=list(sources) if isinstance(sources, list) else [],
    )


def _with_rerank_score(hit: SemanticSearchHit, score: float) -> SemanticSearchHit:
    return SemanticSearchHit(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        filename=hit.filename,
        content=hit.content,
        score=round(score, 6),
        page_number=hit.page_number,
        section_title=hit.section_title,
        vector_score=hit.vector_score,
        keyword_score=hit.keyword_score,
        fusion_score=hit.fusion_score,
        rerank_score=round(score, 6),
        retrieval_sources=hit.retrieval_sources,
    )


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
