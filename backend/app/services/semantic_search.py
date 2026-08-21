import asyncio
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import document as repository
from app.services.embedding import EmbeddingProvider
from app.services.vector_store import QdrantVectorStore


@dataclass(frozen=True, slots=True)
class SemanticSearchHit:
    """向量分数与SQL原文组合后的检索结果。"""

    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    score: float
    page_number: int | None
    section_title: str | None


async def search_knowledge_base(
    session: AsyncSession,
    knowledge_base_id: UUID,
    query: str,
    top_k: int,
    score_threshold: float | None,
    embedding_provider: EmbeddingProvider,
    vector_store: QdrantVectorStore,
) -> list[SemanticSearchHit]:
    """向量召回后按chunk_id从SQL批量读取可靠原文。"""
    query_vector = await asyncio.to_thread(embedding_provider.embed_query, query)
    vector_hits = await asyncio.to_thread(
        vector_store.search,
        query_vector,
        str(knowledge_base_id),
        top_k * 2,
        score_threshold,
    )

    ordered_ids: list[UUID] = []
    hit_by_id = {}
    for hit in vector_hits:
        raw_chunk_id = hit.payload.get("chunk_id", hit.point_id)
        try:
            chunk_id = UUID(str(raw_chunk_id))
        except ValueError:
            continue
        ordered_ids.append(chunk_id)
        hit_by_id[chunk_id] = hit

    records = await repository.get_chunks_with_documents(session, ordered_ids)
    results: list[SemanticSearchHit] = []
    for chunk_id in ordered_ids:
        record = records.get(chunk_id)
        if record is None:
            continue
        chunk, document = record
        if document.knowledge_base_id != knowledge_base_id:
            continue
        hit = hit_by_id[chunk_id]
        results.append(
            SemanticSearchHit(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                content=chunk.content,
                score=round(hit.score, 6),
                page_number=chunk.page_number,
                section_title=chunk.extra_metadata.get("section_title"),
            )
        )
        if len(results) == top_k:
            break
    return results
