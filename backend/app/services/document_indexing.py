import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentStatus
from app.repositories import document as repository
from app.services.embedding import EmbeddingProvider
from app.services.vector_store import QdrantVectorStore, VectorPoint


class DocumentIndexingError(RuntimeError):
    """文档向量化或索引同步失败。"""


class DocumentNotChunkedError(DocumentIndexingError):
    """文档没有可以建立索引的切片。"""


@dataclass(frozen=True, slots=True)
class IndexingSummary:
    """文档索引操作统计。"""

    vector_count: int
    model_name: str
    dimension: int
    elapsed_ms: int


async def index_document(
    session: AsyncSession,
    document: Document,
    embedding_provider: EmbeddingProvider,
    vector_store: QdrantVectorStore,
) -> IndexingSummary:
    """批量生成向量、写入Qdrant并同步SQL索引状态。"""
    chunks = await repository.get_all_document_chunks(session, document.id)
    if not chunks:
        raise DocumentNotChunkedError("文档尚未生成文本切片")

    document_id = document.id
    original_status = document.status
    document.status = DocumentStatus.INDEXING
    document.error_message = None
    await session.commit()

    started_at = perf_counter()
    vectors_replaced = False
    try:
        vectors = await asyncio.to_thread(
            embedding_provider.embed_documents,
            [chunk.content for chunk in chunks],
        )
        if len(vectors) != len(chunks):
            raise DocumentIndexingError("Embedding数量与文档切片数量不一致")

        await asyncio.to_thread(
            vector_store.ensure_collection,
            embedding_provider.dimension,
        )
        await asyncio.to_thread(vector_store.delete_document, str(document_id))
        vectors_replaced = True

        points = [
            VectorPoint(
                point_id=str(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": str(chunk.id),
                    "document_id": str(document_id),
                    "knowledge_base_id": str(document.knowledge_base_id),
                    "page_number": chunk.page_number,
                    "section_title": chunk.extra_metadata.get("section_title"),
                    "filename": document.filename,
                    "embedding_model": embedding_provider.model_name,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await asyncio.to_thread(vector_store.upsert, points)

        for chunk in chunks:
            chunk.vector_id = str(chunk.id)

        document.extra_metadata = {
            **document.extra_metadata,
            "vector_index": {
                "model": embedding_provider.model_name,
                "dimension": embedding_provider.dimension,
                "vector_count": len(chunks),
                "indexed_at": datetime.now(UTC).isoformat(),
            },
        }
        document.status = DocumentStatus.INDEXED
        document.error_message = None
        await session.commit()
    except Exception as exc:
        await session.rollback()
        if vectors_replaced:
            try:
                await asyncio.to_thread(vector_store.delete_document, str(document_id))
            except Exception:
                pass

        current_document = await repository.get_document(session, document_id)
        if current_document is not None:
            current_chunks = await repository.get_all_document_chunks(session, document_id)
            if vectors_replaced:
                for chunk in current_chunks:
                    chunk.vector_id = None
                current_document.status = DocumentStatus.CHUNKED
            else:
                current_document.status = original_status
            current_document.error_message = "文档向量索引失败"
            await session.commit()

        if isinstance(exc, DocumentIndexingError):
            raise
        raise DocumentIndexingError("文档向量索引失败") from exc

    return IndexingSummary(
        vector_count=len(chunks),
        model_name=embedding_provider.model_name,
        dimension=embedding_provider.dimension,
        elapsed_ms=round((perf_counter() - started_at) * 1000),
    )


async def remove_document_index(
    session: AsyncSession,
    document: Document,
    vector_store: QdrantVectorStore,
) -> int:
    """先删除Qdrant向量，再清理SQL关联并恢复文档状态。"""
    chunks = await repository.get_all_document_chunks(session, document.id)
    indexed_count = sum(chunk.vector_id is not None for chunk in chunks)
    await asyncio.to_thread(vector_store.delete_document, str(document.id))

    for chunk in chunks:
        chunk.vector_id = None
    metadata = dict(document.extra_metadata)
    metadata.pop("vector_index", None)
    document.extra_metadata = metadata
    document.status = DocumentStatus.CHUNKED if chunks else DocumentStatus.PARSED
    document.error_message = None
    await session.commit()
    return indexed_count
