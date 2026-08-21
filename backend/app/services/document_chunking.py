import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk, DocumentStatus
from app.repositories import document as repository
from app.services.text_splitter import ChunkingConfig, build_document_chunks

logger = logging.getLogger(__name__)


class DocumentChunkingError(ValueError):
    """当前文档没有可用于分块的解析文本。"""


@dataclass(frozen=True, slots=True)
class ChunkingSummary:
    """一次分块操作的统计结果。"""

    chunk_count: int
    average_characters: float
    max_characters: int


async def chunk_document(
    session: AsyncSession,
    document: Document,
    config: ChunkingConfig,
) -> ChunkingSummary:
    """从DocumentPage生成并原子替换DocumentChunk。"""
    pages = await repository.get_all_document_pages(session, document.id)
    if not pages:
        raise DocumentChunkingError("文档没有可用于分块的解析文本")

    document.status = DocumentStatus.CHUNKING
    document.error_message = None
    await session.commit()

    try:
        drafts = await asyncio.to_thread(
            build_document_chunks,
            pages,
            document.filename,
            config,
        )
        if not drafts:
            raise DocumentChunkingError("文档解析文本为空，无法生成切片")

        chunks = [
            DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=draft.content,
                page_number=draft.page_number,
                token_count=draft.token_count,
                extra_metadata=draft.metadata,
            )
            for index, draft in enumerate(drafts)
        ]
        await repository.replace_document_chunks(session, document.id, chunks)
        document.status = DocumentStatus.CHUNKED
        await session.commit()
    except DocumentChunkingError:
        await session.rollback()
        document.status = DocumentStatus.FAILED
        document.error_message = "文本分块失败"
        await session.commit()
        raise
    except Exception as exc:
        await session.rollback()
        logger.exception("Unexpected error while chunking document %s", document.id)
        document.status = DocumentStatus.FAILED
        document.error_message = "文本分块时发生内部错误"
        await session.commit()
        raise DocumentChunkingError("文本分块时发生内部错误") from exc

    lengths = [len(chunk.content) for chunk in chunks]
    return ChunkingSummary(
        chunk_count=len(chunks),
        average_characters=round(sum(lengths) / len(lengths), 2),
        max_characters=max(lengths),
    )


async def remove_document_chunks(session: AsyncSession, document: Document) -> int:
    """清除切片，并把文档恢复到已解析状态。"""
    deleted_count = await repository.clear_document_chunks(session, document.id)
    document.status = DocumentStatus.PARSED
    document.error_message = None
    await session.commit()
    return deleted_count
