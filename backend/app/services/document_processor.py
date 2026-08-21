import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk, DocumentPage, DocumentStatus
from app.services.document_parser import DocumentParseError, parse_document

logger = logging.getLogger(__name__)


async def process_document(session: AsyncSession, document: Document) -> Document:
    """解析已落盘文档，并将按页文本保存到数据库。"""
    document.status = DocumentStatus.PARSING
    document.error_message = None
    await session.commit()

    try:
        parsed_pages = await asyncio.to_thread(parse_document, document.storage_path)
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
        await session.execute(delete(DocumentPage).where(DocumentPage.document_id == document.id))
        session.add_all(
            [
                DocumentPage(
                    document_id=document.id,
                    page_number=page.page_number,
                    content=page.content,
                    character_count=len(page.content),
                )
                for page in parsed_pages
            ]
        )
        document.page_count = len(parsed_pages)
        document.status = DocumentStatus.PARSED
        document.processed_at = datetime.now(UTC)
    except DocumentParseError as exc:
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)[:2000]
        document.processed_at = datetime.now(UTC)
    except Exception:
        logger.exception("Unexpected error while parsing document %s", document.id)
        document.status = DocumentStatus.FAILED
        document.error_message = "文档解析时发生内部错误"
        document.processed_at = datetime.now(UTC)

    await session.commit()
    await session.refresh(document)
    return document
