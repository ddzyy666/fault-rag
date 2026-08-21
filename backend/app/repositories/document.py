from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk, DocumentPage, DocumentStatus


async def get_document(session: AsyncSession, document_id: UUID) -> Document | None:
    """按主键查询文档。"""
    return await session.get(Document, document_id)


async def get_document_by_hash(
    session: AsyncSession,
    knowledge_base_id: UUID,
    file_hash: str,
) -> Document | None:
    """在同一知识库内按文件哈希查询重复文档。"""
    statement = select(Document).where(
        Document.knowledge_base_id == knowledge_base_id,
        Document.file_hash == file_hash,
    )
    return await session.scalar(statement)


async def list_documents(
    session: AsyncSession,
    knowledge_base_id: UUID,
    page: int,
    page_size: int,
    document_status: DocumentStatus | None = None,
) -> tuple[list[Document], int]:
    """分页查询指定知识库中的文档。"""
    conditions = [Document.knowledge_base_id == knowledge_base_id]
    if document_status is not None:
        conditions.append(Document.status == document_status)

    total = await session.scalar(select(func.count()).select_from(Document).where(*conditions))
    statement = (
        select(Document)
        .where(*conditions)
        .order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = list((await session.scalars(statement)).all())
    return items, total or 0


async def create_document(
    session: AsyncSession,
    *,
    knowledge_base_id: UUID,
    filename: str,
    storage_path: str,
    mime_type: str | None,
    size_bytes: int,
    file_hash: str,
) -> Document:
    """创建待解析的文档记录。"""
    document = Document(
        knowledge_base_id=knowledge_base_id,
        filename=filename,
        storage_path=storage_path,
        mime_type=mime_type,
        size_bytes=size_bytes,
        file_hash=file_hash,
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def list_document_pages(
    session: AsyncSession,
    document_id: UUID,
    page: int,
    page_size: int,
) -> tuple[list[DocumentPage], int]:
    """分页读取文档的原始页文本。"""
    condition = DocumentPage.document_id == document_id
    total = await session.scalar(select(func.count()).select_from(DocumentPage).where(condition))
    statement = (
        select(DocumentPage)
        .where(condition)
        .order_by(DocumentPage.page_number)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = list((await session.scalars(statement)).all())
    return items, total or 0


async def get_all_document_pages(
    session: AsyncSession,
    document_id: UUID,
) -> list[DocumentPage]:
    """按页码读取分块所需的全部原始页面。"""
    statement = (
        select(DocumentPage)
        .where(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number)
    )
    return list((await session.scalars(statement)).all())


async def replace_document_chunks(
    session: AsyncSession,
    document_id: UUID,
    chunks: list[DocumentChunk],
) -> None:
    """清除旧切片并写入本次生成结果。"""
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    session.add_all(chunks)


async def list_document_chunks(
    session: AsyncSession,
    document_id: UUID,
    page: int,
    page_size: int,
) -> tuple[list[DocumentChunk], int]:
    """分页读取文档切片。"""
    condition = DocumentChunk.document_id == document_id
    total = await session.scalar(select(func.count()).select_from(DocumentChunk).where(condition))
    statement = (
        select(DocumentChunk)
        .where(condition)
        .order_by(DocumentChunk.chunk_index)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = list((await session.scalars(statement)).all())
    return items, total or 0


async def clear_document_chunks(session: AsyncSession, document_id: UUID) -> int:
    """删除指定文档的所有切片并返回删除数量。"""
    result = await session.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    return result.rowcount or 0  # type: ignore[attr-defined]


async def delete_document(session: AsyncSession, document: Document) -> None:
    """删除文档记录，数据库外键会级联删除页面和切片。"""
    await session.delete(document)
    await session.commit()
