from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import VectorStoreDependency
from app.db.database import get_db
from app.models.document import Document, DocumentStatus
from app.repositories import document as repository
from app.repositories.knowledge_base import get_knowledge_base
from app.schemas.document import (
    ChunkingRequest,
    ChunkingResult,
    DocumentChunkList,
    DocumentChunkRead,
    DocumentList,
    DocumentPageList,
    DocumentPageRead,
    DocumentRead,
)
from app.schemas.response import ApiResponse
from app.services.document_chunking import (
    DocumentChunkingError,
    chunk_document,
    remove_document_chunks,
)
from app.services.document_indexing import remove_document_index
from app.services.document_processor import process_document
from app.services.document_storage import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    delete_stored_file,
    save_upload,
)
from app.services.text_splitter import ChunkingConfig
from app.services.vector_store import VectorStoreError

router = APIRouter()
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


async def require_document(document_id: UUID, session: AsyncSession) -> Document:
    document = await repository.get_document(session, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=ApiResponse[DocumentRead],
    status_code=status.HTTP_201_CREATED,
    summary="上传并解析文档",
)
async def upload_document(
    knowledge_base_id: UUID,
    session: DatabaseSession,
    file: Annotated[UploadFile, File(description="PDF、DOCX、TXT或Markdown文档")],
) -> ApiResponse[DocumentRead]:
    if await get_knowledge_base(session, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    try:
        stored = await save_upload(file, knowledge_base_id)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = await repository.get_document_by_hash(
        session,
        knowledge_base_id,
        stored.file_hash,
    )
    if existing is not None:
        delete_stored_file(stored.storage_path)
        raise HTTPException(status_code=409, detail="该文件已存在于当前知识库")

    try:
        document = await repository.create_document(
            session,
            knowledge_base_id=knowledge_base_id,
            filename=stored.original_filename,
            storage_path=str(stored.storage_path),
            mime_type=stored.mime_type,
            size_bytes=stored.size_bytes,
            file_hash=stored.file_hash,
        )
    except IntegrityError as exc:
        await session.rollback()
        delete_stored_file(stored.storage_path)
        raise HTTPException(status_code=409, detail="该文件已存在于当前知识库") from exc
    except Exception:
        delete_stored_file(stored.storage_path)
        raise

    document = await process_document(session, document)
    return ApiResponse(data=DocumentRead.model_validate(document))


@router.get(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=ApiResponse[DocumentList],
    summary="分页查询知识库文档",
)
async def get_documents(
    knowledge_base_id: UUID,
    session: DatabaseSession,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    document_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
) -> ApiResponse[DocumentList]:
    if await get_knowledge_base(session, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    items, total = await repository.list_documents(
        session,
        knowledge_base_id,
        page,
        page_size,
        document_status,
    )
    return ApiResponse(
        data=DocumentList(
            items=[DocumentRead.model_validate(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.get(
    "/documents/{document_id}",
    response_model=ApiResponse[DocumentRead],
    summary="查询文档详情",
)
async def get_document(
    document_id: UUID,
    session: DatabaseSession,
) -> ApiResponse[DocumentRead]:
    document = await require_document(document_id, session)
    return ApiResponse(data=DocumentRead.model_validate(document))


@router.get(
    "/documents/{document_id}/content",
    response_model=ApiResponse[DocumentPageList],
    summary="查看文档解析文本",
)
async def get_document_content(
    document_id: UUID,
    session: DatabaseSession,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ApiResponse[DocumentPageList]:
    document = await require_document(document_id, session)
    if document.status not in {
        DocumentStatus.PARSED,
        DocumentStatus.CHUNKED,
        DocumentStatus.INDEXED,
    }:
        raise HTTPException(status_code=409, detail=f"文档当前状态为 {document.status.value}")

    items, total = await repository.list_document_pages(session, document_id, page, page_size)
    return ApiResponse(
        data=DocumentPageList(
            items=[DocumentPageRead.model_validate(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.post(
    "/documents/{document_id}/chunks",
    response_model=ApiResponse[ChunkingResult],
    summary="生成或重建文档切片",
)
async def create_document_chunks(
    document_id: UUID,
    payload: ChunkingRequest,
    session: DatabaseSession,
    vector_store: VectorStoreDependency,
) -> ApiResponse[ChunkingResult]:
    document = await require_document(document_id, session)
    if document.status in {
        DocumentStatus.PENDING,
        DocumentStatus.PARSING,
        DocumentStatus.CHUNKING,
        DocumentStatus.INDEXING,
    }:
        raise HTTPException(status_code=409, detail=f"文档当前状态为 {document.status.value}")

    if document.status == DocumentStatus.INDEXED:
        try:
            await remove_document_index(session, document, vector_store)
        except VectorStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    config = ChunkingConfig(
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        min_chunk_size=payload.min_chunk_size,
    )
    try:
        summary = await chunk_document(session, document, config)
    except DocumentChunkingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ApiResponse(
        data=ChunkingResult(
            document_id=document.id,
            status=document.status,
            chunk_count=summary.chunk_count,
            average_characters=summary.average_characters,
            max_characters=summary.max_characters,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
        )
    )


@router.get(
    "/documents/{document_id}/chunks",
    response_model=ApiResponse[DocumentChunkList],
    summary="分页查看文档切片",
)
async def get_document_chunks(
    document_id: UUID,
    session: DatabaseSession,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ApiResponse[DocumentChunkList]:
    await require_document(document_id, session)
    items, total = await repository.list_document_chunks(session, document_id, page, page_size)
    return ApiResponse(
        data=DocumentChunkList(
            items=[DocumentChunkRead.model_validate(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.delete(
    "/documents/{document_id}/chunks",
    response_model=ApiResponse[None],
    summary="删除文档切片",
)
async def delete_document_chunks(
    document_id: UUID,
    session: DatabaseSession,
    vector_store: VectorStoreDependency,
) -> ApiResponse[None]:
    document = await require_document(document_id, session)
    if document.page_count == 0:
        raise HTTPException(status_code=409, detail="文档没有可恢复的解析文本")
    if document.status == DocumentStatus.INDEXED:
        try:
            await remove_document_index(session, document, vector_store)
        except VectorStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    deleted_count = await remove_document_chunks(session, document)
    return ApiResponse(message=f"已删除 {deleted_count} 个切片", data=None)


@router.delete(
    "/documents/{document_id}",
    response_model=ApiResponse[None],
    summary="删除文档",
)
async def delete_document(
    document_id: UUID,
    session: DatabaseSession,
    vector_store: VectorStoreDependency,
) -> ApiResponse[None]:
    document = await require_document(document_id, session)
    if document.status == DocumentStatus.INDEXED:
        try:
            await remove_document_index(session, document, vector_store)
        except VectorStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    storage_path = document.storage_path
    await repository.delete_document(session, document)
    delete_stored_file(storage_path)
    return ApiResponse(message="文档已删除", data=None)
