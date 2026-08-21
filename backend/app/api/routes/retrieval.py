from dataclasses import asdict
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import EmbeddingDependency, VectorStoreDependency
from app.api.routes.documents import require_document
from app.db.database import get_db
from app.models.document import DocumentStatus
from app.repositories.knowledge_base import get_knowledge_base
from app.schemas.response import ApiResponse
from app.schemas.retrieval import (
    IndexingResult,
    SemanticSearchItem,
    SemanticSearchRequest,
    SemanticSearchResult,
)
from app.services.document_indexing import (
    DocumentIndexingError,
    DocumentNotChunkedError,
    index_document,
    remove_document_index,
)
from app.services.embedding import EmbeddingError
from app.services.semantic_search import search_knowledge_base
from app.services.vector_store import VectorStoreError

router = APIRouter()
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "/documents/{document_id}/index",
    response_model=ApiResponse[IndexingResult],
    summary="生成或重建文档向量索引",
)
async def create_document_index(
    document_id: UUID,
    session: DatabaseSession,
    embedding_provider: EmbeddingDependency,
    vector_store: VectorStoreDependency,
) -> ApiResponse[IndexingResult]:
    document = await require_document(document_id, session)
    if document.status not in {DocumentStatus.CHUNKED, DocumentStatus.INDEXED}:
        raise HTTPException(status_code=409, detail=f"文档当前状态为 {document.status.value}")

    try:
        summary = await index_document(
            session,
            document,
            embedding_provider,
            vector_store,
        )
    except DocumentNotChunkedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DocumentIndexingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ApiResponse(
        data=IndexingResult(
            document_id=document.id,
            status=document.status,
            vector_count=summary.vector_count,
            model_name=summary.model_name,
            dimension=summary.dimension,
            elapsed_ms=summary.elapsed_ms,
        )
    )


@router.delete(
    "/documents/{document_id}/index",
    response_model=ApiResponse[None],
    summary="删除文档向量索引",
)
async def delete_document_index(
    document_id: UUID,
    session: DatabaseSession,
    vector_store: VectorStoreDependency,
) -> ApiResponse[None]:
    document = await require_document(document_id, session)
    if document.status == DocumentStatus.INDEXING:
        raise HTTPException(status_code=409, detail="文档正在建立索引")
    try:
        deleted_count = await remove_document_index(session, document, vector_store)
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ApiResponse(message=f"已解除 {deleted_count} 个向量关联", data=None)


@router.post(
    "/knowledge-bases/{knowledge_base_id}/search",
    response_model=ApiResponse[SemanticSearchResult],
    summary="在知识库中进行语义检索",
)
async def semantic_search(
    knowledge_base_id: UUID,
    payload: SemanticSearchRequest,
    session: DatabaseSession,
    embedding_provider: EmbeddingDependency,
    vector_store: VectorStoreDependency,
) -> ApiResponse[SemanticSearchResult]:
    if await get_knowledge_base(session, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    try:
        hits = await search_knowledge_base(
            session,
            knowledge_base_id,
            payload.query,
            payload.top_k,
            payload.score_threshold,
            embedding_provider,
            vector_store,
        )
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    items = [SemanticSearchItem(**asdict(hit)) for hit in hits]
    return ApiResponse(
        data=SemanticSearchResult(
            query=payload.query,
            items=items,
            total=len(items),
            model_name=embedding_provider.model_name,
        )
    )
