from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    EmbeddingDependency,
    LLMDependency,
    RerankerDependency,
    VectorStoreDependency,
)
from app.api.routes.documents import require_document
from app.core.config import settings
from app.db.database import get_db
from app.models.document import DocumentStatus
from app.repositories.knowledge_base import get_knowledge_base
from app.schemas.response import ApiResponse
from app.schemas.retrieval import (
    IndexingResult,
    LLMUsageRead,
    RagAnswerResult,
    RagAskRequest,
    RagSource,
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
from app.services.hybrid_search import retrieve_knowledge_base
from app.services.llm import LLMError
from app.services.rag_answering import answer_with_knowledge_base, prepare_knowledge_base_rag
from app.services.rag_streaming import stream_prepared_rag
from app.services.sse import ServerSentEventResponse, encode_sse, sse_response
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
    reranker_provider: RerankerDependency,
) -> ApiResponse[SemanticSearchResult]:
    if await get_knowledge_base(session, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    try:
        retrieval = await retrieve_knowledge_base(
            session,
            knowledge_base_id=knowledge_base_id,
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            mode=payload.retrieval_mode,
            rerank=payload.rerank,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            reranker_provider=reranker_provider,
        )
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    items = [SemanticSearchItem(**asdict(hit)) for hit in retrieval.items]
    return ApiResponse(
        data=SemanticSearchResult(
            query=payload.query,
            items=items,
            total=len(items),
            model_name=embedding_provider.model_name,
            retrieval_mode=retrieval.mode,
            reranker_applied=retrieval.reranker_applied,
            reranker_model=retrieval.reranker_model,
        )
    )


@router.post(
    "/knowledge-bases/{knowledge_base_id}/ask",
    response_model=ApiResponse[RagAnswerResult],
    summary="基于知识库生成故障诊断答案",
)
async def ask_knowledge_base(
    knowledge_base_id: UUID,
    payload: RagAskRequest,
    session: DatabaseSession,
    embedding_provider: EmbeddingDependency,
    vector_store: VectorStoreDependency,
    llm_provider: LLMDependency,
    reranker_provider: RerankerDependency,
) -> ApiResponse[RagAnswerResult]:
    if await get_knowledge_base(session, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    try:
        result = await answer_with_knowledge_base(
            session,
            knowledge_base_id=knowledge_base_id,
            question=payload.question,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            max_context_chars=settings.rag_max_context_chars,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            llm_provider=llm_provider,
            reranker_provider=reranker_provider,
            retrieval_mode=payload.retrieval_mode,
            rerank=payload.rerank,
        )
    except (EmbeddingError, VectorStoreError, LLMError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ApiResponse(
        data=RagAnswerResult(
            question=result.question,
            answer=result.answer,
            sources=[RagSource(**asdict(source)) for source in result.sources],
            retrieved_count=len(result.sources),
            embedding_model=result.embedding_model,
            llm_model=result.llm_model,
            llm_called=result.llm_called,
            usage=LLMUsageRead(**asdict(result.usage)),
            retrieval_mode=result.retrieval_mode,
            reranker_applied=result.reranker_applied,
            reranker_model=result.reranker_model,
        )
    )


@router.post(
    "/knowledge-bases/{knowledge_base_id}/ask/stream",
    response_class=ServerSentEventResponse,
    summary="流式生成知识库故障诊断答案",
)
async def stream_knowledge_base_answer(
    knowledge_base_id: UUID,
    payload: RagAskRequest,
    session: DatabaseSession,
    embedding_provider: EmbeddingDependency,
    vector_store: VectorStoreDependency,
    llm_provider: LLMDependency,
    reranker_provider: RerankerDependency,
) -> ServerSentEventResponse:
    if await get_knowledge_base(session, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    async def event_stream() -> AsyncIterator[str]:
        yield encode_sse(
            "retrieval_started",
            {"question": payload.question, "top_k": payload.top_k},
        )
        try:
            prepared = await prepare_knowledge_base_rag(
                session,
                knowledge_base_id=knowledge_base_id,
                question=payload.question,
                top_k=payload.top_k,
                score_threshold=payload.score_threshold,
                max_context_chars=settings.rag_max_context_chars,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                reranker_provider=reranker_provider,
                retrieval_mode=payload.retrieval_mode,
                rerank=payload.rerank,
            )
            async for update in stream_prepared_rag(prepared, llm_provider):
                yield encode_sse(update.event, update.data)
        except (EmbeddingError, VectorStoreError, LLMError) as exc:
            yield encode_sse("error", {"message": str(exc)})
        except Exception:
            yield encode_sse("error", {"message": "流式诊断生成失败"})

    return sse_response(event_stream())
