from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    EmbeddingDependency,
    LLMDependency,
    RerankerDependency,
    SparseEmbeddingDependency,
    VectorStoreDependency,
)
from app.core.config import settings
from app.db.database import get_db
from app.repositories import conversation as repository
from app.repositories.knowledge_base import get_knowledge_base
from app.schemas.conversation import (
    ConversationAskRequest,
    ConversationCreate,
    ConversationList,
    ConversationRead,
    ConversationReply,
    ConversationUpdate,
    MessageList,
    MessageRead,
)
from app.schemas.response import ApiResponse
from app.schemas.retrieval import LLMUsageRead, RagSource
from app.services.conversation_diagnosis import (
    diagnose_in_conversation,
    persist_conversation_answer,
    prepare_conversation_context,
)
from app.services.embedding import EmbeddingError
from app.services.llm import LLMError
from app.services.rag_answering import prepare_knowledge_base_rag
from app.services.rag_streaming import stream_prepared_rag
from app.services.sse import ServerSentEventResponse, encode_sse, sse_response
from app.services.vector_store import VectorStoreError

router = APIRouter(prefix="/conversations")
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


async def require_conversation(conversation_id: UUID, session: AsyncSession):
    conversation = await repository.get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="诊断会话不存在")
    return conversation


@router.post(
    "",
    response_model=ApiResponse[ConversationRead],
    status_code=status.HTTP_201_CREATED,
    summary="创建诊断会话",
)
async def create_conversation(
    payload: ConversationCreate,
    session: DatabaseSession,
) -> ApiResponse[ConversationRead]:
    if await get_knowledge_base(session, payload.knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    conversation = await repository.create_conversation(
        session,
        knowledge_base_id=payload.knowledge_base_id,
        title=payload.title,
    )
    return ApiResponse(data=ConversationRead.model_validate(conversation))


@router.get(
    "",
    response_model=ApiResponse[ConversationList],
    summary="分页查询诊断会话",
)
async def get_conversations(
    session: DatabaseSession,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    knowledge_base_id: UUID | None = None,
) -> ApiResponse[ConversationList]:
    items, total = await repository.list_conversations(
        session,
        page=page,
        page_size=page_size,
        knowledge_base_id=knowledge_base_id,
    )
    return ApiResponse(
        data=ConversationList(
            items=[ConversationRead.model_validate(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.get(
    "/{conversation_id}",
    response_model=ApiResponse[ConversationRead],
    summary="查询诊断会话详情",
)
async def get_conversation(
    conversation_id: UUID,
    session: DatabaseSession,
) -> ApiResponse[ConversationRead]:
    conversation = await require_conversation(conversation_id, session)
    return ApiResponse(data=ConversationRead.model_validate(conversation))


@router.patch(
    "/{conversation_id}",
    response_model=ApiResponse[ConversationRead],
    summary="修改诊断会话标题",
)
async def update_conversation(
    conversation_id: UUID,
    payload: ConversationUpdate,
    session: DatabaseSession,
) -> ApiResponse[ConversationRead]:
    conversation = await require_conversation(conversation_id, session)
    conversation = await repository.update_conversation_title(
        session,
        conversation,
        payload.title,
    )
    return ApiResponse(data=ConversationRead.model_validate(conversation))


@router.delete(
    "/{conversation_id}",
    response_model=ApiResponse[None],
    summary="删除诊断会话",
)
async def delete_conversation(
    conversation_id: UUID,
    session: DatabaseSession,
) -> ApiResponse[None]:
    conversation = await require_conversation(conversation_id, session)
    await repository.delete_conversation(session, conversation)
    return ApiResponse(message="诊断会话已删除", data=None)


@router.get(
    "/{conversation_id}/messages",
    response_model=ApiResponse[MessageList],
    summary="分页查询会话消息",
)
async def get_conversation_messages(
    conversation_id: UUID,
    session: DatabaseSession,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ApiResponse[MessageList]:
    await require_conversation(conversation_id, session)
    items, total = await repository.list_messages(
        session,
        conversation_id,
        page=page,
        page_size=page_size,
    )
    return ApiResponse(
        data=MessageList(
            items=[MessageRead.model_validate(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.post(
    "/{conversation_id}/messages",
    response_model=ApiResponse[ConversationReply],
    summary="发送消息并生成多轮诊断回答",
)
async def create_conversation_message(
    conversation_id: UUID,
    payload: ConversationAskRequest,
    session: DatabaseSession,
    embedding_provider: EmbeddingDependency,
    sparse_embedding_provider: SparseEmbeddingDependency,
    vector_store: VectorStoreDependency,
    llm_provider: LLMDependency,
    reranker_provider: RerankerDependency,
) -> ApiResponse[ConversationReply]:
    conversation = await require_conversation(conversation_id, session)
    if conversation.knowledge_base_id is None:
        raise HTTPException(status_code=409, detail="会话关联的知识库已被删除")
    try:
        exchange = await diagnose_in_conversation(
            session,
            conversation,
            question=payload.question,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            embedding_provider=embedding_provider,
            sparse_embedding_provider=sparse_embedding_provider,
            vector_store=vector_store,
            llm_provider=llm_provider,
            reranker_provider=reranker_provider,
            retrieval_mode=payload.retrieval_mode,
            rerank=payload.rerank,
        )
    except (EmbeddingError, VectorStoreError, LLMError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rag_answer = exchange.rag_answer
    return ApiResponse(
        data=ConversationReply(
            conversation=ConversationRead.model_validate(conversation),
            user_message=MessageRead.model_validate(exchange.user_message),
            assistant_message=MessageRead.model_validate(exchange.assistant_message),
            sources=[RagSource(**asdict(source)) for source in rag_answer.sources],
            embedding_model=rag_answer.embedding_model,
            llm_model=rag_answer.llm_model,
            llm_called=rag_answer.llm_called,
            usage=LLMUsageRead(**asdict(rag_answer.usage)),
            retrieval_mode=rag_answer.retrieval_mode,
            reranker_applied=rag_answer.reranker_applied,
            reranker_model=rag_answer.reranker_model,
        )
    )


@router.post(
    "/{conversation_id}/messages/stream",
    response_class=ServerSentEventResponse,
    summary="发送消息并流式生成多轮诊断回答",
)
async def stream_conversation_message(
    conversation_id: UUID,
    payload: ConversationAskRequest,
    session: DatabaseSession,
    embedding_provider: EmbeddingDependency,
    sparse_embedding_provider: SparseEmbeddingDependency,
    vector_store: VectorStoreDependency,
    llm_provider: LLMDependency,
    reranker_provider: RerankerDependency,
) -> ServerSentEventResponse:
    conversation = await require_conversation(conversation_id, session)
    if conversation.knowledge_base_id is None:
        raise HTTPException(status_code=409, detail="会话关联的知识库已被删除")

    async def event_stream() -> AsyncIterator[str]:
        yield encode_sse(
            "retrieval_started",
            {
                "conversation_id": conversation_id,
                "question": payload.question,
                "top_k": payload.top_k,
            },
        )
        try:
            context = await prepare_conversation_context(
                session,
                conversation,
                payload.question,
            )
            prepared = await prepare_knowledge_base_rag(
                session,
                knowledge_base_id=conversation.knowledge_base_id,
                question=payload.question,
                retrieval_query=context.retrieval_query,
                top_k=payload.top_k,
                score_threshold=payload.score_threshold,
                max_context_chars=settings.rag_max_context_chars,
                embedding_provider=embedding_provider,
                sparse_embedding_provider=sparse_embedding_provider,
                vector_store=vector_store,
                reranker_provider=reranker_provider,
                retrieval_mode=payload.retrieval_mode,
                rerank=payload.rerank,
            )
            async for update in stream_prepared_rag(
                prepared,
                llm_provider,
                context.history,
            ):
                if update.event == "completed" and update.answer is not None:
                    exchange = await persist_conversation_answer(
                        session,
                        conversation,
                        question=payload.question,
                        rag_answer=update.answer,
                    )
                    completed_data = {
                        **update.data,
                        "conversation_id": conversation.id,
                        "user_message_id": exchange.user_message.id,
                        "assistant_message_id": exchange.assistant_message.id,
                    }
                    yield encode_sse("completed", completed_data)
                else:
                    yield encode_sse(update.event, update.data)
        except (EmbeddingError, VectorStoreError, LLMError) as exc:
            await session.rollback()
            yield encode_sse("error", {"message": str(exc)})
        except Exception:
            await session.rollback()
            yield encode_sse("error", {"message": "多轮流式诊断生成失败"})

    return sse_response(event_stream())
