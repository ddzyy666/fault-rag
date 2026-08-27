from dataclasses import asdict, dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.conversation import Conversation, Message, MessageRole
from app.repositories import conversation as repository
from app.services.embedding import EmbeddingProvider
from app.services.hybrid_search import RetrievalMode
from app.services.llm import LLMChatMessage, LLMProvider
from app.services.rag_answering import RagAnswer, RagAnswerSource, answer_with_knowledge_base
from app.services.reranker import RerankerProvider
from app.services.vector_store import QdrantVectorStore


@dataclass(frozen=True, slots=True)
class ConversationExchange:
    user_message: Message
    assistant_message: Message
    rag_answer: RagAnswer


@dataclass(frozen=True, slots=True)
class ConversationRagContext:
    """一轮诊断使用的模型历史和检索查询。"""

    history: list[LLMChatMessage]
    retrieval_query: str


async def diagnose_in_conversation(
    session: AsyncSession,
    conversation: Conversation,
    *,
    question: str,
    top_k: int,
    score_threshold: float | None,
    embedding_provider: EmbeddingProvider,
    vector_store: QdrantVectorStore,
    llm_provider: LLMProvider,
    reranker_provider: RerankerProvider,
    retrieval_mode: RetrievalMode,
    rerank: bool,
) -> ConversationExchange:
    """使用近期历史增强检索和生成，并原子保存一轮用户/助手消息。"""
    if conversation.knowledge_base_id is None:
        raise ValueError("会话关联的知识库已被删除")

    context = await prepare_conversation_context(
        session,
        conversation,
        question,
    )
    rag_answer = await answer_with_knowledge_base(
        session,
        knowledge_base_id=conversation.knowledge_base_id,
        question=question,
        retrieval_query=context.retrieval_query,
        top_k=top_k,
        score_threshold=score_threshold,
        max_context_chars=settings.rag_max_context_chars,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider,
        reranker_provider=reranker_provider,
        retrieval_mode=retrieval_mode,
        rerank=rerank,
        history=context.history,
    )
    return await persist_conversation_answer(
        session,
        conversation,
        question=question,
        rag_answer=rag_answer,
    )


async def prepare_conversation_context(
    session: AsyncSession,
    conversation: Conversation,
    question: str,
) -> ConversationRagContext:
    """读取近期消息，形成模型历史和增强后的检索查询。"""
    recent_messages = await repository.get_recent_messages(
        session,
        conversation.id,
        settings.conversation_history_messages,
    )
    history = [
        LLMChatMessage(role=message.role.value, content=message.content)
        for message in recent_messages
        if message.role in {MessageRole.USER, MessageRole.ASSISTANT}
    ]
    if history and history[0].role == "assistant":
        history = history[1:]
    return ConversationRagContext(
        history=history,
        retrieval_query=build_follow_up_retrieval_query(recent_messages, question),
    )


async def persist_conversation_answer(
    session: AsyncSession,
    conversation: Conversation,
    *,
    question: str,
    rag_answer: RagAnswer,
) -> ConversationExchange:
    """完整回答生成后，原子保存用户消息、助手消息和引用。"""
    citations = [serialize_source(source) for source in rag_answer.sources]
    user_message, assistant_message = await repository.save_exchange(
        session,
        conversation,
        question=question,
        answer=rag_answer.answer,
        citations=citations,
    )
    return ConversationExchange(
        user_message=user_message,
        assistant_message=assistant_message,
        rag_answer=rag_answer,
    )


def build_follow_up_retrieval_query(messages: list[Message], question: str) -> str:
    """将最近两个用户问题加入检索词，补足省略主语的追问。"""
    previous_questions = [
        message.content for message in messages if message.role == MessageRole.USER
    ][-2:]
    return "\n".join([*previous_questions, question])


def serialize_source(source: RagAnswerSource) -> dict[str, object]:
    payload = asdict(source)
    payload["chunk_id"] = str(payload["chunk_id"])
    payload["document_id"] = str(payload["document_id"])
    return payload
