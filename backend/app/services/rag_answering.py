from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embedding import EmbeddingProvider
from app.services.hybrid_search import RetrievalMode, retrieve_knowledge_base
from app.services.llm import LLMChatMessage, LLMProvider, LLMUsage
from app.services.rag_prompt import DiagnosticPrompt, build_diagnostic_prompt
from app.services.reranker import RerankerProvider
from app.services.vector_store import QdrantVectorStore

NO_CONTEXT_ANSWER = (
    "现有知识库信息不足，未检索到能够支持诊断结论的维修资料。"
    "请补充设备型号、故障代码、报警时间、运行参数和已完成的检查结果。"
)


@dataclass(frozen=True, slots=True)
class RagAnswerSource:
    """发送给模型并随回答返回的可追溯资料。"""

    citation: str
    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    score: float
    page_number: int | None
    section_title: str | None
    vector_score: float | None
    keyword_score: float | None
    fusion_score: float | None
    rerank_score: float | None
    retrieval_sources: list[str]


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """一次完整RAG检索和生成的结果。"""

    question: str
    answer: str
    sources: list[RagAnswerSource]
    embedding_model: str
    llm_model: str
    llm_called: bool
    usage: LLMUsage
    retrieval_mode: RetrievalMode
    reranker_applied: bool
    reranker_model: str | None


@dataclass(frozen=True, slots=True)
class PreparedRag:
    """检索结束后、模型生成前的RAG上下文。"""

    question: str
    sources: list[RagAnswerSource]
    prompt: DiagnosticPrompt | None
    embedding_model: str
    retrieval_mode: RetrievalMode
    reranker_applied: bool
    reranker_model: str | None


async def prepare_knowledge_base_rag(
    session: AsyncSession,
    *,
    knowledge_base_id: UUID,
    question: str,
    top_k: int,
    score_threshold: float | None,
    max_context_chars: int,
    embedding_provider: EmbeddingProvider,
    vector_store: QdrantVectorStore,
    reranker_provider: RerankerProvider,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    rerank: bool = True,
    retrieval_query: str | None = None,
) -> PreparedRag:
    """执行向量检索并构造提示词，供同步与流式回答共同复用。"""
    retrieval = await retrieve_knowledge_base(
        session,
        knowledge_base_id=knowledge_base_id,
        query=retrieval_query or question,
        top_k=top_k,
        score_threshold=score_threshold,
        mode=retrieval_mode,
        rerank=rerank,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker_provider=reranker_provider,
    )
    hits = retrieval.items
    if not hits:
        return PreparedRag(
            question=question,
            sources=[],
            prompt=None,
            embedding_model=embedding_provider.model_name,
            retrieval_mode=retrieval.mode,
            reranker_applied=retrieval.reranker_applied,
            reranker_model=retrieval.reranker_model,
        )

    prompt = build_diagnostic_prompt(question, hits, max_context_chars)
    sources = [
        RagAnswerSource(
            citation=f"[资料{index}]",
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            filename=hit.filename,
            content=hit.content,
            score=hit.score,
            page_number=hit.page_number,
            section_title=hit.section_title,
            vector_score=hit.vector_score,
            keyword_score=hit.keyword_score,
            fusion_score=hit.fusion_score,
            rerank_score=hit.rerank_score,
            retrieval_sources=hit.retrieval_sources,
        )
        for index, hit in enumerate(prompt.included_hits, start=1)
    ]
    return PreparedRag(
        question=question,
        sources=sources,
        prompt=prompt if sources else None,
        embedding_model=embedding_provider.model_name,
        retrieval_mode=retrieval.mode,
        reranker_applied=retrieval.reranker_applied,
        reranker_model=retrieval.reranker_model,
    )


async def answer_with_knowledge_base(
    session: AsyncSession,
    *,
    knowledge_base_id: UUID,
    question: str,
    top_k: int,
    score_threshold: float | None,
    max_context_chars: int,
    embedding_provider: EmbeddingProvider,
    vector_store: QdrantVectorStore,
    llm_provider: LLMProvider,
    reranker_provider: RerankerProvider,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    rerank: bool = True,
    history: list[LLMChatMessage] | None = None,
    retrieval_query: str | None = None,
) -> RagAnswer:
    """检索知识切片，构建受约束提示词并生成带来源的诊断答案。"""
    prepared = await prepare_knowledge_base_rag(
        session,
        knowledge_base_id=knowledge_base_id,
        question=question,
        retrieval_query=retrieval_query,
        top_k=top_k,
        score_threshold=score_threshold,
        max_context_chars=max_context_chars,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker_provider=reranker_provider,
        retrieval_mode=retrieval_mode,
        rerank=rerank,
    )
    if prepared.prompt is None:
        return RagAnswer(
            question=question,
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            embedding_model=embedding_provider.model_name,
            llm_model=llm_provider.model_name,
            llm_called=False,
            usage=LLMUsage(),
            retrieval_mode=prepared.retrieval_mode,
            reranker_applied=prepared.reranker_applied,
            reranker_model=prepared.reranker_model,
        )

    generation = await llm_provider.generate(
        prepared.prompt.system_prompt,
        prepared.prompt.user_prompt,
        history,
    )
    return RagAnswer(
        question=question,
        answer=generation.content,
        sources=prepared.sources,
        embedding_model=embedding_provider.model_name,
        llm_model=llm_provider.model_name,
        llm_called=True,
        usage=generation.usage,
        retrieval_mode=prepared.retrieval_mode,
        reranker_applied=prepared.reranker_applied,
        reranker_model=prepared.reranker_model,
    )
