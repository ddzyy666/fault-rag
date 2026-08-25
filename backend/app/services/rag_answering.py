from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embedding import EmbeddingProvider
from app.services.llm import LLMChatMessage, LLMProvider, LLMUsage
from app.services.rag_prompt import build_diagnostic_prompt
from app.services.semantic_search import search_knowledge_base
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
    history: list[LLMChatMessage] | None = None,
    retrieval_query: str | None = None,
) -> RagAnswer:
    """检索知识切片，构建受约束提示词并生成带来源的诊断答案。"""
    hits = await search_knowledge_base(
        session,
        knowledge_base_id,
        retrieval_query or question,
        top_k,
        score_threshold,
        embedding_provider,
        vector_store,
    )
    if not hits:
        return RagAnswer(
            question=question,
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            embedding_model=embedding_provider.model_name,
            llm_model=llm_provider.model_name,
            llm_called=False,
            usage=LLMUsage(),
        )

    prompt = build_diagnostic_prompt(question, hits, max_context_chars)
    generation = await llm_provider.generate(
        prompt.system_prompt,
        prompt.user_prompt,
        history,
    )
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
        )
        for index, hit in enumerate(prompt.included_hits, start=1)
    ]
    return RagAnswer(
        question=question,
        answer=generation.content,
        sources=sources,
        embedding_model=embedding_provider.model_name,
        llm_model=llm_provider.model_name,
        llm_called=True,
        usage=generation.usage,
    )
