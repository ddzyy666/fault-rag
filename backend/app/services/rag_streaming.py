from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.services.llm import LLMChatMessage, LLMProvider, LLMUsage
from app.services.rag_answering import NO_CONTEXT_ANSWER, PreparedRag, RagAnswer


@dataclass(frozen=True, slots=True)
class RagStreamUpdate:
    """内部流事件；completed事件额外携带完整结果供会话落库。"""

    event: Literal[
        "sources", "answer_delta", "completed", "agent_started", "tool_started", "tool_completed"
    ]
    data: dict[str, Any]
    answer: RagAnswer | None = None


async def stream_prepared_rag(
    prepared: PreparedRag,
    llm_provider: LLMProvider,
    history: list[LLMChatMessage] | None = None,
) -> AsyncIterator[RagStreamUpdate]:
    """先发送来源，再逐段生成答案，最终产出完整RAG结果。"""
    yield RagStreamUpdate(
        event="sources",
        data={"items": [asdict(source) for source in prepared.sources]},
    )

    if prepared.prompt is None:
        yield RagStreamUpdate(
            event="answer_delta",
            data={"delta": NO_CONTEXT_ANSWER},
        )
        answer = RagAnswer(
            question=prepared.question,
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            embedding_model=prepared.embedding_model,
            llm_model=llm_provider.model_name,
            llm_called=False,
            usage=LLMUsage(),
            retrieval_mode=prepared.retrieval_mode,
            reranker_applied=prepared.reranker_applied,
            reranker_model=prepared.reranker_model,
        )
        yield RagStreamUpdate(event="completed", data=_completion_data(answer), answer=answer)
        return

    answer_parts: list[str] = []
    usage = LLMUsage()
    async for chunk in llm_provider.stream(
        prepared.prompt.system_prompt,
        prepared.prompt.user_prompt,
        history,
    ):
        if chunk.content:
            answer_parts.append(chunk.content)
            yield RagStreamUpdate(
                event="answer_delta",
                data={"delta": chunk.content},
            )
        if chunk.usage is not None:
            usage = chunk.usage

    answer = RagAnswer(
        question=prepared.question,
        answer="".join(answer_parts),
        sources=prepared.sources,
        embedding_model=prepared.embedding_model,
        llm_model=llm_provider.model_name,
        llm_called=True,
        usage=usage,
        retrieval_mode=prepared.retrieval_mode,
        reranker_applied=prepared.reranker_applied,
        reranker_model=prepared.reranker_model,
    )
    yield RagStreamUpdate(event="completed", data=_completion_data(answer), answer=answer)


def _completion_data(answer: RagAnswer) -> dict[str, Any]:
    return {
        "answer": answer.answer,
        "embedding_model": answer.embedding_model,
        "llm_model": answer.llm_model,
        "llm_called": answer.llm_called,
        "usage": asdict(answer.usage),
        "retrieval_mode": answer.retrieval_mode,
        "reranker_applied": answer.reranker_applied,
        "reranker_model": answer.reranker_model,
    }
