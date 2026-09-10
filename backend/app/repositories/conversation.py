from datetime import timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utc_now
from app.models.agent_run import AgentRun
from app.models.conversation import Conversation, Message, MessageRole


async def get_conversation(
    session: AsyncSession,
    conversation_id: UUID,
) -> Conversation | None:
    return await session.get(Conversation, conversation_id)


async def list_conversations(
    session: AsyncSession,
    *,
    page: int,
    page_size: int,
    knowledge_base_id: UUID | None = None,
) -> tuple[list[Conversation], int]:
    conditions = []
    if knowledge_base_id is not None:
        conditions.append(Conversation.knowledge_base_id == knowledge_base_id)

    total = await session.scalar(select(func.count()).select_from(Conversation).where(*conditions))
    statement = (
        select(Conversation)
        .where(*conditions)
        .order_by(Conversation.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = list((await session.scalars(statement)).all())
    return items, total or 0


async def create_conversation(
    session: AsyncSession,
    *,
    knowledge_base_id: UUID,
    title: str,
) -> Conversation:
    conversation = Conversation(
        knowledge_base_id=knowledge_base_id,
        title=title,
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def update_conversation_title(
    session: AsyncSession,
    conversation: Conversation,
    title: str,
) -> Conversation:
    conversation.title = title
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def delete_conversation(
    session: AsyncSession,
    conversation: Conversation,
) -> None:
    await session.delete(conversation)
    await session.commit()


async def list_messages(
    session: AsyncSession,
    conversation_id: UUID,
    *,
    page: int,
    page_size: int,
) -> tuple[list[Message], int]:
    condition = Message.conversation_id == conversation_id
    total = await session.scalar(select(func.count()).select_from(Message).where(condition))
    statement = (
        select(Message)
        .where(condition)
        .order_by(Message.created_at, Message.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = list((await session.scalars(statement)).all())
    return items, total or 0


async def get_recent_messages(
    session: AsyncSession,
    conversation_id: UUID,
    limit: int,
) -> list[Message]:
    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )
    messages = list((await session.scalars(statement)).all())
    messages.reverse()
    return messages


async def save_exchange(
    session: AsyncSession,
    conversation: Conversation,
    *,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    agent_run_id: UUID | None = None,
    run_elapsed_ms: int | None = None,
) -> tuple[Message, Message]:
    user_time = utc_now()
    assistant_time = user_time + timedelta(microseconds=1)
    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=question,
        citations=[],
        created_at=user_time,
        updated_at=user_time,
    )
    assistant_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=answer,
        citations=citations,
        created_at=assistant_time,
        updated_at=assistant_time,
    )
    if conversation.title == "新诊断":
        conversation.title = make_conversation_title(question)
    conversation.updated_at = assistant_time
    session.add_all([user_message, assistant_message])
    if agent_run_id is not None:
        await session.flush()
        linked = await session.execute(
            update(AgentRun)
            .where(
                AgentRun.id == agent_run_id,
                AgentRun.conversation_id == conversation.id,
                AgentRun.status == "running",
            )
            .values(
                assistant_message_id=assistant_message.id,
                status="succeeded",
                phase="completed",
                finished_at=utc_now(),
                elapsed_ms=run_elapsed_ms,
            )
        )
        if linked.rowcount != 1:
            raise RuntimeError("执行记录已结束，不能保存过期回答")
    await session.commit()
    await session.refresh(user_message)
    await session.refresh(assistant_message)
    await session.refresh(conversation)
    return user_message, assistant_message


def make_conversation_title(question: str, max_length: int = 30) -> str:
    normalized = " ".join(question.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length]}…"
