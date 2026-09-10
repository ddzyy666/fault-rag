from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utc_now
from app.models.agent_run import AgentRun, AgentToolCall


async def recover_expired_runs(session: AsyncSession, conversation_id: UUID) -> None:
    """进程强制退出无法写 finally；查询时将超过最大执行窗口的记录标记中断。"""
    expired = select(AgentRun.id).where(
        AgentRun.conversation_id == conversation_id,
        AgentRun.status == "running",
        AgentRun.expires_at < utc_now(),
    )
    ids = list((await session.scalars(expired)).all())
    if ids:
        await session.execute(
            update(AgentToolCall)
            .where(
                AgentToolCall.run_id.in_(ids),
                AgentToolCall.status == "running",
            )
            .values(status="interrupted", result_summary={"reason": "未收到执行结束记录"})
        )
        await session.execute(
            update(AgentRun)
            .where(
                AgentRun.id.in_(ids),
                AgentRun.status == "running",
            )
            .values(
                status="interrupted", failure_reason="执行窗口已过，未收到结束记录；可能服务中断"
            )
        )
        # 不伪造实际结束时间/耗时。
        await session.commit()


async def list_runs(
    session: AsyncSession,
    conversation_id: UUID,
    page: int,
    page_size: int,
    assistant_message_id: UUID | None = None,
) -> tuple[list[AgentRun], int]:
    await recover_expired_runs(session, conversation_id)
    conditions = [AgentRun.conversation_id == conversation_id]
    if assistant_message_id:
        conditions.append(AgentRun.assistant_message_id == assistant_message_id)
    total = await session.scalar(select(func.count()).select_from(AgentRun).where(*conditions))
    rows = await session.scalars(
        select(AgentRun)
        .where(*conditions)
        .order_by(
            AgentRun.created_at.desc(),
            AgentRun.id.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(rows.all()), total or 0


async def get_run(session: AsyncSession, conversation_id: UUID, run_id: UUID):
    await recover_expired_runs(session, conversation_id)
    run = await session.scalar(
        select(AgentRun).where(
            AgentRun.id == run_id,
            AgentRun.conversation_id == conversation_id,
        )
    )
    if run is None:
        return None, []
    calls = await session.scalars(
        select(AgentToolCall)
        .where(
            AgentToolCall.run_id == run_id,
        )
        .order_by(AgentToolCall.sequence)
    )
    return run, list(calls.all())
