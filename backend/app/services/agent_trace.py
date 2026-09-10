"""独立短事务写执行记录；不保存模型内部推理、完整工具输出或连接异常。"""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import timedelta
from time import perf_counter
from uuid import UUID, uuid4

import anyio
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.db.base import utc_now
from app.models.agent_run import AgentRun, AgentToolCall
from app.services.diagnostic_tools import ToolResult
from app.services.llm import LLMError, LLMUsage


def redact(text: str, limit: int = 1000) -> str:
    for secret in (settings.llm_api_key, settings.reranker_api_key, settings.qdrant_api_key):
        if secret is not None and secret.get_secret_value():
            text = text.replace(secret.get_secret_value(), "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(
        r"(?i)(api[_-]?key|token|password|secret)([\"']?\s*[:=]\s*[\"']?)[^\s,;\"']+",
        r"\1\2[REDACTED]",
        text,
    )
    return text[:limit]


def safe_arguments(name: str, raw: str) -> dict:
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError
    except (ValueError, TypeError):
        return {"invalid_json": True}
    allowed = {
        "get_device_history": {"device_id", "limit"},
        "search_manual": {"query", "knowledge_base_id", "top_k"},
    }.get(name, set())
    result = {
        key: redact(value) if isinstance(value, str) else value
        for key, value in args.items()
        if key in allowed and (value is None or isinstance(value, str | int | float | bool))
    }
    if set(args) - allowed:
        result["omitted_fields"] = True
    return result


def summarize(name: str, result: ToolResult) -> dict:
    if result.status != "ok":
        return {"status": result.status}
    if name == "get_device_history":
        device = result.data.get("device", {})
        return {
            "device_code": redact(str(device.get("code", "")), 50),
            "record_count": len(result.data.get("records", [])),
            "is_simulated": device.get("is_simulated"),
        }
    if name == "search_manual":
        sources = result.data.get("sources", [])
        return {
            "source_count": len(sources),
            "chunk_ids": [str(s["chunk_id"]) for s in sources[:10]],
            "reranker_applied": bool(result.data.get("reranker_applied")),
        }
    return {"status": result.status}


class AgentTrace:
    def __init__(self, session: AsyncSession):
        self.factory = async_sessionmaker(session.bind, expire_on_commit=False)
        self.id = uuid4()
        self.started = perf_counter()
        self.phase = "preparing"
        self.turns = 0
        self.active_tool_started: float | None = None

    @property
    def elapsed_ms(self) -> int:
        return round((perf_counter() - self.started) * 1000)

    async def update(self, **values) -> None:
        async with self.factory() as session, session.begin():
            await session.execute(
                update(AgentRun)
                .where(
                    AgentRun.id == self.id,
                    AgentRun.status == "running",
                )
                .values(**values)
            )

    async def model_started(self) -> None:
        self.phase = "model"
        await self.update(phase=self.phase)

    async def model_completed(self, usage: LLMUsage) -> None:
        self.turns += 1
        await self.update(model_turns=self.turns, usage=asdict(usage))

    async def tool_started(self, sequence: int, step: int, call: dict) -> None:
        self.phase = "tool"
        self.active_tool_started = perf_counter()
        name = call["function"]["name"]
        async with self.factory() as session, session.begin():
            await session.execute(
                update(AgentRun).where(AgentRun.id == self.id).values(phase="tool")
            )
            session.add(
                AgentToolCall(
                    run_id=self.id,
                    sequence=sequence,
                    step=step,
                    call_id=redact(call["id"], 200),
                    tool_name=name
                    if name in {"get_device_history", "search_manual"}
                    else "unknown",
                    arguments=safe_arguments(name, call["function"]["arguments"]),
                )
            )

    async def tool_completed(
        self, sequence: int, name: str, result: ToolResult, elapsed: int
    ) -> None:
        async with self.factory() as session, session.begin():
            await session.execute(
                update(AgentToolCall)
                .where(
                    AgentToolCall.run_id == self.id,
                    AgentToolCall.sequence == sequence,
                )
                .values(
                    status=result.status,
                    result_summary=summarize(name, result),
                    elapsed_ms=elapsed,
                    finished_at=utc_now(),
                )
            )

        self.active_tool_started = None

    async def fail(self, status: str, reason: str) -> None:
        async with self.factory() as session, session.begin():
            changed = await session.execute(
                update(AgentRun)
                .where(
                    AgentRun.id == self.id,
                    AgentRun.status == "running",
                )
                .values(
                    status=status,
                    failure_reason=redact(reason),
                    finished_at=utc_now(),
                    elapsed_ms=self.elapsed_ms,
                )
            )
            if changed.rowcount:
                await session.execute(
                    update(AgentToolCall)
                    .where(
                        AgentToolCall.run_id == self.id,
                        AgentToolCall.status == "running",
                    )
                    .values(
                        status=status,
                        finished_at=utc_now(),
                        result_summary={"reason": "执行未完成，未获得工具结果"},
                        elapsed_ms=(
                            round((perf_counter() - self.active_tool_started) * 1000)
                            if self.active_tool_started is not None
                            else None
                        ),
                    )
                )


@asynccontextmanager
async def record_agent_run(session: AsyncSession, conversation_id: UUID, question: str, model: str):
    trace = AgentTrace(session)
    async with trace.factory() as audit, audit.begin():
        audit.add(
            AgentRun(
                id=trace.id,
                conversation_id=conversation_id,
                question=redact(question),
                model_name=redact(model, 200),
                expires_at=utc_now() + timedelta(seconds=settings.agent_timeout_seconds + 60),
            )
        )
    try:
        yield trace
    except BaseException as exc:
        # SSE 客户端断开时 AnyIO 取消作用域可能仍有效；屏蔽取消以提交结束状态。
        with anyio.CancelScope(shield=True):
            await session.rollback()
            cancelled = isinstance(exc, asyncio.CancelledError | GeneratorExit)
            reason = (
                "请求取消或连接中断"
                if cancelled
                else (redact(str(exc)) if isinstance(exc, LLMError) else f"{trace.phase}阶段失败")
            )
            await trace.fail("cancelled" if cancelled else "failed", reason)
        raise
    else:
        # 已成功的状态不会被改写；生成器被提前结束时留下明确失败记录。
        with anyio.CancelScope(shield=True):
            await trace.fail("failed", "执行未产生已保存的完整回答")
