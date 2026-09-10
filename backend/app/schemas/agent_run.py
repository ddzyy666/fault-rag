from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AgentRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    assistant_message_id: UUID | None
    question: str
    model_name: str
    status: str
    phase: str
    created_at: datetime
    finished_at: datetime | None
    elapsed_ms: int | None
    model_turns: int
    usage: dict[str, Any]
    failure_reason: str | None


class AgentToolCallRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sequence: int
    step: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str
    result_summary: dict[str, Any]
    elapsed_ms: int | None
    created_at: datetime
    finished_at: datetime | None


class AgentRunDetail(AgentRunRead):
    tool_calls: list[AgentToolCallRead]


class AgentRunList(BaseModel):
    items: list[AgentRunRead]
    total: int
    page: int
    page_size: int
