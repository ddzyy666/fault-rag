from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.conversation import MessageRole
from app.schemas.retrieval import LLMUsageRead, RagSource
from app.services.hybrid_search import RetrievalMode


class ConversationCreate(BaseModel):
    knowledge_base_id: UUID
    title: str = Field(default="新诊断", min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("会话标题不能为空")
        return normalized


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("会话标题不能为空")
        return normalized


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID | None
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationList(BaseModel):
    items: list[ConversationRead]
    total: int
    page: int
    page_size: int


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    citations: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class MessageList(BaseModel):
    items: list[MessageRead]
    total: int
    page: int
    page_size: int


class ConversationAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    score_threshold: float | None = Field(default=0.3, ge=0.0, le=1.0)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("诊断问题不能为空")
        return normalized


class ConversationReply(BaseModel):
    conversation: ConversationRead
    user_message: MessageRead
    assistant_message: MessageRead
    sources: list[RagSource]
    embedding_model: str
    llm_model: str
    llm_called: bool
    usage: LLMUsageRead
    retrieval_mode: RetrievalMode
    reranker_applied: bool
    reranker_model: str | None
