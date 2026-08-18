from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class KnowledgeBaseCreate(BaseModel):
    """创建知识库时允许提交的字段。"""

    name: str = Field(min_length=1, max_length=100, examples=["空压机维修知识库"])
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("知识库名称不能为空")
        return normalized


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库时允许提交的字段。"""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("知识库名称不能为空")
        return normalized

    @model_validator(mode="after")
    def reject_explicit_null_name(self) -> "KnowledgeBaseUpdate":
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("知识库名称不能为 null")
        return self


class KnowledgeBaseRead(BaseModel):
    """返回给客户端的知识库数据。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseList(BaseModel):
    """知识库分页数据。"""

    items: list[KnowledgeBaseRead]
    total: int
    page: int
    page_size: int
