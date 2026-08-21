from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.models.document import DocumentStatus


class IndexingResult(BaseModel):
    """文档向量索引结果。"""

    document_id: UUID
    status: DocumentStatus
    vector_count: int
    model_name: str
    dimension: int
    elapsed_ms: int


class SemanticSearchRequest(BaseModel):
    """知识库语义检索参数。"""

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float | None = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("检索问题不能为空")
        return normalized


class SemanticSearchItem(BaseModel):
    """一条包含原文和来源的检索结果。"""

    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    score: float
    page_number: int | None
    section_title: str | None


class SemanticSearchResult(BaseModel):
    """知识库语义检索响应数据。"""

    query: str
    items: list[SemanticSearchItem]
    total: int
    model_name: str
