from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import settings
from app.models.document import DocumentStatus


class DocumentRead(BaseModel):
    """不暴露服务器存储路径的文档响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    filename: str
    mime_type: str | None
    size_bytes: int | None
    file_hash: str | None
    status: DocumentStatus
    page_count: int
    error_message: str | None
    extra_metadata: dict
    processed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    """文档分页数据。"""

    items: list[DocumentRead]
    total: int
    page: int
    page_size: int


class DocumentPageRead(BaseModel):
    """解析后的单页原始文本。"""

    model_config = ConfigDict(from_attributes=True)

    page_number: int
    content: str
    character_count: int


class DocumentPageList(BaseModel):
    """文档原始页文本分页数据。"""

    items: list[DocumentPageRead]
    total: int
    page: int
    page_size: int


class ChunkingRequest(BaseModel):
    """允许针对不同文档实验分块参数。"""

    chunk_size: int = Field(default=settings.default_chunk_size, ge=100, le=4000)
    chunk_overlap: int = Field(default=settings.default_chunk_overlap, ge=0, le=1000)
    min_chunk_size: int = Field(default=settings.min_chunk_size, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_ranges(self) -> "ChunkingRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size不能大于chunk_size")
        return self


class DocumentChunkRead(BaseModel):
    """单条可追溯的文档切片。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    page_number: int | None
    token_count: int | None
    vector_id: str | None
    extra_metadata: dict
    created_at: datetime


class DocumentChunkList(BaseModel):
    """文档切片分页数据。"""

    items: list[DocumentChunkRead]
    total: int
    page: int
    page_size: int


class ChunkingResult(BaseModel):
    """分块完成后的状态与统计信息。"""

    document_id: UUID
    status: DocumentStatus
    chunk_count: int
    average_characters: float
    max_characters: int
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
