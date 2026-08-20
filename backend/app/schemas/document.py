from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

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
