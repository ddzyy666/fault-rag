"""导入所有 ORM 模型，确保 Alembic 能发现表定义。"""

from app.models.conversation import Conversation, Message, MessageRole
from app.models.document import Document, DocumentChunk, DocumentPage, DocumentStatus
from app.models.knowledge_base import KnowledgeBase

__all__ = [
    "Conversation",
    "Document",
    "DocumentChunk",
    "DocumentPage",
    "DocumentStatus",
    "KnowledgeBase",
    "Message",
    "MessageRole",
]
