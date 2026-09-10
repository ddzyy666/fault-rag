"""导入所有 ORM 模型，确保 Alembic 能发现表定义。"""

from app.models.agent_run import AgentRun, AgentToolCall
from app.models.conversation import Conversation, Message, MessageRole
from app.models.device import Device, MaintenanceRecord
from app.models.document import Document, DocumentChunk, DocumentPage, DocumentStatus
from app.models.knowledge_base import KnowledgeBase

__all__ = [
    "AgentRun",
    "AgentToolCall",
    "Conversation",
    "Device",
    "MaintenanceRecord",
    "Document",
    "DocumentChunk",
    "DocumentPage",
    "DocumentStatus",
    "KnowledgeBase",
    "Message",
    "MessageRole",
]
