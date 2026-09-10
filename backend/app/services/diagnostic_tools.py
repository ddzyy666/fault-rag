"""只读工具边界。知识库范围由调用方绑定，不接受模型扩大访问范围。"""

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.device import Device, MaintenanceRecord
from app.models.knowledge_base import KnowledgeBase
from app.services.embedding import EmbeddingProvider
from app.services.hybrid_search import RetrievalMode, retrieve_knowledge_base
from app.services.reranker import RerankerProvider
from app.services.sparse_embedding import SparseEmbeddingProvider
from app.services.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DeviceHistoryArguments(ToolArguments):
    device_id: str = Field(min_length=1, max_length=50, description="设备编号，例如 AC-003")
    limit: int = Field(default=10, ge=1, le=30, strict=True)


class SearchManualArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=1000)
    knowledge_base_id: UUID
    top_k: int = Field(default=5, ge=1, le=10, strict=True)


class ToolResult(BaseModel):
    status: Literal[
        "ok", "not_found", "no_results", "forbidden", "invalid_arguments", "timeout", "error"
    ]
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema.model_json_schema(),
        },
    }
    for name, description, schema in [
        ("get_device_history", "按设备编号查询档案及最近维修记录。", DeviceHistoryArguments),
        ("search_manual", "检索当前知识库维修原文；命中不代表足以诊断。", SearchManualArguments),
    ]
]


class DiagnosticTools:
    def __init__(
        self,
        session: AsyncSession,
        *,
        knowledge_base_id: UUID,
        embedding_provider: EmbeddingProvider,
        sparse_embedding_provider: SparseEmbeddingProvider,
        vector_store: QdrantVectorStore,
        reranker_provider: RerankerProvider,
        timeout_seconds: float = 30,
    ) -> None:
        self.session = session
        self.knowledge_base_id = knowledge_base_id
        self.embedding_provider = embedding_provider
        self.sparse_embedding_provider = sparse_embedding_provider
        self.vector_store = vector_store
        self.reranker_provider = reranker_provider
        self.timeout_seconds = timeout_seconds

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """后续 Agent 统一入口；同一 session 的工具调用应串行执行。"""
        try:
            async with asyncio.timeout(self.timeout_seconds):
                if name == "get_device_history":
                    return await self.get_device_history(
                        DeviceHistoryArguments.model_validate(arguments)
                    )
                if name == "search_manual":
                    return await self.search_manual(SearchManualArguments.model_validate(arguments))
                return ToolResult(status="invalid_arguments", message="未知工具")
        except ValidationError:
            return ToolResult(status="invalid_arguments", message="工具参数不符合 Schema")
        except TimeoutError:
            await self.session.rollback()
            return ToolResult(status="timeout", message="工具调用超时，请稍后重试")
        except Exception:
            await self.session.rollback()
            logger.exception("Diagnostic tool failed: %s", name)
            return ToolResult(status="error", message="工具暂时不可用，不能据此生成诊断结论")

    async def get_device_history(self, args: DeviceHistoryArguments) -> ToolResult:
        device = await self.session.scalar(
            select(Device).where(
                Device.knowledge_base_id == self.knowledge_base_id,
                Device.code == args.device_id,
            )
        )
        if device is None:
            return ToolResult(status="not_found", message="当前知识库下未找到该设备，请核对编号")
        records = (
            await self.session.scalars(
                select(MaintenanceRecord)
                .where(
                    MaintenanceRecord.device_id == device.id,
                )
                .order_by(MaintenanceRecord.occurred_at.desc(), MaintenanceRecord.id)
                .limit(args.limit)
            )
        ).all()
        return ToolResult(
            status="ok",
            message="设备记录查询成功；空记录不表示从未发生故障",
            data={
                "device": {
                    key: getattr(device, key)
                    for key in (
                        "code",
                        "name",
                        "model",
                        "location",
                        "is_simulated",
                    )
                },
                "records": [
                    {
                        "record_id": str(record.id),
                        "occurred_at": record.occurred_at.isoformat(),
                        **{
                            key: getattr(record, key)
                            for key in (
                                "fault_code",
                                "symptom",
                                "action",
                                "outcome",
                                "is_simulated",
                            )
                        },
                    }
                    for record in records
                ],
                "limit": args.limit,
            },
        )

    async def search_manual(self, args: SearchManualArguments) -> ToolResult:
        if args.knowledge_base_id != self.knowledge_base_id:
            return ToolResult(status="forbidden", message="只能检索当前绑定的知识库")
        if await self.session.get(KnowledgeBase, self.knowledge_base_id) is None:
            return ToolResult(status="not_found", message="当前知识库不存在")
        retrieval = await retrieve_knowledge_base(
            self.session,
            knowledge_base_id=self.knowledge_base_id,
            query=args.query,
            top_k=args.top_k,
            score_threshold=None,
            mode=RetrievalMode.HYBRID,
            rerank=True,
            embedding_provider=self.embedding_provider,
            sparse_embedding_provider=self.sparse_embedding_provider,
            vector_store=self.vector_store,
            reranker_provider=self.reranker_provider,
        )
        sources = []
        for index, hit in enumerate(retrieval.items, 1):
            source = asdict(hit)
            source.update(
                chunk_id=str(hit.chunk_id),
                document_id=str(hit.document_id),
                citation=f"[资料{index}]",
            )
            sources.append(source)
        return ToolResult(
            status="ok" if sources else "no_results",
            message="返回候选原文，需判断能否支持结论" if sources else "未找到资料，请补充信息",
            data={"sources": sources, "reranker_applied": retrieval.reranker_applied},
        )
