import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from app.db.base import Base
from app.models.device import Device, MaintenanceRecord
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.services.demo_devices import seed_demo_devices
from app.services.diagnostic_tools import DiagnosticTools
from app.services.document_indexing import index_document
from app.services.hybrid_search import RetrievalMode, RetrievalResult
from app.services.sparse_embedding import HashedLexicalEmbeddingProvider
from backend.tests.conftest import FakeEmbeddingProvider
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
def run_tools(tmp_path, vector_store, reranker_provider):
    def run(check):
        async def scenario():
            engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tools.db'}")
            try:
                async with engine.begin() as connection:
                    await connection.run_sync(Base.metadata.create_all)
                async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                    kb = KnowledgeBase(name="演示知识库")
                    session.add(kb)
                    await session.commit()
                    tools = DiagnosticTools(
                        session,
                        knowledge_base_id=kb.id,
                        embedding_provider=FakeEmbeddingProvider(),
                        sparse_embedding_provider=HashedLexicalEmbeddingProvider(),
                        vector_store=vector_store,
                        reranker_provider=reranker_provider,
                    )
                    await check(session, kb.id, tools)
            finally:
                await engine.dispose()

        asyncio.run(scenario())

    return run


def test_seed_repeat_and_history(run_tools):
    async def check(session, kb_id, tools):
        for _ in range(2):
            await seed_demo_devices(session, kb_id)
            await session.commit()
        assert await session.scalar(select(func.count()).select_from(Device)) == 3
        assert await session.scalar(select(func.count()).select_from(MaintenanceRecord)) == 5
        result = await tools.execute("get_device_history", {"device_id": "AC-003"})
        assert result.status == "ok"
        assert result.data["device"]["is_simulated"] is True
        assert result.data["records"][0]["action"] == "更换空气滤芯"
        assert result.data["records"][0]["outcome"] == "尚未验证高温是否消除"
        result.model_dump_json()
        limited = await tools.execute("get_device_history", {"device_id": "AC-003", "limit": 1})
        assert len(limited.data["records"]) == 1

    run_tools(check)


def test_scope_validation_and_missing(run_tools):
    async def check(session, kb_id, tools):
        other = KnowledgeBase(name="其他知识库")
        session.add(other)
        await session.flush()
        await seed_demo_devices(session, other.id)
        await session.commit()
        assert (
            await tools.execute("get_device_history", {"device_id": "AC-003"})
        ).status == "not_found"
        assert (
            await tools.execute(
                "search_manual",
                {
                    "query": "E101",
                    "knowledge_base_id": str(other.id),
                },
            )
        ).status == "forbidden"
        for name, args in [
            ("get_device_history", {"device_id": " "}),
            ("get_device_history", {"device_id": "AC-003", "limit": 31}),
            ("get_device_history", {"device_id": "AC-003", "knowledge_base_id": str(other.id)}),
            ("search_manual", {"query": "x", "knowledge_base_id": "bad"}),
            ("search_manual", {"query": "x", "knowledge_base_id": str(kb_id), "top_k": 11}),
            ("unknown", {}),
        ]:
            assert (await tools.execute(name, args)).status == "invalid_arguments"

    run_tools(check)


def test_seed_rejects_real_device_and_unknown_kb(run_tools):
    async def check(session, kb_id, tools):
        with pytest.raises(ValueError, match="知识库不存在"):
            await seed_demo_devices(session, uuid4())
        session.add(
            Device(
                knowledge_base_id=kb_id,
                code="AC-003",
                name="真实设备",
                model="Real",
                location="车间",
                is_simulated=False,
            )
        )
        await session.commit()
        with pytest.raises(ValueError, match="冲突"):
            await seed_demo_devices(session, kb_id)
        assert await session.scalar(select(func.count()).select_from(Device)) == 1
        assert await session.scalar(select(func.count()).select_from(MaintenanceRecord)) == 0
        result = await tools.execute("get_device_history", {"device_id": "AC-003"})
        assert result.status == "ok" and result.data["records"] == []

    run_tools(check)


def test_manual_real_index_returns_citations(run_tools):
    async def check(session, kb_id, tools):
        document = Document(
            knowledge_base_id=kb_id,
            filename="演示手册.md",
            storage_path="test-only",
            status=DocumentStatus.CHUNKED,
        )
        session.add(document)
        await session.flush()
        content = "AC-200 E101 高温停机：停机断电泄压后检查冷却器和冷却风扇。"
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content=content,
            page_number=1,
            extra_metadata={"section_title": "E101"},
        )
        session.add(chunk)
        await session.commit()
        await index_document(
            session,
            document,
            tools.embedding_provider,
            tools.sparse_embedding_provider,
            tools.vector_store,
        )
        result = await tools.execute(
            "search_manual",
            {
                "query": "AC-200 E101 高温",
                "knowledge_base_id": str(kb_id),
            },
        )
        assert result.status == "ok"
        source = result.data["sources"][0]
        assert source["content"] == content
        assert source["chunk_id"] == str(chunk.id)
        assert source["filename"] == "演示手册.md"
        assert source["page_number"] == 1 and source["citation"] == "[资料1]"
        result.model_dump_json()

    run_tools(check)


def test_manual_empty_failure_timeout(run_tools):
    async def check(session, kb_id, tools):
        args = {"query": "E101", "knowledge_base_id": str(kb_id)}
        target = "app.services.diagnostic_tools.retrieve_knowledge_base"
        with patch(
            target,
            new=AsyncMock(
                return_value=RetrievalResult(
                    items=[],
                    mode=RetrievalMode.HYBRID,
                    reranker_applied=False,
                    reranker_model=None,
                )
            ),
        ):
            assert (await tools.execute("search_manual", args)).status == "no_results"
        with patch(target, new=AsyncMock(side_effect=RuntimeError("private connection detail"))):
            result = await tools.execute("search_manual", args)
            assert result.status == "error"
            assert "private" not in result.model_dump_json()

        async def slow(*args, **kwargs):
            await asyncio.sleep(1)

        tools.timeout_seconds = 0.02
        with patch(target, new=slow):
            assert (await tools.execute("search_manual", args)).status == "timeout"

    run_tools(check)
