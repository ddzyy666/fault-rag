import asyncio
from collections.abc import Iterator
from math import sqrt

import pytest
from app import models  # noqa: F401
from app.core.config import settings
from app.db.base import Base
from app.db.database import get_db
from app.main import app
from app.services.embedding import get_embedding_provider
from app.services.vector_store import QdrantVectorStore, get_vector_store
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


class FakeEmbeddingProvider:
    """测试专用的确定性语义向量，不下载真实模型。"""

    model_name = "test-keyword-embedding"
    dimension = 4

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        vector = [
            float(sum(lowered.count(term) for term in ("高温", "温度", "冷却", "e101"))),
            float(sum(lowered.count(term) for term in ("压力", "泄漏", "进气"))),
            float(sum(lowered.count(term) for term in ("电机", "过载", "电流", "e201"))),
            0.0,
        ]
        if not any(vector):
            vector[3] = 1.0
        norm = sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def vector_store() -> Iterator[QdrantVectorStore]:
    """为每个测试创建一个独立的内存Qdrant。"""
    test_qdrant_client = QdrantClient(":memory:")
    yield QdrantVectorStore(test_qdrant_client, "test_document_chunks")
    test_qdrant_client.close()


@pytest.fixture
def client(tmp_path, vector_store: QdrantVectorStore) -> Iterator[TestClient]:
    """为每个测试创建一个独立的临时 SQLite 数据库。"""
    database_path = tmp_path / "test.db"
    test_engine = create_async_engine(
        f"sqlite+aiosqlite:///{database_path.as_posix()}",
        poolclass=NullPool,
    )
    test_session_factory = async_sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
    )

    async def create_tables() -> None:
        async with test_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with test_session_factory() as session:
            yield session

    asyncio.run(create_tables())
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_provider] = FakeEmbeddingProvider
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    original_upload_dir = settings.upload_dir
    settings.upload_dir = tmp_path / "uploads"

    with TestClient(app) as test_client:
        yield test_client

    settings.upload_dir = original_upload_dir
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_embedding_provider, None)
    app.dependency_overrides.pop(get_vector_store, None)
    asyncio.run(test_engine.dispose())
