import pytest
from app.services.sparse_embedding import HashedLexicalEmbeddingProvider
from app.services.vector_store import QdrantVectorStore, VectorStoreError
from qdrant_client import QdrantClient, models


def test_sparse_embedding_is_deterministic_and_matches_chinese_terms() -> None:
    provider = HashedLexicalEmbeddingProvider()

    document = provider.embed_documents(["GA-55空压机发生E101高温停机"])[0]
    repeated = provider.embed_documents(["GA-55空压机发生E101高温停机"])[0]
    query = provider.embed_query("E101高温")

    assert document == repeated
    assert set(document.indices) & set(query.indices)
    assert document.indices == sorted(document.indices)
    assert len(document.indices) == len(document.values)
    assert provider.embed_query("！？").indices == []


def test_vector_store_rejects_legacy_single_vector_collection() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "legacy",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    store = QdrantVectorStore(client, "legacy")

    with pytest.raises(VectorStoreError, match="旧版单向量结构"):
        store.ensure_collection(4)

    client.close()
