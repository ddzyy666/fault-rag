from app.services.vector_store import QdrantVectorStore
from fastapi.testclient import TestClient


def create_knowledge_base(client: TestClient, name: str = "空压机语义检索库") -> str:
    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": name},
    )
    assert response.status_code == 201
    return response.json()["data"]["id"]


def upload_and_chunk_document(client: TestClient, knowledge_base_id: str) -> str:
    markdown = """# 空压机维修手册

## 高温停机 E101

空压机出现高温停机时，应检查环境温度、冷却器堵塞、冷却风扇以及润滑油状态。

## 排气压力不足

排气压力不足时，应检查管路泄漏、空气过滤器、用气量和进气阀动作。

## 电机过载 E201

电机过载时，应检查三相电压、电机电流、接触器以及主机机械阻力。
"""
    upload_response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("空压机检索测试.md", markdown.encode("utf-8"), "text/markdown")},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["data"]["id"]

    chunk_response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 300, "chunk_overlap": 40, "min_chunk_size": 20},
    )
    assert chunk_response.status_code == 200
    return document_id


def test_index_search_rechunk_and_delete_index(client: TestClient) -> None:
    knowledge_base_id = create_knowledge_base(client)
    document_id = upload_and_chunk_document(client, knowledge_base_id)

    index_response = client.post(f"/api/v1/documents/{document_id}/index")
    assert index_response.status_code == 200
    index_result = index_response.json()["data"]
    assert index_result["status"] == "indexed"
    assert index_result["vector_count"] == 3
    assert index_result["model_name"] == "test-keyword-embedding"
    assert index_result["dimension"] == 4

    chunks_response = client.get(f"/api/v1/documents/{document_id}/chunks?page_size=100")
    chunks = chunks_response.json()["data"]["items"]
    assert all(chunk["vector_id"] == chunk["id"] for chunk in chunks)

    search_response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/search",
        json={
            "query": "E101高温停机应该检查什么？",
            "top_k": 2,
            "score_threshold": 0.1,
        },
    )
    assert search_response.status_code == 200
    search_result = search_response.json()["data"]
    assert search_result["total"] >= 1
    assert search_result["items"][0]["section_title"] == "高温停机 E101"
    assert "冷却器" in search_result["items"][0]["content"]

    rechunk_response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 400, "chunk_overlap": 50, "min_chunk_size": 20},
    )
    assert rechunk_response.status_code == 200
    assert rechunk_response.json()["data"]["status"] == "chunked"

    stale_search = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/search",
        json={"query": "高温停机", "score_threshold": 0.0},
    )
    assert stale_search.status_code == 200
    assert stale_search.json()["data"]["total"] == 0

    assert client.post(f"/api/v1/documents/{document_id}/index").status_code == 200
    delete_response = client.delete(f"/api/v1/documents/{document_id}/index")
    assert delete_response.status_code == 200
    assert "3" in delete_response.json()["message"]

    document = client.get(f"/api/v1/documents/{document_id}").json()["data"]
    assert document["status"] == "chunked"
    chunks = client.get(f"/api/v1/documents/{document_id}/chunks?page_size=100").json()["data"][
        "items"
    ]
    assert all(chunk["vector_id"] is None for chunk in chunks)


def test_index_requires_chunked_document(client: TestClient) -> None:
    knowledge_base_id = create_knowledge_base(client)
    upload_response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("manual.txt", b"compressor maintenance", "text/plain")},
    )
    document_id = upload_response.json()["data"]["id"]

    response = client.post(f"/api/v1/documents/{document_id}/index")

    assert response.status_code == 409
    assert "parsed" in response.json()["message"]


def test_semantic_search_isolated_by_knowledge_base(client: TestClient) -> None:
    first_knowledge_base = create_knowledge_base(client, "一号空压机知识库")
    second_knowledge_base = create_knowledge_base(client, "二号空压机知识库")
    first_document = upload_and_chunk_document(client, first_knowledge_base)
    second_document = upload_and_chunk_document(client, second_knowledge_base)
    assert client.post(f"/api/v1/documents/{first_document}/index").status_code == 200
    assert client.post(f"/api/v1/documents/{second_document}/index").status_code == 200

    response = client.post(
        f"/api/v1/knowledge-bases/{first_knowledge_base}/search",
        json={"query": "排气压力不足", "top_k": 10, "score_threshold": 0.1},
    )

    assert response.status_code == 200
    items = response.json()["data"]["items"]
    assert items
    assert {item["document_id"] for item in items} == {first_document}


def test_delete_knowledge_base_cleans_vectors(
    client: TestClient,
    vector_store: QdrantVectorStore,
) -> None:
    knowledge_base_id = create_knowledge_base(client)
    document_id = upload_and_chunk_document(client, knowledge_base_id)
    assert client.post(f"/api/v1/documents/{document_id}/index").status_code == 200

    before_delete = vector_store.client.count(
        collection_name=vector_store.collection_name,
        exact=True,
    )
    assert before_delete.count == 3

    response = client.delete(f"/api/v1/knowledge-bases/{knowledge_base_id}")

    assert response.status_code == 200
    after_delete = vector_store.client.count(
        collection_name=vector_store.collection_name,
        exact=True,
    )
    assert after_delete.count == 0
