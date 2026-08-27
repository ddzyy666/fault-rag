from uuid import UUID, uuid4

import pytest
from app.services.retrieval_evaluation import calculate_case_scores
from fastapi.testclient import TestClient


def prepare_indexed_knowledge_base(client: TestClient) -> tuple[str, str]:
    knowledge_base = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "空压机检索评估库"},
    )
    knowledge_base_id = knowledge_base.json()["data"]["id"]
    markdown = """# 空压机维修手册

## E101高温停机

出现E101时，先检查冷却器堵塞、冷却风扇和润滑油状态。

## 排气压力不足

排气压力不足时，检查管路泄漏、空气过滤器和进气阀。

## E201电机过载

出现E201时，检查三相电压、电机电流和主机机械阻力。
"""
    uploaded = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("评估手册.md", markdown.encode(), "text/markdown")},
    )
    document_id = uploaded.json()["data"]["id"]
    chunked = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 200, "chunk_overlap": 20, "min_chunk_size": 20},
    )
    assert chunked.status_code == 200
    assert client.post(f"/api/v1/documents/{document_id}/index").status_code == 200
    return knowledge_base_id, document_id


def get_chunk_id_by_title(client: TestClient, document_id: str, title: str) -> str:
    response = client.get(f"/api/v1/documents/{document_id}/chunks?page_size=100")
    assert response.status_code == 200
    chunks = response.json()["data"]["items"]
    return next(
        chunk["id"] for chunk in chunks if chunk["extra_metadata"]["section_title"] == title
    )


def test_calculate_case_scores_uses_binary_relevance() -> None:
    relevant_first = uuid4()
    relevant_second = uuid4()
    scores = calculate_case_scores(
        [uuid4(), relevant_first, relevant_second],
        [relevant_first, relevant_second],
        top_k=3,
    )

    assert scores.relevant_ranks == [2, 3]
    assert scores.hit is True
    assert scores.precision_at_k == pytest.approx(2 / 3)
    assert scores.recall_at_k == 1.0
    assert scores.reciprocal_rank == 0.5
    assert 0 < scores.ndcg_at_k < 1


def test_retrieval_evaluation_compares_default_variants(client: TestClient) -> None:
    knowledge_base_id, document_id = prepare_indexed_knowledge_base(client)
    high_temperature_chunk = get_chunk_id_by_title(client, document_id, "E101高温停机")
    pressure_chunk = get_chunk_id_by_title(client, document_id, "排气压力不足")

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/evaluations/retrieval",
        json={
            "cases": [
                {
                    "case_id": "compressor-e101",
                    "query": "E101高温停机应该检查什么？",
                    "relevant_chunk_ids": [high_temperature_chunk],
                },
                {
                    "case_id": "compressor-pressure",
                    "query": "排气压力不足并且管路泄漏怎么排查？",
                    "relevant_chunk_ids": [pressure_chunk],
                },
            ],
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["best_variant"] in {"vector", "hybrid_rrf", "hybrid_rerank"}
    assert [variant["name"] for variant in data["variants"]] == [
        "vector",
        "hybrid_rrf",
        "hybrid_rerank",
    ]
    for variant in data["variants"]:
        assert variant["metrics"]["case_count"] == 2
        assert variant["metrics"]["hit_rate_at_k"] == 1.0
        assert variant["metrics"]["recall_at_k"] == 1.0
        assert len(variant["cases"]) == 2
    assert data["variants"][0]["metrics"]["reranker_applied_count"] == 0
    assert data["variants"][2]["metrics"]["reranker_applied_count"] == 2


def test_retrieval_evaluation_rejects_unknown_ground_truth_chunk(
    client: TestClient,
) -> None:
    knowledge_base_id, _ = prepare_indexed_knowledge_base(client)

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/evaluations/retrieval",
        json={
            "cases": [
                {
                    "case_id": "invalid-label",
                    "query": "E101怎么处理？",
                    "relevant_chunk_ids": [str(UUID(int=0))],
                }
            ]
        },
    )

    assert response.status_code == 422
    assert "不属于当前知识库或尚未索引" in response.json()["message"]
