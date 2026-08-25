from app.services.llm import LLMProvider
from fastapi.testclient import TestClient


def create_knowledge_base(client: TestClient) -> str:
    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "RAG故障诊断知识库"},
    )
    assert response.status_code == 201
    return response.json()["data"]["id"]


def prepare_indexed_document(client: TestClient, knowledge_base_id: str) -> str:
    markdown = """# 空压机维修手册

## 高温停机 E101

空压机高温停机时，应检查冷却器堵塞、冷却风扇、环境温度和润滑油状态。

## 排气压力不足

排气压力不足时，应检查管路泄漏、空气过滤器和进气阀动作。
"""
    upload = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("空压机手册.md", markdown.encode("utf-8"), "text/markdown")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["data"]["id"]
    chunks = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 300, "chunk_overlap": 40, "min_chunk_size": 20},
    )
    assert chunks.status_code == 200
    assert client.post(f"/api/v1/documents/{document_id}/index").status_code == 200
    return document_id


def test_rag_ask_returns_grounded_answer_and_sources(
    client: TestClient,
    llm_provider: LLMProvider,
) -> None:
    knowledge_base_id = create_knowledge_base(client)
    document_id = prepare_indexed_document(client, knowledge_base_id)

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/ask",
        json={
            "question": "空压机E101高温停机应该怎么排查？",
            "top_k": 2,
            "score_threshold": 0.1,
        },
    )

    assert response.status_code == 200
    result = response.json()["data"]
    assert result["llm_called"] is True
    assert result["llm_model"] == "test-diagnostic-llm"
    assert result["embedding_model"] == "test-keyword-embedding"
    assert result["retrieved_count"] >= 1
    assert "[资料1]" in result["answer"]
    assert result["sources"][0]["citation"] == "[资料1]"
    assert result["sources"][0]["document_id"] == document_id
    assert result["sources"][0]["section_title"] == "高温停机 E101"
    assert result["usage"]["total_tokens"] == 168

    assert len(llm_provider.calls) == 1  # type: ignore[attr-defined]
    system_prompt, user_prompt, history = llm_provider.calls[0]  # type: ignore[attr-defined]
    assert "只能依据" in system_prompt
    assert "不可信数据" in system_prompt
    assert "[资料1]" in user_prompt
    assert "冷却器堵塞" in user_prompt
    assert history == []


def test_rag_ask_skips_llm_when_no_context(
    client: TestClient,
    llm_provider: LLMProvider,
) -> None:
    knowledge_base_id = create_knowledge_base(client)

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/ask",
        json={"question": "完全未知的故障如何处理？"},
    )

    assert response.status_code == 200
    result = response.json()["data"]
    assert result["llm_called"] is False
    assert result["sources"] == []
    assert result["retrieved_count"] == 0
    assert "知识库信息不足" in result["answer"]
    assert len(llm_provider.calls) == 0  # type: ignore[attr-defined]


def test_rag_ask_requires_existing_knowledge_base(client: TestClient) -> None:
    response = client.post(
        "/api/v1/knowledge-bases/00000000-0000-0000-0000-000000000001/ask",
        json={"question": "空压机为什么高温？"},
    )

    assert response.status_code == 404
    assert response.json()["message"] == "知识库不存在"
