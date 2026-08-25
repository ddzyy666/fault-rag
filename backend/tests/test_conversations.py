from app.services.llm import LLMProvider
from fastapi.testclient import TestClient


def create_knowledge_base(client: TestClient, name: str = "多轮诊断知识库") -> str:
    response = client.post("/api/v1/knowledge-bases", json={"name": name})
    assert response.status_code == 201
    return response.json()["data"]["id"]


def prepare_indexed_document(client: TestClient, knowledge_base_id: str) -> str:
    markdown = """# 空压机维修手册

## 高温停机 E101

空压机高温停机时，应先检查环境温度，再检查冷却器堵塞和冷却风扇，最后检查润滑油状态。

## 排气压力不足

排气压力不足时，应检查管路泄漏、空气过滤器和进气阀动作。
"""
    upload = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("多轮诊断手册.md", markdown.encode("utf-8"), "text/markdown")},
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


def create_conversation(client: TestClient, knowledge_base_id: str) -> str:
    response = client.post(
        "/api/v1/conversations",
        json={"knowledge_base_id": knowledge_base_id},
    )
    assert response.status_code == 201
    return response.json()["data"]["id"]


def test_conversation_crud_and_filter(client: TestClient) -> None:
    first_knowledge_base = create_knowledge_base(client, "一号诊断库")
    second_knowledge_base = create_knowledge_base(client, "二号诊断库")
    first_conversation = create_conversation(client, first_knowledge_base)
    create_conversation(client, second_knowledge_base)

    filtered = client.get(
        "/api/v1/conversations",
        params={"knowledge_base_id": first_knowledge_base},
    )
    assert filtered.status_code == 200
    assert filtered.json()["data"]["total"] == 1
    assert filtered.json()["data"]["items"][0]["id"] == first_conversation

    updated = client.patch(
        f"/api/v1/conversations/{first_conversation}",
        json={"title": "空压机高温诊断"},
    )
    assert updated.status_code == 200
    assert updated.json()["data"]["title"] == "空压机高温诊断"

    detail = client.get(f"/api/v1/conversations/{first_conversation}")
    assert detail.status_code == 200
    assert detail.json()["data"]["title"] == "空压机高温诊断"

    assert client.delete(f"/api/v1/conversations/{first_conversation}").status_code == 200
    assert client.get(f"/api/v1/conversations/{first_conversation}").status_code == 404


def test_multi_turn_diagnosis_persists_history_and_citations(
    client: TestClient,
    llm_provider: LLMProvider,
) -> None:
    knowledge_base_id = create_knowledge_base(client)
    document_id = prepare_indexed_document(client, knowledge_base_id)
    conversation_id = create_conversation(client, knowledge_base_id)

    first_question = "空压机E101高温停机怎么排查？"
    first_reply = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"question": first_question, "top_k": 2, "score_threshold": 0.1},
    )
    assert first_reply.status_code == 200
    first_result = first_reply.json()["data"]
    assert first_result["conversation"]["title"] == first_question
    assert first_result["user_message"]["role"] == "user"
    assert first_result["assistant_message"]["role"] == "assistant"
    assert first_result["assistant_message"]["citations"][0]["document_id"] == document_id
    assert first_result["sources"][0]["citation"] == "[资料1]"

    second_reply = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"question": "那第二步具体检查什么？", "top_k": 2, "score_threshold": 0.1},
    )
    assert second_reply.status_code == 200
    assert second_reply.json()["data"]["sources"][0]["section_title"] == "高温停机 E101"

    assert len(llm_provider.calls) == 2  # type: ignore[attr-defined]
    _, _, second_history = llm_provider.calls[1]  # type: ignore[attr-defined]
    assert [message.role for message in second_history] == ["user", "assistant"]
    assert second_history[0].content == first_question
    assert "[资料1]" in second_history[1].content

    messages = client.get(f"/api/v1/conversations/{conversation_id}/messages")
    assert messages.status_code == 200
    message_data = messages.json()["data"]
    assert message_data["total"] == 4
    assert [item["role"] for item in message_data["items"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_conversation_survives_knowledge_base_deletion(client: TestClient) -> None:
    knowledge_base_id = create_knowledge_base(client)
    conversation_id = create_conversation(client, knowledge_base_id)

    assert client.delete(f"/api/v1/knowledge-bases/{knowledge_base_id}").status_code == 200

    detail = client.get(f"/api/v1/conversations/{conversation_id}")
    assert detail.status_code == 200
    assert detail.json()["data"]["knowledge_base_id"] is None

    reply = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"question": "还能继续诊断吗？"},
    )
    assert reply.status_code == 409
    assert "知识库已被删除" in reply.json()["message"]
