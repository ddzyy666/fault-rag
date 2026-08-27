import json
from collections.abc import AsyncIterator

from app.main import app
from app.services.llm import (
    LLMChatMessage,
    LLMError,
    LLMGeneration,
    LLMProvider,
    LLMStreamChunk,
    get_llm_provider,
)
from fastapi.testclient import TestClient


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data += line.removeprefix("data:").strip()
        events.append((event_name, json.loads(data)))
    return events


def prepare_knowledge_base(client: TestClient) -> tuple[str, str]:
    knowledge_base = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "流式诊断知识库"},
    )
    assert knowledge_base.status_code == 201
    knowledge_base_id = knowledge_base.json()["data"]["id"]
    markdown = """# 空压机维修手册

## 高温停机 E101

空压机高温停机时，应检查冷却器堵塞、冷却风扇、环境温度和润滑油状态。
"""
    upload = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("流式手册.md", markdown.encode("utf-8"), "text/markdown")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["data"]["id"]
    chunks = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 300, "chunk_overlap": 40, "min_chunk_size": 20},
    )
    assert chunks.status_code == 200
    assert client.post(f"/api/v1/documents/{document_id}/index").status_code == 200
    return knowledge_base_id, document_id


def request_stream(
    client: TestClient,
    path: str,
    payload: dict[str, object],
) -> tuple[str, str]:
    with client.stream("POST", path, json=payload) as response:
        assert response.status_code == 200
        content_type = response.headers["content-type"]
        body = "".join(response.iter_text())
    return content_type, body


def test_single_turn_sse_emits_sources_deltas_and_completion(client: TestClient) -> None:
    knowledge_base_id, document_id = prepare_knowledge_base(client)

    content_type, body = request_stream(
        client,
        f"/api/v1/knowledge-bases/{knowledge_base_id}/ask/stream",
        {
            "question": "空压机E101高温停机怎么排查？",
            "top_k": 2,
            "score_threshold": 0.1,
        },
    )

    assert content_type.startswith("text/event-stream")
    events = parse_sse(body)
    event_names = [event for event, _ in events]
    assert event_names == [
        "retrieval_started",
        "sources",
        "answer_delta",
        "answer_delta",
        "answer_delta",
        "completed",
    ]
    sources = events[1][1]["items"]
    assert isinstance(sources, list)
    assert sources[0]["document_id"] == document_id
    streamed_answer = "".join(data["delta"] for event, data in events if event == "answer_delta")
    completed = events[-1][1]
    assert completed["answer"] == streamed_answer
    assert completed["usage"]["total_tokens"] == 160
    assert completed["llm_called"] is True


def test_stream_without_context_skips_llm(
    client: TestClient,
    llm_provider: LLMProvider,
) -> None:
    knowledge_base = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "空流式知识库"},
    )
    knowledge_base_id = knowledge_base.json()["data"]["id"]

    _, body = request_stream(
        client,
        f"/api/v1/knowledge-bases/{knowledge_base_id}/ask/stream",
        {"question": "未知设备发生未知故障"},
    )

    events = parse_sse(body)
    assert [event for event, _ in events] == [
        "retrieval_started",
        "sources",
        "answer_delta",
        "completed",
    ]
    assert events[1][1]["items"] == []
    assert events[-1][1]["llm_called"] is False
    assert len(llm_provider.calls) == 0  # type: ignore[attr-defined]


def test_conversation_stream_persists_only_completed_answer(client: TestClient) -> None:
    knowledge_base_id, _ = prepare_knowledge_base(client)
    conversation = client.post(
        "/api/v1/conversations",
        json={"knowledge_base_id": knowledge_base_id},
    )
    conversation_id = conversation.json()["data"]["id"]

    _, body = request_stream(
        client,
        f"/api/v1/conversations/{conversation_id}/messages/stream",
        {
            "question": "空压机高温怎么排查？",
            "top_k": 2,
            "score_threshold": 0.1,
        },
    )

    events = parse_sse(body)
    completed = events[-1]
    assert completed[0] == "completed"
    assert completed[1]["conversation_id"] == conversation_id
    assert completed[1]["user_message_id"]
    assert completed[1]["assistant_message_id"]

    messages = client.get(f"/api/v1/conversations/{conversation_id}/messages")
    assert messages.status_code == 200
    items = messages.json()["data"]["items"]
    assert [item["role"] for item in items] == ["user", "assistant"]
    assert items[1]["content"] == completed[1]["answer"]
    assert items[1]["citations"][0]["citation"] == "[资料1]"


class FailingStreamLLMProvider:
    model_name = "failing-stream-model"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[LLMChatMessage] | None = None,
    ) -> LLMGeneration:
        raise LLMError("模拟模型故障")

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[LLMChatMessage] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        if False:
            yield LLMStreamChunk()
        raise LLMError("模拟模型流中断")


def test_conversation_stream_error_does_not_persist_partial_messages(
    client: TestClient,
) -> None:
    knowledge_base_id, _ = prepare_knowledge_base(client)
    conversation = client.post(
        "/api/v1/conversations",
        json={"knowledge_base_id": knowledge_base_id},
    )
    conversation_id = conversation.json()["data"]["id"]
    original_override = app.dependency_overrides[get_llm_provider]
    app.dependency_overrides[get_llm_provider] = FailingStreamLLMProvider
    try:
        _, body = request_stream(
            client,
            f"/api/v1/conversations/{conversation_id}/messages/stream",
            {"question": "空压机高温怎么排查？", "score_threshold": 0.1},
        )
    finally:
        app.dependency_overrides[get_llm_provider] = original_override

    events = parse_sse(body)
    assert events[-1] == ("error", {"message": "模拟模型流中断"})
    messages = client.get(f"/api/v1/conversations/{conversation_id}/messages")
    assert messages.json()["data"]["total"] == 0
