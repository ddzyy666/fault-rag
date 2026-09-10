import asyncio
import copy
import json
from unittest.mock import AsyncMock, patch
from uuid import UUID

import httpx
import pytest
from app.core.config import settings
from app.db.database import get_db
from app.main import app
from app.services.demo_devices import seed_demo_devices
from app.services.llm import (
    LLMError,
    LLMToolTurn,
    LLMUsage,
    OpenAICompatibleLLMProvider,
    get_llm_provider,
)
from backend.tests.test_streaming import parse_sse, prepare_knowledge_base


def call(name, args, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


class ScriptedAgent:
    model_name = "test-agent"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    async def tool_turn(self, messages, tools):
        self.messages.append(copy.deepcopy(messages))
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return LLMToolTurn(message={"role": "assistant", **value}, usage=LLMUsage(10, 5, 15))


def conversation(client, kb_id):
    return client.post(
        "/api/v1/conversations",
        json={
            "knowledge_base_id": kb_id,
        },
    ).json()["data"]["id"]


def test_agent_tool_results_citations_and_persistence(client):
    kb_id, _ = prepare_knowledge_base(client)

    async def seed():
        async for session in app.dependency_overrides[get_db]():
            await seed_demo_devices(session, UUID(kb_id))
            await session.commit()

    asyncio.run(seed())
    agent = ScriptedAgent(
        [
            {"content": None, "tool_calls": [call("get_device_history", {"device_id": "AC-003"})]},
            {
                "content": None,
                "tool_calls": [
                    call(
                        "search_manual",
                        {
                            "query": "E101 高温",
                            "knowledge_base_id": kb_id,
                        },
                        "call-2",
                    )
                ],
            },
            {"content": "模拟记录显示更换过空气滤芯。手册建议检查冷却器。[资料1]"},
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: agent
    cid = conversation(client, kb_id)
    response = client.post(
        f"/api/v1/conversations/{cid}/messages/stream",
        json={"question": "3号空压机报E101", "mode": "agent"},
    )
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "agent_started" and names[-1] == "completed"
    assert names.count("tool_completed") == 2
    tool_messages = [m for m in agent.messages[-1] if m["role"] == "tool"]
    history_result = json.loads(tool_messages[0]["content"])
    assert history_result["status"] == "ok"
    assert history_result["data"]["records"][0]["action"] == "更换空气滤芯"
    assert tool_messages[1]["tool_call_id"] == "call-2"
    assert events[-1][1]["usage"]["total_tokens"] == 45
    saved = client.get(f"/api/v1/conversations/{cid}/messages").json()["data"]["items"]
    assert len(saved) == 2
    assert saved[-1]["citations"][0]["citation"] == "[资料1]"


def test_agent_sync_clarification_and_history(client):
    kb_id, _ = prepare_knowledge_base(client)
    agent = ScriptedAgent([{"content": "请补充设备编号。"}, {"content": "请补充温度。"}])
    app.dependency_overrides[get_llm_provider] = lambda: agent
    cid = conversation(client, kb_id)
    for question in ["机器坏了", "3号空压机"]:
        response = client.post(
            f"/api/v1/conversations/{cid}/messages",
            json={
                "question": question,
                "mode": "agent",
            },
        )
        assert response.status_code == 200
        assert response.json()["data"]["sources"] == []
    assert agent.messages[-1][1]["content"] == "机器坏了"
    assert agent.messages[-1][2]["content"] == "请补充设备编号。"


@pytest.mark.parametrize("failure", ["model", "steps", "calls", "timeout", "context"])
def test_agent_failure_does_not_save_partial_answer(client, monkeypatch, failure):
    kb_id, _ = prepare_knowledge_base(client)
    agent = ScriptedAgent([LLMError("模拟失败")])
    if failure in {"steps", "calls"}:
        monkeypatch.setattr(settings, "agent_max_steps", 1)
        monkeypatch.setattr(settings, "agent_max_tool_calls", 1)
        calls = [call("get_device_history", {"device_id": "missing"})]
        if failure == "calls":
            calls.append(call("get_device_history", {"device_id": "missing"}, "call-2"))
        agent = ScriptedAgent([{"tool_calls": calls, "content": "中间分析不应保存"}])
    if failure == "timeout":
        monkeypatch.setattr(settings, "agent_timeout_seconds", 0.01)

        async def slow(*args):
            await asyncio.sleep(1)

        agent.tool_turn = slow
    if failure == "context":
        monkeypatch.setattr(settings, "agent_max_context_chars", 1)
    app.dependency_overrides[get_llm_provider] = lambda: agent
    cid = conversation(client, kb_id)
    response = client.post(
        f"/api/v1/conversations/{cid}/messages/stream",
        json={
            "question": "E101",
            "mode": "agent",
        },
    )
    events = parse_sse(response.text)
    assert events[-1][0] == "error"
    assert "completed" not in [name for name, _ in events]
    assert client.get(f"/api/v1/conversations/{cid}/messages").json()["data"]["total"] == 0


def test_agent_recovers_invalid_json(client):
    kb_id, _ = prepare_knowledge_base(client)
    invalid = call("get_device_history", {})
    invalid["function"]["arguments"] = "{broken"
    agent = ScriptedAgent(
        [
            {"tool_calls": [invalid]},
            {"content": "请核对设备编号。"},
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: agent
    cid = conversation(client, kb_id)
    response = client.post(
        f"/api/v1/conversations/{cid}/messages",
        json={
            "question": "E101",
            "mode": "agent",
        },
    )
    assert response.status_code == 200
    assert json.loads(agent.messages[-1][-1]["content"])["status"] == "invalid_arguments"


def test_agent_tool_failure_rollback_can_still_save(client):
    kb_id, _ = prepare_knowledge_base(client)
    agent = ScriptedAgent(
        [
            {"tool_calls": [call("get_device_history", {"device_id": "AC-003"})]},
            {"content": "暂时无法读取维修记录，请稍后重试。"},
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: agent
    cid = conversation(client, kb_id)
    with patch(
        "app.services.diagnostic_tools.DiagnosticTools.get_device_history",
        new=AsyncMock(side_effect=RuntimeError("test failure")),
    ):
        response = client.post(
            f"/api/v1/conversations/{cid}/messages/stream",
            json={
                "question": "3号机修过什么",
                "mode": "agent",
            },
        )
    assert parse_sse(response.text)[-1][0] == "completed"
    assert client.get(f"/api/v1/conversations/{cid}/messages").json()["data"]["total"] == 2


def test_agent_repeated_retrieval_keeps_citation_identity(client):
    kb_id, _ = prepare_knowledge_base(client)
    agent = ScriptedAgent(
        [
            {"tool_calls": [call("search_manual", {"query": "E101", "knowledge_base_id": kb_id})]},
            {
                "tool_calls": [
                    call("search_manual", {"query": "高温", "knowledge_base_id": kb_id}, "c2")
                ]
            },
            {"content": "检查冷却器。[资料1]"},
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: agent
    cid = conversation(client, kb_id)
    result = client.post(
        f"/api/v1/conversations/{cid}/messages",
        json={
            "question": "E101高温",
            "mode": "agent",
        },
    )
    assert result.status_code == 200
    assert len(result.json()["data"]["sources"]) == 1
    tool_results = [json.loads(m["content"]) for m in agent.messages[-1] if m["role"] == "tool"]
    assert all(r["data"]["sources"][0]["citation"] == "[资料1]" for r in tool_results)


def test_provider_tool_protocol_and_reasoning_content():
    captured = []
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "opaque",
        "tool_calls": [call("get_device_history", {"device_id": "AC-003"})],
    }

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": assistant, "finish_reason": "tool_calls"}]}
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://example.test/v1",
        model_name="test",
        api_key=None,
        timeout_seconds=2,
        temperature=0,
        max_tokens=100,
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(provider.tool_turn([{"role": "user", "content": "E101"}], []))
    assert result.message == assistant
    assert captured[0]["tool_choice"] == "auto"
    assert captured[0]["stream"] is False


@pytest.mark.parametrize(
    "message,reason",
    [
        ({"role": "assistant", "content": "半截"}, "length"),
        ({"role": "assistant", "content": None}, "stop"),
        ({"role": "assistant", "tool_calls": [{"id": "x"}]}, "tool_calls"),
    ],
)
def test_provider_rejects_incomplete_tool_response(message, reason):
    provider = OpenAICompatibleLLMProvider(
        base_url="https://example.test/v1",
        model_name="test",
        api_key=None,
        timeout_seconds=2,
        temperature=0,
        max_tokens=100,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"message": message, "finish_reason": reason}],
                },
            )
        ),
    )
    with pytest.raises(LLMError):
        asyncio.run(provider.tool_turn([], []))
