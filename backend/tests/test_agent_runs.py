import asyncio
import json
from contextlib import aclosing
from datetime import timedelta
from unittest.mock import patch
from uuid import UUID

import pytest
from app.core.config import settings
from app.db.base import utc_now
from app.db.database import get_db
from app.main import app
from app.models.agent_run import AgentRun, AgentToolCall
from app.models.conversation import Conversation, Message, MessageRole
from app.services.conversation_diagnosis import stream_agent_conversation
from app.services.embedding import get_embedding_provider
from app.services.llm import LLMError, get_llm_provider
from app.services.reranker import get_reranker_provider
from app.services.sparse_embedding import get_sparse_embedding_provider
from app.services.vector_store import get_vector_store
from backend.tests.test_agent import ScriptedAgent, call, conversation
from backend.tests.test_streaming import parse_sse, prepare_knowledge_base
from pydantic import SecretStr
from sqlalchemy import func, select


def runs(client, cid):
    return client.get(f"/api/v1/conversations/{cid}/runs").json()["data"]["items"]


def detail(client, cid, run):
    return client.get(f"/api/v1/conversations/{cid}/runs/{run['id']}").json()["data"]


@pytest.mark.parametrize("stream", [False, True])
def test_runs_survive_reopen_and_link_to_answer(client, stream):
    kb, _ = prepare_knowledge_base(client)
    cid = conversation(client, kb)
    agent = ScriptedAgent(
        [
            {
                "reasoning_content": "private reasoning",
                "tool_calls": [
                    call(
                        "search_manual",
                        {
                            "query": "E101",
                            "knowledge_base_id": kb,
                        },
                    )
                ],
            },
            {"content": "检查冷却器。[资料1]"},
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: agent
    endpoint = f"/api/v1/conversations/{cid}/messages" + ("/stream" if stream else "")
    response = client.post(endpoint, json={"mode": "agent", "question": "E101"})
    assert response.status_code == 200
    saved_runs = runs(client, cid)
    assert len(saved_runs) == 1
    run = saved_runs[0]
    assert run["status"] == "succeeded" and run["phase"] == "completed"
    assert run["usage"]["total_tokens"] == 30
    assert run["elapsed_ms"] >= 0
    saved_detail = detail(client, cid, run)
    assert saved_detail["tool_calls"][0]["status"] == "ok"
    assert saved_detail["tool_calls"][0]["arguments"]["query"] == "E101"
    assert saved_detail["tool_calls"][0]["result_summary"]["source_count"] == 1
    serialized = json.dumps(saved_detail, ensure_ascii=False)
    assert "private reasoning" not in serialized
    assert "环境温度和润滑油状态" not in serialized
    messages = client.get(f"/api/v1/conversations/{cid}/messages").json()["data"]["items"]
    assert run["assistant_message_id"] == messages[-1]["id"]
    filtered = client.get(
        f"/api/v1/conversations/{cid}/runs",
        params={
            "assistant_message_id": messages[-1]["id"],
        },
    ).json()["data"]
    assert filtered["total"] == 1
    if stream:
        events = parse_sse(response.text)
        assert events[0][1]["run_id"] == run["id"]
        assert events[-1][1]["run_id"] == run["id"]
    other = conversation(client, kb)
    assert client.get(f"/api/v1/conversations/{other}/runs/{run['id']}").status_code == 404
    assert runs(client, other) == []


def test_model_failure_retains_prior_tools_and_usage(client):
    kb, _ = prepare_knowledge_base(client)
    cid = conversation(client, kb)
    agent = ScriptedAgent(
        [
            {"tool_calls": [call("get_device_history", {"device_id": "missing"})]},
            LLMError("大模型请求超时"),
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: agent
    response = client.post(
        f"/api/v1/conversations/{cid}/messages/stream",
        json={
            "mode": "agent",
            "question": "设备修过什么",
        },
    )
    assert parse_sse(response.text)[-1][0] == "error"
    run = runs(client, cid)[0]
    assert run["status"] == "failed" and run["phase"] == "model"
    assert run["usage"]["total_tokens"] == 15
    assert run["assistant_message_id"] is None
    assert "超时" in run["failure_reason"]
    assert detail(client, cid, run)["tool_calls"][0]["status"] == "not_found"
    assert client.get(f"/api/v1/conversations/{cid}/messages").json()["data"]["total"] == 0


def test_message_save_failure_keeps_run_and_rolls_back_messages(client):
    kb, _ = prepare_knowledge_base(client)
    cid = conversation(client, kb)
    agent = ScriptedAgent([{"content": "请补充编号"}])
    app.dependency_overrides[get_llm_provider] = lambda: agent

    async def fail_save(session, conversation, **kwargs):
        session.add(
            Message(conversation_id=conversation.id, role=MessageRole.USER, content="partial")
        )
        await session.flush()
        raise RuntimeError("private database error")

    with patch("app.repositories.conversation.save_exchange", new=fail_save):
        client.post(
            f"/api/v1/conversations/{cid}/messages/stream",
            json={
                "mode": "agent",
                "question": "坏了",
            },
        )
    run = runs(client, cid)[0]
    assert run["status"] == "failed" and run["phase"] == "persisting"
    assert "private" not in run["failure_reason"]
    assert client.get(f"/api/v1/conversations/{cid}/messages").json()["data"]["total"] == 0


def test_closing_stream_keeps_cancelled_tool(client):
    kb, _ = prepare_knowledge_base(client)
    cid = conversation(client, kb)
    agent = ScriptedAgent([{"tool_calls": [call("get_device_history", {"device_id": "AC-003"})]}])

    async def close_during_tool():
        async for session in app.dependency_overrides[get_db]():
            conv = await session.get(Conversation, UUID(cid))
            async with aclosing(
                stream_agent_conversation(
                    session,
                    conv,
                    question="3号设备",
                    llm_provider=agent,
                    embedding_provider=app.dependency_overrides[get_embedding_provider](),
                    sparse_embedding_provider=app.dependency_overrides[
                        get_sparse_embedding_provider
                    ](),
                    vector_store=app.dependency_overrides[get_vector_store](),
                    reranker_provider=app.dependency_overrides[get_reranker_provider](),
                )
            ) as updates:
                async for event in updates:
                    if event.event == "tool_started":
                        break

    asyncio.run(close_during_tool())
    run = runs(client, cid)[0]
    assert run["status"] == "cancelled"
    assert detail(client, cid, run)["tool_calls"][0]["status"] == "cancelled"


def test_redaction_and_no_raw_unknown_arguments(client, monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", SecretStr("configured-secret"))
    kb, _ = prepare_knowledge_base(client)
    cid = conversation(client, kb)
    agent = ScriptedAgent(
        [
            {
                "tool_calls": [
                    call(
                        "search_manual",
                        {
                            "query": "configured-secret token=private-token sk-test-secret",
                            "knowledge_base_id": kb,
                            "api_key": "unknown-key",
                            "nested": {"secret": "nested-key"},
                        },
                    )
                ]
            },
            {"content": "请重新提供查询"},
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: agent
    client.post(
        f"/api/v1/conversations/{cid}/messages/stream",
        json={
            "mode": "agent",
            "question": "configured-secret Bearer hidden-key",
        },
    )
    run = runs(client, cid)[0]
    serialized = json.dumps(detail(client, cid, run))
    for secret in [
        "configured-secret",
        "private-token",
        "sk-test-secret",
        "unknown-key",
        "nested-key",
        "hidden-key",
    ]:
        assert secret not in serialized
    assert "REDACTED" in serialized


def test_expired_runs_and_cascade_delete(client):
    kb, _ = prepare_knowledge_base(client)
    cid = conversation(client, kb)

    async def seed():
        async for session in app.dependency_overrides[get_db]():
            run = AgentRun(
                conversation_id=UUID(cid),
                question="过期执行",
                model_name="test",
                expires_at=utc_now() - timedelta(seconds=1),
            )
            session.add(run)
            await session.flush()
            session.add(
                AgentToolCall(
                    run_id=run.id,
                    sequence=1,
                    step=1,
                    call_id="1",
                    tool_name="search_manual",
                    arguments={},
                )
            )
            await session.commit()

    asyncio.run(seed())
    run = runs(client, cid)[0]
    assert run["status"] == "interrupted" and run["elapsed_ms"] is None
    assert client.delete(f"/api/v1/conversations/{cid}").status_code == 200

    async def check_deleted():
        async for session in app.dependency_overrides[get_db]():
            assert await session.scalar(select(func.count()).select_from(AgentRun)) == 0
            assert await session.scalar(select(func.count()).select_from(AgentToolCall)) == 0

    asyncio.run(check_deleted())
