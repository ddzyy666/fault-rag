import asyncio
import json

import httpx
from app.services.llm import LLMChatMessage, OpenAICompatibleLLMProvider


def test_openai_compatible_provider_builds_request_and_parses_usage() -> None:
    captured_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["authorization"] = request.headers.get("Authorization")
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "诊断结果"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://llm.example/v1",
        model_name="example-model",
        api_key="secret-key",
        timeout_seconds=10,
        temperature=0.2,
        max_tokens=500,
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        provider.generate(
            "系统提示",
            "当前问题",
            history=[
                LLMChatMessage(role="user", content="上一轮问题"),
                LLMChatMessage(role="assistant", content="上一轮回答"),
            ],
        )
    )

    assert captured_request["url"] == "https://llm.example/v1/chat/completions"
    assert captured_request["authorization"] == "Bearer secret-key"
    request_body = captured_request["body"]
    assert isinstance(request_body, dict)
    assert request_body["model"] == "example-model"
    messages = request_body["messages"]
    assert messages[0] == {  # type: ignore[index]
        "role": "system",
        "content": "系统提示",
    }
    assert messages[1] == {"role": "user", "content": "上一轮问题"}  # type: ignore[index]
    assert messages[2] == {  # type: ignore[index]
        "role": "assistant",
        "content": "上一轮回答",
    }
    assert messages[3] == {"role": "user", "content": "当前问题"}  # type: ignore[index]
    assert result.content == "诊断结果"
    assert result.usage.total_tokens == 14


def test_openai_compatible_provider_parses_stream_chunks() -> None:
    captured_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        stream_body = "".join(
            [
                'data: {"choices":[{"delta":{"content":"第一段"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"第二段"}}]}\n\n',
                (
                    'data: {"choices":[],"usage":{"prompt_tokens":8,'
                    '"completion_tokens":4,"total_tokens":12}}\n\n'
                ),
                "data: [DONE]\n\n",
            ]
        )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=stream_body.encode("utf-8"),
        )

    provider = OpenAICompatibleLLMProvider(
        base_url="https://llm.example/v1",
        model_name="example-model",
        api_key="secret-key",
        timeout_seconds=10,
        temperature=0.2,
        max_tokens=500,
        transport=httpx.MockTransport(handler),
    )

    async def collect_chunks():
        return [chunk async for chunk in provider.stream("系统提示", "用户问题")]

    chunks = asyncio.run(collect_chunks())

    assert captured_body["stream"] is True
    assert "".join(chunk.content for chunk in chunks) == "第一段第二段"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 12
