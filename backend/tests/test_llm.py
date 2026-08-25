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
