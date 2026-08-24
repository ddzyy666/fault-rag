import asyncio
import json

import httpx
from app.services.llm import OpenAICompatibleLLMProvider


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

    result = asyncio.run(provider.generate("系统提示", "用户问题"))

    assert captured_request["url"] == "https://llm.example/v1/chat/completions"
    assert captured_request["authorization"] == "Bearer secret-key"
    request_body = captured_request["body"]
    assert isinstance(request_body, dict)
    assert request_body["model"] == "example-model"
    assert request_body["messages"][0] == {  # type: ignore[index]
        "role": "system",
        "content": "系统提示",
    }
    assert result.content == "诊断结果"
    assert result.usage.total_tokens == 14
