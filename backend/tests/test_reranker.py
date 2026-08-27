import asyncio
import json

import httpx
from app.services.reranker import SiliconFlowRerankerProvider


def test_siliconflow_reranker_builds_request_and_sorts_results() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.35},
                    {"index": 1, "relevance_score": 0.92},
                ]
            },
        )

    provider = SiliconFlowRerankerProvider(
        base_url="https://api.siliconflow.cn/v1",
        model_name="BAAI/bge-reranker-v2-m3",
        api_key="secret-key",
        timeout_seconds=10,
        transport=httpx.MockTransport(handler),
    )

    results = asyncio.run(provider.rerank("E101高温", ["普通维护", "E101高温停机"], 2))

    assert captured["url"] == "https://api.siliconflow.cn/v1/rerank"
    assert captured["authorization"] == "Bearer secret-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["return_documents"] is False
    assert body["top_n"] == 2
    assert [result.index for result in results] == [1, 0]
    assert results[0].score == 0.92
