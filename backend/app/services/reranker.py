from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import httpx

from app.core.config import settings


class RerankerError(RuntimeError):
    """Reranker配置、网络请求或响应解析失败。"""


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


class RerankerProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]: ...


class SiliconFlowRerankerProvider:
    """调用硅基流动文本Rerank接口。"""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def endpoint(self) -> str:
        if self._base_url.endswith("/rerank"):
            return self._base_url
        return f"{self._base_url}/rerank"

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        if not documents:
            return []
        if not self._api_key:
            raise RerankerError("RERANKER_API_KEY和LLM_API_KEY均未配置")

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name,
                        "query": query,
                        "documents": documents,
                        "top_n": min(top_n, len(documents)),
                        "return_documents": False,
                    },
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RerankerError("Reranker请求超时") from exc
        except httpx.HTTPStatusError as exc:
            raise RerankerError(f"Reranker服务返回HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise RerankerError("无法连接Reranker服务") from exc

        try:
            payload = response.json()
            results = [
                RerankResult(
                    index=int(item["index"]),
                    score=float(item["relevance_score"]),
                )
                for item in payload["results"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankerError("Reranker响应格式不正确") from exc

        if any(result.index < 0 or result.index >= len(documents) for result in results):
            raise RerankerError("Reranker返回了无效文档索引")
        if len({result.index for result in results}) != len(results):
            raise RerankerError("Reranker返回了重复文档索引")
        return sorted(results, key=lambda result: result.score, reverse=True)


@lru_cache
def get_reranker_provider() -> RerankerProvider:
    configured_key = (
        settings.reranker_api_key.get_secret_value()
        if settings.reranker_api_key is not None
        else None
    )
    llm_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key is not None else None
    return SiliconFlowRerankerProvider(
        base_url=settings.reranker_base_url,
        model_name=settings.reranker_model_name,
        api_key=configured_key or llm_key,
        timeout_seconds=settings.reranker_timeout_seconds,
    )
