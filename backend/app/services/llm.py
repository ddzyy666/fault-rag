from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import httpx

from app.core.config import settings


class LLMError(RuntimeError):
    """大模型配置、网络请求或响应解析失败。"""


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """模型返回的Token用量；兼容服务未返回时字段为空。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LLMGeneration:
    """一次非流式文本生成结果。"""

    content: str
    usage: LLMUsage


class LLMProvider(Protocol):
    """RAG服务依赖的大模型最小接口。"""

    @property
    def model_name(self) -> str: ...

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMGeneration: ...


class OpenAICompatibleLLMProvider:
    """调用兼容Chat Completions协议的大模型服务。"""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._transport = transport

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def endpoint(self) -> str:
        if self._base_url.endswith("/chat/completions"):
            return self._base_url
        return f"{self._base_url}/chat/completions"

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMGeneration:
        if not self._base_url or not self.model_name:
            raise LLMError("LLM_BASE_URL或LLM_MODEL_NAME尚未配置")

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=request_body,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError("大模型请求超时") from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code in {401, 403}:
                message = "大模型鉴权失败，请检查LLM_API_KEY"
            elif status_code == 429:
                message = "大模型请求过于频繁或额度不足"
            else:
                message = f"大模型服务返回HTTP {status_code}"
            raise LLMError(message) from exc
        except httpx.HTTPError as exc:
            raise LLMError("无法连接大模型服务") from exc

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("模型回答为空")
            raw_usage = payload.get("usage") or {}
            usage = LLMUsage(
                prompt_tokens=_optional_int(raw_usage.get("prompt_tokens")),
                completion_tokens=_optional_int(raw_usage.get("completion_tokens")),
                total_tokens=_optional_int(raw_usage.get("total_tokens")),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("大模型响应格式不正确") from exc
        return LLMGeneration(content=content.strip(), usage=usage)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


@lru_cache
def get_llm_provider() -> LLMProvider:
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key is not None else None
    return OpenAICompatibleLLMProvider(
        base_url=settings.llm_base_url,
        model_name=settings.llm_model_name,
        api_key=api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
