import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol

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


@dataclass(frozen=True, slots=True)
class LLMToolTurn:
    """完整 assistant 消息，包含必须原样回传的工具调用及服务商扩展字段。"""

    message: dict[str, Any]
    usage: LLMUsage


@dataclass(frozen=True, slots=True)
class LLMChatMessage:
    """发送给兼容接口的一条历史对话消息。"""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class LLMStreamChunk:
    """上游流式响应的一段正文或Token用量。"""

    content: str = ""
    usage: LLMUsage | None = None


class LLMProvider(Protocol):
    """RAG服务依赖的大模型最小接口。"""

    @property
    def model_name(self) -> str: ...

    async def tool_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolTurn: ...

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[LLMChatMessage] | None = None,
    ) -> LLMGeneration: ...

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[LLMChatMessage] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]: ...


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

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[LLMChatMessage] | None = None,
    ) -> LLMGeneration:
        self._validate_configuration()
        request_body = self._build_request_body(
            system_prompt,
            user_prompt,
            history,
            stream=False,
        )

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=request_body,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError("大模型请求超时") from exc
        except httpx.HTTPStatusError as exc:
            raise _http_status_error(exc.response.status_code) from exc
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

    async def tool_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMToolTurn:
        self._validate_configuration()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens,
                        "stream": False,
                    },
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError("大模型工具决策请求超时") from exc
        except httpx.HTTPStatusError as exc:
            raise _http_status_error(exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise LLMError("无法连接大模型服务") from exc
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("invalid assistant")
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise ValueError("incomplete response")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("invalid calls")
            ids = set()
            for call in calls:
                call_id = call["id"]
                if not isinstance(call_id, str) or not call_id or call_id in ids:
                    raise ValueError("invalid call id")
                ids.add(call_id)
                if (
                    call["type"] != "function"
                    or not isinstance(call["function"]["name"], str)
                    or not isinstance(call["function"]["arguments"], str)
                ):
                    raise ValueError("invalid function")
            content = message.get("content")
            if not calls and (not isinstance(content, str) or not content.strip()):
                raise ValueError("empty answer")
            return LLMToolTurn(
                message=message, usage=_parse_usage(payload.get("usage")) or LLMUsage()
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise LLMError("大模型工具响应格式不正确或回答未完整结束") from exc

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[LLMChatMessage] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """解析兼容接口返回的data行，并逐段产出正文。"""
        self._validate_configuration()
        request_body = self._build_request_body(
            system_prompt,
            user_prompt,
            history,
            stream=True,
        )
        received_content = False
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=self._headers(),
                    json=request_body,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        raw_data = line.removeprefix("data:").strip()
                        if raw_data == "[DONE]":
                            break
                        chunk = _parse_stream_data(raw_data)
                        if chunk.content:
                            received_content = True
                        if chunk.content or chunk.usage is not None:
                            yield chunk
        except httpx.TimeoutException as exc:
            raise LLMError("大模型流式请求超时") from exc
        except httpx.HTTPStatusError as exc:
            raise _http_status_error(exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise LLMError("无法连接大模型流式服务") from exc

        if not received_content:
            raise LLMError("大模型流式响应没有返回正文")

    def _validate_configuration(self) -> None:
        if not self._base_url or not self.model_name:
            raise LLMError("LLM_BASE_URL或LLM_MODEL_NAME尚未配置")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_request_body(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[LLMChatMessage] | None,
        *,
        stream: bool,
    ) -> dict[str, object]:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": message.role, "content": message.content} for message in history or []
        )
        messages.append({"role": "user", "content": user_prompt})
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": stream,
        }


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _parse_stream_data(raw_data: str) -> LLMStreamChunk:
    try:
        payload = json.loads(raw_data)
        choices = payload.get("choices") or []
        content = ""
        if choices:
            raw_content = choices[0].get("delta", {}).get("content")
            if isinstance(raw_content, str):
                content = raw_content
        usage = _parse_usage(payload.get("usage"))
        return LLMStreamChunk(content=content, usage=usage)
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as exc:
        raise LLMError("大模型流式响应格式不正确") from exc


def _parse_usage(raw_usage: object) -> LLMUsage | None:
    if not isinstance(raw_usage, dict):
        return None
    usage = LLMUsage(
        prompt_tokens=_optional_int(raw_usage.get("prompt_tokens")),
        completion_tokens=_optional_int(raw_usage.get("completion_tokens")),
        total_tokens=_optional_int(raw_usage.get("total_tokens")),
    )
    if all(
        value is None
        for value in (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
    ):
        return None
    return usage


def _http_status_error(status_code: int) -> LLMError:
    if status_code in {401, 403}:
        message = "大模型鉴权失败，请检查LLM_API_KEY"
    elif status_code == 429:
        message = "大模型请求过于频繁或额度不足"
    else:
        message = f"大模型服务返回HTTP {status_code}"
    return LLMError(message)


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
