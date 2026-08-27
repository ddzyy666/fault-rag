import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse


class ServerSentEventResponse(StreamingResponse):
    """让FastAPI在OpenAPI中声明正确的SSE媒体类型。"""

    media_type = "text/event-stream"


def encode_sse(event: str, data: dict[str, Any]) -> str:
    """按照SSE规范编码一个命名JSON事件。"""
    payload = json.dumps(
        jsonable_encoder(data),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event}\ndata: {payload}\n\n"


def sse_response(content: AsyncIterator[str]) -> ServerSentEventResponse:
    return ServerSentEventResponse(
        content,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
