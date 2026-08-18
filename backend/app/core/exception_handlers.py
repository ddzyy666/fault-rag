from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def error_content(code: int, message: str, details: Any = None) -> dict[str, Any]:
    """构造统一错误响应。"""
    return {
        "code": code,
        "message": message,
        "data": details,
    }


def register_exception_handlers(application: FastAPI) -> None:
    """注册应用级异常处理器。"""

    @application.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_content(exc.status_code, str(exc.detail)),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_content(
                422,
                "请求参数校验失败",
                jsonable_encoder(exc.errors()),
            ),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_exception(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_content(500, "服务器内部错误"),
        )
