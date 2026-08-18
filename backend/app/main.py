from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers


def create_application() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        description="基于 RAG 的智能设备故障诊断系统",
    )

    application.include_router(api_router, prefix=settings.api_v1_prefix)
    register_exception_handlers(application)

    return application


app = create_application()
