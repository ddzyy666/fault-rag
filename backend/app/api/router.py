from fastapi import APIRouter

from app.api.routes import health, knowledge_bases

api_router = APIRouter()
api_router.include_router(health.router, tags=["系统状态"])
api_router.include_router(knowledge_bases.router, tags=["知识库"])
