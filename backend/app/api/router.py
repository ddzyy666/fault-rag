from fastapi import APIRouter

from app.api.routes import documents, health, knowledge_bases, retrieval

api_router = APIRouter()
api_router.include_router(health.router, tags=["系统状态"])
api_router.include_router(knowledge_bases.router, tags=["知识库"])
api_router.include_router(documents.router, tags=["文档"])
api_router.include_router(retrieval.router, tags=["向量检索"])
