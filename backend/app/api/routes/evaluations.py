from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    EmbeddingDependency,
    RerankerDependency,
    SparseEmbeddingDependency,
    VectorStoreDependency,
)
from app.db.database import get_db
from app.repositories.knowledge_base import get_knowledge_base
from app.schemas.evaluation import RetrievalEvaluationRequest, RetrievalEvaluationResult
from app.schemas.response import ApiResponse
from app.services.embedding import EmbeddingError
from app.services.retrieval_evaluation import EvaluationDatasetError, evaluate_retrieval
from app.services.vector_store import VectorStoreError

router = APIRouter()
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "/knowledge-bases/{knowledge_base_id}/evaluations/retrieval",
    response_model=ApiResponse[RetrievalEvaluationResult],
    summary="对比并评估多种知识库检索方案",
)
async def run_retrieval_evaluation(
    knowledge_base_id: UUID,
    payload: RetrievalEvaluationRequest,
    session: DatabaseSession,
    embedding_provider: EmbeddingDependency,
    sparse_embedding_provider: SparseEmbeddingDependency,
    vector_store: VectorStoreDependency,
    reranker_provider: RerankerDependency,
) -> ApiResponse[RetrievalEvaluationResult]:
    if await get_knowledge_base(session, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    try:
        result = await evaluate_retrieval(
            session,
            knowledge_base_id=knowledge_base_id,
            payload=payload,
            embedding_provider=embedding_provider,
            sparse_embedding_provider=sparse_embedding_provider,
            vector_store=vector_store,
            reranker_provider=reranker_provider,
        )
    except EvaluationDatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ApiResponse(data=result)
