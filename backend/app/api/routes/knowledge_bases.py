from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.repositories import knowledge_base as repository
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseList,
    KnowledgeBaseRead,
    KnowledgeBaseUpdate,
)
from app.schemas.response import ApiResponse

router = APIRouter(prefix="/knowledge-bases")
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


async def require_knowledge_base(
    knowledge_base_id: UUID,
    session: AsyncSession,
):
    knowledge_base = await repository.get_knowledge_base(session, knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return knowledge_base


@router.post(
    "",
    response_model=ApiResponse[KnowledgeBaseRead],
    status_code=status.HTTP_201_CREATED,
    summary="创建知识库",
)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    session: DatabaseSession,
) -> ApiResponse[KnowledgeBaseRead]:
    existing = await repository.get_knowledge_base_by_name(session, payload.name)
    if existing is not None:
        raise HTTPException(status_code=409, detail="知识库名称已存在")

    try:
        knowledge_base = await repository.create_knowledge_base(
            session,
            name=payload.name,
            description=payload.description,
        )
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="知识库名称已存在") from exc

    return ApiResponse(data=KnowledgeBaseRead.model_validate(knowledge_base))


@router.get(
    "",
    response_model=ApiResponse[KnowledgeBaseList],
    summary="分页查询知识库",
)
async def get_knowledge_bases(
    session: DatabaseSession,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ApiResponse[KnowledgeBaseList]:
    items, total = await repository.list_knowledge_bases(session, page, page_size)
    return ApiResponse(
        data=KnowledgeBaseList(
            items=[KnowledgeBaseRead.model_validate(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.get(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseRead],
    summary="查询知识库详情",
)
async def get_knowledge_base(
    knowledge_base_id: UUID,
    session: DatabaseSession,
) -> ApiResponse[KnowledgeBaseRead]:
    knowledge_base = await require_knowledge_base(knowledge_base_id, session)
    return ApiResponse(data=KnowledgeBaseRead.model_validate(knowledge_base))


@router.patch(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseRead],
    summary="更新知识库",
)
async def update_knowledge_base(
    knowledge_base_id: UUID,
    payload: KnowledgeBaseUpdate,
    session: DatabaseSession,
) -> ApiResponse[KnowledgeBaseRead]:
    knowledge_base = await require_knowledge_base(knowledge_base_id, session)
    changes = payload.model_dump(exclude_unset=True)

    new_name = changes.get("name")
    if isinstance(new_name, str) and new_name.lower() != knowledge_base.name.lower():
        existing = await repository.get_knowledge_base_by_name(session, new_name)
        if existing is not None:
            raise HTTPException(status_code=409, detail="知识库名称已存在")

    try:
        knowledge_base = await repository.update_knowledge_base(session, knowledge_base, changes)
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="知识库名称已存在") from exc

    return ApiResponse(data=KnowledgeBaseRead.model_validate(knowledge_base))


@router.delete(
    "/{knowledge_base_id}",
    response_model=ApiResponse[None],
    summary="删除知识库",
)
async def delete_knowledge_base(
    knowledge_base_id: UUID,
    session: DatabaseSession,
) -> ApiResponse[None]:
    knowledge_base = await require_knowledge_base(knowledge_base_id, session)
    await repository.delete_knowledge_base(session, knowledge_base)
    return ApiResponse(message="知识库已删除", data=None)
