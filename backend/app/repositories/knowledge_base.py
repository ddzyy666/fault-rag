from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase


async def get_knowledge_base(
    session: AsyncSession, knowledge_base_id: UUID
) -> KnowledgeBase | None:
    """按主键查询知识库。"""
    return await session.get(KnowledgeBase, knowledge_base_id)


async def get_knowledge_base_by_name(
    session: AsyncSession,
    name: str,
) -> KnowledgeBase | None:
    """按名称进行不区分大小写的查询。"""
    statement = select(KnowledgeBase).where(func.lower(KnowledgeBase.name) == name.lower())
    return await session.scalar(statement)


async def list_knowledge_bases(
    session: AsyncSession,
    page: int,
    page_size: int,
) -> tuple[list[KnowledgeBase], int]:
    """分页查询知识库。"""
    total = await session.scalar(select(func.count()).select_from(KnowledgeBase))
    statement = (
        select(KnowledgeBase)
        .order_by(KnowledgeBase.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = list((await session.scalars(statement)).all())
    return items, total or 0


async def create_knowledge_base(
    session: AsyncSession,
    name: str,
    description: str | None,
) -> KnowledgeBase:
    """创建并持久化知识库。"""
    knowledge_base = KnowledgeBase(name=name, description=description)
    session.add(knowledge_base)
    await session.commit()
    await session.refresh(knowledge_base)
    return knowledge_base


async def update_knowledge_base(
    session: AsyncSession,
    knowledge_base: KnowledgeBase,
    changes: dict[str, object],
) -> KnowledgeBase:
    """更新知识库的指定字段。"""
    for field, value in changes.items():
        setattr(knowledge_base, field, value)

    await session.commit()
    await session.refresh(knowledge_base)
    return knowledge_base


async def delete_knowledge_base(
    session: AsyncSession,
    knowledge_base: KnowledgeBase,
) -> None:
    """删除知识库。"""
    await session.delete(knowledge_base)
    await session.commit()
