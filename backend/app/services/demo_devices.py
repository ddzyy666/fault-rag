"""固定日期的虚构演示数据；重复初始化不覆盖已有记录。"""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.device import Device, MaintenanceRecord
from app.models.knowledge_base import KnowledgeBase


async def seed_demo_devices(session: AsyncSession, knowledge_base_id: UUID) -> None:
    if await session.get(KnowledgeBase, knowledge_base_id) is None:
        raise ValueError("知识库不存在，请先创建知识库")
    # 先检查冲突，防止把模拟历史挂到真实设备上。
    existing = {
        d.code: d
        for d in (
            await session.scalars(
                select(Device).where(
                    Device.knowledge_base_id == knowledge_base_id,
                    Device.code.in_(["AC-001", "AC-002", "AC-003"]),
                )
            )
        ).all()
    }
    if any(not d.is_simulated for d in existing.values()):
        raise ValueError("演示编号与非模拟设备冲突，请使用独立演示知识库")
    for number in range(1, 4):
        code = f"AC-{number:03d}"
        device = existing.get(code)
        if device is None:
            device = Device(
                knowledge_base_id=knowledge_base_id,
                code=code,
                name=f"{number} 号空压机",
                model="AC-200",
                location=f"演示车间 {number} 号机位",
                is_simulated=True,
            )
            session.add(device)
            await session.flush()
        events = [
            ("demo-inspection", 1, None, "例行巡检", "外观及报警检查", "未发现报警"),
        ]
        if number == 3:
            events += [
                ("demo-e101", 8, "E101", "高温停机", "记录报警，安排排查", "原因尚未确认"),
                (
                    "demo-filter",
                    9,
                    "E101",
                    "高温问题待排查",
                    "更换空气滤芯",
                    "尚未验证高温是否消除",
                ),
            ]
        for key, day, fault, symptom, action, outcome in events:
            found = await session.scalar(
                select(MaintenanceRecord.id).where(
                    MaintenanceRecord.device_id == device.id,
                    MaintenanceRecord.event_key == key,
                )
            )
            if found is None:
                session.add(
                    MaintenanceRecord(
                        device_id=device.id,
                        event_key=key,
                        occurred_at=datetime(2026, 9, day, 1, tzinfo=UTC),
                        fault_code=fault,
                        symptom=symptom,
                        action=action,
                        outcome=outcome,
                        is_simulated=True,
                    )
                )
    await session.flush()
