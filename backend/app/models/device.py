from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Device(TimestampMixin, Base):
    """设备编号只在所属知识库内唯一；演示数据显式标记。"""

    __tablename__ = "devices"
    __table_args__ = (UniqueConstraint("knowledge_base_id", "code"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(100))
    location: Mapped[str] = mapped_column(String(200))
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)


class MaintenanceRecord(TimestampMixin, Base):
    """报警及维修事件；保留结果，不将维修动作等同于故障排除。"""

    __tablename__ = "maintenance_records"
    __table_args__ = (UniqueConstraint("device_id", "event_key"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    device_id: Mapped[UUID] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"), index=True
    )
    event_key: Mapped[str] = mapped_column(String(80))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fault_code: Mapped[str | None] = mapped_column(String(50))
    symptom: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)
