"""Add scoped devices and maintenance records."""

import sqlalchemy as sa
from alembic import op

revision = "c21e6f8a9012"
down_revision = "9bfb4e4f9fe7"
branch_labels = None
depends_on = None


def timestamps():
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "knowledge_base_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("location", sa.String(200), nullable=False),
        sa.Column("is_simulated", sa.Boolean(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("knowledge_base_id", "code"),
    )
    op.create_index("ix_devices_knowledge_base_id", "devices", ["knowledge_base_id"])
    op.create_table(
        "maintenance_records",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "device_id", sa.Uuid(), sa.ForeignKey("devices.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("event_key", sa.String(80), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fault_code", sa.String(50)),
        sa.Column("symptom", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("is_simulated", sa.Boolean(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("device_id", "event_key"),
    )
    op.create_index("ix_maintenance_records_device_id", "maintenance_records", ["device_id"])


def downgrade() -> None:
    op.drop_table("maintenance_records")
    op.drop_table("devices")
