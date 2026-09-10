import os
import subprocess
import sys
from pathlib import Path


def test_device_migration_roundtrip(tmp_path):
    """真实执行迁移，并检查 ORM 与迁移结构一致；不触碰开发数据库。"""
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'migration.db'}"}
    for arguments in [
        ["upgrade", "head"],
        ["check"],
        ["downgrade", "9bfb4e4f9fe7"],
        ["upgrade", "head"],
    ]:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", "backend/alembic.ini", *arguments],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stdout + result.stderr
