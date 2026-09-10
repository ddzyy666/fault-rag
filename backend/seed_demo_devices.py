"""Usage: python backend/seed_demo_devices.py --knowledge-base-id UUID"""

import argparse
import asyncio
from uuid import UUID

from app.db.database import async_session_factory, engine
from app.services.demo_devices import seed_demo_devices


async def main(knowledge_base_id: UUID) -> None:
    try:
        async with async_session_factory() as session, session.begin():
            await seed_demo_devices(session, knowledge_base_id)
        print("演示设备初始化完成：AC-001、AC-002、AC-003（全部为模拟数据）")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="向指定知识库添加模拟设备及维修记录")
    parser.add_argument("--knowledge-base-id", required=True, type=UUID)
    args = parser.parse_args()
    asyncio.run(main(args.knowledge_base_id))
