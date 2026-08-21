from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量和 .env 文件读取应用配置。"""

    app_name: str = "智能故障诊断助手"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    debug: bool = True
    database_url: str = "sqlite+aiosqlite:///./fault_rag.db"
    database_echo: bool = False
    upload_dir: Path = Path("uploads")
    max_upload_size_bytes: int = 20 * 1024 * 1024
    default_chunk_size: int = 700
    default_chunk_overlap: int = 100
    min_chunk_size: int = 80
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512
    embedding_batch_size: int = 32
    qdrant_path: Path = Path("qdrant_storage")
    qdrant_collection: str = "fault_diagnosis_chunks"
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """创建并缓存配置，避免每次请求重复读取环境。"""
    return Settings()


settings = get_settings()
