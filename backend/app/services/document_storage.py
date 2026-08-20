import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.core.config import settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
WRITE_CHUNK_SIZE = 1024 * 1024


class UnsupportedFileTypeError(ValueError):
    """上传了系统不支持的文件格式。"""


class FileTooLargeError(ValueError):
    """上传文件超过配置的大小上限。"""


@dataclass(frozen=True, slots=True)
class StoredFile:
    """安全落盘后的文件信息。"""

    original_filename: str
    storage_path: Path
    mime_type: str | None
    size_bytes: int
    file_hash: str


def sanitize_filename(filename: str) -> str:
    """移除目录部分和危险字符，只保留可展示的安全文件名。"""
    basename = Path(filename).name.strip()
    sanitized = re.sub(r"[^\w.\- ]", "_", basename, flags=re.UNICODE)
    sanitized = sanitized.strip(" .")
    if not sanitized:
        sanitized = "document"
    return sanitized[:200]


def validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UnsupportedFileTypeError(f"仅支持以下文件格式：{allowed}")
    return extension


async def save_upload(upload: UploadFile, knowledge_base_id: UUID) -> StoredFile:
    """以分块方式保存上传文件，同时计算 SHA-256 并限制文件大小。"""
    original_filename = upload.filename or "document"
    validate_extension(original_filename)
    safe_filename = sanitize_filename(original_filename)

    destination_dir = Path(settings.upload_dir) / str(knowledge_base_id)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uuid4().hex}_{safe_filename}"
    temporary = destination.with_name(f"{destination.name}.part")

    digest = hashlib.sha256()
    size_bytes = 0

    try:
        with temporary.open("wb") as output:
            while chunk := await upload.read(WRITE_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > settings.max_upload_size_bytes:
                    raise FileTooLargeError(
                        f"文件大小不能超过 {settings.max_upload_size_bytes // 1024 // 1024} MB"
                    )
                digest.update(chunk)
                output.write(chunk)

        if size_bytes == 0:
            raise ValueError("不能上传空文件")

        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise

    return StoredFile(
        original_filename=original_filename,
        storage_path=destination,
        mime_type=upload.content_type,
        size_bytes=size_bytes,
        file_hash=digest.hexdigest(),
    )


def delete_stored_file(storage_path: str | Path) -> bool:
    """只删除上传根目录内的指定文件，拒绝越界路径。"""
    upload_root = Path(settings.upload_dir).resolve()
    target = Path(storage_path).resolve()
    if target == upload_root or not target.is_relative_to(upload_root):
        return False

    if not target.is_file():
        return False

    target.unlink()
    parent = target.parent
    if parent != upload_root:
        try:
            parent.rmdir()
        except OSError:
            pass
    return True
