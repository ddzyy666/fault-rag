import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", "、", " ")
MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")


class PageLike(Protocol):
    """分块器需要的最小页面接口。"""

    page_number: int
    content: str


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """与具体Embedding模型解耦的字符分块参数。"""

    chunk_size: int = 700
    chunk_overlap: int = 100
    min_chunk_size: int = 80

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size不能小于100")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap必须大于等于0且小于chunk_size")
        if self.min_chunk_size < 1 or self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size必须在1和chunk_size之间")


@dataclass(frozen=True, slots=True)
class TextSection:
    """Markdown标题划分出的语义章节。"""

    content: str
    title: str | None = None
    level: int | None = None


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """尚未写入数据库的文本切片。"""

    content: str
    page_number: int
    token_count: int
    metadata: dict[str, Any]


def estimate_token_count(text: str) -> int:
    """提供轻量近似值，实际Embedding token数在向量化阶段重新计算。"""
    return len(TOKEN_PATTERN.findall(text))


def split_with_separator(text: str, separator: str) -> list[str]:
    """按分隔符切开文本，同时将分隔符保留在前一个片段末尾。"""
    if separator not in text:
        return [text]
    escaped = re.escape(separator)
    return [part for part in re.split(f"(?<={escaped})", text) if part]


def recursive_split(
    text: str,
    max_size: int,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    """优先按段落和句子边界切分，最后才按固定字符数截断。"""
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_size:
        return [normalized]
    if not separators:
        return [
            normalized[start : start + max_size] for start in range(0, len(normalized), max_size)
        ]

    parts = split_with_separator(normalized, separators[0])
    if len(parts) == 1:
        return recursive_split(normalized, max_size, separators[1:])

    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(recursive_split(part, max_size, separators[1:]))
        elif len(current) + len(part) <= max_size:
            current += part
        else:
            if current.strip():
                chunks.append(current.strip())
            current = part

    if current.strip():
        chunks.append(current.strip())
    return chunks


def merge_small_tail(chunks: list[str], min_size: int, max_size: int) -> list[str]:
    """在不超过上限时合并过短的末尾切片。"""
    if len(chunks) < 2 or len(chunks[-1]) >= min_size:
        return chunks
    if len(chunks[-2]) + 1 + len(chunks[-1]) <= max_size:
        return [*chunks[:-2], f"{chunks[-2]}\n{chunks[-1]}".strip()]
    return chunks


def add_overlap(chunks: list[str], overlap: int, max_size: int) -> list[str]:
    """给后续切片补充前一切片尾部上下文，并保证不超过长度上限。"""
    if overlap == 0 or len(chunks) < 2:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        available = max_size - len(chunk) - 1
        prefix_size = min(overlap, max(0, available))
        prefix = overlapped[-1][-prefix_size:] if prefix_size else ""
        overlapped.append(f"{prefix}\n{chunk}".strip() if prefix else chunk)
    return overlapped


def split_markdown_sections(text: str) -> list[TextSection]:
    """识别Markdown标题，保留标题名称和级别。"""
    matches = list(MARKDOWN_HEADING.finditer(text))
    if not matches:
        return [TextSection(content=text)]

    sections: list[TextSection] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(TextSection(content=preamble))

    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(
            TextSection(
                title=match.group(2).strip(),
                level=len(match.group(1)),
                content=text[content_start:content_end].strip(),
            )
        )
    return sections


def split_section(section: TextSection, config: ChunkingConfig) -> list[str]:
    """切分单个章节，并让Markdown标题出现在该章节的每个切片中。"""
    header = ""
    if section.title and section.level:
        header = f"{'#' * section.level} {section.title[:120]}"

    content_limit = max(50, config.chunk_size - len(header) - (1 if header else 0))
    effective_overlap = min(config.chunk_overlap, max(0, content_limit - 1))
    body_limit = max(1, content_limit - effective_overlap)
    body_chunks = recursive_split(section.content, body_limit)
    if not body_chunks:
        return []

    body_chunks = merge_small_tail(body_chunks, config.min_chunk_size, body_limit)
    body_chunks = add_overlap(body_chunks, effective_overlap, content_limit)
    if header:
        return [f"{header}\n{body}".strip() for body in body_chunks]
    return body_chunks


def build_document_chunks(
    pages: list[PageLike],
    filename: str,
    config: ChunkingConfig,
) -> list[ChunkDraft]:
    """将解析页面转换成带页码、章节信息和估算token数的切片。"""
    is_markdown = Path(filename).suffix.lower() == ".md"
    drafts: list[ChunkDraft] = []

    for page in pages:
        sections = (
            split_markdown_sections(page.content)
            if is_markdown
            else [TextSection(content=page.content)]
        )
        for section in sections:
            for content in split_section(section, config):
                if not content.strip():
                    continue
                token_count = estimate_token_count(content)
                drafts.append(
                    ChunkDraft(
                        content=content,
                        page_number=page.page_number,
                        token_count=token_count,
                        metadata={
                            "section_title": section.title,
                            "section_level": section.level,
                            "split_strategy": (
                                "markdown_heading_recursive" if is_markdown else "recursive"
                            ),
                            "chunk_size": config.chunk_size,
                            "chunk_overlap": config.chunk_overlap,
                            "token_count_estimated": True,
                        },
                    )
                )
    return drafts
