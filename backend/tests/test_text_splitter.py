from dataclasses import dataclass

import pytest
from app.services.text_splitter import (
    ChunkingConfig,
    build_document_chunks,
    recursive_split,
)


@dataclass
class SourcePage:
    page_number: int
    content: str


def test_markdown_splitter_preserves_heading_metadata_and_limits() -> None:
    markdown = """# 空压机维修手册

## 排气温度过高

排气温度过高时，应先检查机房通风和环境温度。冷却器堵塞会降低散热能力。
冷却风扇不转、润滑油不足以及温控阀卡滞也可能导致高温停机。

## 排气压力不足

压力不足时，应检查用气量、管路泄漏、空气过滤器和进气阀动作。
如果油气分离器压差过大，也可能造成排气能力下降。
"""
    config = ChunkingConfig(chunk_size=120, chunk_overlap=20, min_chunk_size=20)

    chunks = build_document_chunks(
        [SourcePage(page_number=1, content=markdown)],
        "空压机维修手册.md",
        config,
    )

    assert len(chunks) >= 2
    assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert any(chunk.metadata["section_title"] == "排气温度过高" for chunk in chunks)
    assert any(chunk.metadata["section_title"] == "排气压力不足" for chunk in chunks)
    assert all(chunk.metadata["split_strategy"] == "markdown_heading_recursive" for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)


def test_recursive_split_falls_back_to_fixed_size_for_long_text() -> None:
    text = "A" * 350

    chunks = recursive_split(text, max_size=100)

    assert [len(chunk) for chunk in chunks] == [100, 100, 100, 50]
    assert "".join(chunks) == text


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "min_size"),
    [(100, 100, 20), (100, -1, 20), (100, 10, 101)],
)
def test_invalid_chunking_config_is_rejected(
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_chunk_size=min_size,
        )
