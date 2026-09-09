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


def test_parent_headings_follow_hierarchy_and_repeat_on_long_sections() -> None:
    markdown = "# 手册\n## 压力不足\n### 可能原因\n" + "管路泄漏。" * 80
    markdown += "\n## 高温停机\n### 可能原因\n冷却器堵塞。\n# 新手册\n正文。"
    chunks = build_document_chunks(
        [SourcePage(1, markdown)],
        "manual.md",
        ChunkingConfig(120, 20, 20, include_heading_path=True),
    )
    pressure = [c for c in chunks if "管路泄漏" in c.content]
    assert len(pressure) > 1
    assert all("## 压力不足\n### 可能原因" in c.content for c in pressure)
    hot = next(c for c in chunks if "冷却器堵塞" in c.content)
    assert "## 高温停机" in hot.content and "压力不足" not in hot.content
    assert hot.metadata["heading_path"] == ["手册", "高温停机", "可能原因"]
    assert hot.metadata["section_title"] == "可能原因"
    assert "手册\n" not in chunks[-1].content.replace("新手册\n", "")
    assert all(len(c.content) <= 120 for c in chunks)


def test_long_heading_paths_leave_room_for_body_without_exceeding_limit() -> None:
    markdown = "\n".join("#" * i + " " + "长标题" * 80 for i in range(1, 7))
    chunks = build_document_chunks(
        [SourcePage(1, markdown + "\n" + "正文" * 200)],
        "manual.md",
        ChunkingConfig(100, 20, 20, include_heading_path=True),
    )
    assert chunks and all(len(c.content) <= 100 for c in chunks)
    assert all("正文" in c.content for c in chunks)
    assert len(chunks[0].metadata["heading_path"]) == 6


def test_parent_heading_context_is_disabled_by_default() -> None:
    chunks = build_document_chunks(
        [SourcePage(1, "# 手册\n## 压力不足\n### 可能原因\n管路泄漏。")],
        "manual.md",
        ChunkingConfig(),
    )
    leaf = chunks[-1]
    assert leaf.content.startswith("### 可能原因\n")
    assert "压力不足" not in leaf.content
    assert leaf.metadata["heading_path"] == ["手册", "压力不足", "可能原因"]
    assert leaf.metadata["include_heading_path"] is False


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
