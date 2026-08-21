from pathlib import Path

from app.core.config import settings
from fastapi.testclient import TestClient


def create_knowledge_base(client: TestClient) -> str:
    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "设备维修知识库"},
    )
    assert response.status_code == 201
    return response.json()["data"]["id"]


def test_upload_parse_list_read_content_and_delete_document(client: TestClient) -> None:
    knowledge_base_id = create_knowledge_base(client)
    content = "故障码：E101\n故障原因：冷却风扇异常\n排查步骤：检查风扇供电。"

    upload_response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("维修手册.txt", content.encode("utf-8"), "text/plain")},
    )

    assert upload_response.status_code == 201
    document = upload_response.json()["data"]
    document_id = document["id"]
    assert document["status"] == "parsed"
    assert document["page_count"] == 1
    assert document["filename"] == "维修手册.txt"
    assert "storage_path" not in document

    detail_response = client.get(f"/api/v1/documents/{document_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["file_hash"] == document["file_hash"]

    list_response = client.get(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents?status=parsed"
    )
    assert list_response.status_code == 200
    assert list_response.json()["data"]["total"] == 1

    content_response = client.get(f"/api/v1/documents/{document_id}/content")
    assert content_response.status_code == 200
    page = content_response.json()["data"]["items"][0]
    assert page["page_number"] == 1
    assert "E101" in page["content"]
    assert page["character_count"] == len(page["content"])

    duplicate_response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("副本.txt", content.encode("utf-8"), "text/plain")},
    )
    assert duplicate_response.status_code == 409

    delete_response = client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 200
    assert client.get(f"/api/v1/documents/{document_id}").status_code == 404
    assert not list(Path(settings.upload_dir).rglob("*.txt"))


def test_reject_unsupported_document_type(client: TestClient) -> None:
    knowledge_base_id = create_knowledge_base(client)

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("program.exe", b"not executable", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert "仅支持" in response.json()["message"]


def test_corrupted_pdf_is_recorded_as_failed(client: TestClient) -> None:
    knowledge_base_id = create_knowledge_base(client)

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("damaged.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 201
    document = response.json()["data"]
    assert document["status"] == "failed"
    assert document["error_message"] == "PDF 文件损坏或无法解析"

    content_response = client.get(f"/api/v1/documents/{document['id']}/content")
    assert content_response.status_code == 409


def test_create_rebuild_list_and_delete_document_chunks(client: TestClient) -> None:
    knowledge_base_id = create_knowledge_base(client)
    markdown = """# 空压机知识库

## 高温停机

空压机高温停机时，应检查环境温度、冷却器、冷却风扇和润滑油状态。
如果冷却器表面被粉尘堵塞，散热效率会明显下降。清洁前必须停机、断电并泄压。

## 压力不足

排气压力不足时，应检查用气量是否超过额定排气量，并检查管路是否泄漏。
空气过滤器堵塞或进气阀无法完全打开，也会导致系统压力无法达到设定值。
"""
    upload_response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("空压机知识库.md", markdown.encode("utf-8"), "text/markdown")},
    )
    document_id = upload_response.json()["data"]["id"]

    chunk_response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 120, "chunk_overlap": 20, "min_chunk_size": 20},
    )
    assert chunk_response.status_code == 200
    result = chunk_response.json()["data"]
    assert result["status"] == "chunked"
    assert result["chunk_count"] >= 2
    assert result["max_characters"] <= 120

    list_response = client.get(f"/api/v1/documents/{document_id}/chunks?page_size=100")
    assert list_response.status_code == 200
    first_items = list_response.json()["data"]["items"]
    assert len(first_items) == result["chunk_count"]
    assert [item["chunk_index"] for item in first_items] == list(range(len(first_items)))
    assert any(item["extra_metadata"]["section_title"] == "高温停机" for item in first_items)

    rebuild_response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 200, "chunk_overlap": 30, "min_chunk_size": 30},
    )
    assert rebuild_response.status_code == 200
    rebuilt_count = rebuild_response.json()["data"]["chunk_count"]

    rebuilt_list = client.get(f"/api/v1/documents/{document_id}/chunks?page_size=100")
    assert rebuilt_list.json()["data"]["total"] == rebuilt_count

    delete_response = client.delete(f"/api/v1/documents/{document_id}/chunks")
    assert delete_response.status_code == 200
    assert client.get(f"/api/v1/documents/{document_id}/chunks").json()["data"]["total"] == 0
    assert client.get(f"/api/v1/documents/{document_id}").json()["data"]["status"] == "parsed"


def test_reject_invalid_chunk_overlap(client: TestClient) -> None:
    knowledge_base_id = create_knowledge_base(client)
    upload_response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("manual.txt", b"fault diagnosis text", "text/plain")},
    )
    document_id = upload_response.json()["data"]["id"]

    response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 100, "chunk_overlap": 100, "min_chunk_size": 20},
    )

    assert response.status_code == 422
