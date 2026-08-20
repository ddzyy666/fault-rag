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
