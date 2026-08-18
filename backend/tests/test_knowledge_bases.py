from fastapi.testclient import TestClient


def create_knowledge_base(client: TestClient, name: str = "空压机知识库") -> dict:
    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": name, "description": "空压机维修手册和历史故障案例"},
    )
    assert response.status_code == 201
    return response.json()["data"]


def test_knowledge_base_crud(client: TestClient) -> None:
    created = create_knowledge_base(client)
    knowledge_base_id = created["id"]

    detail_response = client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["name"] == "空压机知识库"

    list_response = client.get("/api/v1/knowledge-bases?page=1&page_size=10")
    assert list_response.status_code == 200
    assert list_response.json()["data"]["total"] == 1
    assert len(list_response.json()["data"]["items"]) == 1

    update_response = client.patch(
        f"/api/v1/knowledge-bases/{knowledge_base_id}",
        json={"name": "空压机故障诊断知识库"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["data"]["name"] == "空压机故障诊断知识库"

    delete_response = client.delete(f"/api/v1/knowledge-bases/{knowledge_base_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["data"] is None

    missing_response = client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}")
    assert missing_response.status_code == 404
    assert missing_response.json()["message"] == "知识库不存在"


def test_duplicate_knowledge_base_name_returns_conflict(client: TestClient) -> None:
    create_knowledge_base(client)

    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "空压机知识库"},
    )

    assert response.status_code == 409
    assert response.json()["message"] == "知识库名称已存在"


def test_invalid_knowledge_base_name_returns_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "   "},
    )

    assert response.status_code == 422
    assert response.json()["message"] == "请求参数校验失败"
