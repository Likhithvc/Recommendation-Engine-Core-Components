import pytest

from fastapi.testclient import TestClient

from api.app import API_KEY, app
from data.database import SessionLocal
from data.models import Content, Interaction, User
import scripts.seed_data as seed_data


@pytest.fixture(scope="module")
def client() -> TestClient:
    seed_data.main()
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_headers() -> dict[str, str]:
    return {"x-api-key": API_KEY}


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "recommendation-api"
    assert "X-Request-ID" in response.headers


def test_recommend_endpoint_success(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/recommend/1?limit=3", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert isinstance(body["recommendations"], list)
    assert len(body["recommendations"]) <= 3
    assert len(body["request_id"]) > 0
    if body["recommendations"]:
        assert "reason" in body["recommendations"][0]


def test_feedback_endpoint_missing_required_field(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/feedback",
        json={"user_id": 1, "rating": 4.2},
        headers=auth_headers,
    )

    assert response.status_code == 400
    body = response.json()
    assert isinstance(body["detail"], list)
    assert any("content_id" in str(item) for item in body["detail"])
    assert len(body["request_id"]) > 0


def test_recommend_endpoint_not_found(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/recommend/999999?limit=3", headers=auth_headers)

    assert response.status_code == 404
    body = response.json()
    assert "not found" in body["detail"].lower()
    assert len(body["request_id"]) > 0


def test_recommend_endpoint_bad_request(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/recommend/1?limit=0", headers=auth_headers)

    assert response.status_code == 400
    body = response.json()
    assert "limit" in body["detail"].lower()
    assert len(body["request_id"]) > 0


def test_recommend_endpoint_invalid_user_id_format(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.get("/recommend/not-an-int?limit=3", headers=auth_headers)

    assert response.status_code == 400
    body = response.json()
    assert isinstance(body["detail"], list)
    assert len(body["request_id"]) > 0


def test_feedback_endpoint_validation_error(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/feedback",
        json={"user_id": 0, "content_id": 1, "rating": 4.2},
        headers=auth_headers,
    )

    assert response.status_code == 400
    body = response.json()
    assert isinstance(body["detail"], list)
    assert len(body["request_id"]) > 0


def test_feedback_endpoint_success(client: TestClient, auth_headers: dict[str, str]) -> None:
    with SessionLocal() as db:
        user = db.query(User).first()
        content = db.query(Content).first()
        assert user is not None
        assert content is not None
        before_count = (
            db.query(Interaction)
            .filter(Interaction.user_id == user.id, Interaction.content_id == content.id)
            .count()
        )
        payload = {"user_id": user.id, "content_id": content.id, "rating": 4.5}

    response = client.post("/feedback", json=payload, headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Feedback recorded"
    assert len(body["request_id"]) > 0

    with SessionLocal() as db:
        after_count = (
            db.query(Interaction)
            .filter(Interaction.user_id == payload["user_id"], Interaction.content_id == payload["content_id"])
            .count()
        )
        assert after_count >= before_count + 1


def test_feedback_endpoint_not_found_user(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/feedback",
        json={"user_id": 999999, "content_id": 1, "rating": 3.5},
        headers=auth_headers,
    )

    assert response.status_code == 404
    body = response.json()
    assert "not found" in body["detail"].lower()
    assert len(body["request_id"]) > 0


def test_metrics_endpoint(client: TestClient, auth_headers: dict[str, str]) -> None:
    before = client.get("/metrics", headers=auth_headers).json()["total_requests"]

    client.get("/health")
    client.get("/health")
    client.get("/recommend/1?limit=3", headers=auth_headers)
    client.get("/recommend/1?limit=3", headers=auth_headers)

    response = client.get("/metrics", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["total_requests"] >= before + 4
    assert body["avg_response_time"] >= 0.0
    assert 0.0 <= body["cache_hit_rate"] <= 1.0
    assert body["cache_hit_rate"] > 0.0


def test_auth_required_for_protected_endpoint(client: TestClient) -> None:
    response = client.get("/recommend/1?limit=3")

    assert response.status_code == 401
    body = response.json()
    assert "api key" in body["detail"].lower()
    assert len(body["request_id"]) > 0


def test_auth_required_for_metrics_endpoint(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 401
    body = response.json()
    assert "api key" in body["detail"].lower()
    assert len(body["request_id"]) > 0
