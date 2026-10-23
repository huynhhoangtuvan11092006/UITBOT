from app.main import app
from fastapi.testclient import TestClient


def test_health_ok(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("app.main.init_pool", lambda: None)
    monkeypatch.setattr("app.main.close_pool", lambda: None)
    monkeypatch.setattr("app.api.health.ping_db", lambda: None)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_unavailable(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("app.main.init_pool", lambda: None)
    monkeypatch.setattr("app.main.close_pool", lambda: None)

    def _fail() -> None:
        raise RuntimeError("down")

    monkeypatch.setattr("app.api.health.ping_db", _fail)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable."
