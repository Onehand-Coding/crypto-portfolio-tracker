from fastapi.testclient import TestClient

from api.main import app


def test_api_routes_still_resolve_when_frontend_absent():
    """The SPA catch-all must never shadow /api paths."""
    assert TestClient(app).get("/api/health").json() == {"status": "ok"}


def test_unknown_api_path_returns_404_not_the_spa():
    assert TestClient(app).get("/api/does-not-exist").status_code == 404
