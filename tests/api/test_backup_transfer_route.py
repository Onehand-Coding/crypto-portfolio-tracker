"""Download and delete for database backups. No src/ changes: both operate
only on paths the core's own list_backups() reports, the same trust model
as the existing restore endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

NAME = "portfolio.db.20260904_120000.bak"
CONTENT = b"fake-sqlite-bytes"


def _seed_backup(mock_read_context, tmp_path) -> Path:
    target = tmp_path / NAME
    target.write_bytes(CONTENT)
    mock_read_context.db_manager.list_backups.return_value = [target]
    return target


def test_download_returns_exact_bytes(mock_read_context, tmp_path):
    target = _seed_backup(mock_read_context, tmp_path)
    resp = TestClient(app).get("/api/system/backup/download", params={"name": NAME})
    assert resp.status_code == 200
    assert resp.content == CONTENT
    assert target.exists()


def test_download_unknown_name_is_404(mock_read_context, tmp_path):
    _seed_backup(mock_read_context, tmp_path)
    resp = TestClient(app).get("/api/system/backup/download", params={"name": "ghost.bak"})
    assert resp.status_code == 404


def test_delete_needs_confirm_and_file_survives(mock_read_context, tmp_path):
    target = _seed_backup(mock_read_context, tmp_path)
    resp = TestClient(app).post("/api/system/backup/delete", json={"name": NAME})
    assert resp.status_code == 400
    assert target.exists()


def test_delete_with_confirm_removes_file(mock_read_context, tmp_path):
    target = _seed_backup(mock_read_context, tmp_path)
    body = TestClient(app).post(
        "/api/system/backup/delete", json={"name": NAME, "confirm": True}
    ).json()
    assert body["deleted"] is True
    assert not target.exists()
