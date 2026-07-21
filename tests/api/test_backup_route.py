"""POST /api/system/backup: an additive, on-demand database backup."""

from fastapi.testclient import TestClient

from api.main import app


def test_reports_the_created_backup_name(mock_read_context):
    mock_read_context.db_manager.backup_database.return_value = (
        "/data/db_backups/portfolio.db.20260101_000000.manual.bak"
    )

    body = TestClient(app).post("/api/system/backup").json()

    assert body["created"] is True
    assert body["name"] == "portfolio.db.20260101_000000.manual.bak"
    # force=True so an explicit request runs even when auto-backup is disabled.
    _, kwargs = mock_read_context.db_manager.backup_database.call_args
    assert kwargs.get("force") is True


def test_none_return_is_a_reported_failure_not_a_crash(mock_read_context):
    mock_read_context.db_manager.backup_database.return_value = None

    body = TestClient(app).post("/api/system/backup").json()

    assert body["created"] is False
    assert body["error"]


def test_exception_is_reported_not_a_500(mock_read_context):
    mock_read_context.db_manager.backup_database.side_effect = OSError("disk full")

    body = TestClient(app).post("/api/system/backup").json()

    assert body["created"] is False
    assert "disk full" in body["error"]
