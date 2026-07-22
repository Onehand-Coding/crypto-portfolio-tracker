"""POST /api/system/restore: reversible restore from a named backup."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def test_restores_a_listed_backup_and_snapshots_current_first(mock_read_context):
    db = mock_read_context.db_manager
    db.list_backups.return_value = [Path("/b/portfolio.db.20260101.manual.bak")]
    db.backup_database.return_value = "/b/portfolio.db.20260102.pre_restore.bak"
    db.restore_from_backup.return_value = True

    body = TestClient(app).post(
        "/api/system/restore", json={"name": "portfolio.db.20260101.manual.bak"}
    ).json()

    assert body["restored"] is True
    # The current DB is snapshotted before being overwritten, so it is undoable.
    assert body["safety_backup"] == "portfolio.db.20260102.pre_restore.bak"
    db.backup_database.assert_called_once()
    db.restore_from_backup.assert_called_once()


def test_unknown_backup_is_404_and_touches_nothing(mock_read_context):
    db = mock_read_context.db_manager
    db.list_backups.return_value = [Path("/b/real.bak")]

    resp = TestClient(app).post("/api/system/restore", json={"name": "ghost.bak"})

    assert resp.status_code == 404
    db.restore_from_backup.assert_not_called()
    db.backup_database.assert_not_called()


def test_path_traversal_name_is_rejected(mock_read_context):
    db = mock_read_context.db_manager
    db.list_backups.return_value = [Path("/b/real.bak")]

    resp = TestClient(app).post(
        "/api/system/restore", json={"name": "../../etc/passwd"}
    )

    assert resp.status_code == 404
    db.restore_from_backup.assert_not_called()


def test_failed_restore_is_reported_not_500(mock_read_context):
    db = mock_read_context.db_manager
    db.list_backups.return_value = [Path("/b/real.bak")]
    db.backup_database.return_value = "/b/safety.bak"
    db.restore_from_backup.return_value = False

    body = TestClient(app).post("/api/system/restore", json={"name": "real.bak"}).json()

    assert body["restored"] is False
    assert body["error"]
