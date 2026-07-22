"""Snapshots, cleanup and import endpoints: gates and wiring (deps mocked)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(autouse=True)
def _cwd(monkeypatch, tmp_path, mock_read_context):
    mock_read_context.config_manager.config = {"database": {"cleanup_days": 90}}
    monkeypatch.chdir(tmp_path)


# --- snapshots ---

def test_lists_snapshots_newest_first(mock_read_context):
    mock_read_context.db_manager.get_all_snapshots.return_value = pd.DataFrame([
        {"timestamp": "2025-01-01T00:00:00", "total_value_usd": 100.0,
         "total_cost_basis_usd": 90.0, "unrealized_pl_usd": 10.0, "unrealized_pl_percent": 11.0},
        {"timestamp": "2025-03-01T00:00:00", "total_value_usd": 120.0,
         "total_cost_basis_usd": 90.0, "unrealized_pl_usd": 30.0, "unrealized_pl_percent": 33.0},
    ])
    body = TestClient(app).get("/api/system/snapshots").json()
    assert body["count"] == 2
    assert body["rows"][0]["timestamp"].startswith("2025-03")


def test_snapshot_delete_requires_confirmation(mock_read_context):
    resp = TestClient(app).post("/api/system/snapshots/delete", json={"confirm": False})
    assert resp.status_code == 400
    mock_read_context.db_manager.delete_snapshot.assert_not_called()


def test_snapshot_delete_forwards_the_row(mock_read_context):
    mock_read_context.db_manager.delete_snapshot.return_value = 1
    body = TestClient(app).post("/api/system/snapshots/delete", json={
        "confirm": True, "timestamp": "2025-01-01T00:00:00", "total_value_usd": 100.0,
        "total_cost_basis_usd": 90.0, "unrealized_pl_usd": 10.0, "unrealized_pl_percent": 11.0,
    }).json()
    assert body["deleted"] == 1
    mock_read_context.db_manager.delete_snapshot.assert_called_once_with(
        "2025-01-01T00:00:00", 100.0, 90.0, 10.0, 11.0
    )


# --- cleanup ---

def test_cleanup_stats_reports_retention(mock_read_context):
    mock_read_context.db_manager.get_cleanup_statistics.return_value = {"old_snapshots": 3}
    body = TestClient(app).get("/api/system/cleanup").json()
    assert body["cleanup_days"] == 90
    assert body["enabled"] is True
    assert body["stats"]["old_snapshots"] == 3


def test_cleanup_requires_confirmation(mock_read_context):
    resp = TestClient(app).post("/api/system/cleanup", json={"confirm": False})
    assert resp.status_code == 400
    mock_read_context.db_manager.cleanup_old_data.assert_not_called()


def test_cleanup_runs_on_confirm(mock_read_context):
    mock_read_context.db_manager.cleanup_old_data.return_value = None
    body = TestClient(app).post("/api/system/cleanup", json={"confirm": True}).json()
    assert body["success"] is True
    mock_read_context.db_manager.cleanup_old_data.assert_called_once()


# --- import ---

def test_import_transactions_parses_and_inserts(mock_read_context):
    mock_read_context.db_manager.bulk_insert_transactions.return_value = 2
    csv = "asset_id,timestamp,type,quantity\n1,2025-01-01,BUY,1\n1,2025-02-01,SELL,1\n"
    body = TestClient(app).post(
        "/api/system/import/transactions",
        files={"file": ("t.csv", csv, "text/csv")},
    ).json()
    assert body["success"] is True
    assert body["rows_affected"] == 2
    mock_read_context.db_manager.bulk_insert_transactions.assert_called_once()


def test_import_holdings_backs_up_then_updates(mock_read_context):
    csv = "symbol,quantity,average_cost_basis\nBTC,1,100\n"
    body = TestClient(app).post(
        "/api/system/import/holdings",
        files={"file": ("h.csv", csv, "text/csv")},
    ).json()
    assert body["success"] is True
    # A backup precedes the write, so a bad import is recoverable.
    mock_read_context.db_manager.backup_database.assert_called_once()
    mock_read_context.db_manager.update_holdings.assert_called_once()


def test_import_rejects_unknown_type(mock_read_context):
    resp = TestClient(app).post(
        "/api/system/import/nope", files={"file": ("x.csv", "a\n1\n", "text/csv")},
    )
    assert resp.status_code == 422
