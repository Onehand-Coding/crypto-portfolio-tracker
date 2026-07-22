"""Reports export generation and download."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

TXNS = pd.DataFrame({
    "symbol": ["BTC", "ETH"],
    "type": ["BUY", "SELL"],
    # Timezone-aware: the Excel writer must not choke on this.
    "timestamp": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"]),
})


@pytest.fixture(autouse=True)
def _cwd(monkeypatch, tmp_path, mock_read_context):
    mock_read_context.config_manager.config = {}
    mock_read_context.db_manager.get_all_transactions.return_value = TXNS
    mock_read_context.db_manager.get_holdings.return_value = pd.DataFrame({"symbol": ["BTC"]})
    monkeypatch.chdir(tmp_path)


def test_generates_a_csv_and_lists_it(tmp_path):
    client = TestClient(app)
    name = client.post("/api/reports/generate",
                       json={"data_type": "transactions", "format": "csv"}).json()["name"]
    assert name.endswith(".csv")
    assert (tmp_path / "data" / "exports" / name).is_file()


def test_excel_export_handles_timezone_aware_dates(tmp_path):
    """The tz-aware timestamp column must not 500 the Excel write."""
    resp = TestClient(app).post(
        "/api/reports/generate", json={"data_type": "transactions", "format": "excel"}
    )
    assert resp.status_code == 200
    assert (tmp_path / "data" / "exports" / resp.json()["name"]).is_file()


def test_download_serves_a_generated_file():
    client = TestClient(app)
    name = client.post("/api/reports/generate",
                       json={"data_type": "holdings", "format": "csv"}).json()["name"]
    resp = client.get(f"/api/reports/download?name={name}")
    assert resp.status_code == 200
    assert "BTC" in resp.text


def test_download_rejects_path_traversal():
    resp = TestClient(app).get("/api/reports/download?name=../../etc/passwd")
    assert resp.status_code == 400


def test_rejects_unknown_type_and_format():
    client = TestClient(app)
    assert client.post("/api/reports/generate",
                       json={"data_type": "nope", "format": "csv"}).status_code == 422
    assert client.post("/api/reports/generate",
                       json={"data_type": "holdings", "format": "pdf"}).status_code == 422
