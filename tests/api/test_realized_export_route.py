"""POST /api/reports/realized: realized FIFO gains as a file."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


def _txns(rows):
    return pd.DataFrame(rows)


GAIN_TXNS = [
    {"timestamp": "2025-01-01T00:00:00Z", "symbol": "BTC", "type": "BUY",
     "quantity": 1.0, "price_usd": 100.0, "fee_usd": 0.0},
    {"timestamp": "2025-02-01T00:00:00Z", "symbol": "BTC", "type": "SELL",
     "quantity": 1.0, "price_usd": 150.0, "fee_usd": 0.0},
]


@pytest.fixture
def realized_ctx(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {"paths": {"export_dir": str(tmp_path)}}
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_realized_export_excel(realized_ctx, mock_read_context):
    mock_read_context.db_manager.get_all_transactions.return_value = _txns(GAIN_TXNS)

    response = TestClient(app).post("/api/reports/realized", json={"format": "excel"})

    assert response.status_code == 200
    assert response.json()["name"].endswith(".xlsx")
    assert (Path(realized_ctx) / response.json()["name"]).is_file()


def test_realized_export_csv(realized_ctx, mock_read_context):
    mock_read_context.db_manager.get_all_transactions.return_value = _txns(GAIN_TXNS)

    response = TestClient(app).post("/api/reports/realized", json={"format": "csv"})

    assert response.status_code == 200
    assert response.json()["name"].endswith(".csv")
    assert (Path(realized_ctx) / response.json()["name"]).is_file()


def test_realized_export_empty_is_422(realized_ctx, mock_read_context):
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    response = TestClient(app).post("/api/reports/realized", json={"format": "csv"})

    assert response.status_code == 422
