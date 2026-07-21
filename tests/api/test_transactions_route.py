"""The /api/transactions route: the global trade log across every asset."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(autouse=True)
def _cwd(monkeypatch, tmp_path, mock_read_context):
    mock_read_context.config_manager.is_testnet_mode = True
    monkeypatch.chdir(tmp_path)


def test_maps_rows_newest_first_with_computed_value(mock_read_context):
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame([
        {"timestamp": "2025-01-01T00:00:00Z", "symbol": "BTC", "type": "BUY",
         "quantity": 2.0, "price_usd": 100.0, "fee_usd": 0.5, "source": "x", "notes": "n"},
        {"timestamp": "2025-03-01T00:00:00Z", "symbol": "ETH", "type": "SELL",
         "quantity": 1.0, "price_usd": 50.0, "fee_usd": 0.0, "source": "y", "notes": None},
    ])

    body = TestClient(app).get("/api/transactions").json()

    assert body["has_data"] is True
    assert body["count"] == 2
    # Newest first.
    assert body["rows"][0]["symbol"] == "ETH"
    # value_usd is quantity * price.
    assert body["rows"][1]["value_usd"] == pytest.approx(200.0)


def test_missing_price_gives_null_value_not_zero(mock_read_context):
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame([
        {"timestamp": "2025-01-01T00:00:00Z", "symbol": "DOGE", "type": "DEPOSIT",
         "quantity": 10.0, "price_usd": None, "fee_usd": None, "source": None, "notes": None},
    ])

    row = TestClient(app).get("/api/transactions").json()["rows"][0]

    assert row["price_usd"] is None
    assert row["value_usd"] is None


def test_no_transactions_reports_no_data(mock_read_context):
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = TestClient(app).get("/api/transactions").json()

    assert body["has_data"] is False
    assert body["count"] == 0
    assert body["rows"] == []
