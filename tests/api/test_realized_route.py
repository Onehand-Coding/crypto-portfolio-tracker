"""The /api/realized route: FIFO realized gains surfaced from the core.

The endpoint wraps calculate_fifo_realized_gains -- the same function the
Streamlit tax report uses -- so these tests pin the mapping and, critically,
the three distinct states: no data, data-but-no-taxable-events, and gains.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


def _txns(rows):
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _cwd(monkeypatch, tmp_path, mock_read_context):
    # cache_path_for reads is_testnet_mode; chdir keeps the (absent) cache file
    # isolated to tmp so staleness reads as "never computed" without touching
    # the real data dir.
    mock_read_context.config_manager.is_testnet_mode = True
    monkeypatch.chdir(tmp_path)


def test_maps_a_realized_gain_from_a_sell(mock_read_context):
    """A BUY at 100 then a SELL at 150 realizes a gain of 50."""
    mock_read_context.db_manager.get_all_transactions.return_value = _txns([
        {"timestamp": "2025-01-01T00:00:00Z", "symbol": "BTC", "type": "BUY",
         "quantity": 1.0, "price_usd": 100.0, "fee_usd": 0.0},
        {"timestamp": "2025-02-01T00:00:00Z", "symbol": "BTC", "type": "SELL",
         "quantity": 1.0, "price_usd": 150.0, "fee_usd": 0.0},
    ])

    body = TestClient(app).get("/api/realized").json()

    assert body["has_data"] is True
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["symbol"] == "BTC"
    assert row["proceeds_usd"] == pytest.approx(150.0)
    assert row["cost_basis_usd"] == pytest.approx(100.0)
    assert row["gain_usd"] == pytest.approx(50.0)
    assert body["total_gain_usd"] == pytest.approx(50.0)
    assert body["by_asset"][0]["symbol"] == "BTC"
    assert body["by_asset"][0]["total_gain_usd"] == pytest.approx(50.0)


def test_transactions_without_sells_is_not_zero_gain(mock_read_context):
    """Holdings that were only ever bought have realized nothing.

    This must be has_data True with an empty rows list -- a real "no taxable
    events yet" state -- not has_data False, and never a fabricated 0.00 gain.
    """
    mock_read_context.db_manager.get_all_transactions.return_value = _txns([
        {"timestamp": "2025-01-01T00:00:00Z", "symbol": "BTC", "type": "BUY",
         "quantity": 1.0, "price_usd": 100.0, "fee_usd": 0.0},
    ])

    body = TestClient(app).get("/api/realized").json()

    assert body["has_data"] is True
    assert body["rows"] == []
    assert body["total_gain_usd"] is None


def test_no_transactions_reports_no_data(mock_read_context):
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = TestClient(app).get("/api/realized").json()

    assert body["has_data"] is False
    assert body["rows"] == []
    assert body["total_gain_usd"] is None
