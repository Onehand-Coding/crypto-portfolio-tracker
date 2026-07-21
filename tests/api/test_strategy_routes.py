"""The rebalance route's mapping from the core's column names.

The core labels these columns for humans -- "Target %", "Drift (pts)" -- and
the route has to translate them. The payload below is copied verbatim from a
real run against live data, because the previous alias list was guessed from
snake_case names the core never emits: every field mapped to null and the
screen rendered eight rows of "?" while still returning 200. Endpoint tests
that only assert a status code cannot catch that.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

# Verbatim from data/api_cache/rebalance_live.json after a successful run.
REAL_ROW = {
    "Symbol": "BTC",
    "Target %": 35.0,
    "Current %": 99.82631899256708,
    "Drift (pts)": 64.82631899256708,
    "Current Value (USD)": 99.50001336000001,
    "TA_Price": 66426.828125,
    "Support": 57747.765625,
    "Resistance": 66890.5,
    "TA_Conditions": "Golden Cross, Price > SMA200, Neutral RSI (40-60)",
    "Signal": "SELL",
    "Suggested Action Detail": "Sell ~$48.46 worth, which is 0.00072953678 BTC",
    "action_usd_value": 48.46081427424205,
    "action_coin_quantity": 0.0007295367796735673,
}


@pytest.fixture
def cached_rebalance(mock_read_context, tmp_path, monkeypatch):
    """Write a rebalance cache the route will read, isolated to tmp_path."""
    mock_read_context.config_manager.is_testnet_mode = True
    monkeypatch.chdir(tmp_path)
    path = Path("data") / "api_cache" / "rebalance_testnet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"suggestions": [REAL_ROW], "_cached_at": 0}))
    return path


def test_maps_the_core_display_column_names(cached_rebalance):
    """Every field the screen shows must survive the translation."""
    response = TestClient(app).get("/api/strategy/rebalance")
    assert response.status_code == 200

    row = response.json()["suggestions"][0]
    assert row["symbol"] == "BTC"
    assert row["action"] == "SELL"
    assert row["target_allocation_pct"] == pytest.approx(35.0)
    assert row["current_allocation_pct"] == pytest.approx(99.826, rel=1e-3)
    assert row["drift_pct"] == pytest.approx(64.826, rel=1e-3)
    assert row["current_value_usd"] == pytest.approx(99.5, rel=1e-3)
    assert row["action_amount_usd"] == pytest.approx(48.46, rel=1e-3)
    assert row["action_quantity"] == pytest.approx(0.00072953, rel=1e-3)
    assert "Sell" in row["reason"]


def test_a_renamed_column_reads_as_unknown_not_zero(cached_rebalance):
    """A column the core renames must go null, never 0.0.

    Zero is a meaningful drift and a meaningful trade size. Rendering an
    unmapped column as zero would state, confidently, that there is nothing
    to do.
    """
    row = dict(REAL_ROW)
    del row["Drift (pts)"]
    cached_rebalance.write_text(json.dumps({"suggestions": [row], "_cached_at": 0}))

    response = TestClient(app).get("/api/strategy/rebalance")

    assert response.json()["suggestions"][0]["drift_pct"] is None


REAL_TECHNICAL = {
    "reports": {
        "swing": {
            "timeframe": "swing",
            "coin_analyses": {
                "BTC-USD": {
                    "symbol": "BTC-USD",
                    "current_price": 66450.0,
                    "rsi": 61.47786118003897,
                    "support_level": 57747.765625,
                    "resistance_level": 66890.5,
                    "active_conditions": ["Golden Cross", "Price > SMA200"],
                }
            },
        }
    },
    "_cached_at": 0,
}


@pytest.fixture
def cached_technical(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    monkeypatch.chdir(tmp_path)
    path = Path("data") / "api_cache" / "technical_testnet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(REAL_TECHNICAL))
    return path


def test_technical_strips_the_quote_currency_suffix(cached_technical):
    """The trend analyzer reports "BTC-USD"; every other screen keys on "BTC".

    Unstripped, the Market screen's join against holdings matches nothing and
    the indicator columns render blank with no error anywhere.
    """
    response = TestClient(app).get("/api/strategy/technical")

    assert response.json()["timeframes"]["swing"][0]["symbol"] == "BTC"


def test_technical_reads_the_level_suffixed_support_keys(cached_technical):
    """The core emits support_level/resistance_level, not support/resistance."""
    row = TestClient(app).get("/api/strategy/technical").json()["timeframes"]["swing"][0]

    assert row["support"] == pytest.approx(57747.77, rel=1e-4)
    assert row["resistance"] == pytest.approx(66890.5, rel=1e-4)
    assert row["rsi"] == pytest.approx(61.478, rel=1e-4)
