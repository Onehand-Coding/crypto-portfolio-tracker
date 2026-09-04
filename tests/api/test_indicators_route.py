"""Cached per-coin indicator history.

The GET serves the per-symbol cache file written by the indicators analysis;
a missing coin is an empty state, not a 404. Adapter validation must raise
before any fetch, so a bad symbol never touches the network.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.analysis_runner import _indicators
from api.main import app

POINTS = [
    {"date": "2026-01-01", "close": 42000.0, "sma_short": 41800.0,
     "sma_long": 41000.0, "rsi": 55.5, "macd": 12.3,
     "macd_signal": 10.1, "macd_hist": 2.2},
    {"date": "2026-01-02", "close": 42100.0, "sma_short": 41900.0,
     "sma_long": 41100.0, "rsi": None, "macd": 12.5,
     "macd_signal": 10.3, "macd_hist": 2.2},
    {"date": "2026-01-03", "close": 41900.0, "sma_short": None,
     "sma_long": None, "rsi": 52.0, "macd": None,
     "macd_signal": None, "macd_hist": None},
]


@pytest.fixture
def cached_indicators(mock_read_context, tmp_path, monkeypatch):
    """Seed the per-symbol indicators cache, isolated to tmp_path."""
    mock_read_context.config_manager.is_testnet_mode = True
    monkeypatch.chdir(tmp_path)
    path = Path("data") / "api_cache" / "indicators_BTC_swing_testnet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"symbol": "BTC", "timeframe": "swing", "points": POINTS, "_cached_at": 0}))
    return path


def test_returns_cached_points_with_null_passthrough(cached_indicators):
    response = TestClient(app).get("/api/strategy/indicators",
                                   params={"symbol": "BTC", "timeframe": "swing"})

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "BTC"
    assert body["timeframe"] == "swing"
    assert len(body["points"]) == 3
    assert body["points"][0]["close"] == pytest.approx(42000.0)
    assert body["points"][1]["rsi"] is None
    assert body["has_data"] is True


def test_unknown_coin_is_empty_state_not_404(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    monkeypatch.chdir(tmp_path)

    response = TestClient(app).get("/api/strategy/indicators",
                                   params={"symbol": "ETH", "timeframe": "swing"})

    assert response.status_code == 200
    assert response.json()["has_data"] is False


@pytest.mark.asyncio
async def test_adapter_rejects_bad_symbol_before_any_fetch():
    with pytest.raises(ValueError):
        await _indicators(Mock(), {"symbol": "!!!"})


@pytest.mark.asyncio
async def test_adapter_rejects_bad_timeframe():
    with pytest.raises(ValueError):
        await _indicators(Mock(), {"symbol": "BTC", "timeframe": "nope"})
