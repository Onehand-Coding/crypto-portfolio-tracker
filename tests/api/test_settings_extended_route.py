"""GET/PUT /api/system/settings: automation, APIs, lookbacks, logging, trend windows.

Seeds the mock config from the repo default_config.json so GET expectations
match the shipped defaults, not hand-written guesses.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

REPO_ROOT = Path(__file__).parents[2]


def _seed(mock_read_context, tmp_path):
    config = json.loads((REPO_ROOT / "config" / "default_config.json").read_text())
    config["exports"] = {"path": str(tmp_path)}
    mock_read_context.config_manager.config = config
    mock_read_context.config_manager.main_api_keys = {"api_key": "SECRET"}
    return mock_read_context.config_manager


def test_get_returns_extended_groups_with_config_values(mock_read_context, tmp_path):
    _seed(mock_read_context, tmp_path)
    body = TestClient(app).get("/api/system/settings").json()
    # Automation mirrors the Streamlit fallbacks (monthly/weekly); the shipped
    # file itself sets rebalancing to quarterly, which must come through as-is.
    assert body["automation"] == {
        "dca_frequency": "monthly",
        "rebalancing_frequency": "quarterly",
    }
    assert body["apis"] == {
        "coingecko_timeout": 30,
        "binance_timeout": 60,
        "binance_recv_window": 20000,
        "binance_delay_ms": 500,
        "coingecko_delay_ms": 1500,
    }
    # The stray "transfers" key in the shipped file is not a real lookback.
    assert body["history_lookback_days"]["trades"] == 90
    assert len(body["history_lookback_days"]) == 12
    assert "transfers" not in body["history_lookback_days"]
    assert body["logging"]["level"] == "INFO"
    assert body["logging"]["file_enabled"] is True
    assert body["logging"]["file_path"].endswith("logs/portfolio_tracker.log")
    assert body["logging"]["console_enabled"] is True
    assert body["trend_timeframes"]["long_term"] == {
        "period": "4y", "sma_short_window": 50, "sma_long_window": 200}
    assert body["trend_timeframes"]["swing"] == {
        "period": "3mo", "sma_short_window": 10, "sma_long_window": 30}
    assert body["trend_timeframes"]["day"] == {
        "period": "60d", "sma_short_window": 5, "sma_long_window": 15}


def test_put_frequencies_round_trip(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    body = TestClient(app).put("/api/system/settings", json={
        "automation": {"dca_frequency": "weekly", "rebalancing_frequency": "daily"},
    }).json()
    assert body["automation"] == {
        "dca_frequency": "weekly", "rebalancing_frequency": "daily"}
    assert cm.config["automation"]["dca"]["frequency"] == "weekly"
    assert cm.config["automation"]["rebalancing"]["frequency"] == "daily"
    cm.save_config.assert_called_once()


def test_put_bad_frequency_is_422(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    resp = TestClient(app).put("/api/system/settings", json={
        "automation": {"dca_frequency": "yearly", "rebalancing_frequency": "daily"},
    })
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_put_apis_round_trip(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    body = TestClient(app).put("/api/system/settings", json={
        "apis": {"coingecko_timeout": 10, "binance_timeout": 20,
                 "binance_recv_window": 5000, "binance_delay_ms": 100,
                 "coingecko_delay_ms": 200},
    }).json()
    assert body["apis"]["coingecko_timeout"] == 10
    assert body["apis"]["binance_recv_window"] == 5000
    assert cm.config["apis"]["binance"]["recv_window"] == 5000
    cm.save_config.assert_called_once()


def test_put_negative_timeout_is_422(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    resp = TestClient(app).put("/api/system/settings", json={
        "apis": {"coingecko_timeout": -5},
    })
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_put_lookback_merges_valid_keys(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    body = TestClient(app).put("/api/system/settings", json={
        "history_lookback_days": {"trades": 30},
    }).json()
    assert body["history_lookback_days"]["trades"] == 30
    assert body["history_lookback_days"]["deposits"] == 90
    assert cm.config["history_lookback_days"]["trades"] == 30


def test_put_unknown_lookback_key_is_422(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    resp = TestClient(app).put("/api/system/settings", json={
        "history_lookback_days": {"transfers": 30},
    })
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_put_lookback_below_min_is_422(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    resp = TestClient(app).put("/api/system/settings", json={
        "history_lookback_days": {"trades": 0},
    })
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_put_logging_round_trip(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    log_path = str(tmp_path / "logs" / "app.log")
    body = TestClient(app).put("/api/system/settings", json={
        "logging": {"level": "DEBUG", "file_enabled": True,
                    "file_path": log_path, "console_enabled": False},
    }).json()
    assert body["logging"] == {
        "level": "DEBUG", "file_enabled": True,
        "file_path": log_path, "console_enabled": False}
    assert cm.config["logging"]["level"] == "DEBUG"
    cm.save_config.assert_called_once()


def test_put_bad_log_level_is_422(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    resp = TestClient(app).put("/api/system/settings", json={
        "logging": {"level": "VERBOSE"},
    })
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_put_trend_windows_round_trip(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    body = TestClient(app).put("/api/system/settings", json={
        "trend_timeframes": {
            "long_term": {"sma_short_window": 40, "sma_long_window": 180},
            "swing": {"sma_short_window": 10, "sma_long_window": 30},
            "day": {"sma_short_window": 5, "sma_long_window": 15}},
    }).json()
    assert body["trend_timeframes"]["long_term"] == {
        "period": "4y", "sma_short_window": 40, "sma_long_window": 180}
    # The Streamlit period strings are untouched by a windows-only patch.
    assert cm.config["trend_analyzer"]["timeframe_settings"]["long_term"]["period"] == "4y"
    cm.save_config.assert_called_once()


def test_put_trend_period_round_trip(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    body = TestClient(app).put("/api/system/settings", json={
        "trend_timeframes": {
            "long_term": {"period": "5y", "sma_short_window": 50, "sma_long_window": 200},
            "swing": {"period": "6mo", "sma_short_window": 10, "sma_long_window": 30},
            "day": {"period": "60d", "sma_short_window": 5, "sma_long_window": 15}},
    }).json()
    assert body["trend_timeframes"]["long_term"]["period"] == "5y"
    assert body["trend_timeframes"]["swing"]["period"] == "6mo"
    assert cm.config["trend_analyzer"]["timeframe_settings"]["day"]["period"] == "60d"
    cm.save_config.assert_called_once()


def test_put_trend_bad_period_is_422(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    for bad in ("decade", "", "  "):
        resp = TestClient(app).put("/api/system/settings", json={
            "trend_timeframes": {
                "long_term": {"period": bad, "sma_short_window": 50, "sma_long_window": 200},
                "swing": {"period": "3mo", "sma_short_window": 10, "sma_long_window": 30},
                "day": {"period": "60d", "sma_short_window": 5, "sma_long_window": 15}},
        })
        assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_put_trend_short_ge_long_is_422(mock_read_context, tmp_path):
    cm = _seed(mock_read_context, tmp_path)
    resp = TestClient(app).put("/api/system/settings", json={
        "trend_timeframes": {
            "long_term": {"sma_short_window": 200, "sma_long_window": 50},
            "swing": {"sma_short_window": 10, "sma_long_window": 30},
            "day": {"sma_short_window": 5, "sma_long_window": 15}},
    })
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_put_response_echoes_updated_groups(mock_read_context, tmp_path):
    _seed(mock_read_context, tmp_path)
    body = TestClient(app).put("/api/system/settings", json={
        "automation": {"dca_frequency": "daily", "rebalancing_frequency": "weekly"},
        "apis": {"coingecko_timeout": 15, "binance_timeout": 45,
                 "binance_recv_window": 10000, "binance_delay_ms": 250,
                 "coingecko_delay_ms": 750},
    }).json()
    assert body["automation"]["dca_frequency"] == "daily"
    assert body["apis"]["coingecko_timeout"] == 15
    assert body["minimum_trade_usd"] == 5.0
