"""Snapshot save, resources and connection-test endpoints (deps mocked)."""

from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import deps
from api.cache import MetricsCache, cache_path_for
from api.main import app

METRICS = {
    "total_value_usd": 1000.0,
    "total_cost_basis_usd": 800.0,
    "unrealized_pl_usd": 200.0,
    "unrealized_pl_percent": 25.0,
}


@pytest.fixture(autouse=True)
def _cwd(monkeypatch, tmp_path, mock_read_context):
    monkeypatch.chdir(tmp_path)
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {"version": "9.9.9-test"}


def _seed_cache(mock_read_context):
    MetricsCache(cache_path_for(mock_read_context.config_manager)).write(dict(METRICS))


def _snapshot_store(mock_read_context):
    """Back the mocked save/list with an in-memory list, so the test checks the
    row through GET /system/snapshots rather than asserting on a mock."""
    store: list[dict] = []

    def _save(**kwargs):
        store.append({
            "timestamp": kwargs["timestamp"],
            "total_value_usd": kwargs["total_value"],
            "total_cost_basis_usd": kwargs["total_cost_basis"],
            "unrealized_pl_usd": kwargs["unrealized_pl"],
            "unrealized_pl_percent": kwargs["unrealized_pl_percent"],
        })

    mock_read_context.db_manager.save_portfolio_snapshot.side_effect = _save
    mock_read_context.db_manager.get_all_snapshots.side_effect = lambda: pd.DataFrame(store)
    return store


# --- snapshot save ---

def test_snapshot_save_persists_cached_metrics(mock_read_context):
    _seed_cache(mock_read_context)
    _snapshot_store(mock_read_context)
    client = TestClient(app)

    assert client.get("/api/system/snapshots").json()["count"] == 0
    body = client.post("/api/system/snapshot/save").json()

    assert body["saved"] is True
    assert body["timestamp"]
    rows = client.get("/api/system/snapshots").json()
    assert rows["count"] == 1
    assert rows["rows"][0]["total_value_usd"] == 1000.0
    mock_read_context.db_manager.save_portfolio_snapshot.assert_called_once()
    kwargs = mock_read_context.db_manager.save_portfolio_snapshot.call_args.kwargs
    assert kwargs["total_value"] == 1000.0
    assert kwargs["total_cost_basis"] == 800.0
    assert kwargs["unrealized_pl"] == 200.0
    assert kwargs["unrealized_pl_percent"] == 25.0
    assert kwargs["timestamp"] == body["timestamp"]


def test_snapshot_save_without_cache_is_422(mock_read_context):
    resp = TestClient(app).post("/api/system/snapshot/save")
    assert resp.status_code == 422
    mock_read_context.db_manager.save_portfolio_snapshot.assert_not_called()


def test_snapshot_save_failure_is_reported_not_500(mock_read_context):
    _seed_cache(mock_read_context)
    mock_read_context.db_manager.save_portfolio_snapshot.side_effect = Exception("disk full")
    body = TestClient(app).post("/api/system/snapshot/save").json()
    assert body["saved"] is False
    assert "disk full" in (body["error"] or "")


# --- resources ---

def test_resources_reports_host_figures(mock_read_context):
    body = TestClient(app).get("/api/system/resources").json()
    assert body["app_version"] == "9.9.9-test"
    assert body["python_version"]
    for field in ("cpu_percent", "ram_percent", "ram_used_gb", "disk_percent"):
        assert body[field] is None or isinstance(body[field], (int, float))


# --- connections ---

def _tracker(binance_client, btc_price: float | None = 65000.0, coingecko_error=None):
    tracker = Mock()
    tracker.binance_client = binance_client
    if coingecko_error is not None:
        tracker.enricher.get_current_prices = AsyncMock(side_effect=coingecko_error)
    else:
        tracker.enricher.get_current_prices = AsyncMock(return_value={"BTC": btc_price})
    deps._tracker = tracker
    return tracker


def test_connections_all_ok():
    _tracker(Mock(), btc_price=65000.0)
    body = TestClient(app).post("/api/system/connections").json()
    assert body["binance"]["ok"] is True
    assert body["coingecko"]["ok"] is True
    assert body["btc_price_usd"] == 65000.0


def test_connections_binance_down_still_probes_coingecko():
    client = Mock()
    client.ping.side_effect = Exception("timeout")
    tracker = _tracker(client, btc_price=65000.0)
    body = TestClient(app).post("/api/system/connections").json()
    assert body["binance"]["ok"] is False
    assert "timeout" in (body["binance"]["detail"] or "")
    assert body["coingecko"]["ok"] is True
    tracker.enricher.get_current_prices.assert_awaited_once_with(["BTC"])


def test_connections_without_keys_skips_binance():
    _tracker(None, btc_price=65000.0)
    body = TestClient(app).post("/api/system/connections").json()
    assert body["binance"]["ok"] is False
    assert "No API keys" in (body["binance"]["detail"] or "")
    assert body["coingecko"]["ok"] is True


def test_connections_coingecko_failure_nulls_btc_price():
    _tracker(Mock(), coingecko_error=Exception("rate limited"))
    body = TestClient(app).post("/api/system/connections").json()
    assert body["binance"]["ok"] is True
    assert body["coingecko"]["ok"] is False
    assert body["btc_price_usd"] is None


def test_connections_without_tracker_reports_both_down(monkeypatch):
    def _boom():
        raise RuntimeError("no keys configured")

    monkeypatch.setattr("api.routes.screens.get_tracker", _boom)
    body = TestClient(app).post("/api/system/connections").json()
    assert body["binance"]["ok"] is False
    assert body["coingecko"]["ok"] is False
    assert body["btc_price_usd"] is None


def test_connections_is_post_only():
    # No GET handler runs the probe: the SPA catch-all 404s unmatched /api/*
    # methods rather than executing anything network-touching.
    assert TestClient(app).get("/api/system/connections").status_code == 404


def test_connections_empty_price_is_not_a_zero_btc_price():
    _tracker(Mock(), btc_price=None)
    tracker = deps._tracker
    tracker.enricher.get_current_prices = AsyncMock(return_value={})
    body = TestClient(app).post("/api/system/connections").json()
    assert body["coingecko"]["ok"] is False
    # Unknown is null, never a confident $0.00.
    assert body["btc_price_usd"] is None
