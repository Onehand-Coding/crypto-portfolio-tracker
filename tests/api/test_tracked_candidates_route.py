"""GET /api/system/tracked-candidates: checkbox universe for the track list."""

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


def _seed(path, symbols):
    from api.cache import MetricsCache
    MetricsCache(path).write({
        "total_value_usd": 1.0,
        "holdings_df": pd.DataFrame([{"symbol": s} for s in symbols]),
    })


def test_candidates_union_all_sources(mock_read_context, tmp_path, monkeypatch):
    live, testnet = tmp_path / "live.json", tmp_path / "testnet.json"
    _seed(live, ["BTC", "DOGE"])
    _seed(testnet, ["BTC", "QUICK"])
    monkeypatch.setattr("api.routes.screens.all_metrics_paths",
                        lambda: [live, testnet])
    mock_read_context.config_manager.config = {
        "portfolio": {"tracked_symbols": ["ETH"], "stablecoin_symbols": ["USDT"]},
        "target_allocation": {"BTC": 0.5, "ETH": 0.5},
    }

    body = TestClient(app).get("/api/system/tracked-candidates").json()

    # BTC is in both caches and targets: listed once. QUICK comes only from
    # the testnet cache even though the server may be running live.
    assert body["symbols"] == ["BTC", "DOGE", "ETH", "QUICK", "USDT"]
    assert body["core"] == ["BTC", "ETH"]
    assert body["stored"] == ["ETH"]


def test_candidates_survive_missing_caches(mock_read_context, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routes.screens.all_metrics_paths",
                        lambda: [tmp_path / "nope.json"])
    mock_read_context.config_manager.config = {
        "portfolio": {"tracked_symbols": [], "stablecoin_symbols": []},
        "target_allocation": {"SOL": 1.0},
    }

    body = TestClient(app).get("/api/system/tracked-candidates").json()

    assert body == {"symbols": ["SOL"], "core": ["SOL"], "stored": [],
                    "warnings": ["SOL has a target weight but no trend ticker "
                                 "(expected SOL-USD in Cryptocurrencies) -- it "
                                 "will be skipped in Rebalancing suggestions."]}


def test_warnings_name_targets_missing_coverage(mock_read_context, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routes.screens.all_metrics_paths",
                        lambda: [tmp_path / "nope.json"])
    mock_read_context.config_manager.config = {
        "portfolio": {"tracked_symbols": ["BTC"], "stablecoin_symbols": []},
        "target_allocation": {"BTC": 0.5, "ETH": 0.5},
        "trend_analyzer": {"cryptocurrencies": ["BTC-USD"]},
    }

    body = TestClient(app).get("/api/system/tracked-candidates").json()

    assert body["warnings"] == [
        "ETH has a target weight but is not tracked -- it will not be priced, "
        "and reads hide it.",
        "ETH has a target weight but no trend ticker (expected ETH-USD in "
        "Cryptocurrencies) -- it will be skipped in Rebalancing suggestions.",
    ]


def test_no_warnings_when_aligned(mock_read_context, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routes.screens.all_metrics_paths",
                        lambda: [tmp_path / "nope.json"])
    mock_read_context.config_manager.config = {
        "portfolio": {"tracked_symbols": [], "stablecoin_symbols": []},
        "target_allocation": {"BTC": 0.5, "ETH": 0.5},
        "trend_analyzer": {"cryptocurrencies": ["BTC-USD", "ETH-USD"]},
    }

    body = TestClient(app).get("/api/system/tracked-candidates").json()

    # Empty list means automatic, and the automatic set covers every target
    # here -- nothing to warn about.
    assert body["warnings"] == []
