"""POST /api/reports/charts: all portfolio charts as PNGs via the core visualizer."""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def charts_ctx(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {
        "paths": {"export_dir": str(tmp_path)},
        "target_allocation": {"BTC": 0.6, "ETH": 0.4},
    }
    monkeypatch.chdir(tmp_path)
    # The endpoint reads snapshots via ctx.db_manager.get_all_snapshots(); the
    # value-history chart needs a non-empty frame with total_value_usd.
    mock_read_context.db_manager.get_all_snapshots.return_value = pd.DataFrame([
        {"timestamp": "2026-01-01T00:00:00", "total_value_usd": 9000.0},
        {"timestamp": "2026-02-01T00:00:00", "total_value_usd": 10000.0},
    ])
    cache = Path("data") / "api_cache" / "metrics_testnet.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "holdings_df": [
            {
                "symbol": "BTC",
                "value_usd": 6000.0,
                "allocation": 0.6,
                "unrealized_pl_usd": 1000.0,
                "current_price": 95000.0,
                "average_cost_basis": 80000.0,
            },
            {
                "symbol": "ETH",
                "value_usd": 4000.0,
                "allocation": 0.4,
                "unrealized_pl_usd": -200.0,
                "current_price": 3500.0,
                "average_cost_basis": 3600.0,
            },
        ],
        "_cached_at": 0,
    }))
    return tmp_path


def test_charts_export_writes_png(charts_ctx):
    response = TestClient(app).post("/api/reports/charts")

    assert response.status_code == 200
    body = response.json()
    assert body["name"].endswith(".png")
    assert (charts_ctx / body["name"]).is_file()
    # Lower bound, not exact: robust to future chart additions.
    assert len([p for p in charts_ctx.rglob("*.png") if p.is_file()]) >= 3


def test_charts_export_without_metrics_is_422(charts_ctx):
    (Path("data") / "api_cache" / "metrics_testnet.json").unlink()

    response = TestClient(app).post("/api/reports/charts")

    assert response.status_code == 422
