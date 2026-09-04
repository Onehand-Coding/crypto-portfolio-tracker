"""POST /api/reports/summary: portfolio summary via the core exporters."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def summary_ctx(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {
        "paths": {"export_dir": str(tmp_path)},
        "target_allocation": {"BTC": 1.0},
    }
    monkeypatch.chdir(tmp_path)
    cache = Path("data") / "api_cache" / "metrics_testnet.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "holdings_df": [{
            "symbol": "BTC",
            "value_usd": 146.49,
            "current_price": 95000.0,
            "average_cost_basis": 80000.0,
        }],
        "_cached_at": 0,
    }))
    return tmp_path


@pytest.mark.parametrize("fmt,ext", [("csv", ".csv"), ("excel", ".xlsx"), ("html", ".html")])
def test_summary_export_writes_file(summary_ctx, fmt, ext):
    response = TestClient(app).post("/api/reports/summary", json={"format": fmt})

    assert response.status_code == 200
    assert response.json()["name"].endswith(ext)


def test_summary_export_rejects_bad_format(summary_ctx):
    response = TestClient(app).post("/api/reports/summary", json={"format": "pdf"})

    assert response.status_code == 422


def test_summary_export_without_metrics_is_422(summary_ctx):
    (Path("data") / "api_cache" / "metrics_testnet.json").unlink()

    response = TestClient(app).post("/api/reports/summary", json={"format": "csv"})

    assert response.status_code == 422
