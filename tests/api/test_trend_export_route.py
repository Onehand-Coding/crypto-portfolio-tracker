"""POST /api/reports/trend: a cached technical report via the core exporter."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def trend_ctx(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {"paths": {"export_dir": str(tmp_path)}}
    monkeypatch.chdir(tmp_path)
    cache = Path("data") / "api_cache" / "technical_testnet.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "reports": {"swing": {"coin_analyses": {"BTC": {
            "symbol": "BTC",
            "current_price": 66450.0,
            "price_change_pct": 1.5,
            "rsi": 61.4,
            "support_level": 57747.0,
            "resistance_level": 66890.0,
            "active_conditions": ["Golden Cross"],
        }}}},
        "_cached_at": 0,
    }))
    return tmp_path


@pytest.mark.parametrize("fmt,ext", [("csv", ".csv"), ("json", ".json")])
def test_trend_export_writes_file(trend_ctx, fmt, ext):
    response = TestClient(app).post(
        "/api/reports/trend", json={"timeframe": "swing", "format": fmt})

    assert response.status_code == 200
    body = response.json()
    assert body["name"].endswith(ext)
    assert (trend_ctx / body["name"]).is_file()


def test_trend_export_csv_content(trend_ctx):
    response = TestClient(app).post(
        "/api/reports/trend", json={"timeframe": "swing", "format": "csv"})

    assert response.status_code == 200
    text = (trend_ctx / response.json()["name"]).read_text(errors="replace")
    assert "Symbol" in text
    assert "BTC" in text


def test_trend_export_json_content(trend_ctx):
    response = TestClient(app).post(
        "/api/reports/trend", json={"timeframe": "swing", "format": "json"})

    assert response.status_code == 200
    path = trend_ctx / response.json()["name"]
    text = path.read_text(errors="replace")
    data = json.loads(text)
    assert "coin_analyses" in data
    assert "BTC" in data["coin_analyses"]
    assert "BTC" in text


def test_trend_export_html_content(trend_ctx):
    response = TestClient(app).post(
        "/api/reports/trend", json={"timeframe": "swing", "format": "html"})

    assert response.status_code == 200
    path = trend_ctx / response.json()["name"]
    assert path.is_file()
    assert "<html" in path.read_text(errors="replace").lower()


def test_trend_export_unknown_timeframe_is_422(trend_ctx):
    response = TestClient(app).post(
        "/api/reports/trend", json={"timeframe": "nope", "format": "csv"})

    assert response.status_code == 422
