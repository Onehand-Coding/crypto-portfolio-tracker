"""GET /api/strategy/completion: the no-sell finish-to-targets plan.

Same math as the CLI/Streamlit surfaces, served offline from the metrics
cache. Worked example (spec §2): BTC 146.49 @ 0.35 + ETH 24.49 @ 0.30 →
anchor BTC, implied total 418.54, ETH need 101.07, additional total 247.56.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

TARGET = {
    "BTC": 0.35, "ETH": 0.30, "SOL": 0.10, "RENDER": 0.06,
    "TAO": 0.06, "AVAX": 0.05, "LINK": 0.05, "ONDO": 0.03,
}

HOLDINGS = [
    {"symbol": "BTC", "value_usd": 146.49, "current_price": 95000.0},
    {"symbol": "ETH", "value_usd": 24.49, "current_price": 3200.0},
]


@pytest.fixture
def completion_setup(mock_read_context, tmp_path, monkeypatch):
    """Seed config + metrics cache the route reads, isolated to tmp_path."""
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {"target_allocation": dict(TARGET)}
    monkeypatch.chdir(tmp_path)
    path = Path("data") / "api_cache" / "metrics_testnet.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def seed(holdings):
        path.write_text(json.dumps({"holdings_df": holdings, "_cached_at": 0}))

    seed(HOLDINGS)
    return seed


def test_worked_example(completion_setup):
    body = TestClient(app).get("/api/strategy/completion").json()
    assert body["valid"] is True
    assert body["anchor_symbol"] == "BTC"
    assert body["implied_total_usd"] == pytest.approx(418.54, rel=1e-4)
    by_symbol = {r["symbol"]: r for r in body["rows"]}
    assert by_symbol["ETH"]["need_usd"] == pytest.approx(101.07, rel=1e-4)
    assert by_symbol["BTC"]["need_usd"] == pytest.approx(0.0, abs=0.01)
    assert body["additional_total_usd"] == pytest.approx(247.56, rel=1e-4)
    assert len(body["rows"]) == len(TARGET)


def test_empty_portfolio_has_no_anchor(completion_setup):
    completion_setup([])
    body = TestClient(app).get("/api/strategy/completion").json()
    assert body["valid"] is False
    assert "anchor" in (body["message"] or "").lower()


def test_unpriced_holding_is_null_not_zero(completion_setup):
    """A present-but-unpriced holding must not anchor and must not read $0."""
    completion_setup([
        {"symbol": "BTC", "value_usd": 146.49, "current_price": 95000.0},
        {"symbol": "ETH", "value_usd": None, "current_price": None},
    ])
    body = TestClient(app).get("/api/strategy/completion").json()
    by_symbol = {r["symbol"]: r for r in body["rows"]}
    assert by_symbol["ETH"]["current_value_usd"] is None
    assert body["anchor_symbol"] == "BTC"


def test_no_target_allocation(completion_setup, mock_read_context):
    mock_read_context.config_manager.config = {"target_allocation": {}}
    body = TestClient(app).get("/api/strategy/completion").json()
    assert body["valid"] is False


def test_zero_weight_asset_never_anchors(completion_setup, mock_read_context):
    mock_read_context.config_manager.config = {
        "target_allocation": {"BTC": 1.0, "DUST": 0.0}
    }
    completion_setup([
        {"symbol": "BTC", "value_usd": 146.49, "current_price": 95000.0},
        {"symbol": "DUST", "value_usd": 50.0, "current_price": 1.0},
    ])
    body = TestClient(app).get("/api/strategy/completion").json()
    assert body["anchor_symbol"] == "BTC"
    assert body["implied_total_usd"] == pytest.approx(146.49, rel=1e-4)
    by_symbol = {r["symbol"]: r for r in body["rows"]}
    assert by_symbol["DUST"]["need_usd"] == pytest.approx(0.0, abs=0.01)
