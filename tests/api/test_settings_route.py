"""GET/PUT /api/system/settings: editable trading, profit-taking, currency."""

from fastapi.testclient import TestClient

from api.main import app

PROFIT = {
    "enabled": True, "min_opportunity_score": 60, "min_unrealized_gain_pct": 20,
    "min_unrealized_gain_usd": 25, "max_gain_take_pct": 50, "default_take_percentage": 30,
}


def _cm(mock_read_context):
    cm = mock_read_context.config_manager
    cm.config = {
        "apis": {"coingecko": {}},
        "portfolio": {"minimum_trade_usd": 5.0, "p2p_fiat_currency": "PHP",
                      "crypto_quotes": [], "stablecoin_symbols": ["USDT"]},
        "profit_taking": dict(PROFIT),
    }
    cm.main_api_keys = {"api_key": "SECRET"}
    return cm


def test_get_reads_current_settings(mock_read_context):
    _cm(mock_read_context)
    body = TestClient(app).get("/api/system/settings").json()
    assert body["minimum_trade_usd"] == 5.0
    assert body["profit_taking"]["default_take_percentage"] == 30
    assert body["stablecoin_symbols"] == ["USDT"]


def test_partial_patch_changes_only_present_fields(mock_read_context):
    cm = _cm(mock_read_context)
    body = TestClient(app).put(
        "/api/system/settings", json={"minimum_trade_usd": 12.5}
    ).json()
    assert body["minimum_trade_usd"] == 12.5
    # Untouched fields survive.
    assert body["p2p_fiat_currency"] == "PHP"
    cm.save_config.assert_called_once()
    assert cm.config["portfolio"]["minimum_trade_usd"] == 12.5


def test_lists_are_normalised_upper_and_trimmed(mock_read_context):
    _cm(mock_read_context)
    body = TestClient(app).put(
        "/api/system/settings",
        json={"stablecoin_symbols": [" usdt ", "usdc", ""]},
    ).json()
    assert body["stablecoin_symbols"] == ["USDT", "USDC"]


def test_restores_secrets_after_save(mock_read_context):
    cm = _cm(mock_read_context)
    TestClient(app).put("/api/system/settings", json={"minimum_trade_usd": 8})
    assert cm.config["main_api_keys"] == {"api_key": "SECRET"}


def test_rejects_non_positive_minimum_trade(mock_read_context):
    cm = _cm(mock_read_context)
    resp = TestClient(app).put("/api/system/settings", json={"minimum_trade_usd": 0})
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_rejects_take_percentage_over_100(mock_read_context):
    cm = _cm(mock_read_context)
    bad = dict(PROFIT, default_take_percentage=150)
    resp = TestClient(app).put("/api/system/settings", json={"profit_taking": bad})
    assert resp.status_code == 422
    cm.save_config.assert_not_called()
