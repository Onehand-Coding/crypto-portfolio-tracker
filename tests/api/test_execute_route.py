"""POST /api/execute/trade: order execution driven by the config switches.

No real order is placed here -- the tracker is mocked. These pin the surviving
gate (explicit confirm), that live_trading_enabled decides is_live, and that
the response reports the true testnet posture -- matching the CLI/Streamlit path.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from api import deps
from api.main import app


def _wire(testnet: bool, is_live: bool, result=None):
    deps._config_manager = Mock(is_testnet_mode=testnet, is_live=is_live)
    tracker = Mock()
    tracker.execute_manual_trade_core = AsyncMock(
        return_value=result or SimpleNamespace(success=True, messages=["ok"], errors=[])
    )
    deps._tracker = tracker
    return tracker


def test_status_reports_testnet_and_live():
    _wire(True, False)
    assert TestClient(app).get("/api/execute/status").json() == {
        "testnet": True, "is_live": False
    }


def test_places_real_order_when_live_enabled():
    tracker = _wire(True, True)

    body = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "buy", "symbol": "btc", "amount": 25,
        "is_quote_qty": True, "confirm": True,
    }).json()

    assert body["success"] is True
    assert body["testnet"] is True
    # Symbol upper-cased, ticker built, is_live taken from the config switch.
    tracker.execute_manual_trade_core.assert_awaited_once_with(
        "BUY", "BTC", "BTCUSDT", 25.0, True, True
    )


def test_simulates_when_live_disabled():
    tracker = _wire(True, False)

    TestClient(app).post("/api/execute/trade", json={
        "trade_type": "BUY", "symbol": "BTC", "amount": 25, "confirm": True,
    })

    # Live trading off -> the core is still called, but with is_live False (dry run).
    tracker.execute_manual_trade_core.assert_awaited_once_with(
        "BUY", "BTC", "BTCUSDT", 25.0, True, False
    )


def test_runs_on_mainnet_and_reports_it():
    tracker = _wire(False, True)

    body = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "BUY", "symbol": "BTC", "amount": 25, "confirm": True,
    }).json()

    # Mainnet is no longer refused; the response reports testnet False honestly.
    assert body["testnet"] is False
    tracker.execute_manual_trade_core.assert_awaited_once_with(
        "BUY", "BTC", "BTCUSDT", 25.0, True, True
    )


def test_requires_explicit_confirmation():
    tracker = _wire(True, True)

    resp = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "BUY", "symbol": "BTC", "amount": 25, "confirm": False,
    })

    assert resp.status_code == 400
    tracker.execute_manual_trade_core.assert_not_called()


def test_rejects_bad_trade_type():
    _wire(True, True)
    resp = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "HODL", "symbol": "BTC", "amount": 25, "confirm": True,
    })
    assert resp.status_code == 422


def test_rejects_non_positive_amount():
    _wire(True, True)
    resp = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "BUY", "symbol": "BTC", "amount": 0, "confirm": True,
    })
    assert resp.status_code == 422
