"""POST /api/execute/trade: testnet-gated order execution.

No real order is placed here -- the tracker is mocked. These pin the two
safety gates (testnet-only, explicit confirm) and the argument wiring.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from api import deps
from api.main import app


def _wire(testnet: bool, result=None):
    deps._config_manager = Mock(is_testnet_mode=testnet)
    tracker = Mock()
    tracker.execute_manual_trade_core = AsyncMock(
        return_value=result or SimpleNamespace(success=True, messages=["ok"], errors=[])
    )
    deps._tracker = tracker
    return tracker


def test_status_reports_testnet():
    _wire(True)
    assert TestClient(app).get("/api/execute/status").json() == {"testnet": True}


def test_places_order_on_testnet_with_correct_args():
    tracker = _wire(True)

    body = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "buy", "symbol": "btc", "amount": 25,
        "is_quote_qty": True, "confirm": True,
    }).json()

    assert body["success"] is True
    assert body["testnet"] is True
    # Symbol upper-cased, ticker built, is_live forced True.
    tracker.execute_manual_trade_core.assert_awaited_once_with(
        "BUY", "BTC", "BTCUSDT", 25.0, True, True
    )


def test_blocked_outside_testnet_and_places_nothing():
    tracker = _wire(False)

    resp = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "BUY", "symbol": "BTC", "amount": 25, "confirm": True,
    })

    assert resp.status_code == 403
    tracker.execute_manual_trade_core.assert_not_called()


def test_requires_explicit_confirmation():
    tracker = _wire(True)

    resp = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "BUY", "symbol": "BTC", "amount": 25, "confirm": False,
    })

    assert resp.status_code == 400
    tracker.execute_manual_trade_core.assert_not_called()


def test_rejects_bad_trade_type():
    _wire(True)
    resp = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "HODL", "symbol": "BTC", "amount": 25, "confirm": True,
    })
    assert resp.status_code == 422


def test_rejects_non_positive_amount():
    _wire(True)
    resp = TestClient(app).post("/api/execute/trade", json={
        "trade_type": "BUY", "symbol": "BTC", "amount": 0, "confirm": True,
    })
    assert resp.status_code == 422
