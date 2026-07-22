"""The batch/transfer/redeem execute endpoints: gates and wiring.

No real orders -- deps are mocked. These pin the two gates on every endpoint
(testnet-only, explicit confirm) and the transfer/redeem argument wiring.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from api import deps
from api.main import app

OK = SimpleNamespace(success=True, messages=["done"], errors=[])
BATCH_ENDPOINTS = ["rebalance", "dca", "profit", "transfer", "redeem"]

BODY = {
    "confirm": True, "asset": "USDT", "amount": 10,
    "from_wallet": "SPOT", "to_wallet": "FUNDING", "trades": [{"asset": "BTC", "amount": 10}],
}


def _wire(testnet: bool):
    deps._config_manager = Mock(is_testnet_mode=testnet)
    tracker = Mock()
    tracker.transfer_spot_to_funding = AsyncMock(return_value=OK)
    tracker.redeem_from_earn = Mock(return_value=OK)
    tracker.execute_dca_trades = AsyncMock(return_value=OK)
    deps._tracker = tracker
    return tracker


def test_every_endpoint_blocked_outside_testnet():
    _wire(False)
    client = TestClient(app)
    for ep in BATCH_ENDPOINTS:
        resp = client.post(f"/api/execute/{ep}", json={**BODY, "confirm": True})
        assert resp.status_code == 403, ep


def test_every_endpoint_requires_confirmation():
    _wire(True)
    client = TestClient(app)
    for ep in BATCH_ENDPOINTS:
        resp = client.post(f"/api/execute/{ep}", json={**BODY, "confirm": False})
        assert resp.status_code == 400, ep


def test_transfer_maps_direction_and_forwards_args():
    tracker = _wire(True)

    body = TestClient(app).post("/api/execute/transfer", json={
        "confirm": True, "asset": "usdt", "amount": 10,
        "from_wallet": "spot", "to_wallet": "funding",
    }).json()

    assert body["success"] is True
    tracker.transfer_spot_to_funding.assert_awaited_once_with(10.0, "USDT", True)


def test_transfer_rejects_unknown_route():
    _wire(True)
    resp = TestClient(app).post("/api/execute/transfer", json={
        "confirm": True, "asset": "USDT", "amount": 10,
        "from_wallet": "SPOT", "to_wallet": "MARS",
    })
    assert resp.status_code == 422


def test_redeem_forwards_args():
    tracker = _wire(True)

    TestClient(app).post("/api/execute/redeem", json={
        "confirm": True, "asset": "usdt", "amount": 7,
    })

    tracker.redeem_from_earn.assert_called_once_with("USDT", 7.0, True)


def test_dca_with_no_trades_is_409():
    _wire(True)
    resp = TestClient(app).post("/api/execute/dca", json={"confirm": True, "trades": []})
    assert resp.status_code == 409
