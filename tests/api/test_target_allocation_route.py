"""PUT /api/system/target-allocation: edit and persist the target allocation.

The config manager is a shared singleton, so save_config() must be called and
the two secret keys it strips from the live dict must be put back afterwards.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


def _cm(mock_read_context):
    cm = mock_read_context.config_manager
    # Real dicts: the route does item assignment the endpoint's Mock cannot do.
    cm.config = {"apis": {"coingecko": {}}, "target_allocation": {}}
    cm.main_api_keys = {"api_key": "SECRET"}
    return cm


def test_writes_upper_cased_allocation_and_persists(mock_read_context):
    cm = _cm(mock_read_context)

    body = TestClient(app).put(
        "/api/system/target-allocation", json={"allocation": {"btc": 0.6, "eth": 0.4}}
    ).json()

    assert body["allocation"] == {"BTC": 0.6, "ETH": 0.4}
    assert body["sum"] == pytest.approx(1.0)
    assert body["sums_to_one"] is True
    cm.save_config.assert_called_once()
    assert cm.config["target_allocation"] == {"BTC": 0.6, "ETH": 0.4}


def test_restores_secrets_stripped_by_save_config(mock_read_context):
    """save_config() deletes these from the live dict; the route restores them."""
    cm = _cm(mock_read_context)

    TestClient(app).put(
        "/api/system/target-allocation", json={"allocation": {"BTC": 1.0}}
    )

    # The shared config the tracker relies on must still carry its keys.
    assert cm.config["main_api_keys"] == {"api_key": "SECRET"}
    assert "coingecko" in cm.config["apis"]


def test_sums_to_one_is_false_when_weights_do_not_total_100(mock_read_context):
    _cm(mock_read_context)

    body = TestClient(app).put(
        "/api/system/target-allocation", json={"allocation": {"BTC": 0.5}}
    ).json()

    assert body["sums_to_one"] is False


def test_rejects_weight_above_one(mock_read_context):
    cm = _cm(mock_read_context)

    resp = TestClient(app).put(
        "/api/system/target-allocation", json={"allocation": {"BTC": 1.5}}
    )

    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_rejects_negative_weight(mock_read_context):
    cm = _cm(mock_read_context)

    resp = TestClient(app).put(
        "/api/system/target-allocation", json={"allocation": {"BTC": -0.2}}
    )

    assert resp.status_code == 422
    cm.save_config.assert_not_called()
