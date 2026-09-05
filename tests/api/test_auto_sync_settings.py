"""GET/PUT /api/system/settings: auto-sync toggle and interval."""

from fastapi.testclient import TestClient

from api.main import app


def _cm(mock_read_context):
    cm = mock_read_context.config_manager
    cm.config = {"automation": {}}
    return cm


def test_auto_sync_defaults_off_when_unconfigured(mock_read_context):
    _cm(mock_read_context)
    body = TestClient(app).get("/api/system/settings").json()
    assert body["automation"]["auto_sync_enabled"] is False
    assert body["automation"]["auto_sync_interval_minutes"] == 5


def test_auto_sync_round_trip(mock_read_context):
    cm = _cm(mock_read_context)
    body = TestClient(app).put(
        "/api/system/settings",
        json={"automation": {"auto_sync_enabled": True,
                             "auto_sync_interval_minutes": 10}},
    ).json()
    assert body["automation"]["auto_sync_enabled"] is True
    assert body["automation"]["auto_sync_interval_minutes"] == 10
    assert cm.config["automation"]["auto_sync"] == {
        "enabled": True, "interval_minutes": 10}


def test_interval_below_minimum_is_rejected(mock_read_context):
    _cm(mock_read_context)
    res = TestClient(app).put(
        "/api/system/settings",
        json={"automation": {"auto_sync_interval_minutes": 1}})
    assert res.status_code == 422
