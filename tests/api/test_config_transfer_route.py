"""Config export/import and log preview. Secrets never reach disk or imports."""

import copy
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

REPO_ROOT = Path(__file__).parents[2]
LIVE_KEYS = {"api_key": "LIVEKEY", "api_secret": "LIVESECRET"}


def _seed(mock_read_context, tmp_path, monkeypatch):
    # Mirror ConfigManager wiring: the live CoinGecko key comes from the
    # environment, which is also what _save_config_preserving_secrets restores.
    monkeypatch.setenv("COINGECKO_API_KEY", "CGSECRET")
    config = json.loads((REPO_ROOT / "config" / "default_config.json").read_text())
    config["paths"] = {"export_dir": str(tmp_path)}
    config["exports"] = {"path": str(tmp_path)}
    config["main_api_keys"] = dict(LIVE_KEYS)
    config["apis"]["coingecko"]["api_key"] = os.getenv("COINGECKO_API_KEY")
    mock_read_context.config_manager.config = config
    mock_read_context.config_manager.main_api_keys = dict(LIVE_KEYS)
    return mock_read_context.config_manager


def _exported_files(tmp_path, prefix):
    return sorted(tmp_path.glob(f"{prefix}_*.json"))


def test_export_masks_secrets_and_drops_coingecko_key(mock_read_context, tmp_path, monkeypatch):
    cm = _seed(mock_read_context, tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/system/config/export")
    assert resp.status_code == 200
    exported = json.loads(resp.content)
    assert set(exported["main_api_keys"].values()) == {"********"}
    assert "api_key" not in exported["apis"]["coingecko"]
    # The live process keeps its real secrets; only the file is masked.
    assert cm.config["main_api_keys"] == LIVE_KEYS
    assert cm.config["apis"]["coingecko"]["api_key"] == "CGSECRET"
    assert len(_exported_files(tmp_path, "config_export")) == 1


def test_import_masked_file_updates_settings_and_preserves_secrets(
    mock_read_context, tmp_path, monkeypatch,
):
    cm = _seed(mock_read_context, tmp_path, monkeypatch)
    incoming = copy.deepcopy(cm.config)
    incoming["main_api_keys"] = {k: "********" for k in incoming["main_api_keys"]}
    del incoming["apis"]["coingecko"]["api_key"]
    incoming["portfolio"]["minimum_trade_usd"] = 9.5
    payload = json.dumps(incoming).encode()
    body = TestClient(app).post(
        "/api/system/config/import",
        files={"file": ("config_export_x.json", payload, "application/json")},
    ).json()
    assert body["minimum_trade_usd"] == 9.5
    assert cm.config["portfolio"]["minimum_trade_usd"] == 9.5
    assert cm.config["main_api_keys"] == LIVE_KEYS
    assert cm.config["apis"]["coingecko"]["api_key"] == "CGSECRET"
    backups = _exported_files(tmp_path, "config_backup")
    assert len(backups) == 1
    backup_text = backups[0].read_text()
    assert "LIVESECRET" not in backup_text
    assert "CGSECRET" not in backup_text
    cm.save_config.assert_called_once()


def test_import_non_json_is_422(mock_read_context, tmp_path, monkeypatch):
    cm = _seed(mock_read_context, tmp_path, monkeypatch)
    resp = TestClient(app).post(
        "/api/system/config/import",
        files={"file": ("config.json", b"not json{{{", "application/json")},
    )
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def test_import_json_list_is_422(mock_read_context, tmp_path, monkeypatch):
    cm = _seed(mock_read_context, tmp_path, monkeypatch)
    resp = TestClient(app).post(
        "/api/system/config/import",
        files={"file": ("config.json", b"[1, 2, 3]", "application/json")},
    )
    assert resp.status_code == 422
    cm.save_config.assert_not_called()


def _seed_log(mock_read_context, tmp_path, monkeypatch, count):
    log_path = tmp_path / "app.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(count)) + "\n")
    cm = _seed(mock_read_context, tmp_path, monkeypatch)
    cm.config["logging"]["file_config"]["path"] = str(log_path)
    return log_path


def test_log_preview_returns_last_n_lines(mock_read_context, tmp_path, monkeypatch):
    log_path = _seed_log(mock_read_context, tmp_path, monkeypatch, 100)
    body = TestClient(app).get(
        "/api/system/logs/preview", params={"lines": 50}).json()
    assert body["path"] == str(log_path)
    assert len(body["lines"]) == 50
    assert body["lines"][0] == "line 50"
    assert body["lines"][-1] == "line 99"
    assert body["truncated"] is True
    assert body["total_lines"] == 100


def test_log_preview_missing_file_is_404(mock_read_context, tmp_path, monkeypatch):
    cm = _seed(mock_read_context, tmp_path, monkeypatch)
    cm.config["logging"]["file_config"]["path"] = str(tmp_path / "ghost.log")
    resp = TestClient(app).get("/api/system/logs/preview")
    assert resp.status_code == 404


def test_log_preview_lines_clamped(mock_read_context, tmp_path, monkeypatch):
    _seed_log(mock_read_context, tmp_path, monkeypatch, 100)
    client = TestClient(app)
    body = client.get("/api/system/logs/preview", params={"lines": 0}).json()
    assert len(body["lines"]) == 1
    assert body["truncated"] is True
    body = client.get("/api/system/logs/preview", params={"lines": 1000}).json()
    assert len(body["lines"]) == 100
    assert body["truncated"] is False
    assert body["total_lines"] == 100


def test_import_partial_apis_deep_merges(mock_read_context, tmp_path, monkeypatch):
    cm = _seed(mock_read_context, tmp_path, monkeypatch)
    assert cm.config["apis"]["yfinance"] == {"request_delay_ms": 1000}
    assert cm.config["apis"]["binance"]["recv_window"] == 20000
    assert cm.config["apis"]["coingecko"]["api_key"] == "CGSECRET"
    payload = json.dumps({"apis": {"coingecko": {"timeout": 5}}}).encode()
    body = TestClient(app).post(
        "/api/system/config/import",
        files={"file": ("partial.json", payload, "application/json")},
    ).json()
    assert body["apis"]["coingecko_timeout"] == 5
    assert cm.config["apis"]["coingecko"]["timeout"] == 5
    assert cm.config["apis"]["yfinance"] == {"request_delay_ms": 1000}
    assert cm.config["apis"]["binance"]["recv_window"] == 20000
    assert cm.config["apis"]["coingecko"]["api_key"] == "CGSECRET"


def test_log_preview_large_file_returns_true_tail(mock_read_context, tmp_path, monkeypatch):
    cm = _seed(mock_read_context, tmp_path, monkeypatch)
    log_path = tmp_path / "big.log"
    total = 3000
    all_lines = [f"line {i:05d} " + "x" * 100 for i in range(total)]
    log_path.write_text("\n".join(all_lines) + "\n")
    assert log_path.stat().st_size > 200_000
    cm.config["logging"]["file_config"]["path"] = str(log_path)
    body = TestClient(app).get(
        "/api/system/logs/preview", params={"lines": 10}).json()
    assert body["lines"] == all_lines[-10:]
    assert body["truncated"] is True
