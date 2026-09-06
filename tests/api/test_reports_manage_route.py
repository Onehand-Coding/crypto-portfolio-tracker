"""Preview and delete for generated exports in the export dir."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def export_dir(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {"paths": {"export_dir": str(tmp_path)}}
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _seed(directory: Path, name: str, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(content)


def test_preview_returns_first_lines(export_dir):
    _seed(export_dir, "a.csv", "\n".join(f"row{i},x" for i in range(100)))
    body = TestClient(app).get("/api/reports/preview", params={"name": "a.csv"}).json()
    assert body["name"] == "a.csv"
    assert body["total_lines"] == 100
    assert len(body["lines"]) == 50
    assert body["truncated"] is True
    assert body["lines"][0] == "row0,x"


def test_preview_short_file_not_truncated(export_dir):
    _seed(export_dir, "b.csv", "h1,h2\n1,2\n")
    body = TestClient(app).get("/api/reports/preview", params={"name": "b.csv"}).json()
    assert body["truncated"] is False
    assert body["total_lines"] == 2


def test_preview_rejects_path_traversal(export_dir):
    response = TestClient(app).get("/api/reports/preview", params={"name": "../a.csv"})
    assert response.status_code == 400


def test_preview_missing_file_404(export_dir):
    response = TestClient(app).get("/api/reports/preview", params={"name": "nope.csv"})
    assert response.status_code == 404


def test_preview_spreadsheet_renders_rows_as_text(export_dir):
    pytest.importorskip("openpyxl")
    import pandas as pd

    pd.DataFrame({"symbol": ["BTC", "ETH"], "value": [1.5, 2.5]}).to_excel(
        export_dir / "holdings.xlsx", index=False)
    body = TestClient(app).get(
        "/api/reports/preview", params={"name": "holdings.xlsx"}).json()
    assert body["kind"] == "sheet"
    assert body["lines"][0] == "symbol,value"
    assert any("BTC" in line for line in body["lines"])
    assert body["truncated"] is False


def test_preview_image_returns_download_url_not_binary(export_dir):
    (export_dir / "chart.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    body = TestClient(app).get(
        "/api/reports/preview", params={"name": "chart.png"}).json()
    assert body["kind"] == "image"
    assert body["image_url"] == "/api/reports/download?name=chart.png"
    assert body["lines"] == []
    assert "PNG" not in str(body)


def test_preview_unsupported_type_says_so_plainly(export_dir):
    (export_dir / "data.bin").write_bytes(b"\x00\x01\x02")
    response = TestClient(app).get(
        "/api/reports/preview", params={"name": "data.bin"})
    assert response.status_code == 422
    assert "not available" in response.json()["detail"]


def test_delete_needs_confirm(export_dir):
    _seed(export_dir, "a.csv", "x\n")
    response = TestClient(app).post("/api/reports/delete", json={"name": "a.csv"})
    assert response.status_code == 400
    assert (export_dir / "a.csv").exists()


def test_delete_removes_file(export_dir):
    _seed(export_dir, "a.csv", "x\n")
    body = TestClient(app).post(
        "/api/reports/delete", json={"name": "a.csv", "confirm": True}).json()
    assert body["deleted"] is True
    assert not (export_dir / "a.csv").exists()
