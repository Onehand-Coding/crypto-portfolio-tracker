# Parity Phase A: Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 9 export/output gaps: reports preview + delete, backup download + delete, portfolio-summary exports, trend exports, realized Excel/CSV, trade-log Excel, snapshots CSV.

**Architecture:** All new files flow through the existing export dir (`_export_dir`) and its list/download endpoints. New binary writes reuse the two established guards: plain-filename check (`name == Path(name).name`) and confirm-gate for deletes (400 without `confirm: true`, mirroring snapshot delete). Core `ReportGenerator` is constructed with config + the read context's `db_manager` only — never a tracker, no network. No `src/` changes.

**Tech Stack:** FastAPI + Pydantic v2, pandas (already used in `api/routes/screens.py`), pytest + TestClient, React + Vitest.

**Files:**
- Modify: `api/schemas/screens.py` (append request/response models)
- Modify: `api/routes/screens.py` (7 new endpoints)
- Create: `tests/api/test_reports_manage_route.py`, `tests/api/test_backup_transfer_route.py`, `tests/api/test_summary_export_route.py`, `tests/api/test_trend_export_route.py`, `tests/api/test_realized_export_route.py`
- Modify: `frontend/src/screens/Reports.tsx`, `frontend/src/screens/SystemHealth.tsx` (or wherever backup restore lives — read first), `frontend/src/screens/TradeLog.tsx`, `frontend/src/screens/Realized.tsx`, `frontend/src/screens/DataManage.tsx`
- Create: `frontend/src/screens/Reports.test.tsx`, `frontend/src/screens/TradeLog.test.tsx`, `frontend/src/screens/Realized.test.tsx`

**Pre-reads (do before touching code):** `ReportGenerator.__init__` signature (`src/crypto_portfolio_tracker/report_generator.py:27`) — pass only `config=` + `db_manager=`; if the constructor requires live objects (fetcher/client), STOP and report BLOCKED rather than inventing arguments. Existing realized test seeding: read `tests/api/test_realized_route.py` for how transactions reach `ctx.db_manager` in tests.

---

### Task 1: Reports preview + delete

**Files:**
- Modify: `api/schemas/screens.py` (append at end)
- Modify: `api/routes/screens.py` (append after `download_report`, ~line 247)
- Create: `tests/api/test_reports_manage_route.py`

- [ ] **Step 1: Add schemas**

```python
class PreviewResponse(BaseModel):
    name: str
    lines: list[str] = []
    truncated: bool = False
    total_lines: int = 0


class DeleteExportRequest(BaseModel):
    name: str
    confirm: bool = False


class DeleteExportResponse(BaseModel):
    deleted: bool
    name: Optional[str] = None
    error: Optional[str] = None
```
`Optional` and `BaseModel` already imported in that file.

- [ ] **Step 2: Write the failing tests**

```python
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
```
Note: `_export_dir` reads `config.paths.export_dir` (screens.py:169-172), so the Mock config above routes it to tmp_path. `mock_read_context` comes from `tests/api/conftest.py`, no import needed.

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/api/test_reports_manage_route.py -q`
Expected: FAIL (no such endpoints).

- [ ] **Step 4: Implement**

Append after `download_report` in `api/routes/screens.py`; extend the schema import with the three new models:
```python
@router.get("/reports/preview", response_model=PreviewResponse)
def preview_report(name: str, ctx=Depends(get_read_context)) -> PreviewResponse:
    """First 50 lines of a generated export for on-screen preview."""
    if name != Path(name).name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    target = _export_dir(ctx) / name
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No such export: {name}")
    try:
        text = target.read_text(errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read export: {exc}")
    if len(text) > 200_000:
        text = text[:200_000]
    lines = text.splitlines()
    return PreviewResponse(
        name=name, lines=lines[:50],
        truncated=len(lines) > 50, total_lines=len(lines))


@router.post("/reports/delete", response_model=DeleteExportResponse)
def delete_report(
    payload: DeleteExportRequest, ctx=Depends(get_read_context)
) -> DeleteExportResponse:
    """Delete a generated export. Confirmation required, same as snapshots."""
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Deletion requires explicit confirmation.")
    if payload.name != Path(payload.name).name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    target = _export_dir(ctx) / payload.name
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No such export: {payload.name}")
    try:
        target.unlink()
    except OSError as exc:
        return DeleteExportResponse(deleted=False, name=payload.name, error=str(exc))
    return DeleteExportResponse(deleted=True, name=payload.name)
```

- [ ] **Step 5: Verify**

Run: `uv run pytest tests/api/test_reports_manage_route.py -q` — Expected: 6 passed.
Run: `uv run ruff check api tests/api run_ui.py` — Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add api/schemas/screens.py api/routes/screens.py tests/api/test_reports_manage_route.py
git commit -m "feat: preview and delete generated exports"
```

---

### Task 2: Backup download + delete

**Files:**
- Modify: `api/schemas/screens.py`, `api/routes/screens.py` (after `restore_backup`)
- Create: `tests/api/test_backup_transfer_route.py`

- [ ] **Step 1: Add schemas**

```python
class BackupDeleteRequest(BaseModel):
    name: str
    confirm: bool = False


class BackupDeleteResponse(BaseModel):
    deleted: bool
    name: Optional[str] = None
    error: Optional[str] = None
```

- [ ] **Step 2: Write the failing tests**

```python
"""Download and delete for database backups. No src/ changes: both operate
only on paths the core's own list_backups() reports, the same trust model
as the existing restore endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def one_backup(mock_read_context, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backup_dir = tmp_path / "db_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    payload = b"fake-sqlite-bytes"
    (backup_dir / "portfolio.db.20260904_120000.bak").write_bytes(payload)
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.db_manager.list_backups.return_value = [
        backup_dir / "portfolio.db.20260904_120000.bak"
    ]
    return backup_dir, payload
```
IMPORTANT — verify first: read `DatabaseManager.list_backups` (`src/crypto_portfolio_tracker/database.py:858`) and the restore route to confirm (a) backups resolve under the process cwd (`db_backups/`), and (b) `ctx.db_manager` in tests is the Mock from `mock_read_context` (conftest) so `list_backups.return_value` works. If `list_backups` derives its dir from config rather than cwd, seed accordingly (set `mock_read_context.config_manager.config` to point at tmp). Then:
```python
def test_download_backup(one_backup):
    backup_dir, payload = one_backup
    response = TestClient(app).get("/api/system/backup/download",
                                   params={"name": "portfolio.db.20260904_120000.bak"})
    assert response.status_code == 200
    assert response.content == payload


def test_download_unknown_backup_404(one_backup):
    response = TestClient(app).get("/api/system/backup/download",
                                   params={"name": "nope.bak"})
    assert response.status_code == 404


def test_delete_backup_needs_confirm(one_backup):
    backup_dir, _payload = one_backup
    response = TestClient(app).post("/api/system/backup/delete",
                                    json={"name": "portfolio.db.20260904_120000.bak"})
    assert response.status_code == 400
    assert (backup_dir / "portfolio.db.20260904_120000.bak").exists()


def test_delete_backup_removes_listed_file(one_backup):
    backup_dir, _payload = one_backup
    body = TestClient(app).post("/api/system/backup/delete",
                                json={"name": "portfolio.db.20260904_120000.bak",
                                      "confirm": True}).json()
    assert body["deleted"] is True
    assert not (backup_dir / "portfolio.db.20260904_120000.bak").exists()
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/api/test_backup_transfer_route.py -q`
Expected: FAIL.

- [ ] **Step 4: Implement** (after `restore_backup`, mirroring its listed-name match):

```python
def _match_backup(ctx, name: str) -> Path:
    """A listed backup path by plain filename, else 404. Same trust model
    as restore: only files the core already reports can be touched."""
    if name != Path(name).name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    backups = ctx.db_manager.list_backups() or []
    match = next((p for p in backups if Path(p).name == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No such backup: {name}")
    return Path(match)


@router.get("/system/backup/download")
def download_backup(name: str, ctx=Depends(get_read_context)) -> FileResponse:
    """Serve a listed database backup for download."""
    return FileResponse(_match_backup(ctx, name),
                        filename=_match_backup(ctx, name).name)


@router.post("/system/backup/delete", response_model=BackupDeleteResponse)
def delete_backup(
    payload: BackupDeleteRequest, ctx=Depends(get_read_context)
) -> BackupDeleteResponse:
    """Delete a listed database backup. Confirmation required."""
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Deletion requires explicit confirmation.")
    target = _match_backup(ctx, payload.name)
    try:
        target.unlink()
    except OSError as exc:
        return BackupDeleteResponse(deleted=False, name=payload.name, error=str(exc))
    return BackupDeleteResponse(deleted=True, name=payload.name)
```
Refactor option (preferred if clean): rewrite `restore_backup`'s inline match to call `_match_backup`. If the diff gets noisy, leave restore untouched.

- [ ] **Step 5: Verify**

Run: `uv run pytest tests/api/test_backup_transfer_route.py -q` — Expected: 4 passed.
Run: `uv run pytest -q` + `uv run ruff check api tests/api run_ui.py`.

- [ ] **Step 6: Commit**

```bash
git add api/schemas/screens.py api/routes/screens.py tests/api/test_backup_transfer_route.py
git commit -m "feat: download and delete database backups"
```

---

### Task 3: Summary, trend, and realized exports

**Files:**
- Modify: `api/schemas/screens.py`, `api/routes/screens.py`
- Create: `tests/api/test_summary_export_route.py`, `tests/api/test_trend_export_route.py`, `tests/api/test_realized_export_route.py`

Shared helper (add near `_export_dir`):
```python
def _new_files_since(export_dir: Path, before: set[str]) -> list[Path]:
    """Files the export just wrote. The core exporters timestamp their own
    filenames, so the newcomer is found by diffing, not by guessing names."""
    return sorted(
        (p for p in export_dir.iterdir()
         if p.is_file() and p.name not in before),
        key=lambda p: p.stat().st_mtime, reverse=True)
```

- [ ] **Step 1: Add schemas**

```python
class SummaryExportRequest(BaseModel):
    format: str  # csv | excel | html


class TrendExportRequest(BaseModel):
    timeframe: str  # long_term | swing | day
    format: str  # csv | json | html


class RealizedExportRequest(BaseModel):
    format: str  # csv | excel
```
All three return the existing `GenerateExportResponse` (name + path), so the Reports screen's download link works unchanged.

- [ ] **Step 2: Tests — summary**

```python
"""POST /reports/summary renders the portfolio summary via the core
exporters, served from the export dir like every other generated file."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

HOLDINGS = [
    {"symbol": "BTC", "value_usd": 146.49, "current_price": 95000.0,
     "average_cost_basis": 80000.0},
]


@pytest.fixture
def summary_setup(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {
        "paths": {"export_dir": str(tmp_path)},
        "target_allocation": {"BTC": 1.0},
    }
    monkeypatch.chdir(tmp_path)
    metrics = Path("data") / "api_cache" / "metrics_testnet.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps({"holdings_df": HOLDINGS, "_cached_at": 0}))


@pytest.mark.parametrize("fmt", ["csv", "excel", "html"])
def test_summary_export_each_format(summary_setup, fmt):
    body = TestClient(app).post("/api/reports/summary", json={"format": fmt}).json()
    assert body["name"]
    assert fmt in body["name"] or body["name"].endswith(
        {"csv": ".csv", "excel": ".xlsx", "html": ".html"}[fmt])


def test_summary_export_bad_format(summary_setup):
    assert TestClient(app).post(
        "/api/reports/summary", json={"format": "pdf"}).status_code == 422


def test_summary_export_without_metrics(summary_setup, mock_read_context, tmp_path):
    (tmp_path / "data" / "api_cache" / "metrics_testnet.json").unlink()
    response = TestClient(app).post("/api/reports/summary", json={"format": "csv"})
    assert response.status_code == 422
```
If any format's exporter is unavailable in this environment (ValueError from the core), narrow that parametrize case to the formats the suite can produce — but first verify it really is environmental (missing dep), not a code bug.

- [ ] **Step 3: Tests — trend**

```python
"""POST /reports/trend renders a cached technical report via the core."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

TECHNICAL = {
    "reports": {
        "swing": {"coin_analyses": {
            "BTC": {"symbol": "BTC", "current_price": 66450.0,
                    "price_change_pct": 1.5, "rsi": 61.4,
                    "support_level": 57747.0, "resistance_level": 66890.0,
                    "active_conditions": ["Golden Cross"]},
        }},
    },
    "_cached_at": 0,
}


@pytest.fixture
def trend_setup(mock_read_context, tmp_path, monkeypatch):
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {"paths": {"export_dir": str(tmp_path)}}
    monkeypatch.chdir(tmp_path)
    path = Path("data") / "api_cache" / "technical_testnet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(TECHNICAL))


@pytest.mark.parametrize("fmt,ext", [("csv", ".csv"), ("json", ".json")])
def test_trend_export_csv_json(trend_setup, fmt, ext):
    body = TestClient(app).post("/api/reports/trend",
                                json={"timeframe": "swing", "format": fmt}).json()
    assert body["name"].endswith(ext)


def test_trend_export_unknown_timeframe(trend_setup):
    assert TestClient(app).post("/api/reports/trend",
                                json={"timeframe": "decade",
                                      "format": "csv"}).status_code == 422
```
HTML is intentionally untested here (needs the core HTML exporter present); the endpoint still accepts it and surfaces a core failure as a 500 with the message, like `generate_export` does.

- [ ] **Step 4: Tests — realized**

Read `tests/api/test_realized_route.py` first for the transaction-seeding pattern, then mirror `realized()` + write the frame. Test: excel produces a `.xlsx` (assert name + file exists in export dir), csv produces `.csv`, empty transactions → 422.

- [ ] **Step 5: Run to verify failure**

Run the three files — Expected: FAIL (no endpoints).

- [ ] **Step 6: Implement** (append near `generate_export`):

```python
def _report_generator(ctx, export_dir: Path):
    """Core exporters pointed at the API export dir. Local import: constructing
    this never touches the network, but keep the import next to the other
    core import below rather than at module top."""
    from crypto_portfolio_tracker.report_generator import ReportGenerator
    config = dict(ctx.config_manager.config or {})
    config["exports"] = {"path": str(export_dir)}
    return ReportGenerator(config=config, db_manager=ctx.db_manager)
```
If `ReportGenerator.__init__` demands more than `config` + `db_manager`, STOP and report BLOCKED (do not invent arguments).

```python
@router.post("/reports/summary", response_model=GenerateExportResponse)
def export_summary(
    payload: SummaryExportRequest, ctx=Depends(get_read_context)
) -> GenerateExportResponse:
    """Portfolio summary via the core exporters (same output as the CLI)."""
    fmt = payload.format.strip().lower()
    if fmt not in ("csv", "excel", "html"):
        raise HTTPException(status_code=422, detail="format must be csv, excel or html.")
    metrics = MetricsCache(cache_path_for(ctx.config_manager)).read()
    if not metrics:
        raise HTTPException(status_code=422, detail="No synced metrics yet -- run a sync first.")
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in export_dir.iterdir() if p.is_file()}
    full = dict(metrics)
    full["holdings_df"] = pd.DataFrame(metrics.get("holdings_df") or [])
    try:
        _report_generator(ctx, export_dir).export_portfolio_summary(full, fmt)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write export: {exc}")
    fresh = _new_files_since(export_dir, before)
    if not fresh:
        raise HTTPException(status_code=500, detail="Exporter reported success but wrote no file.")
    return GenerateExportResponse(name=fresh[0].name, path=str(fresh[0]))


@router.post("/reports/trend", response_model=GenerateExportResponse)
def export_trend(
    payload: TrendExportRequest, ctx=Depends(get_read_context)
) -> GenerateExportResponse:
    """A cached technical report via the core trend exporter."""
    fmt = payload.format.strip().lower()
    if fmt not in ("csv", "json", "html"):
        raise HTTPException(status_code=422, detail="format must be csv, json or html.")
    cache = MetricsCache(analysis_cache_path(ctx.config_manager, "technical"))
    reports = (cache.read() or {}).get("reports") or {}
    report = reports.get(payload.timeframe)
    if not isinstance(report, dict) or "coin_analyses" not in report:
        raise HTTPException(
            status_code=422,
            detail=f"No {payload.timeframe} technical report cached -- run the analysis first.")
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in export_dir.iterdir() if p.is_file()}
    try:
        _report_generator(ctx, export_dir).export_trend_report(report, payload.timeframe, fmt.upper())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write export: {exc}")
    fresh = _new_files_since(export_dir, before)
    if not fresh:
        raise HTTPException(status_code=500, detail="Exporter reported success but wrote no file.")
    return GenerateExportResponse(name=fresh[0].name, path=str(fresh[0]))


@router.post("/reports/realized", response_model=GenerateExportResponse)
def export_realized(
    payload: RealizedExportRequest, ctx=Depends(get_read_context)
) -> GenerateExportResponse:
    """Realized FIFO gains as a file. Same tax_df the /realized screen shows,
    written with the same tz-naive guard as generate_export."""
    fmt = payload.format.strip().lower()
    if fmt not in ("csv", "excel"):
        raise HTTPException(status_code=422, detail="format must be csv or excel.")
    transactions = ctx.db_manager.get_all_transactions()
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise HTTPException(status_code=422, detail="No transactions to export.")
    tax_df = calculate_fifo_realized_gains(transactions)
    if tax_df.empty:
        raise HTTPException(status_code=422, detail="No realized gains to export yet.")
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"realized_{stamp}.{'csv' if fmt == 'csv' else 'xlsx'}"
    path = export_dir / name
    try:
        if fmt == "csv":
            tax_df.to_csv(path, index=False)
        else:
            safe = tax_df.copy()
            for col in safe.columns:
                if isinstance(safe[col].dtype, pd.DatetimeTZDtype):
                    safe[col] = safe[col].dt.tz_localize(None)
            safe.to_excel(path, index=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write export: {exc}")
    return GenerateExportResponse(name=name, path=str(path))
```
`analysis_cache_path` needs adding to the `api.cache` import (line 15); `calculate_fifo_realized_gains` and `datetime` already imported (lines 6, 52).

- [ ] **Step 7: Verify**

Run the three test files, then `uv run pytest -q` (expect prior baseline +13), then `uv run ruff check api tests/api run_ui.py` (clean).

- [ ] **Step 8: Commit**

```bash
git add api/schemas/screens.py api/routes/screens.py tests/api/test_summary_export_route.py tests/api/test_trend_export_route.py tests/api/test_realized_export_route.py
git commit -m "feat: summary, trend and realized exports"
```

---

### Task 4: Frontend wiring for all Phase A endpoints

**Files:**
- Modify: `frontend/src/screens/Reports.tsx` (summary + trend sections, per-file preview + delete)
- Modify: `frontend/src/screens/SystemHealth.tsx` (backup download + delete — first confirm this is where restore lives; if restore lives elsewhere, wire there instead)
- Modify: `frontend/src/screens/TradeLog.tsx` (Excel button via existing generate endpoint)
- Modify: `frontend/src/screens/Realized.tsx` (Excel + CSV buttons via `/reports/realized`)
- Modify: `frontend/src/screens/DataManage.tsx` (snapshots CSV client-side download)
- Create: `frontend/src/screens/Reports.test.tsx`, `frontend/src/screens/TradeLog.test.tsx`, `frontend/src/screens/Realized.test.tsx`

Rules: read each screen fully before editing; match its idioms. New destructive buttons need the repo's existing delete-confirm idiom — mirror how `DataManage.tsx` confirms snapshot deletion (read it first; do not invent a new confirm pattern). Preview renders `<pre>`-style mono text of the returned lines with a truncated note. After any generate/delete, reload the file list the way the screen already reloads it.

- [ ] **Step 1: Reports screen**

Add a "Portfolio summary" section: format picker (CSV/Excel/HTML, same button-group idiom as the existing csv/excel picker at Reports.tsx:19-20) + Generate button → `apiPost('/api/reports/summary', {format})` → reload list (download via the existing per-file link).
Add a "Trend report" section: timeframe picker (long_term/swing/day) + format picker (CSV/JSON/HTML) → `apiPost('/api/reports/trend', {timeframe, format})` → reload list.
Per-file row: Preview button → `apiGet('/api/reports/preview?name=...')` → inline mono block (first lines + "…N more lines" when truncated); Delete button → confirm idiom → `apiPost('/api/reports/delete', {name, confirm: true})` → reload list.

- [ ] **Step 2: System screen backups**

Next to the existing restore control: Download link (`/api/system/backup/download?name=...`, same anchor idiom as Reports.tsx:122) + Delete button (confirm idiom → `apiPost('/api/system/backup/delete', {name, confirm: true})` → reload).

- [ ] **Step 3: TradeLog Excel**

Next to the CSV button (TradeLog.tsx:111-113): Excel button → `apiPost('/api/reports/generate', {data_type: 'transactions', format: 'excel'})` → offer the returned file via the download URL pattern (`/api/reports/download?name=`). No backend change: the endpoint already does this.

- [ ] **Step 4: Realized Excel + CSV**

Buttons → `apiPost('/api/reports/realized', {format})` → download link from the returned name. Same download-URL pattern.

- [ ] **Step 5: DataManage snapshots CSV**

Local `downloadRows(filename, headers, rows)` helper in `DataManage.tsx` mirroring `TradeLog.tsx`'s `downloadCsv` (read it first; same Blob + anchor-click shape, with `""`-quote escaping for commas/quotes/newlines). Button next to the snapshots table exports the currently listed rows (timestamp, total_value_usd, total_cost_basis_usd, unrealized_pl_usd, unrealized_pl_percent; nulls as empty cells).

- [ ] **Step 6: Tests**

`Reports.test.tsx`: fetch-fail renders error not loading; summary generate posts `{format:'excel'}` (assert via fetch-spy URL + body); preview click renders returned lines; delete posts `{name, confirm:true}` then reloads.
`TradeLog.test.tsx`: fetch-fail; Excel button posts `{data_type:'transactions', format:'excel'}`.
`Realized.test.tsx`: fetch-fail; Excel button posts `{format:'excel'}`.
Route fetch by URL substring like `Dca.test.tsx` does. If a screen's fetch shape differs, read its `useApi` calls first and adapt the stub — never weaken assertions to pass.

- [ ] **Step 7: Verify**

`npx vitest run` (full), `npx tsc -b` (clean), `npx oxlint` (no new warnings on touched files).

- [ ] **Step 8: Commit**

```bash
git add frontend/src/screens/Reports.tsx frontend/src/screens/SystemHealth.tsx frontend/src/screens/TradeLog.tsx frontend/src/screens/Realized.tsx frontend/src/screens/DataManage.tsx frontend/src/screens/Reports.test.tsx frontend/src/screens/TradeLog.test.tsx frontend/src/screens/Realized.test.tsx
git commit -m "feat: wire Phase A exports through the React UI"
```
(If restore lives in a different screen than SystemHealth, swap the filename in the commit.)

---

## Out of scope (later phases)

- Per-trade execution selection, backtest custom params, charts, snapshot-save button, wallets allocation-%, system resources, connection test, all Settings-depth items.
- CONTEXT.md changes (additive UI following existing rules).
