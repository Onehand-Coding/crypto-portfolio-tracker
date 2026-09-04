# Parity Phase C: Settings Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 6 settings-depth gaps: trade schedules, API timeouts/delays/lookbacks, logging config + preview, per-timeframe analyzer windows, redacted config export/import.

**Architecture:** Extend the existing `SettingsResponse`/`SettingsUpdate` + PUT-validation pattern (validate → setdefault → `_save_config_preserving_secrets` → return `_read_settings`). Config export/import are file actions in the export dir with secrets always preserved. Log preview reads only the configured log path (no user-supplied paths). No `src/` changes.

**Tech Stack:** FastAPI + Pydantic v2, pytest + TestClient, React + Vitest.

**Files:**
- Modify: `api/schemas/screens.py`, `api/routes/screens.py`
- Create: `tests/api/test_settings_extended_route.py`, `tests/api/test_config_transfer_route.py`
- Modify: `frontend/src/screens/Settings.tsx`
- Create: `frontend/src/screens/Settings.test.tsx`

**Pre-reads:** `settings_page.py:542-556` (exact per-timeframe defaults), the lookback widget's `min_value` (~line 348), `POST /system/import/{data_type}` in screens.py (UploadFile handling pattern to mirror), `Settings.tsx` fully (form + `apiPut` + panel idioms).

---

### Task 1: Backend — extended settings, config transfer, log preview

**Files:**
- Modify: `api/schemas/screens.py` (`SettingsResponse`, `SettingsUpdate`, new models)
- Modify: `api/routes/screens.py` (`_read_settings`, `update_settings`, 3 new endpoints)
- Create: `tests/api/test_settings_extended_route.py`, `tests/api/test_config_transfer_route.py`

- [ ] **Step 1: Add schemas**

```python
FREQUENCIES = ("daily", "weekly", "biweekly", "monthly", "quarterly")
# VERIFY this tuple against settings_page.py freq_options first — mirror it exactly.

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

LOOKBACK_KEYS = (
    "trades", "deposits", "withdrawals", "p2p_buys", "internal_transfers",
    "spot_futures_transfers", "spot_convert_history", "simple_earn_rewards",
    "simple_earn_subscriptions", "simple_earn_redemptions",
    "dividend_history", "staking_history",
)


class AutomationSettings(BaseModel):
    dca_frequency: str = "monthly"
    rebalancing_frequency: str = "weekly"


class ApiSettings(BaseModel):
    coingecko_timeout: float = 30
    binance_timeout: float = 60
    binance_recv_window: int = 20000
    binance_delay_ms: float = 500
    coingecko_delay_ms: float = 1500


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file_enabled: bool = True
    file_path: str = "logs/portfolio_tracker.log"
    console_enabled: bool = True


class TimeframeWindows(BaseModel):
    sma_short_window: int
    sma_long_window: int


class TrendTimeframes(BaseModel):
    long_term: TimeframeWindows
    swing: TimeframeWindows
    day: TimeframeWindows


class LogPreviewResponse(BaseModel):
    path: str
    lines: list[str] = []
    truncated: bool = False
    total_lines: int = 0
```
Extend `SettingsResponse` with `automation: AutomationSettings`, `apis: ApiSettings`, `history_lookback_days: dict[str, int]`, `logging: LoggingSettings`, `trend_timeframes: TrendTimeframes`; extend `SettingsUpdate` with all-Optional counterparts (`history_lookback_days: Optional[dict[str, int]]`).

- [ ] **Step 2: Failing tests — settings groups** (`tests/api/test_settings_extended_route.py`)

Seed via `mock_read_context.config_manager.config = copy of default_config.json` (read the file from repo root in-test: `json.loads(Path("config/default_config.json").read_text())` — tests run with repo cwd? Other tests chdir to tmp... config is in-memory here so cwd doesn't matter; use absolute path via `Path(__file__).parents[2] / "config" / "default_config.json"`).
Tests: GET returns the new groups with config values; PUT automation frequencies round-trips, bad frequency → 422; PUT apis negative timeout → 422; PUT lookback with unknown key → 422, negative → 422 (mirror the widget min); PUT logging bad level → 422; PUT trend windows sma_short >= sma_long → 422 (short must stay below long — Streamlit allows anything? If Streamlit doesn't validate this, do NOT invent the rule; instead require both ≥ 1. CHECK the Streamlit save path first and mirror it exactly); PUT returns the updated groups (persistence proven by re-GET in the same test? config Mock persists per-test — assert response body).

- [ ] **Step 3: Implement `_read_settings` + PUT branches**

`_read_settings`: read `automation.dca/rebalancing.frequency` (defaults monthly/weekly), `apis.coingecko.timeout` (30), `apis.binance.timeout` (60)/`recv_window` (20000)/`request_delay_ms` both (500/1500), `history_lookback_days` filtered to LOOKBACK_KEYS with default 90, `logging.level` (INFO)/`file_config.enabled` (True)/`file_config.path` (logs/portfolio_tracker.log)/`console_config.enabled` (True), `trend_analyzer.timeframe_settings.{long_term,swing,day}.{sma_short_window,sma_long_window}` (defaults from settings_page.py:542-556 — read exact numbers, do not guess).
PUT: frequencies must be in FREQUENCIES else 422; timeouts/recv/delays numeric and > 0 (delays ≥ 0 — VERIFY Streamlit min first; if Streamlit allows 0, allow 0) else 422; lookbacks: reject unknown keys, require ints ≥ widget-min else 422; logging level in LOG_LEVELS, file_path non-empty string (strip; reject empty) else 422; windows ints ≥ 1 (plus short<long ONLY if Streamlit enforces it). Persist via existing `_save_config_preserving_secrets`, return `_read_settings`.

- [ ] **Step 4: Failing tests — config transfer + log preview** (`tests/api/test_config_transfer_route.py`)

```python
def test_export_redacts_secrets(...):
    config with main_api_keys={"k": "SECRET"} and apis.coingecko.api_key="CG"
    GET /api/system/config/export → 200 JSON file in export dir;
    parsed content has main_api_keys values "********" and NO coingecko api_key... or masked?
```
DECISION (locked): export MASKS main_api_keys values with "********" (CLI style) and DROPS `apis.coingecko.api_key` (Streamlit style) — assert exactly that. Import: file with masked/dropped secrets + changed `minimum_trade_usd` → POST multipart → settings updated, live secrets unchanged (assert `cm.config["main_api_keys"]` still has real values via `_save_config_preserving_secrets` mechanics — assert through `mock_read_context.config_manager` state), backup file `config_backup_*.json` written to export dir containing pre-import config. Non-JSON upload → 422. Non-object JSON (list) → 422.
Log preview: seed configured log path with 100 lines → GET with `lines=50` → 50 lines + truncated + total 100; missing file → 404; `lines=0`/`lines=9999` clamped to 1..500.

- [ ] **Step 5: Implement transfer + preview**

```python
@router.get("/system/config/export")
def export_config(ctx=Depends(get_read_context)):
    """Redacted config JSON into the export dir (masked keys, dropped CG key)."""
    config = copy.deepcopy(ctx.config_manager.config or {})
    if isinstance(config.get("main_api_keys"), dict):
        config["main_api_keys"] = {k: "********" for k in config["main_api_keys"]}
    try:
        del config["apis"]["coingecko"]["api_key"]
    except (KeyError, TypeError):
        pass
    export_dir = _export_dir(ctx); export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = export_dir / f"config_export_{stamp}.json"
    path.write_text(json.dumps(config, indent=2))
    return FileResponse(path, filename=path.name)


@router.post("/system/config/import", response_model=SettingsResponse)
async def import_config(file: UploadFile = File(...), ctx=Depends(get_read_context)) -> SettingsResponse:
    """Replace config from an uploaded JSON file. Secrets are never imported:
    masked/dropped secret fields are discarded and live secrets preserved."""
    raw = await file.read()  # mirror /system/import handling; cap at 1MB → 422 if larger
    try:
        incoming = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="Not a valid JSON file.")
    if not isinstance(incoming, dict):
        raise HTTPException(status_code=422, detail="Config must be a JSON object.")
    incoming.pop("main_api_keys", None)
    if isinstance(incoming.get("apis"), dict) and isinstance(incoming["apis"].get("coingecko"), dict):
        incoming["apis"]["coingecko"].pop("api_key", None)
    # Reversible: back up the live config first (mirrors restore's safety backup).
    export_dir = _export_dir(ctx); export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (export_dir / f"config_backup_{stamp}.json").write_text(json.dumps(ctx.config_manager.config or {}, indent=2))
    for key, value in incoming.items():
        ctx.config_manager.config[key] = value
    _save_config_preserving_secrets(ctx.config_manager)
    return _read_settings(ctx.config_manager.config)


@router.get("/system/logs/preview", response_model=LogPreviewResponse)
def preview_logs(lines: int = 50, ctx=Depends(get_read_context)) -> LogPreviewResponse:
    """Last N lines of the configured log file. The path comes from config,
    never from the caller, so this cannot read arbitrary files."""
    count = min(500, max(1, lines))
    log_path = ((ctx.config_manager.config or {}).get("logging", {}) or {}).get("file_config", {}) or {}
    path = Path(str(log_path.get("path", "logs/portfolio_tracker.log")))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No log file at {path}.")
    text = path.read_text(errors="replace")
    if len(text) > 200_000:
        text = text[-200_000:]
    all_lines = text.splitlines()
    return LogPreviewResponse(path=str(path), lines=all_lines[-count:],
                              truncated=len(all_lines) > count, total_lines=len(all_lines))
```
`copy`, `json` imports: check screens.py (json? copy?) — add if missing. `UploadFile`/`File` already imported (line 12). 1MB cap: `if len(raw) > 1_000_000: raise 422`.

- [ ] **Step 6: Verify + commit**

New tests pass, `uv run pytest -q` full, `uv run ruff check api tests/api run_ui.py` clean. Commit `feat: extended settings, config transfer, log preview` (schemas+routes+2 test files).

---

### Task 2: Frontend Settings sections

**Files:**
- Modify: `frontend/src/screens/Settings.tsx`
- Create: `frontend/src/screens/Settings.test.tsx`

Read Settings.tsx fully first (form state shape, `apiPut`, panel idioms, save flow at lines ~80-148).

- [ ] **Step 1: Extend the form + types**

Mirror the backend models in `frontend/src/types.ts` ONLY if SettingsResponse is mirrored there field-for-field (check; if Settings.tsx uses a local form type, extend that instead — do not create a parallel contract).

- [ ] **Step 2: New panels (match existing panel/input idioms)**

Schedules (two pickers from FREQUENCIES), API (five numerics), Lookbacks (compact grid of twelve numerics), Logging (level picker + two toggles + path input + preview-lines input + Preview button rendering `<pre>` lines + truncated note), Trend timeframes (three groups × two numerics), Config transfer (Export download anchor to GET /system/config/export + file input + Import button posting multipart `file` + result message + note that secrets are preserved).
Multipart post: check `lib/api.ts` for a file-post helper; if none, use `fetch` directly with FormData (like apiPost's error handling — read apiPost first and mirror its ok-check).

- [ ] **Step 3: Tests**

`Settings.test.tsx`: fetch-fail renders error not loading; schedules PUT carries frequencies; bad level rejected client-side or posts and surfaces 422 text (assert whichever the implementation does — read the save flow first); import posts FormData with the file. Route fetch by URL.

- [ ] **Step 4: Verify + commit**

`npx vitest run` full, `npx tsc -b` clean, `npx oxlint` no new warnings. Commit `feat: settings depth on React screen`.

---

## Out of scope

- Per-coin indicator charts (deferred live-fetch job).
- CONTEXT.md changes (additive UI following existing rules).
