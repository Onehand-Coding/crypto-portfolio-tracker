# Parity Phase B: Execution Gates, Backtest Params, Charts & System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 8 Phase B gaps: per-trade execution selection, backtest custom params, chart PNG export, per-coin indicator charts, manual snapshot save, wallets allocation-%, system resources, connection test.

**Architecture:** Per-trade gates are frontend-only (the backend already takes `symbols[]` / `trades[]`). Backtest params extend `_backtest_config` plus a merged-config build mirroring Streamlit's `create_custom_config`. New offline reads follow existing cache patterns; the connection test is a POST action through `get_tracker()` like the execute routes (never a GET read). No `src/` changes.

**Tech Stack:** FastAPI + Pydantic v2, pytest + TestClient, React + recharts (already used by Backtest.tsx) + Vitest.

---

### Task 1: Per-trade selection on Rebalance, DCA, Profit (frontend only)

**Files:**
- Modify: `frontend/src/screens/Rebalance.tsx`, `frontend/src/screens/Dca.tsx`, `frontend/src/screens/ProfitTaking.tsx`
- Modify: `frontend/src/screens/Dca.test.tsx` (extend), create-or-extend tests for Rebalance/Profit only if those files exist (check first; if not, add the fail+select test into the same new-file pattern — do not create more than needed)

Backend contract (already live, verified): `POST /api/execute/rebalance {confirm, symbols?}` and `POST /api/execute/profit {confirm, symbols?}` (`ExecuteSelectionRequest`: `symbols: Optional[list[str]]`, null/empty = all); `POST /api/execute/dca {confirm, strategy, trades: [{asset, amount}]}`.

- [ ] **Step 1: Read the three screens' suggestion render + ExecutePanel call sites**

Record: the array each screen maps over (rebalance suggestions with `symbol`/`action`; DCA `preview.allocations` with `symbol`/`amount_usd`; profit `opportunities` with `symbol`), and the exact `apiPost` execute call. `ExecutePanel` already accepts `children` rendered above the confirm gate — checkboxes go there.

- [ ] **Step 2: Failing tests first**

Extend `Dca.test.tsx` + Rebalance/Profit test files: stub a preview with 2+ allocations; assert all checked by default; uncheck one; assert the execute POST body contains only the checked trade(s). For rebalance/profit assert `symbols: [...]` contains only checked symbols. Follow the existing URL-routed fetch-stub + `postBody` patterns.

- [ ] **Step 3: Implement**

Pattern per screen (adapt names to what Step 1 found):
```tsx
const [selected, setSelected] = useState<Record<string, boolean>>({});
const actionable = preview.allocations.filter((a) => a.amount_usd > 0);
// default-checked: treat missing key as checked
const isChecked = (s: string) => selected[s] ?? true;
```
Render a checkbox per actionable row (label `Include ${symbol}`), plus the `ExecutePanel` gets:
```tsx
<ExecutePanel ... execute={() => apiPost('/api/execute/dca', {
  confirm: true, strategy,
  trades: actionable.filter((a) => isChecked(a.symbol))
    .map((a) => ({ asset: a.symbol, amount: a.amount_usd })),
})}>
```
For rebalance/profit: `apiPost('/api/execute/rebalance', { confirm: true, symbols: actionable.filter(isChecked).map(s => s.symbol) })` — when all checked, send the full list explicitly (equivalent to omitted, unambiguous in logs).
Guard: if none checked, disable the panel (`disabled` prop exists) with hint text "Select at least one trade."

- [ ] **Step 4: Verify**

`npx vitest run`, `npx tsc -b` clean, `npx oxlint` no new warnings.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/screens/Rebalance.tsx frontend/src/screens/Dca.tsx frontend/src/screens/ProfitTaking.tsx <test files touched>
git commit -m "feat: per-trade selection on execute panels"
```

---

### Task 2: Backtest custom params

**Files:**
- Modify: `api/analysis_runner.py` (`_backtest_config`, `_backtest`)
- Modify: `frontend/src/screens/Backtest.tsx`, `frontend/src/types.ts` (extend backtest config type only if it names fields — check first)
- Create: `tests/api/test_backtest_custom.py` (check existing `test_backtest_config.py` first; extend it instead if it covers `_backtest_config`)

Streamlit parity reference: `backtest_page.py:create_custom_config` + advanced expander defaults — majors drift (1–20, from config), alts drift (1–20), sell/buy multipliers (0.1–2.0; defaults 0.5/0.75/0.5/1.0), suppress-buys-in-bear checkbox (default True), custom allocation (must sum to 1.0 ± 0.001 or ignored), period presets 1y/2y/3y/4y/5y + Custom freeform like "6y".

- [ ] **Step 1: Failing tests**

```python
"""Custom backtest params ride through to an isolated merged config."""
# _backtest_config unit tests (no tracker needed):
# - custom period "6y"/"90d"?? NO — only ^\d+y$ plus the existing whitelist; "decade" and "" fall back to "2y".
# - numeric clamps: drift 0 → 1.0, 99 → 20.0; multiplier 0 → 0.1, 9 → 2.0; garbage → defaults.
# - allocation summing to 1.0 passes through; summing to 0.5 is dropped (None).
# - suppress flag coerces with bool().
```
Also verify current behavior first: run existing `tests/api/test_backtest_config.py` to learn its harness and avoid duplicating it.

- [ ] **Step 2: Implement backend**

```python
import re  # add if missing

_BACKTEST_CUSTOM_PERIOD = re.compile(r"^\d+y$")

def _clamp(value, lo, hi, default):
    """Float within [lo, hi]; garbage becomes the Streamlit default."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, result))


def _backtest_config(params) -> dict:
    params = params or {}
    ... existing capital/frequency unchanged ...
    period = params.get("period", "2y")
    if period not in _BACKTEST_PERIODS and not (
            isinstance(period, str) and _BACKTEST_CUSTOM_PERIOD.match(period)):
        period = "2y"
    custom = params.get("custom") or {}
    allocation = custom.get("allocation")
    if not (isinstance(allocation, dict) and allocation
            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in allocation.values())
            and abs(sum(allocation.values()) - 1.0) < 0.001):
        allocation = None
    return {
        "initial_capital": capital, "period": period, "frequency": frequency,
        "custom_allocation": allocation,
        "majors_drift": _clamp(custom.get("majors_drift"), 1.0, 20.0, 5.0),
        "alts_drift": _clamp(custom.get("alts_drift"), 1.0, 20.0, 8.0),
        "majors_sell": _clamp(custom.get("majors_sell"), 0.1, 2.0, 0.5),
        "majors_buy": _clamp(custom.get("majors_buy"), 0.1, 2.0, 0.75),
        "alts_sell": _clamp(custom.get("alts_sell"), 0.1, 2.0, 0.5),
        "alts_buy": _clamp(custom.get("alts_buy"), 0.1, 2.0, 1.0),
        "suppress_bear": bool(custom.get("suppress_bear", True)),
    }
```
Drift defaults (5.0/8.0): VERIFY against the repo default config before using — read `config/default_config.json` `rebalance_technical.majors/alts.allocation_drift_threshold_pct`; if different, use the config values as defaults (and note it in the commit). Do not guess.
Merged build in `_backtest` (mirrors `create_custom_config`):
```python
    base = tracker.config or {}
    merged = dict(base)
    merged["rebalance_technical"] = {
        "market_regime_rules": {"suppress_buys_in_bear": config["suppress_bear"]},
        "majors": {
            "allocation_drift_threshold_pct": config["majors_drift"],
            "sell_percentage_multiplier": config["majors_sell"],
            "buy_amount_multiplier": config["majors_buy"],
        },
        "alts": {
            "allocation_drift_threshold_pct": config["alts_drift"],
            "sell_percentage_multiplier": config["alts_sell"],
            "buy_amount_multiplier": config["alts_buy"],
        },
    }
    if config["custom_allocation"] is not None:
        merged["target_allocation"] = config["custom_allocation"]
    backtester = RebalancingBacktester(config=merged)
```
The cached `"config": config` already flows to the UI (`BacktestResponse.config`); extend the TS type only if it enumerates fields.

- [ ] **Step 3: Frontend advanced panel**

Collapsed-by-default "Advanced parameters" section (a `useState` toggle like the DCA completion panel, not a new route): numeric inputs for the six numbers (mirror Streamlit min/max as input constraints where trivial), suppress checkbox (default checked), custom period text input shown when period is "Custom" (add "Custom" to PERIODS; validate `^\d+y$` client-side, else disable Run with hint), allocation editor: fetch nothing new — read current targets from the backtest `data.config`? The GET returns the last run's config, not live targets. Instead: per-asset numeric inputs are unknowable without target keys... CHECK how `Allocation.tsx` fetches live targets (endpoint path) and reuse that same `useApi` call; render one numeric input per asset defaulting to its configured weight; send `custom.allocation` only when the numbers still sum to 1.0 ± 0.001 (else disable Run with "weights must sum to 100%"). POST body becomes `{initial_capital, period, frequency, custom: {...}}`.

- [ ] **Step 4: Verify**

`uv run pytest tests/api/test_backtest_custom.py -q` + full `uv run pytest -q` + `uv run ruff check api tests/api run_ui.py`; `npx vitest run`, `npx tsc -b`, `npx oxlint`.

- [ ] **Step 5: Commit**

```bash
git add api/analysis_runner.py tests/api/test_backtest_custom.py frontend/src/screens/Backtest.tsx frontend/src/types.ts
git commit -m "feat: custom backtest parameters"
```
(Adjust the file list to what actually changed; types.ts only if touched.)

---

### Task 3: Chart PNG export + per-coin indicator charts

**Files:**
- Modify: `api/routes/screens.py`, `api/schemas/screens.py`
- Create: `tests/api/test_chart_export_route.py`
- Modify: `frontend/src/screens/Reports.tsx`, `frontend/src/screens/Market.tsx` (or Technical.tsx — put the viewer where the indicator tables live; read first)
- Create/extend frontend tests for the touched screens

- [ ] **Step 1: PNG export endpoint (spike-then-build, 30 min cap)**

Read `Visualizer.create_portfolio_charts_all` + the 4 chart builders it calls (visualizations.py) and note exactly which metrics keys they consume. Then:
```python
@router.post("/reports/charts", response_model=GenerateExportResponse)
def export_charts(ctx=Depends(get_read_context)) -> GenerateExportResponse:
    """All portfolio charts as PNGs via the core visualizer (same files as CLI-10)."""
```
Implementation: metrics cache required (422 otherwise); `holdings_df` rehydrated to DataFrame; `Visualizer(config with exports.path pointed at the API export dir)`; call `create_portfolio_charts_all(full)`; PNGs land in `<export>/charts/` — MOVE them up to the export root (timestamped names cannot collide) and return... but one response holds one file. Return the FIRST (alphabetical) file or extend? DECISION (locked here to avoid a placeholder): generate all, move all up, return `GenerateExportResponse(name=<first sorted name>, path=...)` and include the rest how? `GenerateExportResponse` has only name/path. Simplest honest shape: reuse the Reports file LIST — after generate, the screen reloads `/api/reports` and shows all new PNGs with download links. The response confirms count via... it can't. Change: return the newest file; the screen reloads the list (already the pattern everywhere). Document this in the endpoint docstring. If `create_portfolio_charts_all` needs live-only inputs (check its body first), STOP and report BLOCKED with the exact missing key instead of faking metrics.
Tests: seed metrics cache with holdings_df records → 200, ≥1 `.png` appears in export dir; no metrics → 422.

- [ ] **Step 2: Indicator series endpoint (spike-then-build, 30 min cap)**

Read `market_page.py:plot_coin_chart` (what series it plots: price+SMA? RSI? MACD?) and the `_technical` adapter output (does the cached technical report contain per-coin HISTORY arrays or only current values?). If history is cached: `GET /api/strategy/indicators?symbol=&timeframe=` returning `{dates[], close[], sma_short[], sma_long[], rsi[], macd[]}` (nulls where unknown), 422 for unknown symbol/timeframe or missing cache. If history is NOT cached and the viewer fetches klines live: STOP and report BLOCKED with findings (that turns this into a live-fetch analysis job like backtest/run — bigger than Phase B; do NOT build live fetching inline in a GET).

- [ ] **Step 3: Frontend**

Reports screen: "Charts" section → Generate button → POST `/reports/charts` → reload list (PNGs appear with download links).
Market (or Technical) screen: coin picker (options = symbols present in the technical timeframes) + timeframe picker → series rendered with recharts (AreaChart/LineChart mirroring Backtest.tsx's equity curve, one panel per indicator group: Price+SMA, RSI, MACD) with em-dash/empty states when the endpoint 422s.

- [ ] **Step 4: Verify + commit**

Backend: new tests pass, full pytest, ruff clean. Frontend: vitest full, tsc, oxlint. Two commits (`feat: chart PNG export`, `feat: per-coin indicator charts`) — or one if the spike fails and only PNG lands (then the indicator half returns to the backlog explicitly in chat, not silently).

---

### Task 4: Snapshot save, wallets %, resources, connection test

**Files:**
- Modify: `api/routes/screens.py`, `api/schemas/screens.py`
- Create: `tests/api/test_system_actions_route.py`
- Modify: `frontend/src/screens/DataManage.tsx` (or SystemHealth — put Save where snapshots live), `frontend/src/screens/Wallets.tsx`, `frontend/src/screens/SystemHealth.tsx`
- Extend frontend tests for the touched screens

- [ ] **Step 1: Snapshot save endpoint**

`ctx.db_manager.save_portfolio_snapshot(timestamp, total_value, ...)` takes scalars — VERIFY its signature in database.py first (keyword names!). Source the figures from the metrics cache (keys like the CLI path: `total_value_usd`, `total_cost_basis_usd`, `unrealized_pl_usd`, `unrealized_pl_percent`, `timestamp` = now UTC if absent):
```python
@router.post("/system/snapshot/save", response_model=SnapshotSaveResponse)
def save_snapshot(ctx=Depends(get_read_context)) -> SnapshotSaveResponse:
    """Persist the cached metrics as a portfolio snapshot (same row CLI-1 writes)."""
```
Schemas: `SnapshotSaveResponse(saved: bool, timestamp: Optional[str], error: Optional[str])`. No metrics → 422 "run a sync first". Tests: seeded cache → row appears via GET /system/snapshots; no cache → 422. Frontend: Save button next to snapshots (DataManage or wherever the snapshots table lives — read first) → POST → reload.

- [ ] **Step 2: Wallets allocation-% (frontend only)**

Read `Wallets.tsx` + its route payload first. Add each wallet's share of total value (value / total * 100, one decimal) next to its balance — a small bar or `Share` column; null-safe (no total → em dash). No backend change. Test: stubbed payload with two wallets asserts both percentages.

- [ ] **Step 3: Resources endpoint + section**

`psutil` is a declared dependency (pyproject). Mirror Streamlit's status tab (app version from config `version`, `platform.python_version()`, `psutil.cpu_percent(interval=1)` — note the 1s block in a comment, `virtual_memory`, `disk_usage("/")`):
```python
@router.get("/system/resources", response_model=ResourcesResponse)
def system_resources(ctx=Depends(get_read_context)) -> ResourcesResponse:
```
Schemas: `ResourcesResponse(app_version: Optional[str], python_version: str, cpu_percent: Optional[float], ram_percent: Optional[float], ram_used_gb: Optional[float], disk_percent: Optional[float])`. Wrap psutil calls so any single failure nulls that field instead of 500ing (unknown-is-null rule). Tests: 200 + python_version present + numeric-or-null fields. Frontend: SystemHealth section rendering the six figures with em dashes for nulls.

- [ ] **Step 4: Connection test (POST action, networked)**

Mirror CLI `test_connections` exactly (`__main__.py:1305-1325`): `tracker.binance_client.ping()` (absent client → skipped, not failed) + `await tracker.enricher.get_current_prices(["BTC"])`:
```python
@router.post("/system/connections", response_model=ConnectionsResponse)
async def test_connections() -> ConnectionsResponse:
    """Live connectivity probe (Binance ping + CoinGecko BTC price). POST because
    it touches the network — same rule as sync and execute."""
```
Schemas: `ConnectionStatus(ok: bool, detail: Optional[str])`, `ConnectionsResponse(binance: ConnectionStatus, coingecko: ConnectionStatus, btc_price_usd: Optional[float])`. `get_tracker()` construction may raise without keys — catch into `{ok: False, detail: "No API keys..."}` (mirror the CLI SKIPPED branch). Tests: mock `api.routes.execute.get_tracker`?? — check how execute tests mock the tracker (read one) and mirror the pattern; test all three states (both ok, binance down, no keys). Frontend: SystemHealth "Run connection test" button + two badges + BTC price (em dash when null).

- [ ] **Step 5: Verify + commit**

Backend: new tests pass, full pytest, ruff clean. Frontend: vitest full, tsc, oxlint. Commit `feat: snapshot save, wallets share, resources, connection test` (split into two commits if the diff exceeds ~400 lines: backend / frontend).

---

## Out of scope (Phase C: settings depth)

- DCA/rebalance frequencies, API timeouts/recvWindow/delays + lookbacks, logging config + preview, per-timeframe analyzer windows, export/import path settings, redacted config dump.
- CONTEXT.md changes (additive UI following existing rules).
