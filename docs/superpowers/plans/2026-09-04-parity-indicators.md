# Per-Coin Indicator Charts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-coin Price+SMA / RSI / MACD history charts in React, fetched live on demand like the Streamlit viewer.

**Architecture:** New `indicators` runner kind (single-flight, like backtest) whose adapter mirrors `market_page.plot_coin_chart` (same period/interval rule, same `_calculate_indicators`, same 30-row minimum, same 500-row tail) and writes a per-symbol+timeframe cache file; a custom GET serves the series; the generic `POST /{kind}/run` already covers the run side once the kind is registered. No `src/` changes.

**Tech Stack:** FastAPI, AnalysisRunner, pandas_ta column names (verified offline: `SMA_{n}`, `RSI_{p}`, `MACD_12_26_9`, `MACDh_12_26_9`, `MACDs_12_26_9`), recharts, Vitest.

**Why not offline:** the technical cache holds current values only (verified in `data/api_cache/technical_live.json`); `fetch_crypto_data_async` caches in-memory only. Live fetch is required — hence a runner job, never a GET.

---

### Task 1: Indicators runner kind + series endpoint + tests

**Files:**
- Modify: `api/analysis_runner.py` (register kind + adapter + constants)
- Modify: `api/routes/strategy.py` (custom GET + cache-path helper + schema import)
- Modify: `api/schemas/screens.py` (point + response models)
- Create: `tests/api/test_indicators_route.py`

- [ ] **Step 1: Adapter + registration in `api/analysis_runner.py`**

```python
_INDICATOR_TIMEFRAMES = {"long_term", "swing", "day"}


async def _indicators(tracker, params=None) -> dict:
    """Per-coin indicator history for charting.

    Same fetch + indicator path as the Streamlit coin viewer
    (market_page.plot_coin_chart): period/interval rule, 30-row minimum,
    500-row tail. Writes its own per-symbol cache file (the generic kind
    file cannot hold every coin) and returns a summary for it.
    """
    from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer

    params = params or {}
    symbol = str(params.get("symbol", "")).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,10}", symbol):
        raise ValueError(f"Unknown symbol: {params.get('symbol')!r}")
    timeframe = str(params.get("timeframe", "swing"))
    if timeframe not in _INDICATOR_TIMEFRAMES:
        raise ValueError(f"Unknown timeframe: {params.get('timeframe')!r}")

    analyzer = CryptoTrendAnalyzer(
        config=tracker.config, binance_client=tracker.binance_client
    )
    settings = analyzer.timeframe_settings.get(timeframe) or {}
    period = settings.get("period", "1mo")
    interval = "1wk" if timeframe == "long_term" else "1d"
    data = await analyzer.fetch_crypto_data_async(symbol, period, interval)
    if data is None or data.empty or len(data) < 30:
        summary = {"symbol": symbol, "timeframe": timeframe, "points": len(data) if data is not None else 0}
        _write_indicators_cache(tracker.config_manager, summary)
        return summary

    # _calculate_indicators is underscore-private; used here deliberately —
    # it is the exact computation the viewer plots, and duplicating the
    # Study assembly would fork the indicator logic.
    frame = analyzer._calculate_indicators(data.copy(), settings)
    if not pd.api.types.is_datetime64_any_dtype(frame.index):
        raise ValueError(f"No datetime index for {symbol}.")
    frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame[~frame.index.isna()].sort_index().tail(500)

    short_len = settings.get("sma_short_window")
    long_len = settings.get("sma_long_window")
    rsi_col = f"RSI_{analyzer.rsi_period}"

    points = []
    for stamp, row in frame.iterrows():
        points.append({
            "date": str(stamp.date()) if hasattr(stamp, "date") else str(stamp),
            "close": _num_or_none(row.get("Close")),
            "sma_short": _num_or_none(row.get(f"SMA_{short_len}")) if short_len else None,
            "sma_long": _num_or_none(row.get(f"SMA_{long_len}")) if long_len else None,
            "rsi": _num_or_none(row.get(rsi_col)),
            "macd": _num_or_none(row.get("MACD_12_26_9")),
            "macd_signal": _num_or_none(row.get("MACDs_12_26_9")),
            "macd_hist": _num_or_none(row.get("MACDh_12_26_9")),
        })
    summary = {"symbol": symbol, "timeframe": timeframe, "points": points}
    _write_indicators_cache(tracker.config_manager, summary)
    return summary
```
Helpers in the same file (check existing `num`-like helper first — analysis_runner may not import `num`; if absent, define tiny local `_num_or_none` mapping None/NaN → None else float, and `_write_indicators_cache(config_manager, payload)` writing `Path("data") / "api_cache" / f"indicators_{symbol}_{timeframe}_{suffix}.json"` via MetricsCache with suffix mirroring `analysis_cache_path`). `re` and `pd` — check imports, add if missing. Register `"indicators": _indicators` in KINDS. The generic `POST /api/strategy/indicators/run` then works with `{symbol, timeframe}` (verify: `run_analysis` passes `params` through to `start` — read it; it does per Backtest usage).

- [ ] **Step 2: Schemas**

```python
class IndicatorPoint(BaseModel):
    date: str
    close: Optional[float] = None
    sma_short: Optional[float] = None
    sma_long: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None


class IndicatorsResponse(BaseModel):
    has_data: bool
    is_running: bool
    error: Optional[str] = None
    staleness: Staleness
    symbol: str
    timeframe: str
    points: list[IndicatorPoint] = []
```
Check how other strategy GETs build `Staleness`/`has_data`/`is_running`/`error` (rebalance pattern with `get_analysis_runner()` + `staleness_for`) and mirror. `Staleness` import path: `api.schemas.system` (per common.py).

- [ ] **Step 3: Custom GET in strategy.py**

```python
def _indicators_cache(ctx, symbol: str, timeframe: str) -> MetricsCache:
    suffix = "testnet" if ctx.config_manager.is_testnet_mode else "live"
    return MetricsCache(
        Path("data") / "api_cache" / f"indicators_{symbol}_{timeframe}_{suffix}.json")


@router.get("/indicators", response_model=IndicatorsResponse)
def indicators(symbol: str, timeframe: str = "swing",
               ctx=Depends(get_read_context)) -> IndicatorsResponse:
    """Cached per-coin indicator history. Run it first via
    POST /api/strategy/indicators/run {symbol, timeframe} (live fetch)."""
    clean_symbol = symbol.strip().upper()
    cache = _indicators_cache(ctx, clean_symbol, timeframe)
    cached = cache.read() or {}
    runner = get_analysis_runner()
    return IndicatorsResponse(
        has_data=bool(cached.get("points")),
        is_running=runner.is_running("indicators"),
        error=runner.last_error("indicators"),
        staleness=staleness_for(cache),
        symbol=clean_symbol, timeframe=timeframe,
        points=[IndicatorPoint(**p) for p in cached.get("points", [])
                if isinstance(p, dict)],
    )
```
`Path` import: check strategy.py imports (it may lack Path — add). `get_analysis_runner` already imported (line 12).

- [ ] **Step 4: Failing tests first** (`tests/api/test_indicators_route.py`)

No network in tests: seed the per-symbol cache file directly (`data/api_cache/indicators_BTC_swing_testnet.json` under monkeypatched cwd with 3 points incl a null) + assert GET returns symbol/points/null passthrough/staleness shape; unknown symbol file → has_data False (200, not 404 — mirrors rebalance's empty state); POST run with bad symbol → surfaces runner error... POST actually STARTS a job (needs tracker → get_tracker in tests? `mock_tracker` fixture wires a Mock tracker; adapter would call Mock methods — Mock returns Mocks, likely exploding somewhere). SCOPE the run test to validation only if cheap: `POST /api/strategy/indicators/run {symbol: "!!!", timeframe: "swing"}` → job fails → GET shows `error` containing "Unknown symbol" (poll with retries — runner task needs an event loop; TestClient runs one... `run_analysis` is async so loop exists. The task runs in background; test must wait: poll GET up to ~5s for error non-null). If flaky, test `_indicators` validation directly instead: call the adapter with a Mock tracker and assert ValueError (no network touched — validation precedes fetch). Prefer the direct-adapter test; do the POST-then-poll test only if stable within 3 tries.

- [ ] **Step 5: Implement, verify, commit**

`uv run pytest tests/api/test_indicators_route.py -q` → green; full `uv run pytest -q`; `uv run ruff check api tests/api run_ui.py` clean. Commit `feat: per-coin indicator history endpoint`.

---

### Task 2: React indicator viewer + tests

**Files:**
- Modify: `frontend/src/screens/Technical.tsx` and/or `Market.tsx` (put the viewer next to the indicator tables — read both first, pick the one showing per-coin rows)
- Modify: `frontend/src/types.ts` (mirror models)
- Create/extend frontend test for the touched screen

- [ ] **Step 1: Read the chosen screen + Backtest.tsx equity curve (recharts pattern to mirror)**

- [ ] **Step 2: Failing tests first**

Stub `/api/strategy/indicators?symbol=BTC&timeframe=swing` with 5 points (one with null rsi) + stub the screen's existing fetches; assert: coin/timeframe pickers render; Run posts `{symbol, timeframe}` to `/api/strategy/indicators/run`; after reload the chart shows (assert a plotted value text or recharts wrapper presence — assert on data-driven DOM like legend/tooltip labels, never on canvas pixels); null point renders gap not crash. Fetch-fail convention test if the screen lacks one.

- [ ] **Step 3: Implement**

Coin picker options from the screen's existing technical payload symbols (strip any `-USD` suffix the same way the screen already displays them — mirror its logic, do not invent a second stripping rule); timeframe picker (long_term/swing/day); Run → `apiPost('/api/strategy/indicators/run', {symbol, timeframe})` → poll via existing `usePollWhile`-style hook if the screen has one, else `reload()` after short delay + manual Refresh button; three recharts panels (Price + SMAs, RSI, MACD + signal + hist) mirroring Backtest.tsx's AreaChart/LineChart idiom; empty state ("Run the viewer to fetch history — needs network") when `!has_data`; error text when `error`.

- [ ] **Step 4: Verify + commit**

`npx vitest run` full, `npx tsc -b` clean, `npx oxlint` no new warnings. Commit `feat: per-coin indicator charts`.

---

## Out of scope

- Indicator computation changes (core `_calculate_indicators` untouched).
- CONTEXT.md changes.
