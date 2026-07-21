# Premium UI — Foundation & Daily Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a working FastAPI + React interface covering the Cockpit, Capital Flow, and Sync screens — the daily driver — on a foundation the remaining twelve screens extend without rework.

**Architecture:** A new `api/` package wraps the existing, unmodified `CryptoPortfolioTracker` facade. Read endpoints serve from SQLite plus an API-owned metrics cache file and never contact Binance; Binance is reached only on explicit user-initiated sync, whose progress streams over SSE. A Vite + React + TypeScript frontend proxies to the API in development and is served by FastAPI as static files in production.

**Tech Stack:** FastAPI, Uvicorn, Pydantic v2, pytest / pytest-asyncio, httpx (already a dependency), Vite 5, React 18, TypeScript 5, Tailwind CSS 3, Vitest, `uv` for Python dependency management, npm for JS.

This plan covers **steps 1–4** of spec §9. Steps 5–8 (the remaining twelve screens) get their own plans, written after this one ships.

## Global Constraints

- **The Python core is never modified.** `src/crypto_portfolio_tracker/**` is read-only for the duration of this plan. Every task's changes live in `api/`, `frontend/`, `tests/api/`, or project config.
- **No endpoint may be written against an assumed method name.** Read the module before binding to it. The verified surface is spec §7; anything outside it must be confirmed by reading source first.
- **Reads never block on Binance.** Any endpoint under `GET /api/**` that can reach the network is a defect.
- **Colour is never the sole carrier of meaning.** Every gain/loss value pairs its colour with an explicit `+`/`-` sign.
- **All numerics render in JetBrains Mono**, right-aligned in tables.
- Design tokens are exactly these values, from spec §5 — copy verbatim, do not re-derive:
  - Surfaces: `level-0 #0B0C0E`, `level-1 #14161A`, `level-2 #1C1F26`, `border #2D3139`
  - Semantic: `positive #00C076`, `negative #E04E4E`, `warning #D9A441`, `action #4F46E5`
  - Text: `primary #F9FAFB`, `secondary #9BA3AF`
  - Radius: `4px` buttons/inputs, `6px` panels/cards, `0px` selection bars
  - Spacing base unit `4px`; panel padding `12px`
- **No shadows, no glow, no glassmorphism.** Depth is tonal layering plus 1px borders.
- The existing test suite must stay green at every commit: `uv run pytest tests/ -q`.
- Commit messages carry no AI attribution, no `Co-authored-by` trailers.

---

## File Structure

| Path | Responsibility |
|---|---|
| `api/__init__.py` | Package marker |
| `api/deps.py` | Process-wide `CryptoPortfolioTracker` and `ConfigManager` singletons |
| `api/cache.py` | API-owned metrics cache: persist/load last good metrics with staleness |
| `api/serialization.py` | DataFrame → JSON-safe primitives; the one place pandas types die |
| `api/schemas/portfolio.py` | Cockpit + holdings Pydantic models |
| `api/schemas/capital.py` | Capital-flow Pydantic models |
| `api/schemas/system.py` | Health, environment, staleness models |
| `api/routes/portfolio.py` | `GET /api/portfolio/*` |
| `api/routes/capital.py` | `GET /api/capital/*` |
| `api/routes/sync.py` | `POST /api/sync`, `GET /api/sync/stream` |
| `api/routes/system.py` | `GET /api/system/*` |
| `api/main.py` | App assembly, router mounting, static serving |
| `tests/api/` | pytest suite for the above, core mocked |
| `frontend/src/tokens.css` | Design tokens as CSS custom properties — single source |
| `frontend/src/lib/api.ts` | Typed fetch client |
| `frontend/src/lib/format.ts` | Number/currency/percent formatting with sign |
| `frontend/src/components/` | Shared primitives (Panel, Metric, DataTable, EnvBanner) |
| `frontend/src/screens/Cockpit.tsx` | Screen 1 |
| `frontend/src/screens/CapitalFlow.tsx` | Screen 2 |
| `frontend/src/screens/Sync.tsx` | Screen 6 |
| `frontend/src/App.tsx` | Shell, routing, global env banner |

---

## Task 1: Design system correction

Spec §9 step 1, and the user's explicit instruction: fix the design system before any screen work. The Stitch design system drifted from its own `designMd` prose (spec §2 defect 3) — gains and losses currently render identically. Correct it at the source, then land the same tokens in the repo as the authoritative copy.

**Files:**
- Create: `docs/design/tokens.md`
- Create: `docs/design/tokens.css`

The token file is staged under `docs/design/` rather than `frontend/src/` because Task 7 runs `npm create vite`, which refuses to scaffold into a non-empty directory. Task 7 copies it into place.

**Interfaces:**
- Produces: CSS custom properties `--surface-0/1/2`, `--border`, `--positive`, `--negative`, `--warning`, `--action`, `--text-primary`, `--text-secondary`, `--radius-control`, `--radius-panel`, `--space` — consumed by every frontend task.

- [ ] **Step 1: Correct the Stitch design system**

The Stitch project is `915468954340907029` ("Crypto Investment Operating System"). Call `mcp__stitch__update_design_system` for that project, setting the palette to the Global Constraints values above — specifically replacing the Material-3 purple primary with `#4F46E5`, and ensuring `positive #00C076` and `negative #E04E4E` are distinct semantic entries.

If `update_design_system` rejects the payload or the tool errors, do not spend more than two attempts on it. Stitch is reference, not source of truth (spec §2) — record the failure in the commit message and proceed. The repo tokens in Step 2 are what actually govern the build.

- [ ] **Step 2: Write the repo token file**

`docs/design/tokens.css`:

```css
:root {
  --surface-0: #0B0C0E;
  --surface-1: #14161A;
  --surface-2: #1C1F26;
  --border: #2D3139;

  --positive: #00C076;
  --negative: #E04E4E;
  --warning: #D9A441;
  --action: #4F46E5;

  --text-primary: #F9FAFB;
  --text-secondary: #9BA3AF;

  --radius-control: 4px;
  --radius-panel: 6px;
  --space: 4px;
  --panel-padding: 12px;

  --font-ui: Inter, system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}
```

- [ ] **Step 3: Document the rules alongside the values**

`docs/design/tokens.md` restates the token table from Global Constraints and the four rules from spec §5 (tonal layering only; mono numerics right-aligned; colour never sole carrier; semantic colours reserved for semantics — categorical series use a separate palette).

- [ ] **Step 4: Commit**

```bash
git add docs/design/tokens.css docs/design/tokens.md
git commit -m "feat: add corrected design tokens

Stitch generated Material-3 purple against a designMd specifying
indigo/mint/crimson, rendering gains and losses identically. These
values follow the prose and are authoritative for the React build."
```

---

## Task 2: API package skeleton and core singletons

**Files:**
- Create: `api/__init__.py`, `api/deps.py`, `api/main.py`
- Create: `tests/api/conftest.py`, `tests/api/test_deps.py` (no `__init__.py` — see Step 4)
- Modify: `pyproject.toml` (add `fastapi`, `uvicorn[standard]`)

**Interfaces:**
- Produces: `get_tracker() -> CryptoPortfolioTracker`, `get_config_manager() -> ConfigManager`, `reset_singletons() -> None`, and `app` (the FastAPI instance).

`CryptoPortfolioTracker.__init__(config_manager: ConfigManager, force_offline: bool = False)` is the composition root — verified at `src/crypto_portfolio_tracker/portfolio_tracker.py:40`. It builds the DatabaseManager, DataSynchronizer, BinanceFetcher, PriceEnricher, PortfolioAnalyzer and TradeExecutor itself. The API constructs exactly one and reuses it; constructing per-request would re-open SQLite and re-init the Binance client on every call.

- [ ] **Step 1: Add dependencies**

```bash
uv add fastapi "uvicorn[standard]"
```

- [ ] **Step 2: Write the failing test**

`tests/api/conftest.py`:

```python
"""Fixtures for API tests. The core is mocked; no network, no real DB."""

import pytest
from unittest.mock import Mock

from api import deps


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Every test starts with a clean singleton slate."""
    deps.reset_singletons()
    yield
    deps.reset_singletons()


@pytest.fixture
def mock_tracker():
    """A CryptoPortfolioTracker stand-in wired into the deps singleton."""
    tracker = Mock()
    tracker.config_manager = Mock()
    tracker.config_manager.is_testnet_mode = True
    deps.set_tracker_for_testing(tracker)
    return tracker
```

`tests/api/test_deps.py`:

```python
from unittest.mock import Mock, patch

from api import deps


def test_get_tracker_returns_same_instance_across_calls():
    with patch("api.deps.CryptoPortfolioTracker") as ctor:
        ctor.return_value = Mock()
        first = deps.get_tracker()
        second = deps.get_tracker()

    assert first is second
    assert ctor.call_count == 1


def test_get_tracker_passes_config_manager():
    with patch("api.deps.CryptoPortfolioTracker") as ctor, \
         patch("api.deps.ConfigManager") as cfg_ctor:
        cfg = Mock()
        cfg_ctor.return_value = cfg
        deps.get_tracker()

    ctor.assert_called_once_with(cfg)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/api/test_deps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 4: Write the implementation**

`api/__init__.py`: empty file.

`api/deps.py`:

```python
"""Process-wide singletons over the untouched core.

CryptoPortfolioTracker opens SQLite and initializes the Binance client in
its constructor, so it is built once per process and reused.
"""

from typing import Optional

from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker

_config_manager: Optional[ConfigManager] = None
_tracker: Optional[CryptoPortfolioTracker] = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_tracker() -> CryptoPortfolioTracker:
    global _tracker
    if _tracker is None:
        _tracker = CryptoPortfolioTracker(get_config_manager())
    return _tracker


def set_tracker_for_testing(tracker) -> None:
    global _tracker
    _tracker = tracker


def reset_singletons() -> None:
    global _config_manager, _tracker
    _config_manager = None
    _tracker = None
```

`api/main.py`:

```python
"""FastAPI application serving the portfolio API and the built frontend."""

from fastapi import FastAPI

app = FastAPI(title="Crypto Portfolio Tracker API", version="1.0.0")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```

Do **not** create `tests/api/__init__.py`. `tests/` has no `__init__.py`, so adding one under `tests/api/` makes pytest import that directory as a top-level package named `api`, colliding with the real `api/` package and breaking collection. Leave `tests/api/` without one.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/api/ -v`
Expected: PASS, 2 tests.

- [ ] **Step 6: Verify the core is still untouched and green**

```bash
git status --porcelain src/crypto_portfolio_tracker/
uv run pytest tests/ -q
```
Expected: the first command prints nothing; the suite passes.

- [ ] **Step 7: Commit**

```bash
git add api/ tests/api/ pyproject.toml uv.lock
git commit -m "feat: add API package skeleton with core singletons"
```

---

## Task 2b: Network-free read context

Discovered during Task 2 review. `CryptoPortfolioTracker.__init__` calls `_init_binance_client()`, which performs live network calls — `client.get_server_time()` (`data_synchronizer.py:160`) and `client.ping()` (`data_synchronizer.py:138`) — and raises `NetworkUnavailableError` when the network is down.

Because `get_tracker()` constructs lazily, the first GET request to any endpoint using `Depends(get_tracker)` would block on Binance and could return 500 while offline. That violates the Global Constraint "Reads never block on Binance."

Read endpoints need only three things, none of which touch the network: the `ConfigManager` (for testnet state and paths), the `DatabaseManager` (SQLite), and a `PortfolioAnalyzer` for `calculate_total_invested_capital()`, which reads only `db_manager.get_invested_capital_transactions()` (verified at `portfolio_analyzer.py:51-86`).

So reads get their own dependency. `get_tracker()` remains, reserved for sync — the one path that is *supposed* to reach Binance.

**Files:**
- Modify: `api/deps.py`
- Create: `tests/api/test_read_context.py`

**Interfaces:**
- Produces: `ReadContext` (attributes `.config_manager`, `.db_manager`, `.portfolio_analyzer`) and `get_read_context() -> ReadContext`. Tasks 4 and 5 depend on this instead of `get_tracker`.

- [ ] **Step 1: Write the failing test**

`tests/api/test_read_context.py`:

```python
from unittest.mock import Mock, patch

from api import deps


def test_read_context_never_constructs_the_binance_client():
    """The whole point: reads must not touch the network."""
    with patch("api.deps.CryptoPortfolioTracker") as tracker_ctor, \
         patch("api.deps.DatabaseManager"), \
         patch("api.deps.PortfolioAnalyzer"), \
         patch("api.deps.ConfigManager"):
        deps.get_read_context()

    tracker_ctor.assert_not_called()


def test_read_context_is_a_singleton():
    with patch("api.deps.DatabaseManager"), \
         patch("api.deps.PortfolioAnalyzer"), \
         patch("api.deps.ConfigManager"):
        first = deps.get_read_context()
        second = deps.get_read_context()

    assert first is second


def test_read_context_builds_analyzer_in_offline_mode():
    with patch("api.deps.DatabaseManager"), \
         patch("api.deps.PortfolioAnalyzer") as analyzer_ctor, \
         patch("api.deps.ConfigManager"):
        deps.get_read_context()

    assert analyzer_ctor.call_args.kwargs["offline_mode"] is True
    assert analyzer_ctor.call_args.kwargs["binance_client"] is None
    assert analyzer_ctor.call_args.kwargs["fetcher"] is None


def test_read_context_uses_the_configured_database_path():
    with patch("api.deps.DatabaseManager") as db_ctor, \
         patch("api.deps.PortfolioAnalyzer"), \
         patch("api.deps.ConfigManager") as cfg_ctor:
        cfg = Mock()
        cfg.get_database_path.return_value = "data/testnet_portfolio.db"
        cfg_ctor.return_value = cfg
        deps.get_read_context()

    assert db_ctor.call_args.kwargs["db_path"] == "data/testnet_portfolio.db"


def test_reset_singletons_clears_the_read_context():
    with patch("api.deps.DatabaseManager"), \
         patch("api.deps.PortfolioAnalyzer"), \
         patch("api.deps.ConfigManager"):
        first = deps.get_read_context()
        deps.reset_singletons()
        second = deps.get_read_context()

    assert first is not second
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/api/test_read_context.py -v`
Expected: FAIL — `AttributeError: module 'api.deps' has no attribute 'get_read_context'`

- [ ] **Step 3: Implement**

Add to `api/deps.py` (keeping everything already there):

```python
from dataclasses import dataclass

from crypto_portfolio_tracker.database import DatabaseManager
from crypto_portfolio_tracker.portfolio_analyzer import PortfolioAnalyzer


@dataclass
class ReadContext:
    """Everything a read endpoint needs, and nothing that touches the network.

    CryptoPortfolioTracker.__init__ pings Binance and syncs server time, so
    read paths must not construct it. They get SQLite and an offline analyzer
    instead. get_tracker() stays reserved for sync.
    """

    config_manager: ConfigManager
    db_manager: DatabaseManager
    portfolio_analyzer: PortfolioAnalyzer


_read_context: Optional[ReadContext] = None


def get_read_context() -> ReadContext:
    global _read_context
    if _read_context is None:
        config_manager = get_config_manager()
        db_manager = DatabaseManager(
            db_path=config_manager.get_database_path(),
            backup_dir=config_manager.get_backup_dir(),
        )
        _read_context = ReadContext(
            config_manager=config_manager,
            db_manager=db_manager,
            portfolio_analyzer=PortfolioAnalyzer(
                config=config_manager.config,
                db_manager=db_manager,
                binance_client=None,
                fetcher=None,
                enricher=None,
                offline_mode=True,
                config_manager=config_manager,
            ),
        )
    return _read_context


def set_read_context_for_testing(context) -> None:
    global _read_context
    _read_context = context
```

Extend the existing `reset_singletons()` to also clear `_read_context`.

Check `DatabaseManager.__init__` at `src/crypto_portfolio_tracker/database.py:29` for its required keyword arguments and pass what it needs. Do not guess the signature — read it.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/api/ -v`
Expected: PASS, all tests including Task 2's.

- [ ] **Step 5: Update the shared fixture**

In `tests/api/conftest.py`, add a `mock_read_context` fixture mirroring `mock_tracker`, wiring a `Mock()` through `deps.set_read_context_for_testing`, and have the autouse reset fixture clear it. Tasks 4 and 5 use this fixture.

- [ ] **Step 6: Commit**

```bash
git add api/deps.py tests/api/
git commit -m "feat: add network-free read context

CryptoPortfolioTracker's constructor pings Binance and syncs server time,
so lazily constructing it inside a GET would block reads on the network
and 500 when offline. Read endpoints now get SQLite and an offline
analyzer; get_tracker stays reserved for sync."
```

---

## Task 3: Metrics cache

The load-bearing piece. `calculate_portfolio_metrics()` is `async` and calls `fetcher.fetch_binance_balances()` and `enricher.get_current_prices()` — verified at `portfolio_analyzer.py:257-380`. Calling it from a GET handler would reproduce Streamlit's latency behind new paint.

Its `offline_mode` branch is **not** a usable substitute: it hardcodes `current_price = 0.0` and `value_usd = 0.0` (lines 275-276), so it would render a $0 portfolio. Confirmed by reading the source.

There is also no persisted current price anywhere in the schema — `holdings` stores only `quantity` and `average_cost_basis`, and `historical_prices` is daily-granularity (`database.py:116-124`). So the API keeps its own cache: the last successful metrics result, written on sync, read by every GET, always accompanied by its age.

**Files:**
- Create: `api/serialization.py`, `api/cache.py`
- Create: `tests/api/test_cache.py`

**Interfaces:**
- Produces: `df_to_records(df) -> list[dict]`, `jsonable(value) -> Any`, `MetricsCache(path)` with `.write(metrics: dict) -> None`, `.read() -> dict | None`, `.age_seconds() -> float | None`, and `cache_path_for(config_manager) -> Path`.

- [ ] **Step 1: Write the failing test**

`tests/api/test_cache.py`:

```python
import datetime
import json

import pandas as pd
import pytest

from api.cache import MetricsCache
from api.serialization import df_to_records, jsonable


def test_df_to_records_converts_nan_to_none():
    df = pd.DataFrame({"symbol": ["BTC"], "value_usd": [float("nan")]})
    assert df_to_records(df) == [{"symbol": "BTC", "value_usd": None}]


def test_df_to_records_on_empty_frame_returns_empty_list():
    assert df_to_records(pd.DataFrame()) == []


def test_jsonable_converts_timestamps_to_iso_strings():
    ts = datetime.datetime(2026, 7, 21, 9, 30, 0)
    assert jsonable(ts) == "2026-07-21T09:30:00"


def test_cache_round_trips_metrics_containing_dataframes(tmp_path):
    cache = MetricsCache(tmp_path / "metrics.json")
    cache.write({
        "total_value_usd": 57.78,
        "holdings_df": pd.DataFrame({"symbol": ["BTC"], "value_usd": [57.78]}),
        "timestamp": datetime.datetime(2026, 7, 21, 9, 30, 0),
    })

    loaded = cache.read()

    assert loaded["total_value_usd"] == 57.78
    assert loaded["holdings_df"] == [{"symbol": "BTC", "value_usd": 57.78}]
    assert loaded["timestamp"] == "2026-07-21T09:30:00"


def test_read_returns_none_when_cache_absent(tmp_path):
    assert MetricsCache(tmp_path / "missing.json").read() is None


def test_read_returns_none_when_cache_corrupt(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{ not json")
    assert MetricsCache(path).read() is None


def test_age_seconds_is_none_when_cache_absent(tmp_path):
    assert MetricsCache(tmp_path / "missing.json").age_seconds() is None


def test_age_seconds_is_small_immediately_after_write(tmp_path):
    cache = MetricsCache(tmp_path / "metrics.json")
    cache.write({"total_value_usd": 1.0})
    assert cache.age_seconds() < 5.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/api/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.cache'`

- [ ] **Step 3: Write the implementation**

`api/serialization.py`:

```python
"""Convert pandas and datetime values into JSON-safe primitives.

The core returns DataFrames and Timestamps throughout. This module is the
single boundary where those types are converted, so no route or schema has
to know about pandas.
"""

import datetime
import math
from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    """Recursively convert a value into something json.dumps accepts."""
    if isinstance(value, pd.DataFrame):
        return df_to_records(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        return None if math.isnan(as_float) else as_float
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NaT or value is None:
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [jsonable(v) for v in value]
    return value


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of JSON-safe dicts. Empty frame -> []."""
    if df is None or df.empty:
        return []
    return [
        {str(col): jsonable(val) for col, val in row.items()}
        for row in df.to_dict(orient="records")
    ]
```

`api/cache.py`:

```python
"""API-owned cache of the last successful portfolio metrics.

calculate_portfolio_metrics() reaches for live prices, so GET endpoints
read this file instead. It is written only by an explicit sync. Its age is
always exposed to the UI rather than hidden -- a stale figure the user can
see is safe; a stale figure presented as current is not.

This lives outside the core database. No core schema is modified.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from api.serialization import jsonable

logger = logging.getLogger(__name__)


class MetricsCache:
    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, metrics: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(k): jsonable(v) for k, v in metrics.items()}
        payload["_cached_at"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)

    def read(self) -> Optional[dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Metrics cache unreadable at %s: %s", self.path, exc)
            return None

    def age_seconds(self) -> Optional[float]:
        cached = self.read()
        if not cached or "_cached_at" not in cached:
            return None
        return time.time() - cached["_cached_at"]


def cache_path_for(config_manager) -> Path:
    """Testnet and live caches are separate files, mirroring the separate DBs.

    config.is_testnet_mode already switches the database path; the cache must
    switch with it or testnet figures would surface as live ones.
    """
    suffix = "testnet" if config_manager.is_testnet_mode else "live"
    return Path("data") / "api_cache" / f"metrics_{suffix}.json"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/api/test_cache.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add api/cache.py api/serialization.py tests/api/test_cache.py
git commit -m "feat: add API-owned metrics cache with staleness tracking

Read endpoints serve from this file so they never block on Binance.
The offline_mode branch of calculate_portfolio_metrics is unusable for
this purpose -- it zeroes current_price, which would render a \$0
portfolio. Cache is keyed by environment so testnet and live never mix."
```

---

## Task 4: Portfolio schemas and the dual-accounting endpoint

The signature element of spec §8.1. Both accounting models, never conflated. The Stitch mock printed them equal; in real data they differ 7.7x.

Verified sources:
- Net invested: `portfolio_analyzer.calculate_total_invested_capital() -> float` (`portfolio_analyzer.py:51`), which sums P2P buys and subtracts withdrawals and P2P sells.
- FIFO: `utils.calculate_fifo_cost_basis(transactions_df) -> tuple[float, float]` returning `(current_quantity, average_cost_basis)` **for a single asset** (`utils.py:211-222`). Portfolio-wide cost basis is therefore the sum of `qty * avg_cost` across symbols — it is not a single call.
- Transactions: `database.get_all_transactions() -> pd.DataFrame` (`database.py:456`).

**Note on the golden values.** Spec §8.1 prints the FIFO basis as `-$142.85 (-71.52%)` on a `$199.75` cost basis against a `$57.78` portfolio. That does not reconcile: `57.78 - 199.75 = -141.97`, and `-141.97 / 199.75 = -71.07%`. The spec's illustrative figure carries a small arithmetic error. This plan asserts the values that follow from the stated inputs. If the real portfolio produces `-142.85`, then one of the three inputs differs from what §8.1 records, and the correct response is to recheck the inputs — not to loosen the assertion.

**Files:**
- Create: `api/schemas/__init__.py`, `api/schemas/portfolio.py`, `api/schemas/system.py`
- Create: `api/accounting.py`, `api/routes/__init__.py`, `api/routes/portfolio.py`
- Modify: `api/main.py`
- Create: `tests/api/test_accounting.py`, `tests/api/test_portfolio_routes.py`

**Interfaces:**
- Consumes: `MetricsCache`, `cache_path_for`, `get_read_context` (Tasks 2b–3).
- Produces: `portfolio_fifo_cost_basis(transactions_df) -> float`, and schemas `Staleness`, `AccountingBasis`, `Holding`, `CockpitResponse`.

- [ ] **Step 1: Write the failing accounting test**

`tests/api/test_accounting.py`:

```python
import pandas as pd

from api.accounting import portfolio_fifo_cost_basis


def test_fifo_cost_basis_sums_across_symbols():
    txs = pd.DataFrame([
        {"symbol": "BTC", "timestamp": "2026-01-01", "type": "BUY",
         "quantity": 1.0, "price_usd": 100.0, "fee_usd": 0.0},
        {"symbol": "ETH", "timestamp": "2026-01-02", "type": "BUY",
         "quantity": 2.0, "price_usd": 50.0, "fee_usd": 0.0},
    ])
    assert portfolio_fifo_cost_basis(txs) == 200.0


def test_fifo_cost_basis_excludes_sold_lots():
    txs = pd.DataFrame([
        {"symbol": "BTC", "timestamp": "2026-01-01", "type": "BUY",
         "quantity": 2.0, "price_usd": 100.0, "fee_usd": 0.0},
        {"symbol": "BTC", "timestamp": "2026-01-02", "type": "SELL",
         "quantity": 1.0, "price_usd": 150.0, "fee_usd": 0.0},
    ])
    assert portfolio_fifo_cost_basis(txs) == 100.0


def test_fifo_cost_basis_on_empty_frame_is_zero():
    assert portfolio_fifo_cost_basis(pd.DataFrame()) == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/api/test_accounting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.accounting'`

- [ ] **Step 3: Implement the accounting helper**

`api/accounting.py`:

```python
"""Portfolio-wide accounting derived from the core's per-asset primitives.

calculate_fifo_cost_basis operates on one asset at a time and returns
(quantity, average_cost_basis). Aggregating it is this module's job; the
core is not modified to do it.
"""

import pandas as pd

from crypto_portfolio_tracker.utils import calculate_fifo_cost_basis


def portfolio_fifo_cost_basis(transactions_df: pd.DataFrame) -> float:
    """Sum the FIFO cost basis of remaining lots across every symbol."""
    if transactions_df is None or transactions_df.empty:
        return 0.0
    if "symbol" not in transactions_df.columns:
        return 0.0

    total = 0.0
    for _symbol, group in transactions_df.groupby("symbol"):
        quantity, average_cost = calculate_fifo_cost_basis(group)
        total += quantity * average_cost
    return float(total)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/api/test_accounting.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Write the schemas**

`api/schemas/__init__.py`: empty file.

`api/schemas/system.py`:

```python
from typing import Optional

from pydantic import BaseModel, Field


class Staleness(BaseModel):
    """How old the served figures are. Never hidden from the UI."""

    cached_at: Optional[str] = Field(
        None, description="ISO timestamp of the last successful sync"
    )
    age_seconds: Optional[float] = Field(
        None, description="Seconds since that sync; null when never synced"
    )
    is_stale: bool = Field(
        description="True when older than the freshness threshold or never synced"
    )


class Environment(BaseModel):
    is_testnet: bool
    database_path: str
    label: str = Field(description="'TESTNET' or 'LIVE' -- rendered globally")
```

`api/schemas/portfolio.py`:

```python
from typing import Optional

from pydantic import BaseModel, Field

from api.schemas.system import Environment, Staleness


class AccountingBasis(BaseModel):
    """One of the two accounting models. Both are correct; they answer
    different questions, and the UI must never present them as equal."""

    label: str = Field(description="'NET INVESTED BASIS' or 'FIFO BASIS'")
    question: str = Field(description="The plain question this basis answers")
    basis_usd: float = Field(description="Denominator: net in, or cost basis")
    pl_usd: float
    pl_percent: float


class Holding(BaseModel):
    symbol: str
    total_quantity: float
    spot_quantity: Optional[float] = None
    earn_quantity: Optional[float] = None
    current_price: Optional[float] = None
    value_usd: Optional[float] = None
    average_cost_basis: Optional[float] = None
    cost_basis_total: Optional[float] = None
    unrealized_pl_usd: Optional[float] = None
    unrealized_pl_percent: Optional[float] = None
    is_core: bool = False


class CockpitResponse(BaseModel):
    total_value_usd: float
    net_invested: AccountingBasis
    fifo: AccountingBasis
    holdings: list[Holding]
    staleness: Staleness
    environment: Environment
    has_data: bool = Field(
        description="False when no sync has ever run; the UI renders an "
                    "explicit empty state rather than zeros"
    )
```

- [ ] **Step 6: Write the failing route test**

`tests/api/test_portfolio_routes.py`:

```python
import time

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


def _client():
    return TestClient(app)


def test_cockpit_returns_empty_state_when_never_synced(mock_read_context, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routes.portfolio.cache_path_for",
                        lambda cm: tmp_path / "metrics.json")

    response = _client().get("/api/portfolio/cockpit")

    assert response.status_code == 200
    body = response.json()
    assert body["has_data"] is False
    assert body["total_value_usd"] == 0.0
    assert body["staleness"]["age_seconds"] is None


def test_cockpit_reports_both_bases_distinctly(mock_read_context, tmp_path, monkeypatch):
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)

    from api.cache import MetricsCache
    MetricsCache(cache_file).write({
        "total_value_usd": 57.78,
        "total_invested_capital": 76.41,
        "holdings_df": pd.DataFrame([{
            "symbol": "BTC", "total_quantity": 0.001, "value_usd": 57.78,
            "average_cost_basis": 100.0, "cost_basis_total": 199.75,
        }]),
    })

    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame([
        {"symbol": "BTC", "timestamp": "2026-01-01", "type": "BUY",
         "quantity": 1.0, "price_usd": 199.75, "fee_usd": 0.0},
    ])

    body = _client().get("/api/portfolio/cockpit").json()

    assert body["total_value_usd"] == 57.78
    assert body["net_invested"]["basis_usd"] == 76.41
    assert body["fifo"]["basis_usd"] == 199.75
    # The defect this test exists to prevent: the two bases printed as equal.
    assert body["net_invested"]["pl_usd"] != body["fifo"]["pl_usd"]


def test_cockpit_pl_math_matches_the_real_portfolio(mock_read_context, tmp_path, monkeypatch):
    """Golden values from spec section 8.1."""
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)

    from api.cache import MetricsCache
    MetricsCache(cache_file).write({
        "total_value_usd": 57.78,
        "total_invested_capital": 76.41,
        "holdings_df": pd.DataFrame(),
    })
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame([
        {"symbol": "BTC", "timestamp": "2026-01-01", "type": "BUY",
         "quantity": 1.0, "price_usd": 199.75, "fee_usd": 0.0},
    ])

    body = _client().get("/api/portfolio/cockpit").json()

    assert round(body["net_invested"]["pl_usd"], 2) == -18.63
    assert round(body["net_invested"]["pl_percent"], 2) == -24.38
    assert round(body["fifo"]["pl_usd"], 2) == -141.97
    assert round(body["fifo"]["pl_percent"], 2) == -71.07


def test_cockpit_marks_data_stale_past_threshold(mock_read_context, tmp_path, monkeypatch):
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)

    from api.cache import MetricsCache
    MetricsCache(cache_file).write({"total_value_usd": 1.0, "holdings_df": pd.DataFrame()})
    stale = __import__("json").loads(cache_file.read_text())
    stale["_cached_at"] = time.time() - 7200
    cache_file.write_text(__import__("json").dumps(stale))

    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = _client().get("/api/portfolio/cockpit").json()
    assert body["staleness"]["is_stale"] is True


def test_cockpit_never_constructs_the_networked_tracker(mock_read_context, tmp_path,
                                                        monkeypatch):
    """CryptoPortfolioTracker's constructor pings Binance. A read that reaches
    for it would block on the network and 500 while offline."""
    from unittest.mock import patch

    monkeypatch.setattr("api.routes.portfolio.cache_path_for",
                        lambda cm: tmp_path / "metrics.json")

    with patch("api.deps.CryptoPortfolioTracker") as tracker_ctor:
        response = _client().get("/api/portfolio/cockpit")

    assert response.status_code == 200
    tracker_ctor.assert_not_called()
```

- [ ] **Step 7: Run it to verify it fails**

Run: `uv run pytest tests/api/test_portfolio_routes.py -v`
Expected: FAIL — 404 on `/api/portfolio/cockpit`, since no router is mounted.

- [ ] **Step 8: Implement the route**

`api/routes/__init__.py`: empty file.

`api/routes/portfolio.py`:

```python
"""Portfolio read endpoints. These never contact Binance."""

import datetime
from typing import Optional

from fastapi import APIRouter, Depends

from api.accounting import portfolio_fifo_cost_basis
from api.cache import MetricsCache, cache_path_for
from api.deps import get_read_context
from api.schemas.portfolio import AccountingBasis, CockpitResponse, Holding
from api.schemas.system import Environment, Staleness

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

STALE_AFTER_SECONDS = 3600.0


def _basis(label: str, question: str, value: float, basis_usd: float) -> AccountingBasis:
    pl = value - basis_usd
    percent = (pl / basis_usd * 100.0) if basis_usd else 0.0
    return AccountingBasis(
        label=label, question=question, basis_usd=basis_usd,
        pl_usd=pl, pl_percent=percent,
    )


def _staleness(age: Optional[float], cached_at: Optional[float]) -> Staleness:
    return Staleness(
        cached_at=(datetime.datetime.fromtimestamp(cached_at).isoformat()
                   if cached_at else None),
        age_seconds=age,
        is_stale=(age is None or age > STALE_AFTER_SECONDS),
    )


def _environment(config_manager) -> Environment:
    is_testnet = bool(config_manager.is_testnet_mode)
    return Environment(
        is_testnet=is_testnet,
        database_path=str(config_manager.get_database_path()),
        label="TESTNET" if is_testnet else "LIVE",
    )


@router.get("/cockpit", response_model=CockpitResponse)
def cockpit(ctx=Depends(get_read_context)) -> CockpitResponse:
    cache = MetricsCache(cache_path_for(ctx.config_manager))
    cached = cache.read()
    environment = _environment(ctx.config_manager)

    if cached is None:
        empty = _basis("", "", 0.0, 0.0)
        return CockpitResponse(
            total_value_usd=0.0,
            net_invested=empty.model_copy(update={
                "label": "NET INVESTED BASIS", "question": "did I make money?"}),
            fifo=empty.model_copy(update={
                "label": "FIFO BASIS", "question": "are my holdings underwater?"}),
            holdings=[],
            staleness=_staleness(None, None),
            environment=environment,
            has_data=False,
        )

    total_value = float(cached.get("total_value_usd") or 0.0)
    net_invested_basis = float(cached.get("total_invested_capital") or 0.0)
    fifo_basis = portfolio_fifo_cost_basis(ctx.db_manager.get_all_transactions())

    holdings = [
        Holding(**{k: v for k, v in row.items() if k in Holding.model_fields})
        for row in (cached.get("holdings_df") or [])
    ]

    return CockpitResponse(
        total_value_usd=total_value,
        net_invested=_basis(
            "NET INVESTED BASIS", "did I make money?", total_value, net_invested_basis),
        fifo=_basis(
            "FIFO BASIS", "are my holdings underwater?", total_value, fifo_basis),
        holdings=holdings,
        staleness=_staleness(cache.age_seconds(), cached.get("_cached_at")),
        environment=environment,
        has_data=True,
    )
```

`api/main.py` — replace the file with:

```python
"""FastAPI application serving the portfolio API and the built frontend."""

from fastapi import FastAPI

from api.routes import portfolio

app = FastAPI(title="Crypto Portfolio Tracker API", version="1.0.0")
app.include_router(portfolio.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/api/ -v`
Expected: PASS. If the golden-value assertions in `test_cockpit_pl_math_matches_the_real_portfolio` fail, do **not** adjust the assertion to match the output. Read the failure: it means the aggregation is wrong. The expected values follow arithmetically from `57.78 - 76.41 = -18.63` and `57.78 - 199.75 = -141.97`.

- [ ] **Step 10: Commit**

```bash
git add api/ tests/api/
git commit -m "feat: add cockpit endpoint with dual accounting bases

Net invested and FIFO cost basis are computed from separate sources and
returned as distinct labelled objects, each carrying the question it
answers. A regression test asserts they are never equal."
```

---

## Task 5: Capital flow endpoint with provenance

Spec §8.2. Justified by a real bug: a failed yfinance PHP/USD lookup silently zeroed `price_usd`, hiding $50.41 of inflow and inverting reported P/L from +122% to -24%. Rows must carry whether a rate was computed or fell back, so that class of bug is visible.

Source: `database.get_invested_capital_transactions() -> pd.DataFrame` with columns `source, type, quantity, price_usd` — verified at `database.py:670-689`. Note it selects only those four columns; there is no timestamp. Provenance is therefore inferred, not read: a `price_usd` of exactly `1.0` on a fiat-sourced row indicates the USDT-peg fallback rather than a computed rate, and `0.0` or null indicates a failed lookup.

**Files:**
- Create: `api/schemas/capital.py`, `api/routes/capital.py`
- Modify: `api/main.py`
- Create: `tests/api/test_capital_routes.py`

**Interfaces:**
- Consumes: `get_read_context`, `Environment` (Tasks 2b, 4).
- Produces: schemas `CapitalFlowRow`, `CapitalFlowResponse`.

- [ ] **Step 1: Write the failing test**

`tests/api/test_capital_routes.py`:

```python
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


def test_capital_flow_classifies_inflows_and_outflows(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 0.9},
        {"source": "Binance P2P Sell", "type": "SELL", "quantity": 20.0, "price_usd": 1.0},
        {"source": "Binance", "type": "WITHDRAWAL", "quantity": 5.0, "price_usd": 2.0},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 60.0

    body = TestClient(app).get("/api/capital/flow").json()

    assert body["net_invested_usd"] == 60.0
    assert [r["direction"] for r in body["rows"]] == ["in", "out", "out"]
    assert body["total_in_usd"] == 90.0
    assert body["total_out_usd"] == 30.0


def test_capital_flow_flags_peg_fallback_provenance(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 1.0},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 100.0

    row = TestClient(app).get("/api/capital/flow").json()["rows"][0]

    assert row["provenance"] == "usdt_peg_fallback"
    assert row["is_suspect"] is True


def test_capital_flow_flags_zero_price_as_failed_lookup(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 0.0},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 0.0

    row = TestClient(app).get("/api/capital/flow").json()["rows"][0]

    assert row["provenance"] == "failed_lookup"
    assert row["is_suspect"] is True
    assert row["value_usd"] == 0.0


def test_capital_flow_marks_computed_rates_as_trusted(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 0.0179},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 1.79

    row = TestClient(app).get("/api/capital/flow").json()["rows"][0]

    assert row["provenance"] == "computed"
    assert row["is_suspect"] is False


def test_capital_flow_empty_when_no_transactions(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame()
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 0.0

    body = TestClient(app).get("/api/capital/flow").json()

    assert body["rows"] == []
    assert body["net_invested_usd"] == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/api/test_capital_routes.py -v`
Expected: FAIL — 404, no capital router mounted.

- [ ] **Step 3: Implement schemas and route**

`api/schemas/capital.py`:

```python
from pydantic import BaseModel, Field


class CapitalFlowRow(BaseModel):
    source: str
    type: str
    direction: str = Field(description="'in' or 'out'")
    quantity: float
    price_usd: float
    value_usd: float
    provenance: str = Field(
        description="'computed', 'usdt_peg_fallback', or 'failed_lookup'"
    )
    is_suspect: bool = Field(
        description="True when the USD value may not reflect the real rate"
    )


class CapitalFlowResponse(BaseModel):
    rows: list[CapitalFlowRow]
    total_in_usd: float
    total_out_usd: float
    net_invested_usd: float
    suspect_count: int
```

`api/routes/capital.py`:

```python
"""Capital flow: fiat -> USDT -> asset provenance.

A failed yfinance fiat lookup silently zeroes price_usd, which once hid
$50.41 of inflow and inverted reported P/L. Provenance makes that class of
failure visible in the UI rather than invisible in the total.
"""

from fastapi import APIRouter, Depends

from api.deps import get_read_context
from api.schemas.capital import CapitalFlowResponse, CapitalFlowRow

router = APIRouter(prefix="/api/capital", tags=["capital"])

INFLOW_SOURCES = {"Binance P2P Buy"}
OUTFLOW_SOURCES = {"Binance P2P Sell"}


def _provenance(price_usd: float) -> str:
    if not price_usd:
        return "failed_lookup"
    if price_usd == 1.0:
        return "usdt_peg_fallback"
    return "computed"


@router.get("/flow", response_model=CapitalFlowResponse)
def capital_flow(ctx=Depends(get_read_context)) -> CapitalFlowResponse:
    df = ctx.db_manager.get_invested_capital_transactions()
    net_invested = float(
        ctx.portfolio_analyzer.calculate_total_invested_capital()
    )

    rows: list[CapitalFlowRow] = []
    if df is not None and not df.empty:
        for record in df.to_dict(orient="records"):
            source = str(record.get("source") or "")
            tx_type = str(record.get("type") or "")
            price_usd = float(record.get("price_usd") or 0.0)
            quantity = float(record.get("quantity") or 0.0)
            direction = "in" if source in INFLOW_SOURCES else "out"
            provenance = _provenance(price_usd)

            rows.append(CapitalFlowRow(
                source=source,
                type=tx_type,
                direction=direction,
                quantity=quantity,
                price_usd=price_usd,
                value_usd=quantity * price_usd,
                provenance=provenance,
                is_suspect=provenance != "computed",
            ))

    return CapitalFlowResponse(
        rows=rows,
        total_in_usd=sum(r.value_usd for r in rows if r.direction == "in"),
        total_out_usd=sum(r.value_usd for r in rows if r.direction == "out"),
        net_invested_usd=net_invested,
        suspect_count=sum(1 for r in rows if r.is_suspect),
    )
```

Add to `api/main.py`:

```python
from api.routes import capital, portfolio

app.include_router(capital.router)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/api/test_capital_routes.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add api/ tests/api/
git commit -m "feat: add capital flow endpoint with rate provenance

Rows carry whether their USD value came from a computed rate, the USDT
peg fallback, or a failed lookup, so a silently zeroed price_usd is
visible rather than absorbed into the total."
```

---

## Task 6: Sync endpoint with SSE progress

Spec §6. `binance_fetcher` already chunks windows into 30-day segments with per-chunk logging; that progress becomes visible instead of hiding behind an indeterminate spinner.

Verified: `CryptoPortfolioTracker.run_full_sync()` is `async` (`portfolio_tracker.py:330`), delegating to `data_synchronizer.run_full_sync(enricher)` (`data_synchronizer.py:673`). `save_snapshot(metrics)` exists at `portfolio_tracker.py:427`. `calculate_portfolio_metrics()` is `async` (`portfolio_tracker.py:322`).

Progress is surfaced by attaching a `logging.Handler` to the `crypto_portfolio_tracker` logger for the duration of the sync and forwarding records to an `asyncio.Queue`. This adds no code to the core — it consumes logging the core already emits.

**Files:**
- Create: `api/sync_runner.py`, `api/routes/sync.py`
- Modify: `api/main.py`
- Create: `tests/api/test_sync.py`

**Interfaces:**
- Consumes: `get_tracker`, `MetricsCache`, `cache_path_for`.
- Produces: `SyncRunner` with `.start() -> bool`, `.is_running -> bool`, `.events() -> AsyncIterator[dict]`, and module-level `get_sync_runner() -> SyncRunner`.

- [ ] **Step 1: Write the failing test**

`tests/api/test_sync.py`:

```python
import asyncio
import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.sync_runner import SyncRunner


@pytest.mark.asyncio
async def test_runner_forwards_core_log_records_as_events(mock_tracker, tmp_path):
    async def fake_sync():
        logging.getLogger("crypto_portfolio_tracker.binance_fetcher").info(
            "Fetching chunk 1 of 3"
        )
        return True

    mock_tracker.run_full_sync = fake_sync
    mock_tracker.calculate_portfolio_metrics = _async_return({
        "total_value_usd": 57.78, "holdings_df": pd.DataFrame(),
    })

    runner = SyncRunner(cache_path=tmp_path / "metrics.json")
    assert runner.start() is True

    messages = []
    async for event in runner.events():
        messages.append(event)
        if event["event"] == "complete":
            break

    assert any("chunk 1 of 3" in e.get("message", "") for e in messages)
    assert messages[-1]["event"] == "complete"


@pytest.mark.asyncio
async def test_runner_refuses_concurrent_syncs(mock_tracker, tmp_path):
    async def slow_sync():
        await asyncio.sleep(0.2)
        return True

    mock_tracker.run_full_sync = slow_sync
    mock_tracker.calculate_portfolio_metrics = _async_return({
        "total_value_usd": 1.0, "holdings_df": pd.DataFrame(),
    })

    runner = SyncRunner(cache_path=tmp_path / "metrics.json")
    assert runner.start() is True
    assert runner.start() is False


@pytest.mark.asyncio
async def test_runner_writes_metrics_cache_on_success(mock_tracker, tmp_path):
    cache_file = tmp_path / "metrics.json"

    async def fake_sync():
        return True

    mock_tracker.run_full_sync = fake_sync
    mock_tracker.calculate_portfolio_metrics = _async_return({
        "total_value_usd": 57.78, "holdings_df": pd.DataFrame(),
    })

    runner = SyncRunner(cache_path=cache_file)
    runner.start()
    async for event in runner.events():
        if event["event"] == "complete":
            break

    from api.cache import MetricsCache
    assert MetricsCache(cache_file).read()["total_value_usd"] == 57.78


@pytest.mark.asyncio
async def test_runner_emits_error_event_and_leaves_cache_untouched(mock_tracker, tmp_path):
    cache_file = tmp_path / "metrics.json"

    async def failing_sync():
        raise RuntimeError("binance unreachable")

    mock_tracker.run_full_sync = failing_sync

    runner = SyncRunner(cache_path=cache_file)
    runner.start()

    events = []
    async for event in runner.events():
        events.append(event)
        if event["event"] in ("error", "complete"):
            break

    assert events[-1]["event"] == "error"
    assert "binance unreachable" in events[-1]["message"]
    assert not cache_file.exists()


def test_post_sync_returns_409_when_already_running(mock_tracker, monkeypatch):
    from api.routes import sync as sync_route

    class AlwaysBusy:
        is_running = True

        def start(self):
            return False

    monkeypatch.setattr(sync_route, "get_sync_runner", lambda: AlwaysBusy())
    assert TestClient(app).post("/api/sync").status_code == 409


def _async_return(value):
    async def _inner():
        return value
    return _inner
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/api/test_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.sync_runner'`

- [ ] **Step 3: Enable asyncio tests**

Confirm `pyproject.toml` has, under `[tool.pytest.ini_options]`, `asyncio_mode = "auto"`. If absent, add it. If the existing suite relies on strict mode, instead keep the `@pytest.mark.asyncio` decorators already written above and leave the setting alone. Verify with `uv run pytest tests/ -q` that no existing test changes behaviour.

- [ ] **Step 4: Implement the runner**

`api/sync_runner.py`:

```python
"""Runs a full sync in the background and streams its progress.

The core already logs per-chunk progress while fetching 30-day windows.
Rather than adding callbacks to the core, this attaches a logging handler
for the duration of the run and forwards records to a queue. The core is
not modified.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Optional

from api.cache import MetricsCache
from api.deps import get_tracker

CORE_LOGGER = "crypto_portfolio_tracker"


class _QueueHandler(logging.Handler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.queue = queue
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        event = {"event": "progress", "message": record.getMessage(),
                 "level": record.levelname}
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)


class SyncRunner:
    def __init__(self, cache_path: Optional[Path] = None):
        self._cache_path = cache_path
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> bool:
        """Begin a sync. Returns False if one is already in flight."""
        if self.is_running:
            return False
        self._queue = asyncio.Queue()
        # get_running_loop, not get_event_loop: the latter is deprecated on
        # Python 3.12+ and this project runs 3.14.
        self._task = asyncio.get_running_loop().create_task(self._run())
        return True

    async def _run(self) -> None:
        tracker = get_tracker()
        cache_path = self._cache_path
        if cache_path is None:
            from api.cache import cache_path_for
            cache_path = cache_path_for(tracker.config_manager)

        logger = logging.getLogger(CORE_LOGGER)
        handler = _QueueHandler(self._queue, asyncio.get_running_loop())
        logger.addHandler(handler)
        try:
            await self._queue.put({"event": "progress", "message": "Starting sync"})
            await tracker.run_full_sync()
            metrics = await tracker.calculate_portfolio_metrics()
            MetricsCache(cache_path).write(metrics)
            await self._queue.put({
                "event": "complete",
                "message": "Sync complete",
                "total_value_usd": float(metrics.get("total_value_usd") or 0.0),
            })
        except Exception as exc:  # surfaced to the UI, never swallowed
            await self._queue.put({"event": "error", "message": str(exc)})
        finally:
            logger.removeHandler(handler)

    async def events(self) -> AsyncIterator[dict]:
        while True:
            event = await self._queue.get()
            yield event
            if event["event"] in ("complete", "error"):
                return


_runner: Optional[SyncRunner] = None


def get_sync_runner() -> SyncRunner:
    global _runner
    if _runner is None:
        _runner = SyncRunner()
    return _runner
```

`api/routes/sync.py`:

```python
"""Explicit, user-initiated sync. The only path that contacts Binance."""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.sync_runner import get_sync_runner

router = APIRouter(prefix="/api/sync", tags=["sync"])


@router.post("")
def start_sync() -> dict:
    runner = get_sync_runner()
    if not runner.start():
        raise HTTPException(status_code=409, detail="A sync is already running")
    return {"status": "started"}


@router.get("/stream")
async def stream_sync() -> StreamingResponse:
    runner = get_sync_runner()

    async def event_source():
        async for event in runner.events():
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
```

Add to `api/main.py`:

```python
from api.routes import capital, portfolio, sync

app.include_router(sync.router)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/api/ -v`
Expected: PASS.

- [ ] **Step 6: Verify the core is still untouched**

```bash
git status --porcelain src/crypto_portfolio_tracker/
uv run pytest tests/ -q
```
Expected: no output from the first; full suite green.

- [ ] **Step 7: Commit**

```bash
git add api/ tests/api/ pyproject.toml
git commit -m "feat: add sync endpoint with SSE progress streaming

Progress is surfaced by forwarding the core's existing per-chunk log
records to an event queue, so the 30-day chunk loop becomes visible
without modifying the core. A successful sync writes the metrics cache."
```

---

## Task 7: Frontend scaffold

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/index.css`
- Create: `frontend/src/lib/format.ts`, `frontend/src/lib/format.test.ts`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `formatUsd(n) -> string`, `formatSigned(n) -> string`, `formatPercent(n) -> string`, `formatQty(n) -> string`, `signOf(n) -> 'positive' | 'negative' | 'zero'`.

- [ ] **Step 1: Scaffold the project**

```bash
cd frontend 2>/dev/null || mkdir -p frontend
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss@3 postcss autoprefixer vitest
npx tailwindcss init -p
cp ../docs/design/tokens.css src/tokens.css
```

The Vite scaffold requires an empty directory, which is why Task 1 staged the tokens under `docs/design/` — the `cp` above puts them in their permanent home.

- [ ] **Step 2: Configure Tailwind against the tokens**

`frontend/tailwind.config.js`:

```js
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'surface-0': 'var(--surface-0)',
        'surface-1': 'var(--surface-1)',
        'surface-2': 'var(--surface-2)',
        border: 'var(--border)',
        positive: 'var(--positive)',
        negative: 'var(--negative)',
        warning: 'var(--warning)',
        action: 'var(--action)',
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
      },
      borderRadius: { control: '4px', panel: '6px' },
      fontFamily: {
        ui: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
};
```

`frontend/src/index.css`:

```css
@import './tokens.css';
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  background: var(--surface-0);
  color: var(--text-primary);
  font-family: var(--font-ui);
}
```

- [ ] **Step 3: Configure the dev proxy**

`frontend/vite.config.ts`:

```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: { outDir: 'dist' },
});
```

This is why no CORS configuration exists anywhere in this plan: in development the browser only ever talks to 5173, and in production only to 8000.

- [ ] **Step 4: Write the failing formatter test**

`frontend/src/lib/format.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { formatPercent, formatQty, formatSigned, formatUsd, signOf } from './format';

describe('formatSigned', () => {
  it('always carries an explicit sign so colour is not the only signal', () => {
    expect(formatSigned(18.63)).toBe('+$18.63');
    expect(formatSigned(-18.63)).toBe('-$18.63');
  });

  it('renders zero without a sign', () => {
    expect(formatSigned(0)).toBe('$0.00');
  });
});

describe('formatUsd', () => {
  it('formats to two decimals with thousands separators', () => {
    expect(formatUsd(1234.5)).toBe('$1,234.50');
  });

  it('renders null as an em dash rather than zero', () => {
    expect(formatUsd(null)).toBe('—');
  });
});

describe('formatPercent', () => {
  it('carries an explicit sign', () => {
    expect(formatPercent(-24.38)).toBe('-24.38%');
    expect(formatPercent(24.38)).toBe('+24.38%');
  });
});

describe('formatQty', () => {
  it('keeps small crypto quantities legible', () => {
    expect(formatQty(0.00012345)).toBe('0.00012345');
  });
});

describe('signOf', () => {
  it('classifies values for semantic styling', () => {
    expect(signOf(1)).toBe('positive');
    expect(signOf(-1)).toBe('negative');
    expect(signOf(0)).toBe('zero');
  });
});
```

- [ ] **Step 5: Run it to verify it fails**

Run: `cd frontend && npx vitest run src/lib/format.test.ts`
Expected: FAIL — cannot resolve `./format`.

- [ ] **Step 6: Implement the formatters**

`frontend/src/lib/format.ts`:

```ts
/**
 * Number formatting. Every signed value renders its sign explicitly so the
 * information survives colour-blindness and greyscale printing -- colour is
 * never the sole carrier of meaning.
 */

export type Sign = 'positive' | 'negative' | 'zero';

const EM_DASH = '—';

export function signOf(value: number | null | undefined): Sign {
  if (value === null || value === undefined || value === 0) return 'zero';
  return value > 0 ? 'positive' : 'negative';
}

export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return `$${Math.abs(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function formatSigned(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  const sign = value > 0 ? '+' : value < 0 ? '-' : '';
  return `${sign}${formatUsd(value)}`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  const sign = value > 0 ? '+' : value < 0 ? '-' : '';
  return `${sign}${Math.abs(value).toFixed(2)}%`;
}

export function formatQty(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return value.toLocaleString('en-US', { maximumFractionDigits: 8 });
}
```

- [ ] **Step 7: Run it to verify it passes**

Run: `cd frontend && npx vitest run`
Expected: PASS.

- [ ] **Step 8: Ignore build output**

Append to `.gitignore`:

```
frontend/node_modules/
frontend/dist/
data/api_cache/
```

- [ ] **Step 9: Commit**

```bash
git add frontend/ .gitignore
git commit -m "feat: scaffold React frontend with design tokens

Tailwind maps to the CSS custom properties rather than redefining colours,
so tokens.css stays the single source. Vite proxies /api to FastAPI in
development, which is why no CORS configuration is needed."
```

---

## Task 8: App shell, routing, and the environment banner

Spec §6: "Testnet and live data must never be presentable as the same thing." The indicator is global and unmissable.

**Files:**
- Create: `frontend/src/lib/api.ts`, `frontend/src/types.ts`
- Create: `frontend/src/components/Panel.tsx`, `frontend/src/components/Metric.tsx`, `frontend/src/components/EnvBanner.tsx`, `frontend/src/components/StalenessNote.tsx`
- Create: `frontend/src/components/EnvBanner.test.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/main.tsx`
- Modify: `frontend/package.json` (add `react-router-dom`, testing libraries)

**Interfaces:**
- Consumes: `format.ts` (Task 7); the response shapes of Tasks 4–6.
- Produces: `apiGet<T>(path) -> Promise<T>`, `apiPost<T>(path) -> Promise<T>`, components `Panel`, `Metric`, `EnvBanner`, `StalenessNote`.

- [ ] **Step 1: Install routing and test libraries**

```bash
cd frontend
npm install react-router-dom
npm install -D @testing-library/react @testing-library/jest-dom jsdom
```

Add to `vite.config.ts` inside `defineConfig({...})`:

```ts
  test: { environment: 'jsdom', globals: true },
```

- [ ] **Step 2: Define the types mirroring the Pydantic schemas**

`frontend/src/types.ts`:

```ts
export interface Staleness {
  cached_at: string | null;
  age_seconds: number | null;
  is_stale: boolean;
}

export interface Environment {
  is_testnet: boolean;
  database_path: string;
  label: string;
}

export interface AccountingBasis {
  label: string;
  question: string;
  basis_usd: number;
  pl_usd: number;
  pl_percent: number;
}

export interface Holding {
  symbol: string;
  total_quantity: number;
  spot_quantity: number | null;
  earn_quantity: number | null;
  current_price: number | null;
  value_usd: number | null;
  average_cost_basis: number | null;
  cost_basis_total: number | null;
  unrealized_pl_usd: number | null;
  unrealized_pl_percent: number | null;
  is_core: boolean;
}

export interface CockpitResponse {
  total_value_usd: number;
  net_invested: AccountingBasis;
  fifo: AccountingBasis;
  holdings: Holding[];
  staleness: Staleness;
  environment: Environment;
  has_data: boolean;
}

export interface CapitalFlowRow {
  source: string;
  type: string;
  direction: 'in' | 'out';
  quantity: number;
  price_usd: number;
  value_usd: number;
  provenance: 'computed' | 'usdt_peg_fallback' | 'failed_lookup';
  is_suspect: boolean;
}

export interface CapitalFlowResponse {
  rows: CapitalFlowRow[];
  total_in_usd: number;
  total_out_usd: number;
  net_invested_usd: number;
  suspect_count: number;
}
```

These are hand-written for now. Spec §10 calls for generating them from the OpenAPI schema; that is added in a later plan once the endpoint surface stops changing daily. Until then, a schema change requires editing this file — note it in the commit if you change a Pydantic model.

- [ ] **Step 3: Write the API client**

`frontend/src/lib/api.ts`:

```ts
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new ApiError(response.status, detail || response.statusText);
  }
  return (await response.json()) as T;
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path);
}

export function apiPost<T>(path: string): Promise<T> {
  return request<T>(path, { method: 'POST' });
}
```

- [ ] **Step 4: Write the failing banner test**

`frontend/src/components/EnvBanner.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { EnvBanner } from './EnvBanner';

describe('EnvBanner', () => {
  it('names the environment in text, not colour alone', () => {
    render(<EnvBanner environment={{
      is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET',
    }} />);
    expect(screen.getByText('TESTNET')).toBeDefined();
  });

  it('shows which database is in use so the two are never confused', () => {
    render(<EnvBanner environment={{
      is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET',
    }} />);
    expect(screen.getByText(/testnet_portfolio\.db/)).toBeDefined();
  });

  it('renders in the live case too, never absent', () => {
    render(<EnvBanner environment={{
      is_testnet: false, database_path: 'data/portfolio.db', label: 'LIVE',
    }} />);
    expect(screen.getByText('LIVE')).toBeDefined();
  });
});
```

Add `import '@testing-library/jest-dom';` to a new `frontend/src/setupTests.ts` and reference it from `vite.config.ts` with `test: { environment: 'jsdom', globals: true, setupFiles: './src/setupTests.ts' }`.

- [ ] **Step 5: Run it to verify it fails**

Run: `cd frontend && npx vitest run src/components/EnvBanner.test.tsx`
Expected: FAIL — cannot resolve `./EnvBanner`.

- [ ] **Step 6: Implement the shared components**

`frontend/src/components/EnvBanner.tsx`:

```tsx
import type { Environment } from '../types';

/**
 * Always rendered, in both environments. Testnet uses the warning colour AND
 * the word TESTNET AND the database filename -- three independent signals,
 * because presenting testnet figures as live is the worst failure this UI has.
 */
export function EnvBanner({ environment }: { environment: Environment }) {
  const isTestnet = environment.is_testnet;
  return (
    <div
      className="flex items-center gap-3 border-b px-3 py-1 font-mono text-xs"
      style={{
        borderColor: 'var(--border)',
        background: isTestnet ? 'var(--warning)' : 'var(--surface-1)',
        color: isTestnet ? 'var(--surface-0)' : 'var(--text-secondary)',
      }}
    >
      <span className="font-bold tracking-wider">{environment.label}</span>
      <span>{environment.database_path}</span>
    </div>
  );
}
```

`frontend/src/components/Panel.tsx`:

```tsx
import type { ReactNode } from 'react';

/** Depth is tonal layering plus a 1px border. No shadows. */
export function Panel({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section
      className="rounded-panel border p-3"
      style={{ background: 'var(--surface-1)', borderColor: 'var(--border)' }}
    >
      {title && (
        <h2
          className="mb-2 font-ui text-xs uppercase tracking-wider"
          style={{ color: 'var(--text-secondary)' }}
        >
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}
```

`frontend/src/components/Metric.tsx`:

```tsx
import { signOf } from '../lib/format';

const COLOUR: Record<string, string> = {
  positive: 'var(--positive)',
  negative: 'var(--negative)',
  zero: 'var(--text-primary)',
};

/**
 * `value` must already carry its sign (use formatSigned/formatPercent).
 * `signal` drives colour only; it never carries meaning by itself.
 */
export function Metric({
  label, value, signal, sub,
}: {
  label: string;
  value: string;
  signal?: number | null;
  sub?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="font-ui text-xs uppercase tracking-wider"
            style={{ color: 'var(--text-secondary)' }}>
        {label}
      </span>
      <span className="font-mono text-2xl tabular-nums"
            style={{ color: signal === undefined ? 'var(--text-primary)' : COLOUR[signOf(signal)] }}>
        {value}
      </span>
      {sub && (
        <span className="font-mono text-xs" style={{ color: 'var(--text-secondary)' }}>
          {sub}
        </span>
      )}
    </div>
  );
}
```

`frontend/src/components/StalenessNote.tsx`:

```tsx
import type { Staleness } from '../types';

/** Staleness is displayed, never hidden behind a spinner. */
export function StalenessNote({ staleness }: { staleness: Staleness }) {
  if (staleness.age_seconds === null) {
    return (
      <span className="font-mono text-xs" style={{ color: 'var(--warning)' }}>
        never synced
      </span>
    );
  }
  const minutes = Math.round(staleness.age_seconds / 60);
  const text = minutes < 1 ? 'synced just now' : `synced ${minutes}m ago`;
  return (
    <span
      className="font-mono text-xs"
      style={{ color: staleness.is_stale ? 'var(--warning)' : 'var(--text-secondary)' }}
    >
      {text}
    </span>
  );
}
```

- [ ] **Step 7: Build the shell**

`frontend/src/App.tsx`:

```tsx
import { NavLink, Route, Routes } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { EnvBanner } from './components/EnvBanner';
import { apiGet } from './lib/api';
import type { CockpitResponse, Environment } from './types';
import { Cockpit } from './screens/Cockpit';
import { CapitalFlow } from './screens/CapitalFlow';
import { Sync } from './screens/Sync';

const NAV = [
  { to: '/', label: 'Cockpit' },
  { to: '/capital', label: 'Capital Flow' },
  { to: '/sync', label: 'Sync' },
];

export default function App() {
  const [environment, setEnvironment] = useState<Environment | null>(null);

  useEffect(() => {
    apiGet<CockpitResponse>('/api/portfolio/cockpit')
      .then((data) => setEnvironment(data.environment))
      .catch(() => setEnvironment(null));
  }, []);

  return (
    <div className="min-h-screen" style={{ background: 'var(--surface-0)' }}>
      {environment && <EnvBanner environment={environment} />}
      <div className="flex">
        <nav className="flex w-48 shrink-0 flex-col gap-1 border-r p-3"
             style={{ borderColor: 'var(--border)' }}>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className="px-2 py-1 font-ui text-sm"
              style={({ isActive }) => ({
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                borderLeft: `2px solid ${isActive ? 'var(--action)' : 'transparent'}`,
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <main className="flex-1 p-4">
          <Routes>
            <Route path="/" element={<Cockpit />} />
            <Route path="/capital" element={<CapitalFlow />} />
            <Route path="/sync" element={<Sync />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
```

`frontend/src/main.tsx`:

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
```

The three screen modules do not exist yet, so the build fails until Tasks 9–11. Create placeholder files now so the shell compiles:

`frontend/src/screens/Cockpit.tsx`, `CapitalFlow.tsx`, `Sync.tsx`, each:

```tsx
export function Cockpit() {
  return null;
}
```

(with the matching export name per file).

- [ ] **Step 8: Run tests and typecheck**

Run: `cd frontend && npx vitest run && npx tsc --noEmit`
Expected: tests PASS, no type errors.

- [ ] **Step 9: Commit**

```bash
git add frontend/
git commit -m "feat: add app shell, routing, and global environment banner

The testnet indicator carries three independent signals -- colour, the
word TESTNET, and the database filename -- since presenting testnet
figures as live is the most damaging failure available to this UI."
```

---

## Task 9: Cockpit screen

The signature screen. Spec §8.1 and §8.3.

**Files:**
- Modify: `frontend/src/screens/Cockpit.tsx`
- Create: `frontend/src/screens/Cockpit.test.tsx`
- Create: `frontend/src/components/HoldingsTable.tsx`

**Interfaces:**
- Consumes: `apiGet`, `Panel`, `Metric`, `StalenessNote`, formatters, `CockpitResponse`.

- [ ] **Step 1: Write the failing test**

`frontend/src/screens/Cockpit.test.tsx`:

```tsx
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Cockpit } from './Cockpit';
import type { CockpitResponse } from '../types';

const POPULATED: CockpitResponse = {
  total_value_usd: 57.78,
  net_invested: {
    label: 'NET INVESTED BASIS', question: 'did I make money?',
    basis_usd: 76.41, pl_usd: -18.63, pl_percent: -24.38,
  },
  fifo: {
    label: 'FIFO BASIS', question: 'are my holdings underwater?',
    basis_usd: 199.75, pl_usd: -141.97, pl_percent: -71.07,
  },
  holdings: [{
    symbol: 'BTC', total_quantity: 0.001, spot_quantity: 0.001, earn_quantity: 0,
    current_price: 57780, value_usd: 57.78, average_cost_basis: 100,
    cost_basis_total: 199.75, unrealized_pl_usd: -141.97,
    unrealized_pl_percent: -71.07, is_core: true,
  }],
  staleness: { cached_at: '2026-07-21T09:30:00', age_seconds: 120, is_stale: false },
  environment: { is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET' },
  has_data: true,
};

const EMPTY: CockpitResponse = {
  ...POPULATED,
  total_value_usd: 0,
  holdings: [],
  staleness: { cached_at: null, age_seconds: null, is_stale: true },
  has_data: false,
};

function mockFetch(payload: CockpitResponse) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true, json: async () => payload,
  }));
}

beforeEach(() => vi.unstubAllGlobals());

describe('Cockpit populated state', () => {
  it('renders both accounting bases with different values', async () => {
    mockFetch(POPULATED);
    render(<Cockpit />);

    // Regex, not exact strings: each basis renders its P/L and percent in a
    // single span, e.g. "-$18.63  (-24.38%)".
    await waitFor(() => expect(screen.getByText('$57.78')).toBeDefined());
    expect(screen.getByText(/-\$18\.63/)).toBeDefined();
    expect(screen.getByText(/-\$141\.97/)).toBeDefined();
  });

  it('labels each basis with the question it answers', async () => {
    mockFetch(POPULATED);
    render(<Cockpit />);

    await waitFor(() => expect(screen.getByText('did I make money?')).toBeDefined());
    expect(screen.getByText('are my holdings underwater?')).toBeDefined();
  });

  it('shows each basis denominator so the two are visibly different', async () => {
    mockFetch(POPULATED);
    render(<Cockpit />);

    await waitFor(() => expect(screen.getByText(/76\.41 net in/)).toBeDefined());
    expect(screen.getByText(/199\.75 cost basis/)).toBeDefined();
  });
});

describe('Cockpit constrained state', () => {
  it('states plainly that no sync has run rather than showing zeros', async () => {
    mockFetch(EMPTY);
    render(<Cockpit />);

    await waitFor(() => expect(screen.getByText(/no data yet/i)).toBeDefined());
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd frontend && npx vitest run src/screens/Cockpit.test.tsx`
Expected: FAIL — the placeholder renders `null`, so no text is found.

- [ ] **Step 3: Implement the holdings table**

`frontend/src/components/HoldingsTable.tsx`:

```tsx
import { formatPercent, formatQty, formatSigned, formatUsd, signOf } from '../lib/format';
import type { Holding } from '../types';

const DUST_THRESHOLD_USD = 0.4;

const COLOUR: Record<string, string> = {
  positive: 'var(--positive)',
  negative: 'var(--negative)',
  zero: 'var(--text-primary)',
};

/**
 * Dust collapses into one aggregate row rather than presenting sub-$0.40
 * positions as meaningful allocations.
 */
export function HoldingsTable({ holdings }: { holdings: Holding[] }) {
  const material = holdings.filter((h) => (h.value_usd ?? 0) >= DUST_THRESHOLD_USD);
  const dust = holdings.filter((h) => (h.value_usd ?? 0) < DUST_THRESHOLD_USD);
  const dustValue = dust.reduce((sum, h) => sum + (h.value_usd ?? 0), 0);

  if (holdings.length === 0) {
    return (
      <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
        No holdings recorded.
      </p>
    );
  }

  return (
    <table className="w-full font-mono text-sm tabular-nums">
      <thead>
        <tr style={{ color: 'var(--text-secondary)' }}>
          <th className="text-left font-normal">Asset</th>
          <th className="text-right font-normal">Quantity</th>
          <th className="text-right font-normal">Price</th>
          <th className="text-right font-normal">Value</th>
          <th className="text-right font-normal">Unrealized</th>
        </tr>
      </thead>
      <tbody>
        {material.map((h) => (
          <tr key={h.symbol} className="border-t" style={{ borderColor: 'var(--border)' }}>
            <td className="text-left">{h.symbol}</td>
            <td className="text-right">{formatQty(h.total_quantity)}</td>
            <td className="text-right">{formatUsd(h.current_price)}</td>
            <td className="text-right">{formatUsd(h.value_usd)}</td>
            <td className="text-right" style={{ color: COLOUR[signOf(h.unrealized_pl_usd)] }}>
              {formatSigned(h.unrealized_pl_usd)} ({formatPercent(h.unrealized_pl_percent)})
            </td>
          </tr>
        ))}
        {dust.length > 0 && (
          <tr className="border-t" style={{ borderColor: 'var(--border)',
                                            color: 'var(--text-secondary)' }}>
            <td className="text-left">{dust.length} dust positions</td>
            <td className="text-right">—</td>
            <td className="text-right">—</td>
            <td className="text-right">{formatUsd(dustValue)}</td>
            <td className="text-right">—</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 4: Implement the Cockpit**

`frontend/src/screens/Cockpit.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { HoldingsTable } from '../components/HoldingsTable';
import { Metric } from '../components/Metric';
import { Panel } from '../components/Panel';
import { StalenessNote } from '../components/StalenessNote';
import { apiGet } from '../lib/api';
import { formatPercent, formatSigned, formatUsd } from '../lib/format';
import type { AccountingBasis, CockpitResponse } from '../types';

function BasisBlock({ basis, denominator }: { basis: AccountingBasis; denominator: string }) {
  return (
    <div className="flex flex-col gap-1">
      <Metric
        label={basis.label}
        value={`${formatSigned(basis.pl_usd)}  (${formatPercent(basis.pl_percent)})`}
        signal={basis.pl_usd}
        sub={denominator}
      />
      <span className="font-ui text-xs italic" style={{ color: 'var(--text-secondary)' }}>
        {basis.question}
      </span>
    </div>
  );
}

export function Cockpit() {
  const [data, setData] = useState<CockpitResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<CockpitResponse>('/api/portfolio/cockpit')
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return <p style={{ color: 'var(--negative)' }}>{error}</p>;
  }
  if (!data) {
    return <p style={{ color: 'var(--text-secondary)' }}>Loading…</p>;
  }

  if (!data.has_data) {
    return (
      <Panel title="Cockpit">
        <p className="font-ui text-sm" style={{ color: 'var(--warning)' }}>
          No data yet — run a sync to populate the portfolio.
        </p>
      </Panel>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <Panel>
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-4xl tabular-nums">
            {formatUsd(data.total_value_usd)}
          </span>
          <span className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
            portfolio value
          </span>
          <span className="ml-auto">
            <StalenessNote staleness={data.staleness} />
          </span>
        </div>

        {/*
          Both bases, side by side, each with its denominator and its question.
          They are computed from different sources and routinely differ several
          fold; rendering them as one number would be a lie.
        */}
        <div className="mt-4 grid grid-cols-2 gap-8">
          <BasisBlock
            basis={data.net_invested}
            denominator={`on ${formatUsd(data.net_invested.basis_usd)} net in`}
          />
          <BasisBlock
            basis={data.fifo}
            denominator={`on ${formatUsd(data.fifo.basis_usd)} cost basis`}
          />
        </div>
      </Panel>

      <Panel title="Holdings">
        <HoldingsTable holdings={data.holdings} />
      </Panel>
    </div>
  );
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run && npx tsc --noEmit`
Expected: PASS, no type errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: add Cockpit screen with dual accounting headline

Both bases render side by side with their denominators and the question
each answers. Tests assert the two values are distinct and that the
never-synced state says so rather than displaying zeros."
```

---

## Task 10: Capital Flow screen

**Files:**
- Modify: `frontend/src/screens/CapitalFlow.tsx`
- Create: `frontend/src/screens/CapitalFlow.test.tsx`

- [ ] **Step 1: Write the failing test**

`frontend/src/screens/CapitalFlow.test.tsx`:

```tsx
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { CapitalFlow } from './CapitalFlow';
import type { CapitalFlowResponse } from '../types';

const RESPONSE: CapitalFlowResponse = {
  rows: [
    { source: 'Binance P2P Buy', type: 'BUY', direction: 'in', quantity: 100,
      price_usd: 0.0179, value_usd: 1.79, provenance: 'computed', is_suspect: false },
    { source: 'Binance P2P Buy', type: 'BUY', direction: 'in', quantity: 50,
      price_usd: 0, value_usd: 0, provenance: 'failed_lookup', is_suspect: true },
  ],
  total_in_usd: 1.79,
  total_out_usd: 0,
  net_invested_usd: 1.79,
  suspect_count: 1,
};

function mockFetch(payload: CapitalFlowResponse) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => payload }));
}

beforeEach(() => vi.unstubAllGlobals());

describe('CapitalFlow', () => {
  it('warns when any row has suspect provenance', async () => {
    mockFetch(RESPONSE);
    render(<CapitalFlow />);
    await waitFor(() =>
      expect(screen.getByText(/1 row.*could not be priced|suspect/i)).toBeDefined());
  });

  it('labels the provenance of each row in text', async () => {
    mockFetch(RESPONSE);
    render(<CapitalFlow />);
    await waitFor(() => expect(screen.getByText('failed lookup')).toBeDefined());
    expect(screen.getByText('computed')).toBeDefined();
  });

  it('renders an explicit empty state', async () => {
    mockFetch({ rows: [], total_in_usd: 0, total_out_usd: 0,
                net_invested_usd: 0, suspect_count: 0 });
    render(<CapitalFlow />);
    await waitFor(() => expect(screen.getByText(/no capital flow/i)).toBeDefined());
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd frontend && npx vitest run src/screens/CapitalFlow.test.tsx`
Expected: FAIL — placeholder renders `null`.

- [ ] **Step 3: Implement the screen**

`frontend/src/screens/CapitalFlow.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { Metric } from '../components/Metric';
import { Panel } from '../components/Panel';
import { apiGet } from '../lib/api';
import { formatQty, formatUsd } from '../lib/format';
import type { CapitalFlowResponse } from '../types';

const PROVENANCE_LABEL: Record<string, string> = {
  computed: 'computed',
  usdt_peg_fallback: 'USDT peg fallback',
  failed_lookup: 'failed lookup',
};

export function CapitalFlow() {
  const [data, setData] = useState<CapitalFlowResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<CapitalFlowResponse>('/api/capital/flow')
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <p style={{ color: 'var(--negative)' }}>{error}</p>;
  if (!data) return <p style={{ color: 'var(--text-secondary)' }}>Loading…</p>;

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Capital flow">
        <div className="grid grid-cols-3 gap-8">
          <Metric label="Total in" value={formatUsd(data.total_in_usd)} />
          <Metric label="Total out" value={formatUsd(data.total_out_usd)} />
          <Metric label="Net invested" value={formatUsd(data.net_invested_usd)} />
        </div>
        {data.suspect_count > 0 && (
          <p className="mt-3 font-ui text-sm" style={{ color: 'var(--warning)' }}>
            {data.suspect_count} row{data.suspect_count === 1 ? '' : 's'} could not be
            priced from a real exchange rate. Net invested may understate actual inflow.
          </p>
        )}
      </Panel>

      <Panel title="Transactions">
        {data.rows.length === 0 ? (
          <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
            No capital flow recorded yet.
          </p>
        ) : (
          <table className="w-full font-mono text-sm tabular-nums">
            <thead>
              <tr style={{ color: 'var(--text-secondary)' }}>
                <th className="text-left font-normal">Source</th>
                <th className="text-left font-normal">Dir</th>
                <th className="text-right font-normal">Quantity</th>
                <th className="text-right font-normal">Rate</th>
                <th className="text-right font-normal">Value</th>
                <th className="text-left font-normal">Provenance</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, index) => (
                <tr key={index} className="border-t" style={{ borderColor: 'var(--border)' }}>
                  <td className="text-left">{row.source}</td>
                  <td className="text-left">{row.direction === 'in' ? '+ in' : '- out'}</td>
                  <td className="text-right">{formatQty(row.quantity)}</td>
                  <td className="text-right">{formatQty(row.price_usd)}</td>
                  <td className="text-right">{formatUsd(row.value_usd)}</td>
                  <td className="text-left"
                      style={{ color: row.is_suspect ? 'var(--warning)'
                                                     : 'var(--text-secondary)' }}>
                    {PROVENANCE_LABEL[row.provenance]}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run && npx tsc --noEmit`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: add Capital Flow screen with provenance column"
```

---

## Task 11: Sync screen

**Files:**
- Modify: `frontend/src/screens/Sync.tsx`
- Create: `frontend/src/screens/Sync.test.tsx`

- [ ] **Step 1: Write the failing test**

`frontend/src/screens/Sync.test.tsx`:

```tsx
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Sync } from './Sync';

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  onmessage: ((event: { data: string }) => void) | null = null;
  closed = false;

  constructor(public url: string) {
    FakeEventSource.instances.push(this);
  }

  close() {
    this.closed = true;
  }

  emit(payload: object) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }
}

beforeEach(() => {
  vi.unstubAllGlobals();
  FakeEventSource.instances = [];
  vi.stubGlobal('EventSource', FakeEventSource);
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true, json: async () => ({ status: 'started' }),
  }));
});

describe('Sync', () => {
  it('streams progress lines from the core rather than a blank spinner', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({ event: 'progress', message: 'Fetching chunk 1 of 3' });

    await waitFor(() => expect(screen.getByText('Fetching chunk 1 of 3')).toBeDefined());
  });

  it('surfaces errors instead of appearing to succeed', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({ event: 'error', message: 'binance unreachable' });

    await waitFor(() => expect(screen.getByText(/binance unreachable/)).toBeDefined());
  });

  it('closes the stream once complete', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({ event: 'complete', message: 'Sync complete' });

    await waitFor(() => expect(FakeEventSource.instances[0].closed).toBe(true));
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd frontend && npx vitest run src/screens/Sync.test.tsx`
Expected: FAIL — no button rendered.

- [ ] **Step 3: Implement the screen**

`frontend/src/screens/Sync.tsx`:

```tsx
import { useRef, useState } from 'react';
import { Panel } from '../components/Panel';
import { apiPost } from '../lib/api';

interface SyncEvent {
  event: 'progress' | 'complete' | 'error';
  message: string;
}

export function Sync() {
  const [events, setEvents] = useState<SyncEvent[]>([]);
  const [running, setRunning] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  async function start() {
    setEvents([]);
    setRunning(true);
    try {
      await apiPost('/api/sync');
    } catch (e) {
      setEvents([{ event: 'error', message: String(e) }]);
      setRunning(false);
      return;
    }

    const source = new EventSource('/api/sync/stream');
    sourceRef.current = source;
    source.onmessage = (message) => {
      const parsed: SyncEvent = JSON.parse(message.data);
      setEvents((previous) => [...previous, parsed]);
      if (parsed.event === 'complete' || parsed.event === 'error') {
        source.close();
        setRunning(false);
      }
    };
  }

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Sync">
        <p className="mb-3 font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
          Sync is the only action that contacts Binance. Everything else reads
          local data.
        </p>
        <button
          onClick={start}
          disabled={running}
          className="rounded-control px-3 py-1 font-ui text-sm"
          style={{
            background: running ? 'var(--surface-2)' : 'var(--action)',
            color: 'var(--text-primary)',
            cursor: running ? 'not-allowed' : 'pointer',
          }}
        >
          {running ? 'Syncing…' : 'Start sync'}
        </button>
      </Panel>

      {events.length > 0 && (
        <Panel title="Progress">
          <ul className="flex flex-col gap-1 font-mono text-xs">
            {events.map((event, index) => (
              <li
                key={index}
                style={{
                  color: event.event === 'error' ? 'var(--negative)'
                       : event.event === 'complete' ? 'var(--positive)'
                       : 'var(--text-secondary)',
                }}
              >
                {event.event === 'error' ? `error: ${event.message}` : event.message}
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run && npx tsc --noEmit`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: add Sync screen streaming per-chunk progress"
```

---

## Task 12: Production serving and single entry point

**Files:**
- Modify: `api/main.py`
- Create: `run_ui.py`
- Create: `tests/api/test_static_serving.py`
- Modify: `CONTEXT.md`

- [ ] **Step 1: Write the failing test**

`tests/api/test_static_serving.py`:

```python
from fastapi.testclient import TestClient

from api.main import app


def test_api_routes_still_resolve_when_frontend_absent():
    """The SPA catch-all must never shadow /api paths."""
    assert TestClient(app).get("/api/health").json() == {"status": "ok"}


def test_unknown_api_path_returns_404_not_the_spa():
    assert TestClient(app).get("/api/does-not-exist").status_code == 404
```

- [ ] **Step 2: Run it to verify the second test fails**

Run: `uv run pytest tests/api/test_static_serving.py -v`
Expected: the first PASSes; the second may pass now but must keep passing after Step 3 — that is the point of writing it first.

- [ ] **Step 3: Serve the built frontend**

Replace `api/main.py`:

```python
"""FastAPI application serving the portfolio API and the built frontend.

One process, one port. In development Vite proxies /api here; in production
this serves the built bundle too. Neither arrangement needs CORS.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import capital, portfolio, sync

app = FastAPI(title="Crypto Portfolio Tracker API", version="1.0.0")

app.include_router(portfolio.router)
app.include_router(capital.router)
app.include_router(sync.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        # Registered after the routers, so /api/* has already matched. An
        # unmatched /api path must 404 rather than silently return the SPA.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(FRONTEND_DIST / "index.html")
```

- [ ] **Step 4: Add the entry point**

`run_ui.py`:

```python
"""Start the React UI. Build the frontend first with: npm --prefix frontend run build"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
```

Do **not** add a `[project.scripts]` entry for this. The hatch build config packages only `src/crypto_portfolio_tracker`, so `api` is not installed into the wheel and a console script pointing at `api.main` would fail on any installed copy. `run_ui.py` at the repo root is the entry point; the existing `cpt` and `track-portfolio-cli` scripts are unaffected.

- [ ] **Step 5: Build and verify end to end**

```bash
npm --prefix frontend run build
uv run pytest tests/ -q
uv run python run_ui.py &
sleep 3
curl -s http://127.0.0.1:8000/api/health
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/nope
kill %1
```

Expected: `{"status":"ok"}`, then `200`, then `404`.

Do not claim this task complete without pasting the actual output of these commands. If the build fails or a status differs, stop and investigate rather than adjusting the expectation.

- [ ] **Step 6: Update CONTEXT.md**

Add a section documenting: the `api/` + `frontend/` architecture; that the core is wrapped and unmodified; that reads serve from `data/api_cache/metrics_{env}.json` and only sync contacts Binance; that the cache is per-environment because testnet uses a separate database; and the two run commands (dev: `uv run python run_ui.py` plus `npm --prefix frontend run dev`; prod: build then `run_ui.py`).

Update only the affected sections. Do not append a session log.

- [ ] **Step 7: Commit**

```bash
git add api/ run_ui.py tests/api/ CONTEXT.md
git commit -m "feat: serve built frontend from FastAPI on a single port

The SPA catch-all is registered after the API routers and explicitly
404s unmatched /api paths, so a typo in a frontend fetch surfaces as an
error rather than as silently-returned HTML."
```

---

## Verification checklist for the whole plan

Run before declaring the plan complete. Paste real output; do not assert from memory.

- [ ] `uv run pytest tests/ -q` — full suite green, including the pre-existing tests
- [ ] `git status --porcelain src/crypto_portfolio_tracker/` — empty; the core was never modified
- [ ] `grep -rl "import streamlit" src/crypto_portfolio_tracker/*.py | wc -l` — unchanged from before this plan
- [ ] `cd frontend && npx vitest run && npx tsc --noEmit` — green, no type errors
- [ ] `uv run ruff check api/ tests/api/` — clean
- [ ] Manual: Cockpit shows two visibly different accounting bases against real data
- [ ] Manual: the environment banner reads TESTNET and names the testnet database
- [ ] Manual: a sync streams individual chunk lines, not an indeterminate spinner
- [ ] Manual: with `data/api_cache/` deleted, the Cockpit says "no data yet" rather than showing $0.00 as though it were real

## Deferred to later plans

- Screens 3–5, 7–15 (spec §9 steps 5–8)
- OpenAPI-generated TypeScript types replacing the hand-written `types.ts` (spec §10)
- The parity gate against each Streamlit page (spec §10)
- Charting (spec §4 defers the `visualizations.py` port; charts are rebuilt client-side when the first charting screen lands)
