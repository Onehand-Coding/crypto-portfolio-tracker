# Completion Plan API + React Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only no-sell completion plan to the API (`GET /api/strategy/completion`) and the React DCA screen (collapsed by default).

**Architecture:** The endpoint mirrors `dca_preview` in `api/routes/strategy.py` (offline compute over the metrics cache + `target_allocation`, same `num`/`opt` null-rules) but takes no input and returns the implied total as output. The React section mirrors the DCA preview table, fetches on mount via `useApi`, and hides behind a `useState` toggle (no disclosure primitive exists in `components/`).

**Tech Stack:** FastAPI + Pydantic v2, pytest + TestClient, React 19 + TS, Vitest + Testing Library.

**Semantics (locked, from the approved spec):** `T = max(current/weight)` over holdings with known value > 0 and weight > 0; `need = max(0, T*w - current)`; total = `sum(needs)`. Worked example: BTC 146.49 @ 0.35 + ETH 24.49 @ 0.30 (SOL .10, RENDER .06, TAO .06, AVAX .05, LINK .05, ONDO .03 at zero) → anchor BTC, T = 418.54, ETH need = 101.07, total = 247.56. Absent holding = known 0.0; present-but-unpriced (null/NaN) = unknown (null in API, em dash in UI), excluded from anchors; empty anchors → invalid with plain message; every target asset renders a row including $0 anchors; no dust filter; no sells ever.

---

### Task 1: `GET /api/strategy/completion` + schemas + tests

**Files:**
- Modify: `api/schemas/screens.py:186` (append after `DcaPreviewResponse`)
- Modify: `api/routes/strategy.py:253` (append endpoint + helper after `dca_preview`/`_allocation`; import the new schemas)
- Create: `tests/api/test_completion_route.py`
- Modify: `docs/superpowers/specs/2026-09-04-completion-plan-design.md` (addendum: React/API approved)

- [ ] **Step 1: Record the addendum in the spec**

Append to the end of `docs/superpowers/specs/2026-09-04-completion-plan-design.md`:
```markdown
## Addendum 2026-09-04 (parity branch)

The deferred half is approved: `GET /api/strategy/completion` (offline,
input-less, mirroring `dca_preview`'s null-rules) plus a collapsed-by-default
section on the React DCA screen. Same semantics as §2; unknown renders as
`null` in the API and an em dash in the UI.
```

- [ ] **Step 2: Add the schemas**

Append after `DcaPreviewResponse` (`api/schemas/screens.py:186`):
```python
class CompletionRow(BaseModel):
    symbol: str
    target_allocation_pct: float
    target_value_usd: float
    # None, not zero, when the holding exists but could not be priced.
    current_value_usd: Optional[float] = None
    need_usd: float


class CompletionResponse(BaseModel):
    """No-sell completion plan. Computed on read, so like DcaPreviewResponse
    it carries its own validity instead of AnalysisState staleness."""

    valid: bool
    message: Optional[str] = None
    anchor_symbol: Optional[str] = None
    implied_total_usd: Optional[float] = None
    additional_total_usd: float = 0.0
    rows: list[CompletionRow] = []
```
`Optional` is already imported in that file (used by `DcaAllocation`).

- [ ] **Step 3: Write the failing test**

Create `tests/api/test_completion_route.py`:
```python
"""GET /api/strategy/completion: the no-sell finish-to-targets plan.

Same math as the CLI/Streamlit surfaces, served offline from the metrics
cache. Worked example (spec §2): BTC 146.49 @ 0.35 + ETH 24.49 @ 0.30 →
anchor BTC, implied total 418.54, ETH need 101.07, additional total 247.56.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

TARGET = {
    "BTC": 0.35, "ETH": 0.30, "SOL": 0.10, "RENDER": 0.06,
    "TAO": 0.06, "AVAX": 0.05, "LINK": 0.05, "ONDO": 0.03,
}

HOLDINGS = [
    {"symbol": "BTC", "value_usd": 146.49, "current_price": 95000.0},
    {"symbol": "ETH", "value_usd": 24.49, "current_price": 3200.0},
]


@pytest.fixture
def completion_setup(mock_read_context, tmp_path, monkeypatch):
    """Seed config + metrics cache the route reads, isolated to tmp_path."""
    mock_read_context.config_manager.is_testnet_mode = True
    mock_read_context.config_manager.config = {"target_allocation": dict(TARGET)}
    monkeypatch.chdir(tmp_path)
    path = Path("data") / "api_cache" / "metrics_testnet.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def seed(holdings):
        path.write_text(json.dumps({"holdings_df": holdings, "_cached_at": 0}))

    seed(HOLDINGS)
    return seed


def test_worked_example(completion_setup):
    body = TestClient(app).get("/api/strategy/completion").json()
    assert body["valid"] is True
    assert body["anchor_symbol"] == "BTC"
    assert body["implied_total_usd"] == pytest.approx(418.54, rel=1e-4)
    by_symbol = {r["symbol"]: r for r in body["rows"]}
    assert by_symbol["ETH"]["need_usd"] == pytest.approx(101.07, rel=1e-4)
    assert by_symbol["BTC"]["need_usd"] == pytest.approx(0.0, abs=0.01)
    assert body["additional_total_usd"] == pytest.approx(247.56, rel=1e-4)
    assert len(body["rows"]) == len(TARGET)


def test_empty_portfolio_has_no_anchor(completion_setup):
    completion_setup([])
    body = TestClient(app).get("/api/strategy/completion").json()
    assert body["valid"] is False
    assert "anchor" in (body["message"] or "").lower()


def test_unpriced_holding_is_null_not_zero(completion_setup):
    """A present-but-unpriced holding must not anchor and must not read $0."""
    completion_setup([
        {"symbol": "BTC", "value_usd": 146.49, "current_price": 95000.0},
        {"symbol": "ETH", "value_usd": None, "current_price": None},
    ])
    body = TestClient(app).get("/api/strategy/completion").json()
    by_symbol = {r["symbol"]: r for r in body["rows"]}
    assert by_symbol["ETH"]["current_value_usd"] is None
    assert body["anchor_symbol"] == "BTC"


def test_no_target_allocation(completion_setup, mock_read_context):
    mock_read_context.config_manager.config = {"target_allocation": {}}
    body = TestClient(app).get("/api/strategy/completion").json()
    assert body["valid"] is False


def test_zero_weight_asset_never_anchors(completion_setup, mock_read_context):
    mock_read_context.config_manager.config = {
        "target_allocation": {"BTC": 1.0, "DUST": 0.0}
    }
    completion_setup([
        {"symbol": "BTC", "value_usd": 146.49, "current_price": 95000.0},
        {"symbol": "DUST", "value_usd": 50.0, "current_price": 1.0},
    ])
    body = TestClient(app).get("/api/strategy/completion").json()
    assert body["anchor_symbol"] == "BTC"
    assert body["implied_total_usd"] == pytest.approx(146.49, rel=1e-4)
    by_symbol = {r["symbol"]: r for r in body["rows"]}
    assert by_symbol["DUST"]["need_usd"] == pytest.approx(0.0, abs=0.01)
```
Note: `mock_read_context` is function-scoped (fresh `Mock` per test), so reassigning `.config` per test is isolated. `cache_path_for` keys off `is_testnet_mode` → `metrics_testnet.json` under the monkeypatched cwd, mirroring `test_strategy_routes.py`'s `cached_rebalance` pattern.

- [ ] **Step 4: Run the new test to verify it fails**

Run: `uv run pytest tests/api/test_completion_route.py -q`
Expected: FAIL — `test_completion_route.py` errors on collection or 404s, because `GET /api/strategy/completion` does not exist yet.

- [ ] **Step 5: Implement the endpoint**

In `api/routes/strategy.py`, extend the schema import (line 16-27) with `CompletionResponse` and `CompletionRow`, then append after `_allocation` (line 268):
```python
@router.get("/completion", response_model=CompletionResponse)
def completion(ctx=Depends(get_read_context)) -> CompletionResponse:
    """No-sell finish-to-targets plan. Computed offline on every read.

    The implied finished total is set by whichever holding demands the
    biggest total, so nothing already held ever needs selling. Takes no
    input: the money needed is the output.
    """
    config = ctx.config_manager.config
    target = config.get("target_allocation", {}) or {}
    if not target:
        return CompletionResponse(valid=False, message="No target allocation configured.")

    metrics = MetricsCache(cache_path_for(ctx.config_manager)).read() or {}
    holdings = {str(r.get("symbol")).upper(): r
                for r in (metrics.get("holdings_df") or []) if isinstance(r, dict)}

    values: dict[str, float] = {}
    unknown: set[str] = set()
    for symbol in target:
        row = holdings.get(str(symbol).upper())
        if row is None:
            # Never held: known zero, not unknown.
            values[symbol] = 0.0
            continue
        # opt() maps None/NaN to None; a missing price must never read as $0.
        current = opt(row.get("value_usd"))
        if current is None:
            unknown.add(symbol)
            values[symbol] = 0.0
        else:
            values[symbol] = current

    anchors = {
        s: values[s] / float(w) for s, w in target.items()
        if s not in unknown and values[s] > 0 and float(w) > 0
    }
    if not anchors:
        return CompletionResponse(
            valid=False, message="No holdings to anchor from yet.")
    total = max(anchors.values())
    anchor = max(anchors, key=lambda s: anchors[s])

    rows = []
    additional = 0.0
    for symbol, weight in target.items():
        goal = total * float(weight)
        need = max(0.0, goal - values[symbol])
        additional += need
        rows.append(CompletionRow(
            symbol=symbol,
            target_allocation_pct=float(weight) * 100.0,
            target_value_usd=round(goal, 2),
            current_value_usd=(None if symbol in unknown
                               else round(values[symbol], 2)),
            need_usd=round(need, 2),
        ))
    rows.sort(key=lambda r: r.need_usd, reverse=True)
    return CompletionResponse(
        valid=True, anchor_symbol=anchor,
        implied_total_usd=round(total, 2),
        additional_total_usd=round(additional, 2), rows=rows)
```
`MetricsCache`, `cache_path_for`, `num`/`opt` are already imported/used in this file (see `dca_preview`). `float(w)` unguarded matches existing convention (`_allocation` line 267 does the same; weights are config-validated numerics).

- [ ] **Step 6: Run the new test to verify it passes**

Run: `uv run pytest tests/api/test_completion_route.py -q`
Expected: 5 passed. (Rounding check: 146.49/0.35 = 418.542857 → 418.54; ETH 418.542857*0.30-24.49 = 101.072857 → 101.07; total 247.562857 → 247.56 — all within rel=1e-4.)

- [ ] **Step 7: Run the full suite + lint**

Run: `uv run pytest -q` — Expected: all pass (baseline on this branch: 188 passed, 9 skipped; now 193+).
Run: `uv run ruff check api tests/api run_ui.py` — Expected: clean (this scope is lint-clean).

- [ ] **Step 8: Commit**

```bash
git add api/schemas/screens.py api/routes/strategy.py tests/api/test_completion_route.py docs/superpowers/specs/2026-09-04-completion-plan-design.md
git commit -m "feat: add no-sell completion plan endpoint"
```

---

### Task 2: React section on the DCA screen + types + tests

**Files:**
- Modify: `frontend/src/types.ts:257` (append after `DcaPreviewResponse`)
- Modify: `frontend/src/screens/Dca.tsx` (collapsed `useState` section; fetch via `useApi`)
- Create: `frontend/src/screens/Dca.test.tsx` (fetch-fail test + completion render test)

- [ ] **Step 1: Add the TypeScript mirror types**

Append after `DcaPreviewResponse` (`frontend/src/types.ts:257`), field for field including nullability:
```ts
export interface CompletionRow {
  symbol: string;
  target_allocation_pct: number;
  target_value_usd: number;
  current_value_usd: number | null;
  need_usd: number;
}

export interface CompletionResponse {
  valid: boolean;
  message: string | null;
  anchor_symbol: string | null;
  implied_total_usd: number | null;
  additional_total_usd: number;
  rows: CompletionRow[];
}
```

- [ ] **Step 2: Write the failing frontend tests**

Create `frontend/src/screens/Dca.test.tsx`. The screen mounts three GET hooks (`/api/strategy/dca`, `/api/execute/status`, plus the new `/api/strategy/completion`), so the fetch stub must route by URL:
```tsx
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Dca } from './Dca';
import type { CompletionResponse } from '../types';

const DCA_STATE = {
  has_data: true, is_running: false, error: null,
  staleness: { cached_at: null, age_seconds: null, is_stale: true },
  available_usdt: 100, spot_usdt: 60, earn_usdt: 40, minimum_trade_usd: 5,
};

const STATUS = { is_live: false, testnet: true };

const COMPLETION: CompletionResponse = {
  valid: true, message: null, anchor_symbol: 'BTC',
  implied_total_usd: 418.54, additional_total_usd: 247.56,
  rows: [
    { symbol: 'ETH', target_allocation_pct: 30, target_value_usd: 125.56, current_value_usd: 24.49, need_usd: 101.07 },
    { symbol: 'SOL', target_allocation_pct: 10, target_value_usd: 41.85, current_value_usd: 0, need_usd: 41.85 },
    { symbol: 'BTC', target_allocation_pct: 35, target_value_usd: 146.49, current_value_usd: 146.49, need_usd: 0 },
  ],
};

function stubFetch(completion: CompletionResponse | null) {
  vi.stubGlobal('fetch', vi.fn(async (url: unknown) => {
    const path = String(url);
    const payload = path.includes('/api/strategy/completion') ? completion
      : path.includes('/api/execute/status') ? STATUS : DCA_STATE;
    return { ok: true, json: async () => payload };
  }));
}

beforeEach(() => vi.unstubAllGlobals());

describe('Dca fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Dca />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});

describe('Dca completion plan', () => {
  it('stays collapsed until opened, then shows anchor, total and per-asset needs', async () => {
    stubFetch(COMPLETION);
    render(<Dca />);
    await waitFor(() => expect(screen.getByText(/completion plan/i)).toBeDefined());
    expect(screen.queryByText('101.07')).toBeNull();
    (screen.getByRole('button', { name: /completion plan/i }) as HTMLButtonElement).click();
    await waitFor(() => expect(screen.getByText('101.07')).toBeDefined());
    expect(screen.getByText(/anchored by BTC/i)).toBeDefined();
    expect(screen.getByText('247.56')).toBeDefined();
  });

  it('renders the invalid message instead of a table when there is no anchor', async () => {
    stubFetch({ valid: false, message: 'No holdings to anchor from yet.',
                anchor_symbol: null, implied_total_usd: null,
                additional_total_usd: 0, rows: [] });
    render(<Dca />);
    (await screen.findAllByRole('button', { name: /completion plan/i }));
    screen.getByRole('button', { name: /completion plan/i }).click();
    await waitFor(() =>
      expect(screen.getByText(/no holdings to anchor/i)).toBeDefined());
  });
});
```
Caveat for the implementer: match the toggle button's accessible name to `/completion plan/i` (label it e.g. `Show completion plan` / `Hide completion plan` — both match). If the screen renders the button label differently, adjust the test's regex, not the component, only if the label still clearly names the feature. `formatUsd(101.07)` must render exactly `101.07` somewhere — check `lib/format.ts` (`formatUsd` renders `$101.07`; the test looks for `'101.07'` which is a substring — `getByText('101.07')` needs an element whose FULL text is `101.07`. If `formatUsd` yields `$101.07`, use `screen.getByText(/101\.07/)` instead. Read `format.ts` before finalising and prefer regex matchers throughout.)

- [ ] **Step 3: Run the new test to verify it fails**

Run: `npx vitest run src/screens/Dca.test.tsx`
Expected: FAIL — `Dca.test.tsx` references `/api/strategy/completion` data the component never fetches, and no toggle button exists.

- [ ] **Step 4: Implement the section in `Dca.tsx`**

Requirements (follow the file's existing idioms — `Panel`, `Button`, `formatUsd`, `formatPercentPlain`, preview table markup):
1. Import the new types: `CompletionResponse` alongside the existing type imports (line 10-12).
2. Add hooks next to the existing ones (after line 23):
```tsx
const completion = useApi<CompletionResponse>('/api/strategy/completion');
const [showCompletion, setShowCompletion] = useState(false);
```
(`useState` is already imported at line 1.)
3. Render after the "Available to deploy" panel (after its closing `</Panel>`, before the "Plan a contribution" panel), collapsed by default:
```tsx
<Panel title="Completion plan">
  <p className="font-ui" style={{ color: 'var(--text-secondary)', fontSize: '13px', marginTop: 0 }}>
    How much of each asset is still needed to finish your targets without
    selling anything. {completion.data?.valid && completion.data.anchor_symbol
      ? `Anchored by ${completion.data.anchor_symbol} at ${formatUsd(completion.data.implied_total_usd)} implied total.`
      : 'The money needed is the output.'}
  </p>
  <Button onClick={() => setShowCompletion((v) => !v)}>
    {showCompletion ? 'Hide completion plan' : 'Show completion plan'}
  </Button>
  {showCompletion && !completion.data && !completion.error && (
    <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>Loading…</p>
  )}
  {showCompletion && completion.error && (
    <p className="font-mono text-sm" style={{ color: 'var(--negative)', marginBottom: 0 }}>
      Completion plan unavailable: {completion.error}
    </p>
  )}
  {showCompletion && completion.data && !completion.data.valid && (
    <p className="font-ui text-sm" style={{ color: 'var(--warning)', marginBottom: 0 }}>
      {completion.data.message ?? 'No holdings to anchor from yet.'}
    </p>
  )}
  {showCompletion && completion.data?.valid && (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th className="text-left">Asset</th>
            <th className="text-right">Target</th>
            <th className="text-right">Target value</th>
            <th className="text-right">Current</th>
            <th className="text-right">Still to buy</th>
          </tr>
        </thead>
        <tbody>
          {completion.data.rows.map((r) => (
            <tr key={r.symbol}>
              <td className="text-left" style={{ fontWeight: 500 }}>{r.symbol}</td>
              <td className="text-right">{formatPercentPlain(r.target_allocation_pct)}</td>
              <td className="text-right">{formatUsd(r.target_value_usd)}</td>
              <td className="text-right">{formatUsd(r.current_value_usd)}</td>
              <td className="text-right">{formatUsd(r.need_usd)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="font-ui text-sm" style={{ marginBottom: 0 }}>
        Additional cash needed: {formatUsd(completion.data.additional_total_usd)}
      </p>
    </div>
  )}
</Panel>
```
Notes: `formatUsd(null)` renders an em dash (see `lib/format.ts` — verify before using; the invalid/unknown path depends on it). `formatPercentPlain` takes a percent number (preview passes `a.target_allocation_pct` which is `weight*100`, line 170) — the API sends `target_allocation_pct` already multiplied by 100, so pass through directly. `Button` takes `onClick` + children (line 126 usage). The completion fetch failure must NOT blank the whole screen: it renders inline (`completion.error` branch) while the screen-level `error` (from `/api/strategy/dca`) keeps its existing `ErrorPanel` behavior — this is why the completion hook needs its own `useApi` instance rather than reusing `error`.
4. Keep the fetch-fail contract: with fetch rejected, the screen still returns the existing `ErrorPanel` (`Failed to load: …`) and no `Loading…` remains — the new section must not introduce its own permanent loading state on the error path (it doesn't: `showCompletion` starts false, and the screen returns early on `error`).

- [ ] **Step 5: Run the frontend tests + typecheck + lint**

Run: `npx vitest run src/screens/Dca.test.tsx` — Expected: all pass (adjust only the text matchers per the `format.ts` caveat in Step 2, never weaken the assertions to pass vacuously).
Run: `npx vitest run` — Expected: all files pass (baseline: 7 files, 56 tests).
Run: `npx tsc -b` — Expected: clean.
Run: `npx oxlint` — Expected: no new warnings on touched files.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types.ts frontend/src/screens/Dca.tsx frontend/src/screens/Dca.test.tsx
git commit -m "feat: add no-sell completion plan to React DCA screen"
```

---

## Out of scope (do not build)

- Executing the plan, CSV export, hypothetical-anchor input, new routes/nav entries (the section lives inside the existing DCA screen).
- Changes to `dca_preview`, rebalance, or any `src/` core file.
- CONTEXT.md changes (additive UI following existing rules; the pandas_ta fix on `main` is separate work).
