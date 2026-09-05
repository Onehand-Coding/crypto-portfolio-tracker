# Allocation Chart Readability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Allocation page readable for a concentrated portfolio: fitted P/L axis, visible dust bars, legible tooltips, and a donut the drift table can be matched to.

**Architecture:** Frontend-only changes in `frontend/src/screens/Allocation.tsx` plus a new test file. Two pure exported helpers (`plDomain`, `groupDust`) carry all the new logic so it is unit-testable; the JSX wires them in. No backend, schema, or type changes — `CockpitResponse` already carries every field.

**Tech Stack:** React + recharts (existing), Vitest + Testing Library, `tsc -b`, `oxlint`.

**Spec:** `docs/superpowers/specs/2026-09-05-allocation-charts-design.md` (approved; Section 1 = Task 1+2, Section 2 = Task 3).

---

### Task 1: `plDomain` helper and unit tests

**Files:**
- Modify: `frontend/src/screens/Allocation.tsx` (append helper + export)
- Test: `frontend/src/screens/Allocation.test.tsx` (create)

- [ ] **Step 1: Write the failing test**

Create `frontend/src/screens/Allocation.test.tsx` starting with only this (the render/fetch tests come in Task 2 — do not add them yet):

```tsx
import { describe, expect, it } from 'vitest';
import { plDomain } from './Allocation';

describe('plDomain', () => {
  it('fits mixed signs with a zero baseline and padding', () => {
    const [lo, hi] = plDomain([-34.5, -3.59, 0.15, 0, -0.01]);
    expect(lo).toBeLessThanOrEqual(-34.5);
    expect(hi).toBeGreaterThanOrEqual(0.15);
    expect(lo).toBeLessThanOrEqual(0);
    expect(hi).toBeGreaterThanOrEqual(0);
  });

  it('pins zero baseline for all-positive data', () => {
    expect(plDomain([1, 2, 3])[0]).toBe(0);
    expect(plDomain([1, 2, 3])[1]).toBeGreaterThan(3);
  });

  it('pins zero baseline for all-negative data', () => {
    const [lo, hi] = plDomain([-5, -2]);
    expect(hi).toBe(0);
    expect(lo).toBeLessThan(-5);
  });

  it('never degenerates for all-zero data', () => {
    expect(plDomain([0, 0])).toEqual([-1, 1]);
  });

  it('never degenerates for empty input', () => {
    const [lo, hi] = plDomain([]);
    expect(hi).toBeGreaterThan(lo);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/screens/Allocation.test.tsx` (from `frontend/`)
Expected: FAIL — `plDomain` is not exported (import error).

- [ ] **Step 3: Write minimal implementation**

Append to `frontend/src/screens/Allocation.tsx` (after the imports/colour block, before the component):

```ts
/** Y-axis domain for the P/L chart that always contains the data and zero.
    Recharts' auto domain demonstrably clipped a -$34.50 bar at -$24, so the
    domain is owned here and unit-tested with real magnitudes. */
export function plDomain(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo;
  if (span === 0) return [Math.min(0, lo - 1), Math.max(0, hi + 1)];
  const pad = span * 0.05;
  return [Math.min(0, lo - pad), Math.max(0, hi + pad)];
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/screens/Allocation.test.tsx` (from `frontend/`)
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/screens/Allocation.tsx frontend/src/screens/Allocation.test.tsx
git commit -m "test: plDomain helper for fitted P/L axis"
```

---

### Task 2: P/L chart wiring (domain, min bars, tooltip, axis)

**Files:**
- Modify: `frontend/src/screens/Allocation.tsx` (BarChart block only)
- Test: extend `frontend/src/screens/Allocation.test.tsx` (fetch-fail + render smoke)

- [ ] **Step 1: Write the failing tests**

Append to `frontend/src/screens/Allocation.test.tsx`:

```tsx
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, vi } from 'vitest';
import { Allocation } from './Allocation';

const STALENESS = { cached_at: null, age_seconds: null, is_stale: false };

const COCKPIT = {
  total_value_usd: 175.02,
  holdings: [
    { symbol: 'BTC', value_usd: 149.47, unrealized_pl_usd: -34.5 },
    { symbol: 'ETH', value_usd: 24.58, unrealized_pl_usd: -3.59 },
    { symbol: 'USDT', value_usd: 0.15, unrealized_pl_usd: 0 },
  ],
  staleness: STALENESS,
};

const HEALTH = { target_allocation: { BTC: 0.35, ETH: 0.3, USDT: 0.05 } };

function stubFetch() {
  vi.stubGlobal('fetch', vi.fn(async (url: unknown) => {
    const path = String(url);
    if (path.includes('/api/portfolio/cockpit')) {
      return { ok: true, json: async () => COCKPIT };
    }
    if (path.includes('/api/system/health')) {
      return { ok: true, json: async () => HEALTH };
    }
    throw new Error(`unexpected fetch: ${path}`);
  }));
}

beforeEach(() => vi.unstubAllGlobals());

describe('Allocation fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Allocation />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load/i)).toBeDefined();
  });
});

describe('Allocation P/L chart', () => {
  it('renders every holding so each bar is hoverable', async () => {
    stubFetch();
    const { container } = render(<Allocation />);
    await screen.findByText('Unrealized P/L by asset');
    // All three symbols reach the chart's tooltip items (recharts renders
    // one Tooltip item per datum even for sub-pixel bars).
    const items = container.querySelectorAll('.recharts-tooltip-item');
    const names = Array.from(items).map((el) => el.textContent ?? '');
    expect(names.some((t) => t.includes('BTC'))).toBe(true);
    expect(names.some((t) => t.includes('ETH'))).toBe(true);
    expect(names.some((t) => t.includes('USDT'))).toBe(true);
  });
});
```

Caveat for the implementer: recharts only renders `.recharts-tooltip-item` nodes for the *hovered* datum by default — the assertion above as written will find zero items without a hover. Before running, verify how recharts behaves in jsdom: if tooltip items are absent pre-hover, replace the item assertion with firing a `mouseEnter`/`mouseOver` on the chart container (`container.querySelector('.recharts-wrapper')`) and then asserting. If hover simulation proves flaky in jsdom, fall back to asserting the bar cells render one `Cell` per datum: `container.querySelectorAll('.recharts-bar .recharts-cell').length === 3`. Either assertion pins "every holding reaches the chart". Document which one you used in the commit message.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/screens/Allocation.test.tsx` (from `frontend/`)
Expected: FAIL — the fetch-failure test fails (screen has no error path handling? It does have ErrorPanel — it should already pass). Honest expectation: the fetch-fail test PASSES already (ErrorPanel exists), the chart test fails or errors on the tooltip assertion. TDD still holds for the chart behaviour: if the chart test passes pre-change, strengthen it (e.g. assert `minPointSize` via the Bar element is not directly queryable — instead assert dust-bar visibility through the hover path). Do not force a red that isn't real; record what failed in the commit message.

- [ ] **Step 3: Write minimal implementation**

In `frontend/src/screens/Allocation.tsx`, replace the BarChart block's axis/tooltip/bar props (data mapping `pl` stays identical):

```tsx
<BarChart data={pl} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
  <XAxis dataKey="name" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
         stroke="var(--border)" interval={0} angle={-35} textAnchor="end"
         height={56} />
  <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
         stroke="var(--border)" width={64}
         domain={plDomain(pl.map((d) => d.value))}
         tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
  <Tooltip
    cursor={{ fill: 'var(--surface-2)' }}
    contentStyle={{ background: 'var(--surface-2)',
                    border: '1px solid var(--border-strong)',
                    borderRadius: 'var(--radius-control)', fontSize: '12px' }}
    labelStyle={{ color: 'var(--text-primary)' }}
    itemStyle={{ color: 'var(--text-secondary)' }}
    formatter={(v) => [formatSigned(typeof v === 'number' ? v : null), 'Unrealized']}
  />
  <Bar dataKey="value" radius={[2, 2, 0, 0]} minPointSize={3}>
    {pl.map((d, i) => (
      <Cell key={i} fill={d.value >= 0 ? 'var(--positive)' : 'var(--negative)'} />
    ))}
  </Bar>
</BarChart>
```

Changes vs current code, nothing else: `domain={...}`, `minPointSize={3}`, `labelStyle` + `itemStyle`, XAxis `interval/angle/textAnchor/height`, bottom margin 0 → 8. Keep `tickFormatter`, cursor, contentStyle, formatter text, Cell colours exactly as they are.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/screens/Allocation.test.tsx` then `npx tsc -b` (from `frontend/`)
Expected: all PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/screens/Allocation.tsx frontend/src/screens/Allocation.test.tsx
git commit -m "fix: fitted P/L axis, visible dust bars, legible tooltip"
```

---

### Task 3: Donut grouping, shared colours, table dots

**Files:**
- Modify: `frontend/src/screens/Allocation.tsx` (pie memo, cells, drift table)
- Test: extend `frontend/src/screens/Allocation.test.tsx`

- [ ] **Step 1: Write the failing tests**

Append to the `plDomain` describe block's file (new top-level describes):

```tsx
import { DUST_THRESHOLD_PCT, groupDust } from './Allocation';

describe('groupDust', () => {
  const slices = [
    { name: 'BTC', value: 149.47 },
    { name: 'ETH', value: 24.58 },
    { name: 'USDT', value: 0.15 },
    { name: 'SOL', value: 0.03 },
  ];

  it('groups sub-threshold holdings into one named slice', () => {
    expect(DUST_THRESHOLD_PCT).toBe(1);
    const out = groupDust(slices, 175.02);
    expect(out.map((s) => s.name)).toEqual(['BTC', 'ETH', 'Others (2)']);
    const others = out.find((s) => s.name === 'Others (2)');
    expect(others?.value).toBeCloseTo(0.18, 10);
  });

  it('passes holdings through when nothing is dust', () => {
    expect(groupDust(
      [{ name: 'BTC', value: 60 }, { name: 'ETH', value: 40 }], 100))
      .toEqual([{ name: 'BTC', value: 60 }, { name: 'ETH', value: 40 }]);
  });

  it('collapses to a single slice when everything is dust', () => {
    expect(groupDust(
      [{ name: 'A', value: 0.4 }, { name: 'B', value: 0.3 }], 100))
      .toEqual([{ name: 'Others (2)', value: 0.7 }]);
  });

  it('does not divide by a zero total', () => {
    expect(groupDust([{ name: 'A', value: 0 }], 0))
      .toEqual([{ name: 'A', value: 0 }]);
  });
});

describe('Allocation colour dots', () => {
  it('shows a dot per drift row matching the ring order', async () => {
    stubFetch();
    const { container } = render(<Allocation />);
    await screen.findByText('Current vs target');
    // One dot per drift row (BTC, ETH, USDT in the stub health payload).
    const dots = container.querySelectorAll('[data-testid="drift-dot"]');
    expect(dots.length).toBe(3);
  });
});
```

The dot test requires the implementation to render dots with `data-testid="drift-dot"` (specified in Step 3).

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/screens/Allocation.test.tsx` (from `frontend/`)
Expected: FAIL — `groupDust` not exported.

- [ ] **Step 3: Write minimal implementation**

In `frontend/src/screens/Allocation.tsx`:

(a) Next to `SLICE_COLOURS`, add:

```ts
// Ring slices below this share of total value merge into one "Others (n)"
// slice. The drift table still lists every asset: grouping is ring-only.
export const DUST_THRESHOLD_PCT = 1;

export interface PieSlice { name: string; value: number; }

export function groupDust(slices: PieSlice[], total: number): PieSlice[] {
  const isDust = (s: PieSlice) => total > 0 && (s.value / total) * 100 < DUST_THRESHOLD_PCT;
  const dust = slices.filter(isDust);
  if (dust.length === 0) return slices;
  const kept = slices.filter((s) => !isDust(s));
  const sum = dust.reduce((acc, s) => acc + s.value, 0);
  return [...kept, { name: `Others (${dust.length})`, value: sum }];
}
```

(b) Extend the `pie` memo to group (total from `cockpit.data.total_value_usd ?? 0`):

```ts
  const pie = useMemo(() => {
    const holdings = cockpit.data?.holdings ?? [];
    const total = cockpit.data?.total_value_usd ?? 0;
    const slices = holdings
      .filter((h) => (h.value_usd ?? 0) > 0)
      .map((h) => ({ name: h.symbol, value: h.value_usd as number }))
      .sort((a, b) => b.value - a.value);
    return groupDust(slices, total);
  }, [cockpit.data]);
```

(c) Shared colour map + cells (replaces index-based fills so ring and table agree):

```tsx
  const colourOf = new Map(
    pie.map((s, i) => [s.name, SLICE_COLOURS[i % SLICE_COLOURS.length]]),
  );
```

Place it after the memos (plain const in the component body, before the early returns — hooks order: it is not a hook, so position is free, but keep it with the memos for readability). Cells become:

```tsx
{pie.map((s) => (
  <Cell key={s.name} fill={colourOf.get(s.name)} />
))}
```

(d) Drift-table Asset cell gains the dot (only change in that table):

```tsx
<td className="text-left" style={{ fontWeight: 500 }}>
  <span data-testid="drift-dot" style={{
    display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
    background: colourOf.get(d.name) ?? 'var(--text-tertiary)',
    marginRight: 8,
  }} />
  {d.name}
</td>
```

A target with no priced holding (not in the ring) gets the tertiary dot — honest fallback, never a wrong colour.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/screens/Allocation.test.tsx` then `npx tsc -b` (from `frontend/`)
Expected: all PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/screens/Allocation.tsx frontend/src/screens/Allocation.test.tsx
git commit -m "fix: donut dust grouping with table colour dots"
```

---

### Task 4: Full verification and live check

- [ ] **Step 1: Frontend suite + typecheck + lint**

Run (from `frontend/`): `npx vitest run`, `npx tsc -b`, `npx oxlint`
Expected: all test files pass; tsc clean; oxlint shows only the pre-existing warnings (nav.tsx fast-refresh and any others present before this branch — confirm via `git stash`-free check: `git show main:frontend/src/screens/Allocation.tsx` never had lint issues; new code must add zero).

- [ ] **Step 2: Backend suite untouched but green**

Run (from root): `uv run pytest -q`
Expected: 298 passed, 9 skipped (no backend changes in this branch; this guards accidents).

- [ ] **Step 3: Rebuild and live browser check**

```bash
npm --prefix frontend run build
```

Then restart detached (NEVER foreground `./start.sh` — it blocks on `wait`):
`setsid nohup ./start.sh --skip-build > /tmp/tracker.log 2>&1 < /dev/null & disown`, then probe `curl -s -m 8 http://localhost:8000/api/portfolio/cockpit` in a separate call. With Playwright: open `/allocation` and confirm (a) the BTC bar sits fully inside the axis (no clipping), (b) dust bars show as small ticks with hover tooltips in light text, (c) the ring shows BTC/ETH/Others with matching table dots. Screenshot for the report.

- [ ] **Step 4: Report, no merge, no push.** Leave the branch for owner review and merge.

---

## Self-review

- Spec coverage: Section 1.1 domain helper → Task 1 + Task 2 wiring; 1.2 min bars → Task 2 (`minPointSize`, zero stays zero pinned how? The Task 2 chart test covers datum presence, not zero-height. Gap: add to Task 2 Step 1 an explicit case — recharts `minPointSize` skips exact 0, but asserting rendered pixel height in jsdom is unreliable (no layout). Resolution: the USDT stub datum has value 0 and the cell-count assertion (fallback) proves it reaches the chart; zero-height rendering is recharts-owned behaviour, out of scope to test. Noted here instead of pretending otherwise.
- Section 1.3 tooltip/axis → Task 2 (labelStyle/itemStyle/angle/height/margin all specified). Section 2.1 colour map + dots → Task 3 (map, fallback, testid all specified). Section 2.2 grouping → Task 3 (threshold, naming, table-untouched, all three edge cases tested). Section 2.3 fetch-fail test → Task 2 Step 1. Non-goals respected: no backend/types changes anywhere in the plan; Backtest/Technical explicitly excluded.
- Placeholder scan: every step has exact code, exact commands, exact expected output. The single adaptive instruction (Task 2 Step 2 recharts-tooltip-in-jsdom caveat) gives both the primary and the fallback assertion plus where to record the choice — a bounded decision, not a placeholder.
- Type consistency: `plDomain(values: number[]): [number, number]`; `PieSlice { name: string; value: number }`; `groupDust(slices: PieSlice[], total: number): PieSlice[]`; `DUST_THRESHOLD_PCT = 1` (number, compared against percent-scale `(s.value / total) * 100`); `colourOf: Map<string, string>`; drift rows use `d.name` (existing field). `data-testid="drift-dot"` consistent between implementation and test.
