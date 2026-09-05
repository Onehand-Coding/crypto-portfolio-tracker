# Dashboard and Overview Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate Cockpit and Overview into a single Dashboard with historical value and FIFO-basis performance data, while preserving `/overview` as a redirect.

**Architecture:** Keep the existing `Cockpit` component and its two read-only API requests. Extend its Performance panel with Overview's stored cost-basis series and the unique first-snapshot change metric. Delete the separate screen from the route tree, rename the visible navigation destination to Dashboard, and use React Router's `Navigate` to preserve deep links.

**Tech Stack:** React 19, TypeScript, React Router 7, Recharts, Vitest, Testing Library.

---

## File Structure

- Modify: `frontend/src/screens/Cockpit.tsx` — combine the existing range-filtered value series with historical snapshot cost basis and show the first-snapshot delta.
- Modify: `frontend/src/screens/Cockpit.test.tsx` — stub both dashboard reads and pin the consolidated chart/KPI/error states.
- Modify: `frontend/src/App.tsx` — replace the Overview screen route with a redirect and use Dashboard in user-visible screen copy.
- Modify: `frontend/src/nav.tsx` — rename Cockpit and remove the visible Overview tab.
- Modify: `frontend/src/App.test.tsx` — prove the navigation label and `/overview` redirect behavior.
- Delete: `frontend/src/screens/Overview.tsx` — its only view is absorbed by the Dashboard.

### Task 1: Pin Consolidated Dashboard Behavior

**Files:**
- Modify: `frontend/src/screens/Cockpit.test.tsx`

- [ ] **Step 1: Add a reusable overview fixture and path-aware fetch stub**

Add the overview response fixture after `EMPTY` and replace the single-payload fetch helper. The dashboard makes three requests, so the fixture must explicitly answer both portfolio and history requests:

```tsx
const OVERVIEW: OverviewResponse = {
  has_data: true,
  staleness: { cached_at: '2026-07-21T09:30:00', age_seconds: 120, is_stale: false },
  points: [
    {
      timestamp: '2026-07-01T09:30:00', total_value_usd: 40,
      total_cost_basis_usd: 100, unrealized_pl_usd: -60, unrealized_pl_percent: -60,
    },
    {
      timestamp: '2026-07-21T09:30:00', total_value_usd: 57.78,
      total_cost_basis_usd: 199.75, unrealized_pl_usd: -141.97, unrealized_pl_percent: -71.07,
    },
  ],
};

function mockFetch(
  cockpit: CockpitResponse = POPULATED,
  overview: OverviewResponse = OVERVIEW,
) {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes('/api/portfolio/cockpit')) {
      return Promise.resolve({ ok: true, json: async () => cockpit });
    }
    if (path.includes('/api/overview')) {
      return Promise.resolve({ ok: true, json: async () => overview });
    }
    if (path.includes('/api/system/health')) {
      return Promise.resolve({ ok: true, json: async () => ({ target_allocation: {} }) });
    }
    return Promise.reject(new Error(`Unexpected request: ${path}`));
  }));
}
```

Import `OverviewResponse` with `CockpitResponse` from `../types`. Update existing `mockFetch(POPULATED)` calls to use the default or pass it as the first parameter.

- [ ] **Step 2: Add failing tests for the history metric and failed history request**

Add these cases to `Cockpit.test.tsx`:

```tsx
it('shows change since the first snapshot without duplicating the current value', async () => {
  mockFetch();
  renderCockpit();

  expect(await screen.findByText('Change since first snapshot')).toBeDefined();
  expect(screen.getByText(/\+\$17\.78/)).toBeDefined();
  expect(screen.getByText(/from 2026-07-01/)).toBeDefined();
  expect(screen.queryByText('Latest value')).toBeNull();
});

it('shows an explicit history error when the overview request fails', async () => {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes('/api/portfolio/cockpit')) {
      return Promise.resolve({ ok: true, json: async () => POPULATED });
    }
    if (path.includes('/api/overview')) return Promise.reject(new TypeError('Failed to fetch'));
    if (path.includes('/api/system/health')) {
      return Promise.resolve({ ok: true, json: async () => ({ target_allocation: {} }) });
    }
    return Promise.reject(new Error(`Unexpected request: ${path}`));
  }));

  renderCockpit();

  expect(await screen.findByText(/could not load history/i)).toBeDefined();
});
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `cd frontend && npx vitest run src/screens/Cockpit.test.tsx`

Expected: FAIL because the Performance panel does not yet render `Change since first snapshot` or the historical cost-basis line.

### Task 2: Extend the Dashboard Performance Panel

**Files:**
- Modify: `frontend/src/screens/Cockpit.tsx:1-358`
- Test: `frontend/src/screens/Cockpit.test.tsx`

- [ ] **Step 1: Add the cost-basis and history-change calculations**

Import `Line` from `recharts`. Keep the current timestamp filtering and make each point carry the nullable stored snapshot basis:

```tsx
const series = useMemo(() => {
  const points = overview.data?.points ?? [];
  const days = RANGES.find((r) => r.id === range)?.days ?? Infinity;
  const cutoff = days === Infinity ? 0 : Date.now() - days * 86_400_000;
  return points
    .filter((p) => p.timestamp !== null && p.total_value_usd !== null)
    .filter((p) => new Date(p.timestamp!).getTime() >= cutoff)
    .map((p) => ({
      t: new Date(p.timestamp!).getTime(),
      date: p.timestamp!.slice(0, 10),
      value: p.total_value_usd as number,
      basis: p.total_cost_basis_usd,
    }));
}, [overview.data, range]);

const historyChange = useMemo(() => {
  if (series.length < 2) return null;
  return series[series.length - 1].value - series[0].value;
}, [series]);
```

Use the selected range for this metric, so it always describes the chart in front of the user. Use the mapped ISO `date` string for the first point so the visible date is stable across the browser's locale and timezone.

- [ ] **Step 2: Render the unique history KPI and the snapshot FIFO-basis line**

Change the panel heading area to include the metric under the existing range buttons. It must be absent when fewer than two valid snapshots exist:

```tsx
<div className="flex shrink-0 items-center justify-between" style={{ marginBottom: 'var(--space-3)' }}>
  <div>
    <h2 className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px', fontWeight: 500,
                                      letterSpacing: '0.08em', textTransform: 'uppercase', margin: 0 }}>
      Performance
    </h2>
    {historyChange !== null && (
      <p className="font-mono" style={{ color: historyChange >= 0 ? 'var(--positive)' : 'var(--negative)',
                                         fontSize: '11px', margin: '4px 0 0' }}>
        Change since first snapshot: {formatSigned(historyChange)}
        {' '}from {series[0].date}
      </p>
    )}
  </div>
  {/* existing range buttons */}
</div>
```

Add the line after the existing `Area` inside `AreaChart`:

```tsx
<Line
  type="monotone"
  dataKey="basis"
  name="FIFO cost basis at snapshot"
  stroke="var(--text-tertiary)"
  strokeWidth={1.5}
  strokeDasharray="3 3"
  dot={false}
/>
```

Update the tooltip formatter to use `name` so both series are explicit:

```tsx
formatter={(value, name) => [
  formatUsd(typeof value === 'number' ? value : null),
  name === 'FIFO cost basis at snapshot' ? 'FIFO cost basis at snapshot' : 'Value',
]}
```

Do not add a latest-value or current-cost-basis metric: those duplicate the main Dashboard accounting band and can disagree with a stored snapshot.

Change the three user-visible Cockpit state labels to Dashboard while retaining
the `Cockpit` component export: `ErrorPanel title="Dashboard"`,
`<Panel title="Dashboard">` for loading, and `<Panel title="Dashboard">` for
the no-data state.

- [ ] **Step 3: Run the Cockpit tests to verify they pass**

Run: `cd frontend && npx vitest run src/screens/Cockpit.test.tsx`

Expected: PASS. The existing no-data, unpriced, accounting-basis, and failed cockpit-fetch tests remain green alongside the new history cases.

- [ ] **Step 4: Commit the Dashboard performance change**

```bash
git add frontend/src/screens/Cockpit.tsx frontend/src/screens/Cockpit.test.tsx
git commit -m "feat: consolidate overview performance into Dashboard"
```

### Task 3: Replace Overview Navigation With a Compatible Redirect

**Files:**
- Modify: `frontend/src/App.tsx:1-217`
- Modify: `frontend/src/nav.tsx:39-114`
- Modify: `frontend/src/App.test.tsx`
- Delete: `frontend/src/screens/Overview.tsx`

- [ ] **Step 1: Add failing navigation and redirect tests**

In `App.test.tsx`, import `Navigate` is not necessary in the test. Use `MemoryRouter` with `initialEntries` and stub the three App/Dashboard requests:

```tsx
function stubAppFetch() {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path.includes('/api/portfolio/cockpit')) {
      return Promise.resolve({ ok: true, json: async () => ({
        total_value_usd: 0,
        net_invested: { label: '', question: '', basis_usd: 0, pl_usd: 0, pl_percent: null },
        fifo: { label: '', question: '', basis_usd: 0, pl_usd: 0, pl_percent: null },
        holdings: [], staleness: { cached_at: null, age_seconds: null, is_stale: false },
        environment: { is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET' },
        has_data: false, unpriced_count: 0,
      }) });
    }
    if (path.includes('/api/overview')) {
      return Promise.resolve({ ok: true, json: async () => ({ has_data: false, points: [], staleness: {} }) });
    }
    if (path.includes('/api/system/health')) {
      return Promise.resolve({ ok: true, json: async () => ({ target_allocation: {} }) });
    }
    if (path.includes('/api/sync/status')) {
      return Promise.resolve({ ok: true, json: async () => ({ staleness: {} }) });
    }
    return Promise.reject(new Error(`Unexpected request: ${path}`));
  }));
}

it('labels the root portfolio destination Dashboard and removes the Overview tab', async () => {
  stubAppFetch();
  render(<MemoryRouter><App /></MemoryRouter>);

  expect(screen.getByRole('link', { name: 'Dashboard' })).toHaveAttribute('href', '/');
  expect(screen.queryByRole('link', { name: 'Overview' })).toBeNull();
});

it('redirects the legacy overview route to Dashboard', async () => {
  stubAppFetch();
  render(<MemoryRouter initialEntries={['/overview']}><App /></MemoryRouter>);

  expect(await screen.findByText(/no data yet/i)).toBeDefined();
  expect(screen.queryByText('Portfolio overview')).toBeNull();
});
```

If `toHaveAttribute` is unavailable in the configured Vitest setup, assert the returned link element's `getAttribute('href')` equals `/` instead.

- [ ] **Step 2: Run navigation tests to verify they fail**

Run: `cd frontend && npx vitest run src/App.test.tsx`

Expected: FAIL because Cockpit and Overview are still separately visible and `/overview` renders the old Overview screen.

- [ ] **Step 3: Make the minimal route and navigation changes**

In `App.tsx`, import `Navigate`, remove the Overview import, and replace the route:

```tsx
import { Navigate, NavLink, Route, Routes, useLocation } from 'react-router-dom';

// Remove: import { Overview } from './screens/Overview';

<Route path="/" element={<Cockpit />} />
<Route path="/overview" element={<Navigate to="/" replace />} />
```

In `nav.tsx`, replace the first two Portfolio items with the single root destination:

```tsx
items: [
  { to: '/', label: 'Dashboard' },
  { to: '/realized', label: 'Realized P/L' },
  { to: '/wallets', label: 'Wallets' },
  { to: '/capital', label: 'Capital Flow' },
],
```

Delete `frontend/src/screens/Overview.tsx`. Do not change `/api/overview`; the Dashboard continues to use it for historical snapshots.

- [ ] **Step 4: Run navigation tests to verify they pass**

Run: `cd frontend && npx vitest run src/App.test.tsx`

Expected: PASS. `/overview` renders the root Dashboard's no-data state, and the only root Portfolio tab is Dashboard.

- [ ] **Step 5: Commit the route consolidation**

```bash
git add frontend/src/App.tsx frontend/src/nav.tsx frontend/src/App.test.tsx frontend/src/screens/Overview.tsx
git commit -m "feat: route Overview links to Dashboard"
```

### Task 4: Verify the Frontend Suite and Build

**Files:**
- Verify only.

- [ ] **Step 1: Run all frontend tests**

Run: `cd frontend && npx vitest run`

Expected: PASS. This checks the shared App shell, Dashboard, and unrelated screens against the updated navigation.

- [ ] **Step 2: Run static checks and production build**

Run: `cd frontend && npx tsc -b && npx oxlint && npm run build`

Expected: all commands exit 0. The build confirms the deleted Overview module has no remaining imports.

- [ ] **Step 3: Inspect the final change set**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and a clean working tree after the two feature commits.

## Plan Self-Review

- Spec coverage: Dashboard composition is Task 2; retained range controls and two-series chart are Task 2; no duplicate current KPIs is explicitly constrained in Task 2; navigation/removal/redirect is Task 3; error and no-data behavior is covered by Tasks 1 and 2; tests and full verification are Tasks 1, 3, and 4.
- Placeholder scan: no deferred implementation language or unspecified test cases remain.
- Type consistency: the plan uses existing `OverviewResponse`, `SnapshotPoint`, `CockpitResponse`, `Navigate`, Recharts `Line`, and `formatSigned` APIs; no new API contract is introduced.
