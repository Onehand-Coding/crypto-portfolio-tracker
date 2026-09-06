import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const lineSpy = vi.hoisted(() => vi.fn());

vi.mock('recharts', () => {
  const Passthrough = ({ children }: { children?: ReactNode }) => <>{children}</>;
  const Empty = () => null;
  return {
    Area: Empty,
    AreaChart: Passthrough,
    CartesianGrid: Empty,
    Line: (props: unknown) => {
      lineSpy(props);
      return null;
    },
    ResponsiveContainer: Passthrough,
    Tooltip: Empty,
    XAxis: Empty,
    YAxis: Empty,
  };
});

import { Cockpit } from './Cockpit';
import type { CockpitResponse, OverviewResponse, ProfitOpportunity, ProfitResponse, SystemHealthResponse } from '../types';

/** The alert cards link to the screens that resolve them, so the cockpit
 *  requires router context. */
const renderCockpit = () =>
  render(<MemoryRouter><Cockpit /></MemoryRouter>);

const POPULATED: CockpitResponse = {
  total_value_usd: 57.78,
  net_invested: {
    label: 'Cash profit', question: 'did I make money?',
    basis_usd: 76.41, pl_usd: -18.63, pl_percent: -24.38,
  },
  fifo: {
    label: 'Holdings profit (FIFO)', question: 'are my holdings underwater?',
    basis_usd: 199.75, pl_usd: -141.97, pl_percent: -71.07,
  },
  holdings: [
    {
      symbol: 'BTC', total_quantity: 0.0004, spot_quantity: 0.0004, earn_quantity: 0,
      current_price: 95000, value_usd: 38.00, average_cost_basis: 330000,
      cost_basis_total: 132.00, unrealized_pl_usd: -94.00,
      unrealized_pl_percent: -71.21, is_core: true, price_unavailable: false,
    },
    {
      symbol: 'ETH', total_quantity: 0.006, spot_quantity: 0.006, earn_quantity: 0,
      current_price: 3200, value_usd: 19.20, average_cost_basis: 11291.67,
      cost_basis_total: 67.75, unrealized_pl_usd: -48.55,
      unrealized_pl_percent: -71.66, is_core: true, price_unavailable: false,
    },
    {
      symbol: 'DOGE', total_quantity: 2.5, spot_quantity: 2.5, earn_quantity: 0,
      current_price: 0.232, value_usd: 0.58, average_cost_basis: 0.4,
      cost_basis_total: 1.00, unrealized_pl_usd: -0.42,
      unrealized_pl_percent: -42.00, is_core: false, price_unavailable: false,
    },
  ],
  staleness: { cached_at: '2026-07-21T09:30:00', age_seconds: 120, is_stale: false },
  environment: { is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET' },
  has_data: true,
  unpriced_count: 0,
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

const HEALTH: SystemHealthResponse = {
  environment_label: 'TESTNET',
  is_testnet: true,
  database_path: 'data/testnet_portfolio.db',
  database_exists: true,
  database_size_bytes: 1,
  transaction_count: 1,
  asset_count: 1,
  snapshot_count: 2,
  target_allocation: {},
  live_trading_enabled: false,
  minimum_trade_usd: 10,
  backups: [],
  metrics_cache_age_seconds: 120,
  binance_configured: true,
};

function mockDashboardFetch(overview?: OverviewResponse | Error, cockpit = POPULATED, profit: ProfitResponse | Error = PROFIT_EMPTY) {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url === '/api/portfolio/cockpit') {
      return Promise.resolve({ ok: true, json: async () => cockpit });
    }
    if (url === '/api/overview') {
      if (overview === undefined) return new Promise<never>(() => {});
      if (overview instanceof Error) return Promise.reject(overview);
      return Promise.resolve({ ok: true, json: async () => overview });
    }
    if (url === '/api/system/health') {
      return Promise.resolve({ ok: true, json: async () => HEALTH });
    }
    if (url === '/api/strategy/profit') {
      if (profit instanceof Error) return Promise.reject(profit);
      return Promise.resolve({ ok: true, json: async () => profit });
    }
    return Promise.reject(new Error(`Unexpected request: ${url}`));
  }));
}

/** A fresh run that found nothing: the neutral profit state, so existing
 *  dashboard tests keep asserting a card-free alerts section. */
const PROFIT_EMPTY: ProfitResponse = {
  opportunities: [],
  has_data: true,
  is_running: false,
  error: null,
  staleness: POPULATED.staleness,
};

function profitWith(opportunity: ProfitOpportunity, staleness = POPULATED.staleness, is_running = false): ProfitResponse {
  return { opportunities: [opportunity], has_data: true, is_running, error: null, staleness };
}

const SCORED: ProfitOpportunity = {
  symbol: 'SOL',
  unrealized_gain_usd: 42.50,
  unrealized_gain_pct: 61.20,
  opportunity_score: 82,
  rsi_score: 80,
  pl_score: 80,
  resistance_score: 90,
  market_context_score: 70,
  current_price: 103.36,
  support_level: 90.10,
  resistance_level: 105.00,
  reasons: ['Near resistance'],
};

beforeEach(() => vi.unstubAllGlobals());
afterEach(() => vi.useRealTimers());

describe('Cockpit unpriced holdings', () => {
  it('caveats the total when a holding could not be priced', async () => {
    // Without this the total reads as a confident figure while silently
    // excluding a position of unknown -- possibly dominant -- size.
    mockFetch({ ...POPULATED, unpriced_count: 1 });
    renderCockpit();
    expect(await screen.findByText(/could not be priced/)).toBeDefined();
  });

  it('shows no caveat when every holding is priced', async () => {
    mockFetch(POPULATED);
    renderCockpit();
    await screen.findByText('$57.78');
    expect(screen.queryByText(/could not be priced/)).toBeNull();
  });
});

describe('Cockpit populated state', () => {
  it('renders both accounting bases with different values', async () => {
    mockFetch(POPULATED);
    renderCockpit();

    // Regex, not exact strings: each basis renders its P/L and percent in a
    // single span, e.g. "-$18.63  (-24.38%)".
    await waitFor(() => expect(screen.getByText('$57.78')).toBeDefined());
    expect(screen.getByText(/-\$18\.63/)).toBeDefined();
    expect(screen.getByText(/-\$141\.97/)).toBeDefined();
  });

  it('labels each basis with the question it answers', async () => {
    mockFetch(POPULATED);
    renderCockpit();

    await waitFor(() => expect(screen.getByText('did I make money?')).toBeDefined());
    expect(screen.getByText('are my holdings underwater?')).toBeDefined();
    expect(screen.getByText('Cash profit')).toBeDefined();
    expect(screen.getByText('Holdings profit (FIFO)')).toBeDefined();
  });

  it('shows each basis denominator so the two are visibly different', async () => {
    mockFetch(POPULATED);
    renderCockpit();

    await waitFor(() => expect(screen.getByText(/76\.41 net in/)).toBeDefined());
    expect(screen.getByText(/199\.75 cost basis/)).toBeDefined();
  });
});

describe('Cockpit performance history', () => {
  it('retains history when current cache data is absent', async () => {
    const now = Date.now();
    lineSpy.mockClear();
    mockDashboardFetch({
      has_data: true,
      staleness: POPULATED.staleness,
      points: [
        { timestamp: new Date(now - 7 * 86_400_000).toISOString(), total_value_usd: 100, total_cost_basis_usd: 80,
          unrealized_pl_usd: 20, unrealized_pl_percent: 25 },
        { timestamp: new Date(now - 86_400_000).toISOString(), total_value_usd: 125, total_cost_basis_usd: 90,
          unrealized_pl_usd: 35, unrealized_pl_percent: 38.89 },
      ],
    }, EMPTY);
    renderCockpit();

    expect(await screen.findByText(/no data yet/i)).toBeDefined();
    expect(await screen.findByText(/Change since first snapshot: \+\$25\.00 from/)).toBeDefined();
    expect(lineSpy).toHaveBeenCalled();
  });

  it('shows the selected-range change from overview snapshots', async () => {
    const now = Date.now();
    mockDashboardFetch({
      has_data: true,
      staleness: POPULATED.staleness,
      points: [
        {
          timestamp: new Date(now - 7 * 86_400_000).toISOString(),
          total_value_usd: 100,
          total_cost_basis_usd: 80,
          unrealized_pl_usd: 20,
          unrealized_pl_percent: 25,
        },
        {
          timestamp: new Date(now - 86_400_000).toISOString(),
          total_value_usd: 125,
          total_cost_basis_usd: 90,
          unrealized_pl_usd: 35,
          unrealized_pl_percent: 38.89,
        },
      ],
    });
    renderCockpit();

    expect(await screen.findByText(/Change since first snapshot: \+\$25\.00 from/)).toBeDefined();
    expect(screen.queryByText(/Latest value/i)).toBeNull();
    expect(screen.queryByText(/Current cost basis/i)).toBeNull();
  });

  it('recomputes the signed change from snapshots in the selected range', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-15T12:00:00Z'));
    mockDashboardFetch({
      has_data: true,
      staleness: POPULATED.staleness,
      points: [
        { timestamp: '2025-01-01T00:00:00Z', total_value_usd: 50, total_cost_basis_usd: 40,
          unrealized_pl_usd: 10, unrealized_pl_percent: 25 },
        { timestamp: '2025-12-20T00:00:00Z', total_value_usd: 80, total_cost_basis_usd: 60,
          unrealized_pl_usd: 20, unrealized_pl_percent: 33.33 },
        { timestamp: '2026-02-20T00:00:00Z', total_value_usd: 120, total_cost_basis_usd: 90,
          unrealized_pl_usd: 30, unrealized_pl_percent: 33.33 },
        { timestamp: '2026-03-10T00:00:00Z', total_value_usd: 150, total_cost_basis_usd: 100,
          unrealized_pl_usd: 50, unrealized_pl_percent: 50 },
      ],
    });
    renderCockpit();

    await act(async () => {});
    expect(screen.getByText(/Change since first snapshot: \+\$70\.00 from 2025-12-20/)).toBeDefined();
    fireEvent.click(screen.getByRole('button', { name: '1M' }));

    expect(screen.getByText((_, element) =>
      element?.textContent === 'Change since first snapshot: +$30.00 from 2026-02-20',
    )).toBeDefined();
  });

  it('keeps history loading explicit before the overview request resolves', async () => {
    mockDashboardFetch();
    renderCockpit();

    expect(await screen.findByText('Loading history…')).toBeDefined();
  });

  it('keeps fewer than two snapshots in the selected range explicit', async () => {
    mockDashboardFetch({
      has_data: true,
      staleness: POPULATED.staleness,
      points: [
        { timestamp: '2026-03-10T00:00:00Z', total_value_usd: 150, total_cost_basis_usd: 100,
          unrealized_pl_usd: 50, unrealized_pl_percent: 50 },
      ],
    });
    renderCockpit();

    expect(await screen.findByText('Not enough snapshots in this range to draw a line.')).toBeDefined();
  });

  it('configures the FIFO cost basis line with its tooltip name and dashed series props', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-15T12:00:00Z'));
    lineSpy.mockClear();
    mockDashboardFetch({
      has_data: true,
      staleness: POPULATED.staleness,
      points: [
        { timestamp: '2026-03-01T00:00:00Z', total_value_usd: 100, total_cost_basis_usd: 80,
          unrealized_pl_usd: 20, unrealized_pl_percent: 25 },
        { timestamp: '2026-03-10T00:00:00Z', total_value_usd: 125, total_cost_basis_usd: 90,
          unrealized_pl_usd: 35, unrealized_pl_percent: 38.89 },
      ],
    });
    renderCockpit();

    await act(async () => {});
    expect(lineSpy).toHaveBeenCalled();
    expect(lineSpy.mock.calls.at(-1)?.[0]).toMatchObject({
      dataKey: 'basis', name: 'FIFO cost basis at snapshot', strokeDasharray: '4 4',
    });
  });

  it('shows an explicit error when overview history cannot load', async () => {
    mockDashboardFetch(new TypeError('History unavailable'));
    renderCockpit();

    expect(await screen.findByText(/Could not load history: Cannot reach the API server/)).toBeDefined();
  });
});

describe('Cockpit constrained state', () => {
  it('states plainly that no sync has run rather than showing zeros', async () => {
    mockFetch(EMPTY);
    renderCockpit();

    await waitFor(() => expect(screen.getByText(/no data yet/i)).toBeDefined());
  });
});

describe('Cockpit profit-taking alert', () => {
  it('shows a trim card for a fresh high-scoring opportunity', async () => {
    mockDashboardFetch(undefined, POPULATED, profitWith(SCORED));
    renderCockpit();

    expect(await screen.findByText('Alerts & review items (1)')).toBeDefined();
    expect(screen.getByText('SOL scored 82 — trim candidate')).toBeDefined();
    expect(screen.getByText('Up +$42.50 (+61.20%) at $103.36.')).toBeDefined();
    expect(screen.getByRole('link', { name: 'Review profit-taking' }))
      .toHaveAttribute('href', '/profit');
  });

  it('stays silent when the top score is below the positive boundary', async () => {
    mockDashboardFetch(undefined, POPULATED, profitWith({ ...SCORED, opportunity_score: 60 }));
    renderCockpit();

    await screen.findByText('$57.78');
    expect(screen.queryByText(/Alerts & review items/)).toBeNull();
  });

  it('stays silent on a stale analysis run rather than nudging on dead signal', async () => {
    mockDashboardFetch(undefined, POPULATED, profitWith(
      SCORED, { ...POPULATED.staleness, age_seconds: 7200, is_stale: true }));
    renderCockpit();

    await screen.findByText('$57.78');
    expect(screen.queryByText(/Alerts & review items/)).toBeNull();
  });

  it('stays silent while an analysis run is in flight', async () => {
    mockDashboardFetch(undefined, POPULATED, profitWith(SCORED, POPULATED.staleness, true));
    renderCockpit();

    await screen.findByText('$57.78');
    expect(screen.queryByText(/Alerts & review items/)).toBeNull();
  });

  it('stays silent when no analysis has ever run', async () => {
    mockDashboardFetch(undefined, POPULATED, { ...PROFIT_EMPTY, has_data: false });
    renderCockpit();

    await screen.findByText('$57.78');
    expect(screen.queryByText(/Alerts & review items/)).toBeNull();
  });

  it('renders the dashboard without the card when the profit fetch fails', async () => {
    // The alerts section must not depend on the analysis pipeline being up:
    // every other fetch succeeds here, only /api/strategy/profit rejects.
    mockDashboardFetch(undefined, POPULATED, new Error('Analysis unavailable'));
    renderCockpit();

    await screen.findByText('$57.78');
    expect(screen.queryByText(/Alerts & review items/)).toBeNull();
    expect(screen.queryByText(/cannot reach|failed|error|unable/i)).toBeNull();
  });
});

describe('Cockpit error state', () => {
  it('renders a visible error message when the fetch rejects, not a blank panel or permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    renderCockpit();

    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });

    // Something a user would read as "this failed" must be visible -- not an
    // empty panel and not a stuck loading indicator.
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});
