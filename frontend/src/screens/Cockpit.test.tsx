import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Cockpit } from './Cockpit';
import type { CockpitResponse, OverviewResponse, SystemHealthResponse } from '../types';

/** The alert cards link to the screens that resolve them, so the cockpit
 *  requires router context. */
const renderCockpit = () =>
  render(<MemoryRouter><Cockpit /></MemoryRouter>);

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

function mockDashboardFetch(overview: OverviewResponse | Error) {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url === '/api/portfolio/cockpit') {
      return Promise.resolve({ ok: true, json: async () => POPULATED });
    }
    if (url === '/api/overview') {
      if (overview instanceof Error) return Promise.reject(overview);
      return Promise.resolve({ ok: true, json: async () => overview });
    }
    if (url === '/api/system/health') {
      return Promise.resolve({ ok: true, json: async () => HEALTH });
    }
    return Promise.reject(new Error(`Unexpected request: ${url}`));
  }));
}

beforeEach(() => vi.unstubAllGlobals());

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
  });

  it('shows each basis denominator so the two are visibly different', async () => {
    mockFetch(POPULATED);
    renderCockpit();

    await waitFor(() => expect(screen.getByText(/76\.41 net in/)).toBeDefined());
    expect(screen.getByText(/199\.75 cost basis/)).toBeDefined();
  });
});

describe('Cockpit performance history', () => {
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
