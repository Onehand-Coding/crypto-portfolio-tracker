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
    // getAllByText, not getByText: this fixture has exactly one holding, so
    // total_value_usd and that holding's value_usd are legitimately equal
    // (57.78) -- "$57.78" correctly renders in two places (header total and
    // the holdings row), not a bug.
    await waitFor(() => expect(screen.getAllByText('$57.78').length).toBeGreaterThan(0));
    expect(screen.getByText(/-\$18\.63/)).toBeDefined();
    // getAllByText: the fixture's single holding's unrealized P/L (-141.97)
    // legitimately equals the FIFO basis P/L for the whole portfolio, so it
    // correctly renders in both the summary and the holdings row.
    expect(screen.getAllByText(/-\$141\.97/).length).toBeGreaterThan(0);
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

describe('Cockpit error state', () => {
  it('renders a visible error message when the fetch rejects, not a blank panel or permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(<Cockpit />);

    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });

    // Something a user would read as "this failed" must be visible -- not an
    // empty panel and not a stuck loading indicator.
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});
