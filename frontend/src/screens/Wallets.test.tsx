import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Wallets } from './Wallets';

const STALENESS = { cached_at: null, age_seconds: null, is_stale: true };

function walletsPayload(total: number) {
  return {
    has_data: true,
    spot_earn_value_usd: 800,
    futures_value_usd: 200,
    funding_value_usd: 0,
    total_value_usd: total,
    spot_holdings: [
      { symbol: 'BTC', quantity: 0.5, value_usd: 250 },
      { symbol: 'ETH', quantity: 1, value_usd: null },
    ],
    futures_balances: [],
    funding_balances: [],
    staleness: STALENESS,
  };
}

function stubFetch(payload: unknown) {
  const fetchMock = vi.fn(async (url: unknown) => {
    const path = String(url);
    if (path.includes('/api/wallets')) {
      return { ok: true, json: async () => payload };
    }
    if (path.includes('/api/execute/status')) {
      return { ok: true, json: async () => ({ testnet: true, is_live: false }) };
    }
    throw new Error(`unexpected fetch: ${path}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => vi.unstubAllGlobals());

describe('Wallets fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Wallets />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load wallets/i)).toBeDefined();
  });
});

describe('Wallets share of total', () => {
  it('shows each balance as a one-decimal share of the total value', async () => {
    stubFetch(walletsPayload(1000));
    render(<Wallets />);
    await waitFor(() => {
      expect(screen.getByText('25.0%')).toBeDefined();
    });
  });

  it('renders an em dash, never 0.0%, for an unpriced balance', async () => {
    stubFetch(walletsPayload(1000));
    render(<Wallets />);
    await screen.findByText('25.0%');
    // The ETH row has a null value: its share cell must be an em dash.
    const ethRow = screen.getByText('ETH').closest('tr');
    expect(ethRow).not.toBeNull();
    expect(ethRow?.textContent).toContain('—');
    expect(ethRow?.textContent).not.toContain('0.0%');
  });

  it('renders em dashes when there is no total to share against', async () => {
    stubFetch(walletsPayload(0));
    render(<Wallets />);
    await screen.findByText('BTC');
    expect(screen.queryByText(/%/)).toBeNull();
  });
});
