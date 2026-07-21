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

describe('CapitalFlow error state', () => {
  it('renders a visible error message when the fetch rejects, not a blank panel or permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(<CapitalFlow />);

    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });

    // Something a user would read as "this failed" must be visible -- not an
    // empty panel and not a stuck loading indicator.
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});

describe('CapitalFlow unpriced inflow caption', () => {
  it('shows "excludes 1 unpriced row" caption on Total in when one inflow row is unpriced', async () => {
    const payload: CapitalFlowResponse = {
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
    mockFetch(payload);
    render(<CapitalFlow />);
    await waitFor(() =>
      expect(screen.getByText('excludes 1 unpriced row')).toBeDefined());
  });

  it('shows "excludes 2 unpriced rows" (plural) caption on Total in when two inflow rows are unpriced', async () => {
    const payload: CapitalFlowResponse = {
      rows: [
        { source: 'Binance P2P Buy', type: 'BUY', direction: 'in', quantity: 100,
          price_usd: 0.0179, value_usd: 1.79, provenance: 'computed', is_suspect: false },
        { source: 'Binance P2P Buy', type: 'BUY', direction: 'in', quantity: 50,
          price_usd: 0, value_usd: 0, provenance: 'failed_lookup', is_suspect: true },
        { source: 'Binance P2P Buy', type: 'BUY', direction: 'in', quantity: 30,
          price_usd: 0, value_usd: 0, provenance: 'failed_lookup', is_suspect: true },
      ],
      total_in_usd: 1.79,
      total_out_usd: 0,
      net_invested_usd: 1.79,
      suspect_count: 2,
    };
    mockFetch(payload);
    render(<CapitalFlow />);
    await waitFor(() =>
      expect(screen.getByText('excludes 2 unpriced rows')).toBeDefined());
  });

  it('does not show unpriced row caption on Total in when no inflow rows are unpriced', async () => {
    const payload: CapitalFlowResponse = {
      rows: [
        { source: 'Binance P2P Buy', type: 'BUY', direction: 'in', quantity: 100,
          price_usd: 0.0179, value_usd: 1.79, provenance: 'computed', is_suspect: false },
      ],
      total_in_usd: 1.79,
      total_out_usd: 0,
      net_invested_usd: 1.79,
      suspect_count: 0,
    };
    mockFetch(payload);
    render(<CapitalFlow />);
    await waitFor(() => {
      expect(screen.queryByText(/excludes.*unpriced row/)).toBeNull();
    });
  });

  it('does not show unpriced row caption on Total in when only outflow rows are unpriced', async () => {
    const payload: CapitalFlowResponse = {
      rows: [
        { source: 'Binance P2P Buy', type: 'BUY', direction: 'in', quantity: 100,
          price_usd: 0.0179, value_usd: 1.79, provenance: 'computed', is_suspect: false },
        { source: 'Binance Withdrawal', type: 'SELL', direction: 'out', quantity: 50,
          price_usd: 0, value_usd: 0, provenance: 'failed_lookup', is_suspect: true },
      ],
      total_in_usd: 1.79,
      total_out_usd: 0,
      net_invested_usd: 1.79,
      suspect_count: 1,
    };
    mockFetch(payload);
    render(<CapitalFlow />);
    await waitFor(() => {
      expect(screen.queryByText(/excludes.*unpriced row/)).toBeNull();
    });
  });
});
