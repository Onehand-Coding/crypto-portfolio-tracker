import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { TradeLog } from './TradeLog';
import type { TransactionsResponse } from '../types';

const DATA: TransactionsResponse = {
  has_data: true,
  count: 2,
  rows: [
    {
      timestamp: '2026-01-01T12:00:00', symbol: 'BTC', type: 'BUY',
      quantity: 0.5, price_usd: 40000, value_usd: 20000, fee_usd: 1,
      source: 'spot', notes: null,
    },
    {
      timestamp: '2026-02-01T12:00:00', symbol: 'ETH', type: 'SELL',
      quantity: 1, price_usd: 2500, value_usd: 2500, fee_usd: 1,
      source: 'spot', notes: null,
    },
  ],
  staleness: { cached_at: null, age_seconds: null, is_stale: false },
};

function stubFetch() {
  const fetchMock = vi.fn(async (url: unknown, _init?: RequestInit) => {
    const path = String(url);
    if (path.includes('/api/reports/generate')) {
      return { ok: true, json: async () => ({ name: 'transactions_20260101.xlsx', path: '/exports/transactions_20260101.xlsx' }) };
    }
    if (path.includes('/api/transactions')) return { ok: true, json: async () => DATA };
    throw new Error(`unexpected fetch: ${path}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => vi.unstubAllGlobals());

describe('TradeLog fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<TradeLog />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load/i)).toBeDefined();
  });
});

describe('TradeLog Excel export', () => {
  it('posts transactions plus excel and offers the returned file for download', async () => {
    const fetchMock = stubFetch();
    render(<TradeLog />);
    fireEvent.click(await screen.findByRole('button', { name: /export excel/i }));
    await waitFor(() => {
      expect(screen.getByRole('link', { name: /download/i })).toBeDefined();
    });
    const post = fetchMock.mock.calls.find(([url]) => String(url).includes('/api/reports/generate'));
    expect(post).toBeDefined();
    expect(JSON.parse(String((post?.[1] as RequestInit | undefined)?.body)))
      .toEqual({ data_type: 'transactions', format: 'excel' });
    expect(screen.getByRole('link', { name: /download/i }).getAttribute('href'))
      .toContain('transactions_20260101.xlsx');
  });
});
