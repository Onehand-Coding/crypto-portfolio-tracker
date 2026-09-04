import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Realized } from './Realized';
import type { RealizedResponse } from '../types';

const DATA: RealizedResponse = {
  has_data: true,
  rows: [
    {
      date: '2026-02-01T12:00:00', year: 2026, symbol: 'ETH', quantity: 1,
      proceeds_usd: 2500, cost_basis_usd: 2000, gain_usd: 500,
    },
  ],
  by_asset: [
    {
      symbol: 'ETH', total_gain_usd: 500,
      total_proceeds_usd: 2500, total_cost_basis_usd: 2000,
    },
  ],
  total_gain_usd: 500,
  total_proceeds_usd: 2500,
  total_cost_basis_usd: 2000,
  staleness: { cached_at: null, age_seconds: null, is_stale: false },
};

function stubFetch() {
  const fetchMock = vi.fn(async (url: unknown, _init?: RequestInit) => {
    const path = String(url);
    if (path.includes('/api/reports/realized')) {
      return { ok: true, json: async () => ({ name: 'realized_20260101.xlsx', path: '/exports/realized_20260101.xlsx' }) };
    }
    if (path.includes('/api/realized')) return { ok: true, json: async () => DATA };
    throw new Error(`unexpected fetch: ${path}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => vi.unstubAllGlobals());

describe('Realized fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Realized />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load/i)).toBeDefined();
  });
});

describe('Realized Excel export', () => {
  it('posts excel and offers the returned file for download', async () => {
    const fetchMock = stubFetch();
    render(<Realized />);
    fireEvent.click(await screen.findByRole('button', { name: /export excel/i }));
    await waitFor(() => {
      expect(screen.getByRole('link', { name: /download/i })).toBeDefined();
    });
    const post = fetchMock.mock.calls.find(([url]) => String(url).includes('/api/reports/realized'));
    expect(post).toBeDefined();
    expect(JSON.parse(String((post?.[1] as RequestInit | undefined)?.body)))
      .toEqual({ format: 'excel' });
    expect(screen.getByRole('link', { name: /download/i }).getAttribute('href'))
      .toContain('realized_20260101.xlsx');
  });
});
