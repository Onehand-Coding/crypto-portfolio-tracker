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
    await waitFor(() => expect(screen.getByText(/show completion plan/i)).toBeDefined());
    expect(screen.queryByText(/101\.07/)).toBeNull();
    screen.getByRole('button', { name: /completion plan/i }).click();
    await waitFor(() => expect(screen.getByText(/101\.07/)).toBeDefined());
    expect(screen.getByText(/anchored by BTC/i)).toBeDefined();
    expect(screen.getByText(/247\.56/)).toBeDefined();
  });

  it('renders the invalid message instead of a table when there is no anchor', async () => {
    stubFetch({ valid: false, message: 'No holdings to anchor from yet.',
                anchor_symbol: null, implied_total_usd: null,
                additional_total_usd: 0, rows: [] });
    render(<Dca />);
    await screen.findAllByRole('button', { name: /completion plan/i });
    screen.getByRole('button', { name: /completion plan/i }).click();
    await waitFor(() =>
      expect(screen.getByText(/no holdings to anchor/i)).toBeDefined());
  });
});
