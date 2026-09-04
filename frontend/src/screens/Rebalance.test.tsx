import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Rebalance } from './Rebalance';

const STATUS = { is_live: false, testnet: true };

const STATE = {
  has_data: true, is_running: false, error: null,
  staleness: { cached_at: null, age_seconds: null, is_stale: false },
  suggestions: [
    { symbol: 'BTC', action: 'BUY', current_value_usd: 100,
      current_allocation_pct: 10, target_allocation_pct: 35, drift_pct: -25,
      action_amount_usd: 50, action_quantity: 0.0005,
      reason: 'Below target', raw: {} },
    { symbol: 'ETH', action: 'SELL', current_value_usd: 200,
      current_allocation_pct: 40, target_allocation_pct: 30, drift_pct: 10,
      action_amount_usd: 20, action_quantity: 0.01,
      reason: 'Above target', raw: {} },
    { symbol: 'XRP', action: 'HOLD', current_value_usd: 10,
      current_allocation_pct: 5, target_allocation_pct: 5, drift_pct: 0,
      action_amount_usd: 0, action_quantity: 0,
      reason: null, raw: {} },
  ],
};

function stubFetch(captured: { body?: Record<string, unknown> }) {
  vi.stubGlobal('fetch', vi.fn(async (url: unknown, init?: { body?: unknown }) => {
    const path = String(url);
    if (path.includes('/api/execute/rebalance')) {
      if (typeof init?.body === 'string') captured.body = JSON.parse(init.body);
      return { ok: true, json: async () => ({ success: true, testnet: true, messages: [], errors: [] }) };
    }
    const payload = path.includes('/api/execute/status') ? STATUS : STATE;
    return { ok: true, json: async () => payload };
  }));
}

beforeEach(() => vi.unstubAllGlobals());

describe('Rebalance fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Rebalance />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});

describe('Rebalance per-trade selection', () => {
  it('checks every actionable suggestion by default and sends only checked symbols', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Rebalance />);
    const btc = await screen.findByRole('checkbox', { name: 'Include BTC' });
    const eth = screen.getByRole('checkbox', { name: 'Include ETH' });
    expect(btc).toBeChecked();
    expect(eth).toBeChecked();
    // A HOLD row is not actionable, so it gets no checkbox.
    expect(screen.queryByRole('checkbox', { name: 'Include XRP' })).toBeNull();
    fireEvent.click(eth);
    expect(eth).not.toBeChecked();
    fireEvent.change(screen.getByPlaceholderText('EXECUTE'), { target: { value: 'EXECUTE' } });
    fireEvent.click(screen.getByRole('button', { name: /simulate rebalance/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body?.symbols).toEqual(['BTC']);
  });

  it('disables execution and hints when nothing is selected', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Rebalance />);
    await screen.findByRole('checkbox', { name: 'Include BTC' });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include BTC' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include ETH' }));
    expect(await screen.findByText(/select at least one trade/i)).toBeDefined();
    expect(screen.getByRole('button', { name: /simulate rebalance/i })).toBeDisabled();
    expect(captured.body).toBeUndefined();
  });
});
