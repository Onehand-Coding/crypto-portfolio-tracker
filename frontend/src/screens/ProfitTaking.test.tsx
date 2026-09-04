import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ProfitTaking } from './ProfitTaking';

const STATUS = { is_live: false, testnet: true };

const STATE = {
  has_data: true, is_running: false, error: null,
  staleness: { cached_at: null, age_seconds: null, is_stale: false },
  opportunities: [
    { symbol: 'BTC', unrealized_gain_usd: 100, unrealized_gain_pct: 25,
      opportunity_score: 80, rsi_score: 70, pl_score: 80, resistance_score: 75,
      market_context_score: 60, current_price: 50000, support_level: 45000,
      resistance_level: 55000, reasons: ['RSI overbought'] },
    { symbol: 'ETH', unrealized_gain_usd: 50, unrealized_gain_pct: 15,
      opportunity_score: 70, rsi_score: 65, pl_score: 70, resistance_score: 60,
      market_context_score: 55, current_price: 3000, support_level: 2800,
      resistance_level: 3200, reasons: ['Near resistance'] },
  ],
};

function stubFetch(captured: { body?: Record<string, unknown> }) {
  vi.stubGlobal('fetch', vi.fn(async (url: unknown, init?: { body?: unknown }) => {
    const path = String(url);
    if (path.includes('/api/execute/profit')) {
      if (typeof init?.body === 'string') captured.body = JSON.parse(init.body);
      return { ok: true, json: async () => ({ success: true, testnet: true, messages: [], errors: [] }) };
    }
    const payload = path.includes('/api/execute/status') ? STATUS : STATE;
    return { ok: true, json: async () => payload };
  }));
}

beforeEach(() => vi.unstubAllGlobals());

describe('ProfitTaking fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<ProfitTaking />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});

describe('ProfitTaking per-trade selection', () => {
  it('checks every opportunity by default and sends only checked symbols', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<ProfitTaking />);
    const btc = await screen.findByRole('checkbox', { name: 'Include BTC' });
    const eth = screen.getByRole('checkbox', { name: 'Include ETH' });
    expect(btc).toBeChecked();
    expect(eth).toBeChecked();
    fireEvent.click(eth);
    expect(eth).not.toBeChecked();
    fireEvent.change(screen.getByPlaceholderText('EXECUTE'), { target: { value: 'EXECUTE' } });
    fireEvent.click(screen.getByRole('button', { name: /simulate profit-taking/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body?.symbols).toEqual(['BTC']);
  });

  it('disables execution and hints when nothing is selected', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<ProfitTaking />);
    await screen.findByRole('checkbox', { name: 'Include BTC' });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include BTC' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include ETH' }));
    expect(await screen.findByText(/select at least one trade/i)).toBeDefined();
    expect(screen.getByRole('button', { name: /simulate profit-taking/i })).toBeDisabled();
    expect(captured.body).toBeUndefined();
  });
});
