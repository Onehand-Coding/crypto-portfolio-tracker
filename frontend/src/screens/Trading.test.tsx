import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Trading } from './Trading';

const COCKPIT = {
  total_value_usd: 5000,
  holdings: [
    { symbol: 'BTC', value_usd: 1000, current_price: 50000, total_quantity: 0.02 },
    { symbol: 'ETH', value_usd: 500, current_price: 3000, total_quantity: 0.166 },
  ],
};

const HEALTH = { minimum_trade_usd: 5, target_allocation: { BTC: 0.5, ETH: 0.5 } };
const STATUS = { is_live: false, testnet: true };

beforeEach(() => vi.unstubAllGlobals());

describe('Trading fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Trading />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});

describe('Trading units', () => {
  async function executeWith(captured: { body?: Record<string, unknown> }) {
    vi.stubGlobal('fetch', vi.fn(async (url: unknown, init?: { body?: unknown }) => {
      const path = String(url);
      if (path.includes('/api/execute/trade')) {
        if (typeof init?.body === 'string') captured.body = JSON.parse(init.body);
        return { ok: true, json: async () => ({ success: true, testnet: true, messages: [], errors: [] }) };
      }
      if (path.includes('/api/portfolio/cockpit')) {
        return { ok: true, json: async () => COCKPIT };
      }
      if (path.includes('/api/system/health')) {
        return { ok: true, json: async () => HEALTH };
      }
      return { ok: true, json: async () => STATUS };
    }));
    render(<Trading />);
    await screen.findByText(/portfolio impact/i);
    fireEvent.change(screen.getByPlaceholderText('EXECUTE'), { target: { value: 'EXECUTE' } });
    fireEvent.click(screen.getByRole('button', { name: /simulate buy/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
  }

  it('posts USD amounts as quote quantity by default', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    await executeWith(captured);
    expect(captured.body).toMatchObject({
      trade_type: 'BUY', symbol: 'BTC', amount: 50, is_quote_qty: true, confirm: true,
    });
  });

  it('posts coin quantities with is_quote_qty false after switching units', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    vi.stubGlobal('fetch', vi.fn(async (url: unknown, init?: { body?: unknown }) => {
      const path = String(url);
      if (path.includes('/api/execute/trade')) {
        if (typeof init?.body === 'string') captured.body = JSON.parse(init.body);
        return { ok: true, json: async () => ({ success: true, testnet: true, messages: [], errors: [] }) };
      }
      if (path.includes('/api/portfolio/cockpit')) {
        return { ok: true, json: async () => COCKPIT };
      }
      if (path.includes('/api/system/health')) {
        return { ok: true, json: async () => HEALTH };
      }
      return { ok: true, json: async () => STATUS };
    }));
    render(<Trading />);
    await screen.findByText(/portfolio impact/i);
    fireEvent.click(screen.getByRole('button', { name: /btc units/i }));
    fireEvent.change(screen.getByPlaceholderText('EXECUTE'), { target: { value: 'EXECUTE' } });
    fireEvent.click(screen.getByRole('button', { name: /simulate buy/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body).toMatchObject({
      trade_type: 'BUY', symbol: 'BTC', amount: 50, is_quote_qty: false, confirm: true,
    });
  });
});
