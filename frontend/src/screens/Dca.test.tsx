import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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

describe('Dca preview pending state', () => {
  it('disables the button and swaps the label while the preview is in flight', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: unknown) => {
      const path = String(url);
      if (path.includes('/api/strategy/dca/preview')) {
        return new Promise<never>(() => {});
      }
      const payload = path.includes('/api/strategy/completion') ? COMPLETION
        : path.includes('/api/execute/status') ? STATUS : DCA_STATE;
      return { ok: true, json: async () => payload };
    }));
    render(<Dca />);
    fireEvent.click(await screen.findByRole('button', { name: /preview allocation/i }));
    expect(await screen.findByRole('button', { name: /previewing/i })).toBeDisabled();
  });
});

describe('Dca per-trade selection', () => {
  const PREVIEW = {
    strategy: 'target_weight', amount_usd: 50, valid: true, message: null,
    allocations: [
      { symbol: 'BTC', amount_usd: 30, quantity: 0.0003,
        current_allocation_pct: 10, target_allocation_pct: 35 },
      { symbol: 'ETH', amount_usd: 20, quantity: 0.01,
        current_allocation_pct: 5, target_allocation_pct: 30 },
    ],
  };

  function stubFetchWithExecute(captured: { body?: Record<string, unknown> }) {
    vi.stubGlobal('fetch', vi.fn(async (url: unknown, init?: { body?: unknown }) => {
      const path = String(url);
      if (path.includes('/api/execute/dca')) {
        if (typeof init?.body === 'string') captured.body = JSON.parse(init.body);
        return { ok: true, json: async () => ({ success: true, testnet: true, messages: [], errors: [] }) };
      }
      if (path.includes('/api/strategy/dca/preview')) {
        return { ok: true, json: async () => PREVIEW };
      }
      const payload = path.includes('/api/strategy/completion') ? COMPLETION
        : path.includes('/api/execute/status') ? STATUS : DCA_STATE;
      return { ok: true, json: async () => payload };
    }));
  }

  async function renderWithPreview(captured: { body?: Record<string, unknown> }) {
    stubFetchWithExecute(captured);
    render(<Dca />);
    fireEvent.click(await screen.findByRole('button', { name: /preview allocation/i }));
    await screen.findByRole('checkbox', { name: 'Include BTC' });
  }

  it('checks every allocation by default and sends only checked trades', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    await renderWithPreview(captured);
    const btc = screen.getByRole('checkbox', { name: 'Include BTC' });
    const eth = screen.getByRole('checkbox', { name: 'Include ETH' });
    expect(btc).toBeChecked();
    expect(eth).toBeChecked();
    fireEvent.click(eth);
    expect(eth).not.toBeChecked();
    fireEvent.change(screen.getByPlaceholderText('EXECUTE'), { target: { value: 'EXECUTE' } });
    fireEvent.click(screen.getByRole('button', { name: /simulate dca/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body?.trades).toEqual([{ asset: 'BTC', amount: 30 }]);
  });

  it('disables execution and hints when nothing is selected', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    await renderWithPreview(captured);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include BTC' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include ETH' }));
    expect(await screen.findByText(/select at least one trade/i)).toBeDefined();
    expect(screen.getByRole('button', { name: /simulate dca/i })).toBeDisabled();
    expect(captured.body).toBeUndefined();
  });
});
