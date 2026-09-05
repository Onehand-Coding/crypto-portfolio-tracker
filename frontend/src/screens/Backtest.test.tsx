import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Backtest } from './Backtest';

const BT_STATE = {
  has_data: false, is_running: false, error: null,
  staleness: { cached_at: null, age_seconds: null, is_stale: true },
  result: null, trade_log: null, value_history: null, config: null,
};

const HEALTH = {
  target_allocation: { BTC: 0.5, ETH: 0.5 },
};

function stubFetch(captured: { body?: Record<string, unknown> }) {
  vi.stubGlobal('fetch', vi.fn(async (url: unknown, init?: { body?: unknown }) => {
    const path = String(url);
    if (path.includes('/api/strategy/backtest/run')) {
      if (typeof init?.body === 'string') captured.body = JSON.parse(init.body);
      return { ok: true, json: async () => ({ status: 'started' }) };
    }
    const payload = path.includes('/api/system/health') ? HEALTH : BT_STATE;
    return { ok: true, json: async () => payload };
  }));
}

beforeEach(() => vi.unstubAllGlobals());

describe('Backtest fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Backtest />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});

describe('Backtest run payload', () => {
  it('omits custom entirely for a plain run', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Backtest />);
    fireEvent.click(await screen.findByRole('button', { name: /run backtest/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body).not.toHaveProperty('custom');
    expect(captured.body?.period).toBe('2y');
  });

  it('posts a clamped custom block for an advanced run', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Backtest />);
    await screen.findByRole('button', { name: /run backtest/i });
    fireEvent.click(screen.getByRole('button', { name: /advanced parameters/i }));
    fireEvent.change(screen.getByLabelText('Majors drift threshold (%)'), { target: { value: '0' } });
    fireEvent.change(screen.getByLabelText('Alts buy multiplier'), { target: { value: '9' } });
    fireEvent.change(screen.getByLabelText('Majors sell multiplier'), { target: { value: 'garbage' } });
    fireEvent.click(screen.getByRole('button', { name: /run backtest/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    const custom = captured.body?.custom as Record<string, unknown>;
    expect(custom['majors_drift']).toBe(1.0);
    expect(custom['alts_buy']).toBe(2.0);
    expect(custom['majors_sell']).toBe(0.5);
    expect(custom['suppress_bear']).toBe(true);
    expect(custom['allocation']).toEqual({ BTC: 0.5, ETH: 0.5 });
  });

  it('disables Run and hints on a bad custom period', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Backtest />);
    await screen.findByRole('button', { name: /run backtest/i });
    fireEvent.click(screen.getByRole('button', { name: 'Custom' }));
    fireEvent.change(screen.getByLabelText('Custom period'), { target: { value: 'decade' } });
    expect(screen.getByRole('button', { name: /run backtest/i })).toBeDisabled();
    expect(screen.getByText(/must look like 6y/i)).toBeDefined();
    expect(captured.body).toBeUndefined();
    fireEvent.change(screen.getByLabelText('Custom period'), { target: { value: '6y' } });
    expect(screen.getByRole('button', { name: /run backtest/i })).not.toBeDisabled();
  });

  it('disables Run when custom weights do not sum to 100%', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Backtest />);
    await screen.findByRole('button', { name: /run backtest/i });
    fireEvent.click(screen.getByRole('button', { name: /advanced parameters/i }));
    await screen.findByLabelText('BTC weight (%)');
    fireEvent.change(screen.getByLabelText('BTC weight (%)'), { target: { value: '10' } });
    expect(screen.getByRole('button', { name: /run backtest/i })).toBeDisabled();
    expect(screen.getByText(/must sum to 100%/i)).toBeDefined();
    expect(captured.body).toBeUndefined();
  });
});

describe('Backtest custom allocation assets', () => {
  it('adds a new asset and includes it in the run payload', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Backtest />);
    await screen.findByRole('button', { name: /run backtest/i });
    fireEvent.click(screen.getByRole('button', { name: /advanced parameters/i }));
    await screen.findByLabelText('BTC weight (%)');
    fireEvent.change(screen.getByPlaceholderText('DOGE'), { target: { value: 'doge' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add asset' }));
    expect(await screen.findByLabelText('DOGE weight (%)')).toBeDefined();
    fireEvent.change(screen.getByLabelText('DOGE weight (%)'), { target: { value: '0' } });
    fireEvent.click(screen.getByRole('button', { name: /run backtest/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    const custom = captured.body?.custom as Record<string, unknown>;
    expect(custom['allocation']).toMatchObject({ BTC: 0.5, ETH: 0.5, DOGE: 0 });
  });

  it('rejects invalid and duplicate symbols with a hint', async () => {
    stubFetch({});
    render(<Backtest />);
    await screen.findByRole('button', { name: /run backtest/i });
    fireEvent.click(screen.getByRole('button', { name: /advanced parameters/i }));
    await screen.findByLabelText('BTC weight (%)');
    fireEvent.change(screen.getByPlaceholderText('DOGE'), { target: { value: 'x' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add asset' }));
    expect(await screen.findByText(/2–10 letters/i)).toBeDefined();
    fireEvent.change(screen.getByPlaceholderText('DOGE'), { target: { value: 'BTC' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add asset' }));
    expect(await screen.findByText(/already in the allocation/i)).toBeDefined();
  });

  it('resets added assets back to configured targets', async () => {
    stubFetch({});
    render(<Backtest />);
    await screen.findByRole('button', { name: /run backtest/i });
    fireEvent.click(screen.getByRole('button', { name: /advanced parameters/i }));
    await screen.findByLabelText('BTC weight (%)');
    fireEvent.change(screen.getByPlaceholderText('DOGE'), { target: { value: 'DOGE' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add asset' }));
    await screen.findByLabelText('DOGE weight (%)');
    fireEvent.click(screen.getByRole('button', { name: 'Reset to defaults' }));
    await waitFor(() => {
      expect(screen.queryByLabelText('DOGE weight (%)')).toBeNull();
    });
  });
});
