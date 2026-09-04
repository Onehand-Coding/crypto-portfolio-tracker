import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Technical } from './Technical';

function techRow(symbol: string) {
  return {
    symbol,
    price: 67000,
    rsi: 55.5,
    sma_short: 66000,
    sma_long: 60000,
    support: 64000,
    resistance: 70000,
    conditions: ['ABOVE_SMA_SHORT'],
  };
}

const TECHNICAL = {
  has_data: true, is_running: false, error: null,
  staleness: { cached_at: null, age_seconds: null, is_stale: false },
  timeframes: {
    swing: [techRow('BTC'), techRow('ETH')],
    long_term: [techRow('BTC')],
    day: [],
  },
  bear_market: null,
};

const POINTS = [
  { date: '2026-08-01', close: 65000, sma_short: 64000, sma_long: 60000, rsi: 50.0, macd: 100, macd_signal: 90, macd_hist: 10 },
  { date: '2026-08-02', close: 65500, sma_short: 64200, sma_long: 60100, rsi: 52.0, macd: 110, macd_signal: 95, macd_hist: 15 },
  { date: '2026-08-03', close: 66000, sma_short: 64400, sma_long: 60200, rsi: null, macd: 120, macd_signal: 100, macd_hist: 20 },
  { date: '2026-08-04', close: 66500, sma_short: 64600, sma_long: 60300, rsi: 58.0, macd: 130, macd_signal: 105, macd_hist: 25 },
  { date: '2026-08-05', close: 67543, sma_short: 64800, sma_long: 60400, rsi: 60.0, macd: 140, macd_signal: 110, macd_hist: 30 },
];

const INDICATORS = {
  has_data: true, is_running: false, error: null,
  staleness: { cached_at: null, age_seconds: null, is_stale: false },
  symbol: 'BTC', timeframe: 'swing', points: POINTS,
};

let technicalOverride: Record<string, unknown> | null = null;
let indicatorsOverride: Record<string, unknown> | null = null;

function stubFetch(captured: { body?: Record<string, unknown> }) {
  vi.stubGlobal('fetch', vi.fn(async (url: unknown, init?: { body?: unknown }) => {
    const path = String(url);
    if (path.includes('/api/strategy/indicators/run')) {
      if (typeof init?.body === 'string') captured.body = JSON.parse(init.body);
      return { ok: true, json: async () => ({ status: 'started' }) };
    }
    if (path.includes('/api/strategy/indicators')) {
      return { ok: true, json: async () => indicatorsOverride ?? INDICATORS };
    }
    return { ok: true, json: async () => technicalOverride ?? TECHNICAL };
  }));
}

beforeEach(() => {
  vi.unstubAllGlobals();
  technicalOverride = null;
  indicatorsOverride = null;
});

describe('Technical fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Technical />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/cannot reach|failed|error|unable/i)).toBeDefined();
  });
});

describe('Per-coin indicator history', () => {
  it('renders coin and timeframe pickers', async () => {
    stubFetch({});
    render(<Technical />);
    const coin = await screen.findByLabelText('Coin');
    expect(coin).toBeDefined();
    expect(screen.getByLabelText('Timeframe')).toBeDefined();
    // Picker options come from the screen's existing technical payload.
    expect(screen.getByRole('option', { name: 'BTC' })).toBeDefined();
    expect(screen.getByRole('option', { name: 'ETH' })).toBeDefined();
  });

  it('posts symbol and timeframe to the run endpoint', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Technical />);
    fireEvent.click(await screen.findByRole('button', { name: /run indicators/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body).toEqual({ symbol: 'BTC', timeframe: 'swing' });
  });

  it('posts the newly picked coin and timeframe', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    stubFetch(captured);
    render(<Technical />);
    await screen.findByLabelText('Coin');
    fireEvent.change(screen.getByLabelText('Coin'), { target: { value: 'ETH' } });
    fireEvent.change(screen.getByLabelText('Timeframe'), { target: { value: 'day' } });
    fireEvent.click(screen.getByRole('button', { name: /run indicators/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body).toEqual({ symbol: 'ETH', timeframe: 'day' });
  });

  it('strips a -USD suffix before sending the symbol', async () => {
    const captured: { body?: Record<string, unknown> } = {};
    technicalOverride = {
      ...TECHNICAL,
      timeframes: { swing: [techRow('BTC-USD')], long_term: [], day: [] },
    };
    stubFetch(captured);
    render(<Technical />);
    fireEvent.click(await screen.findByRole('button', { name: /run indicators/i }));
    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body).toEqual({ symbol: 'BTC', timeframe: 'swing' });
  });

  it('plots fetched points with data-driven labels, surviving a null rsi', async () => {
    stubFetch({});
    render(<Technical />);
    // Latest close of the 5-point fixture, rendered as a value label.
    await screen.findByText(/\$67,543\.00/);
    expect(screen.getByText(/5 points/)).toBeDefined();
    // Chart panels render despite the null-rsi point (nulls gap, never crash).
    expect(screen.getByRole('heading', { name: 'Price and moving averages' })).toBeDefined();
    expect(screen.getByRole('heading', { name: 'RSI' })).toBeDefined();
    expect(screen.getByRole('heading', { name: 'MACD' })).toBeDefined();
  });

  it('explains empty history as needing network', async () => {
    indicatorsOverride = { ...INDICATORS, has_data: false, points: [] };
    stubFetch({});
    render(<Technical />);
    const empty = await screen.findByText(/needs network/i);
    expect(empty).toBeDefined();
  });

  it('shows the backend error text when the run failed', async () => {
    indicatorsOverride = { ...INDICATORS, error: 'boom' };
    stubFetch({});
    render(<Technical />);
    await screen.findByText(/boom/);
  });
});
