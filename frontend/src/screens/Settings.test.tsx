import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Settings } from './Settings';
import type { SettingsResponse } from '../types';

const SETTINGS: SettingsResponse = {
  minimum_trade_usd: 5,
  testnet_mode: false,
  live_trading_enabled: false,
  profit_taking: {
    enabled: false, min_opportunity_score: 60, min_unrealized_gain_pct: 10,
    min_unrealized_gain_usd: 50, max_gain_take_pct: 50, default_take_percentage: 25,
  },
  p2p_fiat_currency: 'USD',
  crypto_quotes: ['USDT'],
  stablecoin_symbols: ['USDT', 'USDC'],
  trend_analyzer: { rsi_period: 14, rsi_oversold: 30, rsi_overbought: 70, cryptocurrencies: ['BTC-USD'] },
  cleanup_days: 90,
  automation: {
    dca_frequency: 'monthly',
    rebalancing_frequency: 'weekly',
    auto_sync_enabled: false,
    auto_sync_interval_minutes: 5,
  },
  apis: {
    coingecko_timeout: 30, binance_timeout: 60, binance_recv_window: 20000,
    binance_delay_ms: 500, coingecko_delay_ms: 1500,
  },
  history_lookback_days: {
    trades: 90, deposits: 90, withdrawals: 90, p2p_buys: 90,
    internal_transfers: 90, spot_futures_transfers: 90, spot_convert_history: 90,
    simple_earn_rewards: 90, simple_earn_subscriptions: 90, simple_earn_redemptions: 90,
    dividend_history: 90, staking_history: 90,
  },
  logging: {
    level: 'INFO', file_enabled: true,
    file_path: 'logs/portfolio_tracker.log', console_enabled: true,
  },
  trend_timeframes: {
    long_term: { period: '4y', sma_short_window: 10, sma_long_window: 30 },
    swing: { period: '90d', sma_short_window: 10, sma_long_window: 30 },
    day: { period: '7d', sma_short_window: 10, sma_long_window: 30 },
  },
};

const PREVIEW = {
  path: 'logs/portfolio_tracker.log',
  lines: ['2026-01-01 INFO started', '2026-01-01 INFO synced'],
  truncated: true,
  total_lines: 103,
};

function stubFetch() {
  const fetchMock = vi.fn(async (url: unknown, init?: RequestInit) => {
    const path = String(url);
    const method = init?.method ?? 'GET';
    if (path.includes('/api/system/logs/preview')) {
      return { ok: true, json: async () => PREVIEW };
    }
    if (path.includes('/api/system/config/import')) {
      return { ok: true, json: async () => SETTINGS };
    }
    if (path.includes('/api/system/settings')) {
      if (method === 'PUT') {
        return { ok: true, json: async () => SETTINGS };
      }
      return { ok: true, json: async () => SETTINGS };
    }
    throw new Error(`unexpected fetch: ${path} ${method}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function putBody(fetchMock: ReturnType<typeof stubFetch>): Record<string, unknown> {
  const put = fetchMock.mock.calls.find(
    ([url, init]) => String(url).includes('/api/system/settings')
      && (init as RequestInit | undefined)?.method === 'PUT',
  );
  expect(put).toBeDefined();
  return JSON.parse(String((put?.[1] as RequestInit | undefined)?.body)) as Record<string, unknown>;
}

beforeEach(() => vi.unstubAllGlobals());

describe('Settings fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Settings />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load/i)).toBeDefined();
  });
});

describe('Settings schedules save', () => {
  it('carries both frequencies in the PUT payload', async () => {
    const fetchMock = stubFetch();
    render(<Settings />);
    await screen.findByText('Schedules');

    fireEvent.change(screen.getByLabelText('DCA frequency'), { target: { value: 'weekly' } });
    fireEvent.change(screen.getByLabelText('Rebalancing frequency'), { target: { value: 'daily' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));

    await waitFor(() => {
      expect(screen.getByText(/settings saved/i)).toBeDefined();
    });
    const body = putBody(fetchMock);
    expect(body['automation']).toEqual({
      dca_frequency: 'weekly',
      rebalancing_frequency: 'daily',
      auto_sync_enabled: false,
      auto_sync_interval_minutes: 5,
    });
  });
});

describe('Settings auto-sync save', () => {
  it('carries the auto-sync toggle and interval in the PUT payload', async () => {
    const fetchMock = stubFetch();
    render(<Settings />);
    await screen.findByText('Schedules');

    fireEvent.click(screen.getByLabelText('Sync automatically'));
    fireEvent.change(screen.getByLabelText('Every N minutes'), { target: { value: '10' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));

    await waitFor(() => {
      expect(screen.getByText(/settings saved/i)).toBeDefined();
    });
    const body = putBody(fetchMock);
    expect(body['automation']).toMatchObject({
      auto_sync_enabled: true,
      auto_sync_interval_minutes: 10,
    });
  });
});

describe('Settings trend periods', () => {
  it('carries edited period strings in the PUT payload', async () => {
    const fetchMock = stubFetch();
    render(<Settings />);
    await screen.findByText('Trend timeframes');

    fireEvent.change(screen.getByLabelText('long_term period'), { target: { value: '5y' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));

    await waitFor(() => {
      expect(screen.getByText(/settings saved/i)).toBeDefined();
    });
    const body = putBody(fetchMock) as Record<string, Record<string, Record<string, unknown>>>;
    expect(body['trend_timeframes']['long_term']['period']).toBe('5y');
    expect(body['trend_timeframes']['swing']['period']).toBe('90d');
  });
});

describe('Settings config import', () => {  it('posts the picked file as multipart FormData under the file key', async () => {
    const fetchMock = stubFetch();
    render(<Settings />);
    await screen.findByText('Config transfer');

    const file = new File(['{}'], 'config.json', { type: 'application/json' });
    fireEvent.change(screen.getByLabelText(/config file/i), { target: { files: [file] } });
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));
    fireEvent.click(screen.getByRole('button', { name: 'Confirm import' }));

    await waitFor(() => {
      expect(screen.getByText(/config imported/i)).toBeDefined();
    });
    const post = fetchMock.mock.calls.find(([url]) => String(url).includes('/api/system/config/import'));
    expect(post).toBeDefined();
    const body = (post?.[1] as RequestInit | undefined)?.body;
    expect(body).toBeInstanceOf(FormData);
    expect((body as FormData).get('file')).toBeDefined();
  });

  it('requires confirmation before posting the import', async () => {
    const fetchMock = stubFetch();
    render(<Settings />);
    await screen.findByText('Config transfer');

    const file = new File(['{}'], 'config.json', { type: 'application/json' });
    fireEvent.change(screen.getByLabelText(/config file/i), { target: { files: [file] } });
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/api/system/config/import')))
      .toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/api/system/config/import')))
      .toBe(false);
  });
});

describe('Settings log preview', () => {
  it('renders the returned log lines', async () => {
    stubFetch();
    render(<Settings />);
    await screen.findByText('Logging');

    fireEvent.click(screen.getByRole('button', { name: 'Preview' }));

    await waitFor(() => {
      expect(screen.getByText(/started/)).toBeDefined();
    });
    expect(screen.getByText(/synced/)).toBeDefined();
  });
});
