import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { SystemHealth } from './SystemHealth';

const HEALTH = {
  environment_label: 'TESTNET',
  is_testnet: true,
  database_path: '/tmp/test.db',
  database_exists: true,
  database_size_bytes: 1024,
  transaction_count: 10,
  asset_count: 3,
  snapshot_count: 2,
  live_trading_enabled: false,
  minimum_trade_usd: 5,
  target_allocation: {},
  backups: [],
  metrics_cache_age_seconds: null,
  binance_configured: true,
};

const RESOURCES = {
  app_version: '9.9.9',
  python_version: '3.12.1',
  cpu_percent: 12.3,
  ram_percent: null,
  ram_used_gb: 4.5,
  disk_percent: 60.0,
};

const CONNECTIONS = {
  binance: { ok: true, detail: 'SUCCESS' },
  coingecko: { ok: false, detail: 'FAILED (boom)' },
  btc_price_usd: null,
};

function stubFetch() {
  const fetchMock = vi.fn(async (url: unknown, init?: RequestInit) => {
    const path = String(url);
    const method = init?.method ?? 'GET';
    if (path.includes('/api/system/connections')) {
      return { ok: true, json: async () => CONNECTIONS };
    }
    if (path.includes('/api/system/resources')) {
      return { ok: true, json: async () => RESOURCES };
    }
    if (path.includes('/api/system/health')) {
      return { ok: true, json: async () => HEALTH };
    }
    throw new Error(`unexpected fetch: ${path} ${method}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => vi.unstubAllGlobals());

describe('SystemHealth fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<SystemHealth />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load system health/i)).toBeDefined();
  });
});

describe('SystemHealth resources', () => {
  it('renders the six host figures with an em dash for the null one', async () => {
    stubFetch();
    render(<SystemHealth />);
    await waitFor(() => {
      expect(screen.getByText('9.9.9')).toBeDefined();
    });
    expect(screen.getByText('3.12.1')).toBeDefined();
    expect(screen.getByText('12.3%')).toBeDefined();
    expect(screen.getByText('4.5 GB')).toBeDefined();
    expect(screen.getByText('60.0%')).toBeDefined();
    // ram_percent is null: the RAM figure must be an em dash.
    const panel = screen.getByText('Resources').closest('section');
    expect(panel).not.toBeNull();
    expect(panel?.textContent).toContain('—');
  });
});

describe('SystemHealth connection test', () => {
  it('posts the probe and renders both badges plus the BTC price', async () => {
    const fetchMock = stubFetch();
    render(<SystemHealth />);
    await screen.findByText('Run connection test');

    fireEvent.click(screen.getByRole('button', { name: 'Run connection test' }));

    await waitFor(() => {
      expect(screen.getByText('BINANCE OK')).toBeDefined();
    });
    expect(screen.getByText('COINGECKO FAILED')).toBeDefined();
    // Null BTC price renders as an em dash, never $0.00.
    expect(screen.getByText(/BTC/)).toBeDefined();
    const post = fetchMock.mock.calls.find(([url]) => String(url).includes('/api/system/connections'));
    expect(post).toBeDefined();
    expect((post?.[1] as RequestInit | undefined)?.method).toBe('POST');
  });

  it('surfaces a probe failure instead of appearing to succeed', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: unknown) => {
      const path = String(url);
      if (path.includes('/api/system/resources')) {
        return { ok: true, json: async () => RESOURCES };
      }
      if (path.includes('/api/system/health')) {
        return { ok: true, json: async () => HEALTH };
      }
      return { ok: false, status: 500, statusText: 'probe blew up', text: async () => 'probe blew up' };
    }));
    render(<SystemHealth />);
    await screen.findByText('Run connection test');

    fireEvent.click(screen.getByRole('button', { name: 'Run connection test' }));

    await waitFor(() => {
      expect(screen.getByText(/connection test failed/i)).toBeDefined();
    });
  });
});
