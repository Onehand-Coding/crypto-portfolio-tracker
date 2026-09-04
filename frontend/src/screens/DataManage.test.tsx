import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DataManage } from './DataManage';

const ROW = {
  timestamp: '2026-01-01T12:00:00', total_value_usd: 1000.0,
  total_cost_basis_usd: 800.0, unrealized_pl_usd: 200.0, unrealized_pl_percent: 25.0,
};

function stubFetch(
  saveResult: unknown = { saved: true, timestamp: '2026-02-01T12:00:00', error: null },
) {
  const fetchMock = vi.fn(async (url: unknown, init?: RequestInit) => {
    const path = String(url);
    const method = init?.method ?? 'GET';
    if (path.includes('/api/system/snapshot/save')) {
      return { ok: true, json: async () => saveResult };
    }
    if (path.includes('/api/system/snapshots/delete')) {
      return { ok: true, json: async () => ({ deleted: 1, error: null }) };
    }
    if (path.includes('/api/system/snapshots')) {
      return { ok: true, json: async () => ({ count: 1, rows: [ROW] }) };
    }
    if (path.includes('/api/system/cleanup')) {
      if (method === 'POST') {
        return { ok: true, json: async () => ({ success: true, message: 'Cleanup complete.', error: null }) };
      }
      return { ok: true, json: async () => ({ cleanup_days: 0, enabled: false, stats: {} }) };
    }
    throw new Error(`unexpected fetch: ${path} ${method}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => vi.unstubAllGlobals());

describe('DataManage fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<DataManage />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load/i)).toBeDefined();
  });
});

describe('DataManage snapshot save', () => {
  it('posts to the save endpoint, confirms, and reloads the table', async () => {
    const fetchMock = stubFetch();
    render(<DataManage />);
    await screen.findByText('Save snapshot');

    fireEvent.click(screen.getByRole('button', { name: 'Save snapshot' }));

    await waitFor(() => {
      expect(screen.getByText(/snapshot saved/i)).toBeDefined();
    });
    const post = fetchMock.mock.calls.find(([url]) => String(url).includes('/api/system/snapshot/save'));
    expect(post).toBeDefined();
    expect((post?.[1] as RequestInit | undefined)?.method).toBe('POST');
    // The table reloads after a save: snapshots fetched again on top of the
    // two mount-time reads (the screen and the panel each fetch once).
    const gets = fetchMock.mock.calls.filter(([url]) => String(url) === '/api/system/snapshots');
    expect(gets.length).toBeGreaterThan(2);
  });

  it('surfaces a save failure instead of appearing to succeed', async () => {
    stubFetch({ saved: false, timestamp: null, error: 'disk full' });
    render(<DataManage />);
    await screen.findByText('Save snapshot');

    fireEvent.click(screen.getByRole('button', { name: 'Save snapshot' }));

    await waitFor(() => {
      expect(screen.getByText(/save failed/i)).toBeDefined();
    });
    expect(screen.getByText(/disk full/)).toBeDefined();
  });
});
