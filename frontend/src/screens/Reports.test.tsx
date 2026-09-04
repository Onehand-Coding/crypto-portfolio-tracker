import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Reports } from './Reports';

const FILE_NAME = 'transactions_20260101_120000.csv';

const FILES = {
  files: [
    {
      name: FILE_NAME,
      path: `/exports/${FILE_NAME}`,
      size_bytes: 1234,
      modified: '2026-01-01T12:00:00',
    },
  ],
  export_dir: '/exports',
};

const PREVIEW = {
  name: FILE_NAME,
  lines: ['Date,Asset,Type', '2026-01-01,BTC,BUY'],
  truncated: true,
  total_lines: 103,
};

function stubFetch() {
  const fetchMock = vi.fn(async (url: unknown, _init?: RequestInit) => {
    const path = String(url);
    if (path.includes('/api/reports/preview')) {
      return { ok: true, json: async () => PREVIEW };
    }
    if (path.includes('/api/reports/delete')) {
      return { ok: true, json: async () => ({ deleted: true, name: FILE_NAME, error: null }) };
    }
    if (path.includes('/api/reports/charts')) {
      return { ok: true, json: async () => ({ name: 'portfolio_allocation_pie_20260101_120000.png', path: '/exports/portfolio_allocation_pie_20260101_120000.png' }) };
    }
    if (path.includes('/api/reports/summary')) {
      return { ok: true, json: async () => ({ name: 'summary_20260101.xlsx', path: '/exports/summary_20260101.xlsx' }) };
    }
    if (path.includes('/api/reports/trend')) {
      return { ok: true, json: async () => ({ name: 'trend_20260101.csv', path: '/exports/trend_20260101.csv' }) };
    }
    if (path.includes('/api/reports/generate')) {
      return { ok: true, json: async () => ({ name: 'transactions_x.csv', path: '/exports/transactions_x.csv' }) };
    }
    if (path.includes('/api/reports')) return { ok: true, json: async () => FILES };
    throw new Error(`unexpected fetch: ${path}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function postBody(fetchMock: ReturnType<typeof stubFetch>, part: string): unknown {
  const post = fetchMock.mock.calls.find(([url]) => String(url).includes(part));
  expect(post).toBeDefined();
  return JSON.parse(String((post?.[1] as RequestInit | undefined)?.body));
}

beforeEach(() => vi.unstubAllGlobals());

describe('Reports fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Reports />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load reports/i)).toBeDefined();
  });
});

describe('Reports summary export', () => {
  it('posts the picked summary format and reloads the file list', async () => {
    const fetchMock = stubFetch();
    render(<Reports />);
    await screen.findByText(FILE_NAME);
    const panel = screen.getByText('Portfolio summary').closest('section');
    expect(panel).not.toBeNull();
    const scope = within(panel as HTMLElement);
    fireEvent.click(scope.getByRole('button', { name: 'Excel' }));
    fireEvent.click(scope.getByRole('button', { name: /generate summary/i }));
    await waitFor(() => {
      expect(screen.getByText(/generated summary_20260101\.xlsx/i)).toBeDefined();
    });
    expect(postBody(fetchMock, '/api/reports/summary')).toEqual({ format: 'excel' });
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === '/api/reports').length)
      .toBeGreaterThan(1);
  });
});

describe('Reports trend export', () => {
  it('posts the picked timeframe and format and confirms the file', async () => {
    const fetchMock = stubFetch();
    render(<Reports />);
    await screen.findByText(FILE_NAME);
    const panel = screen.getByText('Trend report').closest('section');
    expect(panel).not.toBeNull();
    const scope = within(panel as HTMLElement);
    fireEvent.click(scope.getByRole('button', { name: 'Swing' }));
    fireEvent.click(scope.getByRole('button', { name: 'JSON' }));
    fireEvent.click(scope.getByRole('button', { name: /generate trend/i }));
    await waitFor(() => {
      expect(screen.getByText(/generated trend_20260101\.csv/i)).toBeDefined();
    });
    expect(postBody(fetchMock, '/api/reports/trend'))
      .toEqual({ timeframe: 'swing', format: 'json' });
  });
});

describe('Reports charts export', () => {
  it('posts to the charts endpoint and reloads the file list', async () => {
    const fetchMock = stubFetch();
    render(<Reports />);
    await screen.findByText(FILE_NAME);
    const panel = screen.getByText('Charts').closest('section');
    expect(panel).not.toBeNull();
    const scope = within(panel as HTMLElement);
    fireEvent.click(scope.getByRole('button', { name: /generate charts/i }));
    await waitFor(() => {
      expect(screen.getByText(/generated portfolio_allocation_pie_20260101_120000\.png/i)).toBeDefined();
    });
    expect(fetchMock.mock.calls.some(([url]) => String(url) === '/api/reports/charts')).toBe(true);
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === '/api/reports').length)
      .toBeGreaterThan(1);
  });
});

describe('Reports preview', () => {
  it('renders the returned lines and the remaining-line count', async () => {
    stubFetch();
    render(<Reports />);
    await screen.findByText(FILE_NAME);
    fireEvent.click(screen.getByRole('button', { name: 'Preview' }));
    await waitFor(() => {
      expect(screen.getByText(/date,asset,type/i)).toBeDefined();
    });
    expect(screen.getByText(/101 more lines/)).toBeDefined();
  });
});

describe('Reports delete', () => {
  it('posts the name plus confirmation and reloads the list', async () => {
    const fetchMock = stubFetch();
    render(<Reports />);
    await screen.findByText(FILE_NAME);
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
    await waitFor(() => {
      expect(screen.getByText(/deleted transactions_20260101_120000\.csv/i)).toBeDefined();
    });
    expect(postBody(fetchMock, '/api/reports/delete'))
      .toEqual({ name: FILE_NAME, confirm: true });
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === '/api/reports').length)
      .toBeGreaterThan(1);
  });
});
