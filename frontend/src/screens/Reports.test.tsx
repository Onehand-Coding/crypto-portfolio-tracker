import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Reports } from './Reports';

const FILE_NAME = 'transactions_20260101_120000.csv';

const IMAGE_NAME = 'chart_20260101_120000.png';

const HTML_NAME = 'portfolio_report_20260101_120000.html';

const JSON_NAME = 'trend_report_long_term_20260101_120000.json';

const FILES = {
  files: [
    {
      name: FILE_NAME,
      path: `/exports/${FILE_NAME}`,
      size_bytes: 1234,
      modified: '2026-01-01T12:00:00',
    },
    {
      name: IMAGE_NAME,
      path: `/exports/${IMAGE_NAME}`,
      size_bytes: 250000,
      modified: '2026-01-01T12:00:00',
    },
    {
      name: HTML_NAME,
      path: `/exports/${HTML_NAME}`,
      size_bytes: 9500,
      modified: '2026-01-01T12:00:00',
    },
    {
      name: JSON_NAME,
      path: `/exports/${JSON_NAME}`,
      size_bytes: 4000,
      modified: '2026-01-01T12:00:00',
    },
  ],
  export_dir: '/exports',
};

const TABLE_PREVIEW = {
  name: FILE_NAME,
  lines: [],
  truncated: true,
  total_lines: 103,
  kind: 'table',
  columns: ['Date', 'Asset', 'Type'],
  rows: [['2026-01-01', 'BTC', 'BUY'], ['2026-01-02', 'ETH', null]],
  image_url: null,
};

const IMAGE_PREVIEW = {
  name: IMAGE_NAME,
  lines: [],
  truncated: false,
  total_lines: 0,
  kind: 'image',
  columns: [],
  rows: [],
  image_url: `/api/reports/download?name=${IMAGE_NAME}`,
};

const HTML_PREVIEW = {
  name: HTML_NAME,
  lines: [],
  truncated: false,
  total_lines: 0,
  kind: 'html',
  columns: [],
  rows: [],
  image_url: null,
};

const JSON_PREVIEW = {
  name: JSON_NAME,
  lines: ['{"a":1,"b":[1,2]}'],
  truncated: false,
  total_lines: 1,
  kind: 'json',
  columns: [],
  rows: [],
  image_url: null,
};

function stubFetch() {
  const fetchMock = vi.fn(async (url: unknown, _init?: RequestInit) => {
    const path = String(url);
    if (path.includes('/api/reports/preview')) {
      // Each kind arrives shaped for its viewer: tables as columns plus
      // rows, images as a download URL, never binary as text lines.
      if (path.includes(IMAGE_NAME)) {
        return { ok: true, json: async () => IMAGE_PREVIEW };
      }
      if (path.includes(HTML_NAME)) {
        return { ok: true, json: async () => HTML_PREVIEW };
      }
      if (path.includes(JSON_NAME)) {
        return { ok: true, json: async () => JSON_PREVIEW };
      }
      return { ok: true, json: async () => TABLE_PREVIEW };
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
  async function openPreview(fileName: string) {
    await screen.findByText(fileName);
    const row = screen.getByText(fileName).closest('tr');
    expect(row).not.toBeNull();
    fireEvent.click(within(row as HTMLElement).getByRole('button', { name: 'Preview' }));
    await screen.findByRole('dialog', { name: `Preview of ${fileName}` });
  }

  it('renders tabular previews as a table with unknown cells marked', async () => {
    stubFetch();
    render(<Reports />);
    await openPreview(FILE_NAME);
    const dialog = screen.getByRole('dialog', { name: `Preview of ${FILE_NAME}` });
    const scope = within(dialog as HTMLElement);
    expect(scope.getByText('Date')).toBeDefined();
    expect(scope.getByText('BTC')).toBeDefined();
    // Unknown is never blank or zero: the null cell renders N/A.
    expect(scope.getByText('N/A')).toBeDefined();
    expect(scope.getByText(/101 more rows/)).toBeDefined();
  });

  it('renders image previews as an image, not binary text', async () => {
    stubFetch();
    render(<Reports />);
    await openPreview(IMAGE_NAME);
    const img = await screen.findByAltText(`Preview of ${IMAGE_NAME}`);
    expect(img.getAttribute('src')).toBe(IMAGE_PREVIEW.image_url);
  });

  it('renders HTML reports instead of showing their source', async () => {
    stubFetch();
    render(<Reports />);
    await openPreview(HTML_NAME);
    const dialog = screen.getByRole('dialog', { name: `Preview of ${HTML_NAME}` });
    const frame = within(dialog as HTMLElement).getByTitle(`Preview of ${HTML_NAME}`);
    expect(frame.getAttribute('src')).toBe(
      `/api/reports/download?name=${encodeURIComponent(HTML_NAME)}`);
    expect(frame.getAttribute('sandbox')).not.toBeNull();
  });

  it('pretty-prints JSON previews', async () => {
    stubFetch();
    render(<Reports />);
    await openPreview(JSON_NAME);
    const dialog = screen.getByRole('dialog', { name: `Preview of ${JSON_NAME}` });
    expect(within(dialog as HTMLElement).getByText(/"a": 1/)).toBeDefined();
  });

  it('closes the modal on Close, backdrop click, and Escape', async () => {
    stubFetch();
    const { container } = render(<Reports />);
    await openPreview(FILE_NAME);

    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull();
    });

    await openPreview(FILE_NAME);
    fireEvent.click(screen.getByRole('dialog'));
    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull();
    });

    await openPreview(FILE_NAME);
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull();
    });
    expect(container).toBeDefined();
  });
});

describe('Reports delete', () => {
  it('posts the name plus confirmation and reloads the list', async () => {
    const fetchMock = stubFetch();
    render(<Reports />);
    await screen.findByText(FILE_NAME);
    const row = screen.getByText(FILE_NAME).closest('tr');
    expect(row).not.toBeNull();
    const scope = within(row as HTMLElement);
    fireEvent.click(scope.getByRole('button', { name: 'Delete' }));
    fireEvent.click(scope.getByRole('button', { name: 'Confirm' }));
    await waitFor(() => {
      expect(screen.getByText(/deleted transactions_20260101_120000\.csv/i)).toBeDefined();
    });
    expect(postBody(fetchMock, '/api/reports/delete'))
      .toEqual({ name: FILE_NAME, confirm: true });
    expect(fetchMock.mock.calls.filter(([url]) => String(url) === '/api/reports').length)
      .toBeGreaterThan(1);
  });
});
