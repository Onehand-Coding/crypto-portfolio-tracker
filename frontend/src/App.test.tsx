import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';

function LocationProbe() {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <>
      <output data-testid="location">{location.pathname}</output>
      <button onClick={() => navigate(-1)}>Back</button>
    </>
  );
}

describe('App', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows ENVIRONMENT UNKNOWN when the cockpit fetch fails, never hiding the banner', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByText('ENVIRONMENT UNKNOWN')).toBeDefined();
    });
  });

  it('shows TESTNET and the database path when the cockpit fetch resolves with a testnet environment', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          total_value_usd: 0,
          net_invested: { label: '', question: '', basis_usd: 0, pl_usd: 0, pl_percent: null },
          fifo: { label: '', question: '', basis_usd: 0, pl_usd: 0, pl_percent: null },
          holdings: [],
          staleness: { cached_at: null, age_seconds: null, is_stale: false },
          environment: { is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET' },
          has_data: false,
        }),
      }),
    );

    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );

    // Deliberately more than one: the top-bar chip and the persistent bottom
    // status strip both carry it, because showing testnet figures as live is
    // the worst failure this UI has.
    await waitFor(() => {
      expect(screen.getAllByText('TESTNET').length).toBeGreaterThanOrEqual(2);
    });
    expect(screen.getByText(/testnet_portfolio\.db/)).toBeDefined();
  });

  it('renders a banner in both the failure and success cases -- never absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    const { container } = render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByText('ENVIRONMENT UNKNOWN')).toBeDefined();
    });
    expect(container.querySelector('.font-mono')).not.toBeNull();
  });

  it('links Dashboard to the root without exposing Overview', () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: 'Dashboard' }).getAttribute('href')).toBe('/');
    expect(screen.queryByRole('link', { name: 'Overview' })).toBeNull();
  });

    it('refetches the environment when notified of a settings flip, without a page reload', async () => {
    const base = {
      total_value_usd: 0,
      net_invested: { label: '', question: '', basis_usd: 0, pl_usd: 0, pl_percent: null },
      fifo: { label: '', question: '', basis_usd: 0, pl_usd: 0, pl_percent: null },
      holdings: [],
      staleness: { cached_at: null, age_seconds: null, is_stale: false },
      has_data: false,
    };
    // Must match ENVIRONMENT_CHANGED_EVENT in lib/environment.ts.
    const NOTIFY = 'environment-changed';
    let live = false;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: unknown) => {
        if (String(url).includes('/api/portfolio/cockpit')) {
          return {
            ok: true,
            json: async () => ({
              ...base,
              environment: live
                ? { is_testnet: false, database_path: 'data/portfolio.db', label: 'LIVE' }
                : { is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET' },
            }),
          };
        }
        throw new TypeError('Failed to fetch');
      }),
    );

    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getAllByText('TESTNET').length).toBeGreaterThanOrEqual(2);
    });
    live = true;
    window.dispatchEvent(new Event(NOTIFY));
    await waitFor(() => {
      expect(screen.getAllByText('LIVE').length).toBeGreaterThanOrEqual(2);
    });
  });

  it('replaces the legacy overview route with Dashboard', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(
      <MemoryRouter initialEntries={['/prior', '/overview']} initialIndex={1}>
        <App />
        <LocationProbe />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Dashboard' })).toBeDefined();
      expect(screen.getByTestId('location').textContent).toBe('/');
    });
    expect(screen.queryByText('Portfolio overview')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Back' }));

    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/prior');
    });
  });
});
