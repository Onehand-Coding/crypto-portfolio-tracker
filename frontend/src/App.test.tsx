import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';

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

    await waitFor(() => {
      expect(screen.getByText('TESTNET')).toBeDefined();
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
});
