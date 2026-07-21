import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { HoldingsTable } from './HoldingsTable';
import type { Holding } from '../types';

function holding(overrides: Partial<Holding>): Holding {
  return {
    symbol: 'XXX', total_quantity: 1, spot_quantity: 1, earn_quantity: 0,
    current_price: 1, value_usd: 1, average_cost_basis: 1,
    cost_basis_total: 1, unrealized_pl_usd: 0,
    unrealized_pl_percent: 0, is_core: false,
    ...overrides,
  };
}

describe('HoldingsTable', () => {
  it('renders a holding at or above the dust threshold as its own row', () => {
    render(<HoldingsTable holdings={[holding({ symbol: 'BTC', value_usd: 38.0 })]} />);
    expect(screen.getByText('BTC')).toBeDefined();
  });

  it('does not render dust holdings as individual rows', () => {
    render(<HoldingsTable holdings={[holding({ symbol: 'DOGE', value_usd: 0.1 })]} />);
    expect(screen.queryByText('DOGE')).toBeNull();
  });

  it('collapses multiple dust holdings into one aggregate row with count and summed value', () => {
    render(<HoldingsTable holdings={[
      holding({ symbol: 'DOGE', value_usd: 0.1 }),
      holding({ symbol: 'SHIB', value_usd: 0.05 }),
    ]} />);
    expect(screen.getByText('2 dust positions')).toBeDefined();
    expect(screen.getByText('$0.15')).toBeDefined();
    expect(screen.queryByText('DOGE')).toBeNull();
    expect(screen.queryByText('SHIB')).toBeNull();
  });

  it('shows no aggregate row when there are no dust holdings', () => {
    render(<HoldingsTable holdings={[holding({ symbol: 'BTC', value_usd: 38.0 })]} />);
    expect(screen.queryByText(/dust position/)).toBeNull();
  });

  it('renders "No holdings recorded." for an empty holdings array, not an empty table', () => {
    render(<HoldingsTable holdings={[]} />);
    expect(screen.getByText('No holdings recorded.')).toBeDefined();
    expect(screen.queryByRole('table')).toBeNull();
  });

  it('treats a holding worth exactly the dust threshold (0.40) as material, not dust', () => {
    render(<HoldingsTable holdings={[holding({ symbol: 'DOGE', value_usd: 0.4 })]} />);
    expect(screen.getByText('DOGE')).toBeDefined();
    expect(screen.queryByText(/dust position/)).toBeNull();
  });

  it('treats a holding with value_usd: null as dust (0 via ?? 0), not material, and does not crash', () => {
    render(<HoldingsTable holdings={[holding({ symbol: 'NULLCOIN', value_usd: null })]} />);
    expect(screen.queryByText('NULLCOIN')).toBeNull();
    expect(screen.getByText('1 dust positions')).toBeDefined();
    expect(screen.getByText('$0.00')).toBeDefined();
  });
});
