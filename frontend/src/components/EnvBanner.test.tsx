import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { EnvBanner } from './EnvBanner';

describe('EnvBanner', () => {
  it('names the environment in text, not colour alone', () => {
    render(<EnvBanner environment={{
      is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET',
    }} />);
    expect(screen.getByText('TESTNET')).toBeDefined();
  });

  it('shows which database is in use so the two are never confused', () => {
    render(<EnvBanner environment={{
      is_testnet: true, database_path: 'data/testnet_portfolio.db', label: 'TESTNET',
    }} />);
    expect(screen.getByText(/testnet_portfolio\.db/)).toBeDefined();
  });

  it('renders in the live case too, never absent', () => {
    render(<EnvBanner environment={{
      is_testnet: false, database_path: 'data/portfolio.db', label: 'LIVE',
    }} />);
    expect(screen.getByText('LIVE')).toBeDefined();
  });

  it('renders an explicit unknown state instead of disappearing when environment is null', () => {
    render(<EnvBanner environment={null} />);
    expect(screen.getByText('ENVIRONMENT UNKNOWN')).toBeDefined();
    expect(screen.getByText(/cannot reach API/)).toBeDefined();
  });
});
