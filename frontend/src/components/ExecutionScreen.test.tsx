import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ExecutionScreen } from './ExecutionScreen';
import type { ExecutionStatus } from '../types';

const LIVE: ExecutionStatus = { testnet: false, is_live: true } as ExecutionStatus;

describe('ExecutionScreen', () => {
  it('renders the posture strip between the header and the content', () => {
    render(
      <ExecutionScreen title="Probe" subtitle="probe screen" status={LIVE}>
        <p>body content</p>
      </ExecutionScreen>,
    );
    const banner = screen.getByText(/ORDERS WILL BE PLACED/);
    const body = screen.getByText('body content');
    const heading = screen.getByRole('heading', { name: 'Probe' });
    // Header first, posture strip second, content last. No screen using this
    // wrapper can drift the banner below its figures again.
    expect(heading.compareDocumentPosition(banner)
      & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(banner.compareDocumentPosition(body)
      & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('renders header and content with no strip while status is unknown', () => {
    render(
      <ExecutionScreen title="Probe" status={null}>
        <p>body content</p>
      </ExecutionScreen>,
    );
    expect(screen.getByRole('heading', { name: 'Probe' })).toBeDefined();
    expect(screen.getByText('body content')).toBeDefined();
    expect(screen.queryByText(/ORDERS WILL BE PLACED|SIMULATION MODE/)).toBeNull();
  });
});
