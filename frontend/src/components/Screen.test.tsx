import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AnalysisBar } from './Screen';
import type { AnalysisState } from '../types';
import { ApiError } from '../lib/api';

const IDLE: AnalysisState = {
  has_data: true,
  is_running: false,
  error: null,
  staleness: { cached_at: '2026-09-06T10:00:00', age_seconds: 60, is_stale: false },
};

describe('AnalysisBar run feedback', () => {
  it('shows a disabled Starting state immediately, before the server roundtrip', async () => {
    // Never resolves: the button must flip on click, not on reload.
    render(<AnalysisBar state={IDLE} onRun={() => new Promise(() => {})} label="Rebalancing analysis" />);

    fireEvent.click(screen.getByRole('button', { name: 'Run analysis' }));

    const pending = await screen.findByRole('button', { name: 'Starting…' });
    expect(pending).toBeDisabled();
  });

  it('reverts and names the failure when the start request rejects', async () => {
    const onRun = vi.fn().mockRejectedValue(new ApiError(409, 'rebalance analysis already running'));
    render(<AnalysisBar state={IDLE} onRun={onRun} label="Rebalancing analysis" />);

    fireEvent.click(screen.getByRole('button', { name: 'Run analysis' }));

    // A rejected start used to be silent: try/finally with no catch anywhere.
    await waitFor(() =>
      expect(screen.getByText(/this analysis is already running/i)).toBeDefined());
    expect(screen.getByRole('button', { name: 'Run analysis' })).not.toBeDisabled();
  });

  it('prefers the server Running state once a run is in flight', async () => {
    render(<AnalysisBar state={{ ...IDLE, is_running: true }} onRun={() => {}} label="Rebalancing analysis" />);

    const running = screen.getByRole('button', { name: 'Running…' });
    expect(running).toBeDisabled();
  });
});
