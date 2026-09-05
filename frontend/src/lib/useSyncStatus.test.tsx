import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useSyncStatus } from './useSyncStatus';

function probe() {
  function Probe() {
    const status = useSyncStatus(1000);
    return <span>{status ? `age:${status.staleness.age_seconds}` : 'waiting'}</span>;
  }
  return render(<Probe />);
}

function stubFetch(ages: Array<number | null>) {
  let calls = 0;
  vi.stubGlobal('fetch', vi.fn(async () => {
    const age = ages[Math.min(calls++, ages.length - 1)];
    return {
      ok: true,
      json: async () => ({
        is_running: false,
        staleness: { cached_at: null, age_seconds: age, is_stale: false },
      }),
    };
  }));
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.unstubAllGlobals();
});
afterEach(() => vi.useRealTimers());

describe('useSyncStatus', () => {
  it('fetches on mount and re-polls on the interval', async () => {
    stubFetch([10, 70]);
    probe();
    await vi.waitFor(() => {
      expect(screen.getByText('age:10')).toBeDefined();
    });
    await vi.advanceTimersByTimeAsync(1000);
    await vi.waitFor(() => {
      expect(screen.getByText('age:70')).toBeDefined();
    });
  });

  it('keeps the last known age when a poll fails', async () => {
    let calls = 0;
    vi.stubGlobal('fetch', vi.fn(async () => {
      calls += 1;
      if (calls > 1) throw new TypeError('Failed to fetch');
      return {
        ok: true,
        json: async () => ({
          is_running: false,
          staleness: { cached_at: null, age_seconds: 10, is_stale: false },
        }),
      };
    }));
    probe();
    await vi.waitFor(() => {
      expect(screen.getByText('age:10')).toBeDefined();
    });
    await vi.advanceTimersByTimeAsync(1000);
    // No error UI, no blanking: the last known age stands until sync recovers.
    expect(screen.getByText('age:10')).toBeDefined();
  });
});
