import { beforeEach, describe, expect, it, vi } from 'vitest';
import { cloneElement, isValidElement, type ReactElement } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { plDomain, DUST_THRESHOLD_PCT, groupDust, Allocation } from './Allocation';

// jsdom has no layout, so recharts' ResponsiveContainer (ResizeObserver +
// measured width) renders a 0x0 box and no chart at all — not even
// `.recharts-wrapper`. Injecting fixed dimensions is the documented
// workaround; the charts under test are unchanged.
vi.mock('recharts', async (importOriginal) => {
  const orig = await importOriginal<typeof import('recharts')>();
  return {
    ...orig,
    ResponsiveContainer: ({ children }: { children: ReactElement }) =>
      isValidElement(children)
        ? cloneElement(children, { width: 800, height: 300 } as never)
        : children,
  };
});

describe('plDomain', () => {
  it('fits mixed signs with a zero baseline and padding', () => {
    const [lo, hi] = plDomain([-34.5, -3.59, 0.15, 0, -0.01]);
    expect(lo).toBeLessThanOrEqual(-34.5);
    expect(hi).toBeGreaterThanOrEqual(0.15);
    expect(lo).toBeLessThanOrEqual(0);
    expect(hi).toBeGreaterThanOrEqual(0);
  });

  it('pins zero baseline for all-positive data', () => {
    expect(plDomain([1, 2, 3])[0]).toBe(0);
    expect(plDomain([1, 2, 3])[1]).toBeCloseTo(3.1, 10);
  });

  it('pins zero baseline for all-negative data', () => {
    const [lo, hi] = plDomain([-5, -2]);
    expect(hi).toBe(0);
    expect(lo).toBeLessThan(-5);
  });

  it('never degenerates for all-zero data', () => {
    expect(plDomain([0, 0])).toEqual([-1, 1]);
  });

  it('never degenerates for empty input', () => {
    expect(plDomain([])).toEqual([0, 1]);
  });

  it('never degenerates for single-element span-0 input', () => {
    expect(plDomain([5])).toEqual([0, 6]);
    expect(plDomain([-5])).toEqual([-6, 0]);
    expect(plDomain([0])).toEqual([-1, 1]);
  });
});

const STALENESS = { cached_at: null, age_seconds: null, is_stale: false };

const COCKPIT = {
  total_value_usd: 175.02,
  holdings: [
    { symbol: 'BTC', value_usd: 149.47, unrealized_pl_usd: -34.5 },
    { symbol: 'ETH', value_usd: 24.58, unrealized_pl_usd: -3.59 },
    { symbol: 'USDT', value_usd: 0.15, unrealized_pl_usd: 0 },
  ],
  staleness: STALENESS,
};

const HEALTH = { target_allocation: { BTC: 0.35, ETH: 0.3, USDT: 0.05 } };

function stubFetch() {
  vi.stubGlobal('fetch', vi.fn(async (url: unknown) => {
    const path = String(url);
    if (path.includes('/api/portfolio/cockpit')) {
      return { ok: true, json: async () => COCKPIT };
    }
    if (path.includes('/api/system/health')) {
      return { ok: true, json: async () => HEALTH };
    }
    throw new Error(`unexpected fetch: ${path}`);
  }));
}

beforeEach(() => vi.unstubAllGlobals());

describe('Allocation fetch failure', () => {
  it('renders a visible error when fetch rejects, not a permanent loading state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    render(<Allocation />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).toBeNull();
    });
    expect(screen.getByText(/failed to load/i)).toBeDefined();
  });
});

describe('Allocation P/L chart', () => {
  it('renders every holding so each bar is hoverable', async () => {
    stubFetch();
    const { container } = render(<Allocation />);
    await screen.findByText('Unrealized P/L by asset');
    // Caveat outcome (see commit message): recharts only renders
    // `.recharts-tooltip-item` for the hovered datum, and in jsdom the bars
    // render no geometry at all (empty `.recharts-bar-rectangle` groups, no
    // `.recharts-cell` class in recharts v3), so neither the tooltip-item
    // assertion nor mouseEnter/mouseOver hover simulation can observe the
    // data. The XAxis renders one tick label per datum instead — asserting
    // all three stub symbols there pins "every holding reaches the chart".
    const barChart = Array.from(container.querySelectorAll('.recharts-wrapper'))
      .find((w) => w.querySelector('.recharts-bar'));
    expect(barChart).toBeDefined();
    const labels = Array.from(
      barChart!.querySelectorAll('.recharts-cartesian-axis-tick-value'),
    ).map((el) => el.textContent ?? '');
    expect(labels.some((t) => t.includes('BTC'))).toBe(true);
    expect(labels.some((t) => t.includes('ETH'))).toBe(true);
    expect(labels.some((t) => t.includes('USDT'))).toBe(true);
  });

  it('fits the Y axis to the data with headroom above zero', async () => {
    stubFetch();
    const { container } = render(<Allocation />);
    await screen.findByText('Unrealized P/L by asset');
    // Stub P/L max is exactly 0 (USDT) and min -34.5 (BTC). Recharts' auto
    // domain tops out at $0; the fitted plDomain pads above zero, so the top
    // Y tick must exceed $0. Pins the `domain={plDomain(...)}` wiring, which
    // is the actual Task 2 behaviour change (the x-label test above passes
    // pre-change; this one is the red-to-green).
    const barChart = Array.from(container.querySelectorAll('.recharts-wrapper'))
      .find((w) => w.querySelector('.recharts-bar'));
    expect(barChart).toBeDefined();
    const ticks = Array.from(
      barChart!.querySelectorAll('.recharts-cartesian-axis-tick-value'),
    )
      .map((el) => el.textContent ?? '')
      .filter((t) => t.startsWith('$'))
      .map((t) => Number(t.replace('$', '')));
    expect(ticks.length).toBeGreaterThan(0);
    expect(Math.max(...ticks)).toBeGreaterThan(0);
  });
});

describe('groupDust', () => {
  const slices = [
    { name: 'BTC', value: 149.47 },
    { name: 'ETH', value: 24.58 },
    { name: 'USDT', value: 0.15 },
    { name: 'SOL', value: 0.03 },
  ];

  it('groups sub-threshold holdings into one named slice', () => {
    expect(DUST_THRESHOLD_PCT).toBe(1);
    const out = groupDust(slices, 175.02);
    expect(out.map((s) => s.name)).toEqual(['BTC', 'ETH', 'Others (2)']);
    const others = out.find((s) => s.name === 'Others (2)');
    expect(others?.value).toBeCloseTo(0.18, 10);
  });

  it('passes holdings through when nothing is dust', () => {
    expect(groupDust(
      [{ name: 'BTC', value: 60 }, { name: 'ETH', value: 40 }], 100))
      .toEqual([{ name: 'BTC', value: 60 }, { name: 'ETH', value: 40 }]);
  });

  it('collapses to a single slice when everything is dust', () => {
    expect(groupDust(
      [{ name: 'A', value: 0.4 }, { name: 'B', value: 0.3 }], 100))
      .toEqual([{ name: 'Others (2)', value: 0.7 }]);
  });

  it('does not divide by a zero total', () => {
    expect(groupDust([{ name: 'A', value: 0 }], 0))
      .toEqual([{ name: 'A', value: 0 }]);
  });
});

describe('Allocation colour dots', () => {
  it('shows a dot per drift row matching the ring order', async () => {
    stubFetch();
    const { container } = render(<Allocation />);
    await screen.findByText('Current vs target');
    // One dot per drift row (BTC, ETH, USDT in the stub health payload).
    const dots = container.querySelectorAll('[data-testid="drift-dot"]');
    expect(dots.length).toBe(3);
  });
});
