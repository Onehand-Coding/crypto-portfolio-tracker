import { describe, expect, it } from 'vitest';
import { plDomain } from './Allocation';

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
    expect(plDomain([1, 2, 3])[1]).toBeGreaterThan(3);
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
    const [lo, hi] = plDomain([]);
    expect(hi).toBeGreaterThan(lo);
  });
});
