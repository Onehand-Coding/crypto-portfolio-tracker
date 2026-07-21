import { describe, expect, it } from 'vitest';
import { formatPercent, formatQty, formatSigned, formatUsd, signOf } from './format';

describe('formatSigned', () => {
  it('always carries an explicit sign so colour is not the only signal', () => {
    expect(formatSigned(18.63)).toBe('+$18.63');
    expect(formatSigned(-18.63)).toBe('-$18.63');
  });

  it('renders zero without a sign', () => {
    expect(formatSigned(0)).toBe('$0.00');
  });
});

describe('formatUsd', () => {
  it('formats to two decimals with thousands separators', () => {
    expect(formatUsd(1234.5)).toBe('$1,234.50');
  });

  it('renders null as an em dash rather than zero', () => {
    expect(formatUsd(null)).toBe('—');
  });
});

describe('formatPercent', () => {
  it('carries an explicit sign', () => {
    expect(formatPercent(-24.38)).toBe('-24.38%');
    expect(formatPercent(24.38)).toBe('+24.38%');
  });
});

describe('formatQty', () => {
  it('keeps small crypto quantities legible', () => {
    expect(formatQty(0.00012345)).toBe('0.00012345');
  });
});

describe('signOf', () => {
  it('classifies values for semantic styling', () => {
    expect(signOf(1)).toBe('positive');
    expect(signOf(-1)).toBe('negative');
    expect(signOf(0)).toBe('zero');
  });
});
