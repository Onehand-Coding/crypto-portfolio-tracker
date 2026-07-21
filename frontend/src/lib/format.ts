/**
 * Number formatting. Every signed value renders its sign explicitly so the
 * information survives colour-blindness and greyscale printing -- colour is
 * never the sole carrier of meaning.
 */

export type Sign = 'positive' | 'negative' | 'zero';

const EM_DASH = '—';

export function signOf(value: number | null | undefined): Sign {
  if (value === null || value === undefined || value === 0) return 'zero';
  return value > 0 ? 'positive' : 'negative';
}

export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return `$${Math.abs(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function formatSigned(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  const sign = value > 0 ? '+' : value < 0 ? '-' : '';
  return `${sign}${formatUsd(value)}`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  const sign = value > 0 ? '+' : value < 0 ? '-' : '';
  return `${sign}${Math.abs(value).toFixed(2)}%`;
}

export function formatQty(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return value.toLocaleString('en-US', { maximumFractionDigits: 8 });
}
