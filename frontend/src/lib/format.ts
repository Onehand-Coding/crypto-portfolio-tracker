/**
 * Number formatting. Every signed value renders its sign explicitly so the
 * information survives colour-blindness and greyscale printing -- colour is
 * never the sole carrier of meaning.
 */

export type Sign = 'positive' | 'negative' | 'zero';

/* Rendered wherever a figure is unknown. Deliberately a word, not a dash:
   a hyphen in a numeric column reads as a minus sign, which is worse than
   a gap. */
const NULL_GLYPH = 'N/A';

export { NULL_GLYPH };

export function signOf(value: number | null | undefined): Sign {
  if (value === null || value === undefined || value === 0) return 'zero';
  return value > 0 ? 'positive' : 'negative';
}

export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NULL_GLYPH;
  return `$${Math.abs(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function formatSigned(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NULL_GLYPH;
  const sign = value > 0 ? '+' : value < 0 ? '-' : '';
  return `${sign}${formatUsd(value)}`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NULL_GLYPH;
  const sign = value > 0 ? '+' : value < 0 ? '-' : '';
  return `${sign}${Math.abs(value).toFixed(2)}%`;
}

/**
 * A percentage that is a level, not a change: target weights, allocations,
 * RSI. formatPercent signs its output because a P/L of +2% and -2% must never
 * be confused; a target allocation of "+35%" is just wrong.
 */
export function formatPercentPlain(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NULL_GLYPH;
  return `${value.toFixed(2)}%`;
}

export function formatQty(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NULL_GLYPH;
  return value.toLocaleString('en-US', { maximumFractionDigits: 8 });
}
