import { formatPercent, formatQty, formatSigned, formatUsd, signOf } from '../lib/format';
import type { Holding } from '../types';

const DUST_THRESHOLD_USD = 0.4;

const COLOUR: Record<string, string> = {
  positive: 'var(--positive)',
  negative: 'var(--negative)',
  zero: 'var(--text-primary)',
};

/**
 * Dust collapses into one aggregate row rather than presenting sub-$0.40
 * positions as meaningful allocations.
 */
export function HoldingsTable({ holdings }: { holdings: Holding[] }) {
  const material = holdings.filter((h) => (h.value_usd ?? 0) >= DUST_THRESHOLD_USD);
  const dust = holdings.filter((h) => (h.value_usd ?? 0) < DUST_THRESHOLD_USD);
  const dustValue = dust.reduce((sum, h) => sum + (h.value_usd ?? 0), 0);

  if (holdings.length === 0) {
    return (
      <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
        No holdings recorded.
      </p>
    );
  }

  return (
    <table className="w-full font-mono text-sm tabular-nums">
      <thead>
        <tr style={{ color: 'var(--text-secondary)' }}>
          <th className="text-left font-normal">Asset</th>
          <th className="text-right font-normal">Quantity</th>
          <th className="text-right font-normal">Price</th>
          <th className="text-right font-normal">Value</th>
          <th className="text-right font-normal">Unrealized</th>
        </tr>
      </thead>
      <tbody>
        {material.map((h) => (
          <tr key={h.symbol} className="border-t" style={{ borderColor: 'var(--border)' }}>
            <td className="text-left">{h.symbol}</td>
            <td className="text-right">{formatQty(h.total_quantity)}</td>
            <td className="text-right">{formatUsd(h.current_price)}</td>
            <td className="text-right">{formatUsd(h.value_usd)}</td>
            <td className="text-right" style={{ color: COLOUR[signOf(h.unrealized_pl_usd)] }}>
              {formatSigned(h.unrealized_pl_usd)} ({formatPercent(h.unrealized_pl_percent)})
            </td>
          </tr>
        ))}
        {dust.length > 0 && (
          <tr className="border-t" style={{ borderColor: 'var(--border)',
                                            color: 'var(--text-secondary)' }}>
            <td className="text-left">{dust.length} dust positions</td>
            <td className="text-right">—</td>
            <td className="text-right">—</td>
            <td className="text-right">{formatUsd(dustValue)}</td>
            <td className="text-right">—</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
