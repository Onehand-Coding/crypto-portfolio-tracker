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
 *
 * Holdings whose price could not be fetched are excluded from that collapse
 * entirely. Their value is unknown, not small: a failed lookup on a large
 * position would otherwise hide it inside the dust row, and the position would
 * read as "too small to matter" while being the bulk of the portfolio.
 */
export function HoldingsTable({ holdings }: { holdings: Holding[] }) {
  // A null value_usd counts as unknown too, not just the flagged case: the two
  // are independent in the schema, and `?? 0` here would classify an unknown
  // value as dust and make the row vanish -- the same defect the flag exists
  // to prevent, arriving through the other door.
  const unpriced = holdings.filter((h) => h.price_unavailable || h.value_usd == null);
  const priced = holdings.filter((h) => !h.price_unavailable && h.value_usd != null);
  const material = priced.filter((h) => (h.value_usd ?? 0) >= DUST_THRESHOLD_USD);
  const dust = priced.filter((h) => (h.value_usd ?? 0) < DUST_THRESHOLD_USD);
  const dustValue = dust.reduce((sum, h) => sum + (h.value_usd ?? 0), 0);

  if (holdings.length === 0) {
    return (
      <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
        No holdings recorded.
      </p>
    );
  }

  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th className="text-left">Asset</th>
            <th className="text-right">Quantity</th>
            <th className="text-right">Price</th>
            <th className="text-right">Value</th>
            <th className="text-right">Unrealized</th>
          </tr>
        </thead>
        <tbody>
          {material.map((h) => (
            <tr key={h.symbol}>
              <td className="text-left" style={{ fontWeight: 500 }}>{h.symbol}</td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatQty(h.total_quantity)}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(h.current_price)}
              </td>
              <td className="text-right">{formatUsd(h.value_usd)}</td>
              <td className="text-right" style={{ color: COLOUR[signOf(h.unrealized_pl_usd)] }}>
                {formatSigned(h.unrealized_pl_usd)} ({formatPercent(h.unrealized_pl_percent)})
              </td>
            </tr>
          ))}
          {unpriced.map((h) => (
            <tr key={h.symbol}>
              <td className="text-left" style={{ fontWeight: 500 }}>{h.symbol}</td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatQty(h.total_quantity)}
              </td>
              <td className="text-right" style={{ color: 'var(--warning)' }}>
                price unavailable
              </td>
              <td className="text-right" style={{ color: 'var(--warning)' }}>—</td>
              <td className="text-right" style={{ color: 'var(--warning)' }}>—</td>
            </tr>
          ))}
          {dust.length > 0 && (
            <tr style={{ color: 'var(--text-tertiary)' }}>
              <td className="text-left">{dust.length} dust positions</td>
              <td className="text-right">—</td>
              <td className="text-right">—</td>
              <td className="text-right">{formatUsd(dustValue)}</td>
              <td className="text-right">—</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
