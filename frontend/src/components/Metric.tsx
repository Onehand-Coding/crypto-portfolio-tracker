import { signOf } from '../lib/format';

const COLOUR: Record<string, string> = {
  positive: 'var(--positive)',
  negative: 'var(--negative)',
  zero: 'var(--text-primary)',
};

/**
 * `value` must already carry its sign (use formatSigned/formatPercent).
 * `signal` drives colour only; it never carries meaning by itself.
 */
export function Metric({
  label, value, signal, sub,
}: {
  label: string;
  value: string;
  signal?: number | null;
  sub?: string;
}) {
  return (
    <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
      <span
        className="font-ui"
        style={{
          color: 'var(--text-tertiary)',
          fontSize: '11px',
          fontWeight: 500,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </span>
      <span
        className="font-mono"
        style={{
          fontSize: '26px',
          lineHeight: 1.1,
          letterSpacing: '-0.01em',
          color: signal === undefined ? 'var(--text-primary)' : COLOUR[signOf(signal)],
        }}
      >
        {value}
      </span>
      {sub && (
        <span className="font-ui" style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
          {sub}
        </span>
      )}
    </div>
  );
}
