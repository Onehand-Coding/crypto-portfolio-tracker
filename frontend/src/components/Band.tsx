import type { ReactNode } from 'react';

/**
 * Horizontal KPI band: figures inline, not stacked cards. Sits inside a Panel.
 *
 * This is the control-room density the cockpit established - the total anchored
 * left with its supporting figures beside it, everything above the fold -
 * rather than a row of tall cards that pushes the table below it off-screen.
 */
export function KpiBand({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end" style={{ gap: 'var(--space-6)' }}>
      {children}
    </div>
  );
}

/**
 * One figure in a KPI band: label above, value below, no card chrome.
 *
 * `signal` drives colour only (P/L direction); it never carries meaning by
 * itself. `emphasis` marks the screen's headline number, rendered larger.
 * `value` must already carry its sign (use formatSigned/formatPercent).
 */
export function BandMetric({
  label, value, signal, sub, note, emphasis,
}: {
  label: string;
  value: string;
  signal?: number | null;
  sub?: string;
  note?: string;
  emphasis?: boolean;
}) {
  const colour = signal === undefined || signal === null || signal === 0
    ? 'var(--text-primary)'
    : signal > 0 ? 'var(--positive)' : 'var(--negative)';
  return (
    <div className="flex flex-col" style={{ gap: '3px', minWidth: 0 }}>
      <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '10px',
                                         fontWeight: 700, letterSpacing: '0.06em',
                                         textTransform: 'uppercase' }}>
        {label}
      </span>
      <span className="font-mono" style={emphasis
        ? { fontSize: '28px', lineHeight: '34px', letterSpacing: '-0.02em', color: colour }
        : { fontSize: '18px', lineHeight: '24px', letterSpacing: '-0.01em', color: colour }}>
        {value}
      </span>
      {sub && (
        <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px' }}>
          {sub}
        </span>
      )}
      {/* The plain-language question a figure answers, where its label alone is
          ambiguous (e.g. two accounting bases that differ several fold). */}
      {note && (
        <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                           fontStyle: 'italic' }}>
          {note}
        </span>
      )}
    </div>
  );
}
