import type { ExecutionStatus } from '../types';

/**
 * The three-cell trading posture strip, ported from the Streamlit dashboard.
 *
 * Two independent switches drive three readouts: whether live trading is armed,
 * which exchange endpoint is in use, and - the one that matters most - whether
 * an order placed now is real or simulated. Every execution screen shows this so
 * the posture is never a surprise at the moment of confirming.
 *
 * Placement rule (the component cannot enforce it, so it is stated here):
 * render it FIRST inside the screen's content column, directly under the
 * ScreenHeader and above every figure. A posture strip below the figures it
 * qualifies is a safety bug, not a layout choice.
 */
function Cell({ text, tone }: { text: string; tone: 'negative' | 'warning' | 'positive' | 'info' }) {
  const color = {
    negative: 'var(--negative)', warning: 'var(--warning)',
    positive: 'var(--positive)', info: 'var(--action)',
  }[tone];
  return (
    <div
      className="flex flex-1 items-center justify-center font-mono"
      style={{
        gap: 'var(--space-2)', minWidth: '150px',
        padding: 'var(--space-2) var(--space-3)',
        borderRadius: 'var(--radius-control)',
        border: `1px solid color-mix(in srgb, ${color} 35%, transparent)`,
        background: `color-mix(in srgb, ${color} 12%, transparent)`,
        color, fontSize: '11px', fontWeight: 700, letterSpacing: '0.06em',
        textAlign: 'center',
      }}
    >
      {text}
    </div>
  );
}

export function TradingStatusBanner({ status }: { status: ExecutionStatus | null }) {
  if (!status) return null;
  const { is_live, testnet } = status;
  return (
    <div className="flex flex-wrap" style={{ gap: 'var(--space-2)' }}>
      <Cell
        text={is_live ? '🔴 LIVE TRADING ENABLED' : '🟡 LIVE TRADING DISABLED'}
        tone={is_live ? 'negative' : 'warning'}
      />
      <Cell
        text={testnet ? '🧪 TESTNET CONNECTION' : '🌐 MAINNET CONNECTION'}
        tone="info"
      />
      <Cell
        text={is_live ? '⚠️ ORDERS WILL BE PLACED' : '✅ SIMULATION MODE'}
        tone={is_live ? 'negative' : 'positive'}
      />
    </div>
  );
}
