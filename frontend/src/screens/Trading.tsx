import { useState } from 'react';
import { Panel } from '../components/Panel';
import { Metric } from '../components/Metric';
import { Badge, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { formatPercentPlain, formatQty, formatUsd } from '../lib/format';
import type { CockpitResponse, SystemHealthResponse } from '../types';

/**
 * Order review with portfolio-impact preview.
 *
 * Execution is deliberately not wired. Placing a real order is irreversible and
 * spends real money, and this screen has not been through the review that the
 * accounting paths got. The impact preview below is the useful, safe half; use
 * the CLI or Streamlit to actually place the order until execution here has
 * been reviewed as carefully as the figures were.
 */
export function Trading() {
  const cockpit = useApi<CockpitResponse>('/api/portfolio/cockpit');
  const health = useApi<SystemHealthResponse>('/api/system/health');

  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [symbol, setSymbol] = useState('BTC');
  const [amount, setAmount] = useState('50');

  const error = cockpit.error ?? health.error;
  if (error) return <ErrorPanel title="Trading" message={`Failed to load: ${error}`} />;
  if (!cockpit.data || !health.data) {
    return <Panel title="Trading"><Empty>Loading…</Empty></Panel>;
  }

  const holding = cockpit.data.holdings.find((h) => h.symbol === symbol.toUpperCase());
  const amountUsd = Number(amount);
  const valid = Number.isFinite(amountUsd) && amountUsd >= health.data.minimum_trade_usd;

  const total = cockpit.data.total_value_usd;
  const currentValue = holding?.value_usd ?? 0;
  const afterValue = side === 'BUY' ? currentValue + amountUsd : currentValue - amountUsd;
  const afterTotal = side === 'BUY' ? total + amountUsd : total;
  const price = holding?.current_price ?? null;

  return (
    <>
      <ScreenHeader title="Trading" subtitle="Review an order and its effect on your allocation" />

      <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
        <Panel>
          <div className="flex items-center" style={{ gap: 'var(--space-3)' }}>
            <Badge text="REVIEW ONLY" tone="warning" />
            <span className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
              This screen never places an order. Execution is intentionally not wired here —
              use the CLI or Streamlit to trade.
            </span>
          </div>
        </Panel>

        <Panel title="Order">
          <div className="flex items-end" style={{ gap: 'var(--space-4)', flexWrap: 'wrap' }}>
            <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
              <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                                 letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                Side
              </span>
              <div className="flex" style={{ gap: 'var(--space-2)' }}>
                {(['BUY', 'SELL'] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSide(s)}
                    className="font-ui transition-colors"
                    style={{
                      background: side === s
                        ? (s === 'BUY' ? 'color-mix(in srgb, var(--positive) 20%, transparent)'
                                       : 'color-mix(in srgb, var(--negative) 20%, transparent)')
                        : 'transparent',
                      color: side === s
                        ? (s === 'BUY' ? 'var(--positive)' : 'var(--negative)')
                        : 'var(--text-secondary)',
                      border: `1px solid ${side === s ? 'transparent' : 'var(--border)'}`,
                      borderRadius: 'var(--radius-control)',
                      padding: 'var(--space-2) var(--space-4)', fontSize: '13px',
                      fontWeight: 600, cursor: 'pointer',
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
              <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                                 letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                Asset
              </span>
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="font-mono"
                style={{
                  background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
                  borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                  padding: 'var(--space-2) var(--space-3)', fontSize: '14px', minWidth: '120px',
                }}
              >
                {Array.from(new Set([
                  ...Object.keys(health.data.target_allocation),
                  ...cockpit.data.holdings.map((h) => h.symbol),
                ])).sort().map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>

            <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
              <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                                 letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                Amount (USD)
              </span>
              <input
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                inputMode="decimal"
                className="font-mono"
                style={{
                  background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
                  borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                  padding: 'var(--space-2) var(--space-3)', width: '140px', fontSize: '14px',
                }}
              />
            </label>
          </div>

          {!valid && (
            <p className="font-ui text-sm"
               style={{ color: 'var(--warning)', marginTop: 'var(--space-3)', marginBottom: 0 }}>
              Amount must be at least the {formatUsd(health.data.minimum_trade_usd)} minimum
              trade size.
            </p>
          )}
        </Panel>

        {valid && (
          <Panel title="Portfolio impact">
            <div className="grid grid-cols-4" style={{ gap: 'var(--space-5)' }}>
              <Metric label="Est. quantity"
                      value={price ? formatQty(amountUsd / price) : '—'}
                      sub={price ? `at ${formatUsd(price)}` : 'price unknown'} />
              <Metric label={`${symbol} value now`} value={formatUsd(currentValue)} />
              <Metric label={`${symbol} value after`} value={formatUsd(afterValue)} />
              <Metric
                label="Allocation after"
                value={afterTotal ? formatPercentPlain((afterValue / afterTotal) * 100) : '—'}
                sub={health.data.target_allocation[symbol] !== undefined
                  ? `target ${formatPercentPlain(health.data.target_allocation[symbol] * 100)}`
                  : 'not a core asset'}
              />
            </div>
            {side === 'SELL' && amountUsd > currentValue && (
              <p className="font-ui text-sm"
                 style={{ color: 'var(--negative)', marginTop: 'var(--space-4)', marginBottom: 0 }}>
                You hold {formatUsd(currentValue)} of {symbol} — this sell is larger than
                the position.
              </p>
            )}
          </Panel>
        )}
      </div>
    </>
  );
}
