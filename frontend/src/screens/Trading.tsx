import { useState } from 'react';
import { Panel } from '../components/Panel';
import { BandMetric, KpiBand } from '../components/Band';
import { Badge, Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { TradingStatusBanner } from '../components/TradingStatusBanner';
import { useApi } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatPercentPlain, formatQty, formatUsd } from '../lib/format';
import type {
  CockpitResponse, ExecutionStatus, SystemHealthResponse, TradeExecuteResponse,
} from '../types';

/**
 * Order review with portfolio-impact preview and execution.
 *
 * Placing an order is irreversible, so it stays behind a typed confirmation and
 * the posture strip is always in view. Whether the order is real or simulated is
 * decided by the live-trading switch (Settings), exactly as in the CLI: with it
 * off the order is a dry run on whichever endpoint testnet mode selects.
 */
export function Trading() {
  const cockpit = useApi<CockpitResponse>('/api/portfolio/cockpit');
  const health = useApi<SystemHealthResponse>('/api/system/health');
  const status = useApi<ExecutionStatus>('/api/execute/status');

  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [symbol, setSymbol] = useState('BTC');
  const [amount, setAmount] = useState('50');
  const [confirmText, setConfirmText] = useState('');
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<TradeExecuteResponse | null>(null);

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

  const testnet = status.data?.testnet ?? false;
  const isLive = status.data?.is_live ?? false;

  async function execute() {
    setExecuting(true);
    setResult(null);
    try {
      const res = await apiPost<TradeExecuteResponse>('/api/execute/trade', {
        trade_type: side, symbol: symbol.toUpperCase(), amount: amountUsd,
        is_quote_qty: true, confirm: true,
      });
      setResult(res);
      setConfirmText('');
    } catch (e) {
      setResult({ success: false, testnet, messages: [],
                  errors: [e instanceof Error ? e.message : String(e)] });
    } finally {
      setExecuting(false);
    }
  }

  return (
    <>
      <ScreenHeader title="Trading"
                    subtitle="Review an order, then execute it — live or simulated per your settings" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <TradingStatusBanner status={status.data ?? null} />

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
            <KpiBand>
              <BandMetric
                emphasis
                label="Allocation after"
                value={afterTotal ? formatPercentPlain((afterValue / afterTotal) * 100) : '—'}
                sub={health.data.target_allocation[symbol] !== undefined
                  ? `target ${formatPercentPlain(health.data.target_allocation[symbol] * 100)}`
                  : 'not a core asset'}
              />
              <BandMetric label="Est. quantity"
                          value={price ? formatQty(amountUsd / price) : '—'}
                          sub={price ? `at ${formatUsd(price)}` : 'price unknown'} />
              <BandMetric label={`${symbol} value now`} value={formatUsd(currentValue)} />
              <BandMetric label={`${symbol} value after`} value={formatUsd(afterValue)} />
            </KpiBand>
            {side === 'SELL' && amountUsd > currentValue && (
              <p className="font-ui text-sm"
                 style={{ color: 'var(--negative)', marginTop: 'var(--space-4)', marginBottom: 0 }}>
                You hold {formatUsd(currentValue)} of {symbol} — this sell is larger than
                the position.
              </p>
            )}
          </Panel>
        )}

        {valid && (
          <Panel title={isLive ? 'Execute' : 'Execute (simulated)'}>
            <p className="font-ui text-sm"
               style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-3) 0' }}>
              This {isLive ? 'places' : 'simulates'} a market {side} of {formatUsd(amountUsd)} {symbol} on
              the Binance {testnet ? 'testnet' : 'mainnet'}
              {isLive ? '' : ' (live trading is off, so no order is sent)'}. To confirm,
              type <span className="font-mono"
              style={{ color: 'var(--text-primary)' }}>EXECUTE</span> below.
            </p>
            <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-3)' }}>
              <input
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value)}
                placeholder="EXECUTE"
                className="font-mono"
                style={{ background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
                         borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                         padding: 'var(--space-2) var(--space-3)', width: '160px', fontSize: '14px' }}
              />
              <Button onClick={execute} disabled={executing || confirmText.trim() !== 'EXECUTE'}>
                {executing ? 'Placing…'
                           : `${isLive ? 'Execute' : 'Simulate'} ${side} on ${testnet ? 'testnet' : 'mainnet'}`}
              </Button>
            </div>

            {result && (
              <div style={{ marginTop: 'var(--space-4)' }}>
                <Badge text={result.success ? 'ORDER PLACED' : 'ORDER FAILED'}
                       tone={result.success ? 'positive' : 'negative'} />
                <ul className="flex flex-col font-mono"
                    style={{ gap: '4px', fontSize: '12px', margin: 'var(--space-3) 0 0 0',
                             padding: 0, listStyle: 'none' }}>
                  {result.messages.map((m, i) => (
                    <li key={`m${i}`} style={{ color: 'var(--text-secondary)',
                                               wordBreak: 'break-all' }}>{m}</li>
                  ))}
                  {result.errors.map((m, i) => (
                    <li key={`e${i}`} style={{ color: 'var(--negative)' }}>{m}</li>
                  ))}
                </ul>
              </div>
            )}
          </Panel>
        )}
      </div>
    </>
  );
}
