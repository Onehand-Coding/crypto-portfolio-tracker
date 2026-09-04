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
  const [unit, setUnit] = useState<'USD' | 'COIN'>('USD');
  const [confirmText, setConfirmText] = useState('');
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<TradeExecuteResponse | null>(null);

  const error = cockpit.error ?? health.error;
  if (error) return <ErrorPanel title="Trading" message={`Failed to load: ${error}`} />;
  if (!cockpit.data || !health.data) {
    return <Panel title="Trading"><Empty>Loading…</Empty></Panel>;
  }

  const holding = cockpit.data.holdings.find((h) => h.symbol === symbol.toUpperCase());
  const price = holding?.current_price ?? null;
  const raw = Number(amount);
  const isQuote = unit === 'USD';
  // Coin mode values the order at the last known price; unknown price means
  // unknown impact, never $0. The backend revalidates either way.
  const estUsd = isQuote ? raw : (price !== null && Number.isFinite(raw) ? raw * price : null);
  const valid = Number.isFinite(raw) && raw > 0
    && (isQuote ? raw >= health.data.minimum_trade_usd : true);

  const total = cockpit.data.total_value_usd;
  const currentValue = holding?.value_usd ?? 0;
  const deltaUsd = isQuote ? raw : estUsd;
  const afterValue = deltaUsd === null
    ? null : side === 'BUY' ? currentValue + deltaUsd : currentValue - deltaUsd;
  const afterTotal = deltaUsd === null
    ? null : side === 'BUY' ? total + deltaUsd : total;

  const testnet = status.data?.testnet ?? false;
  const isLive = status.data?.is_live ?? false;

  async function execute() {
    setExecuting(true);
    setResult(null);
    try {
      const res = await apiPost<TradeExecuteResponse>('/api/execute/trade', {
        trade_type: side, symbol: symbol.toUpperCase(), amount: raw,
        is_quote_qty: isQuote, confirm: true,
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
                Amount ({isQuote ? 'USD' : `${symbol} units`})
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
              <div className="flex" style={{ gap: 'var(--space-2)' }}>
                {(['USD', 'COIN'] as const).map((u) => (
                  <button
                    key={u}
                    onClick={() => setUnit(u)}
                    className="font-ui transition-colors"
                    style={{
                      background: unit === u ? 'var(--surface-2)' : 'transparent',
                      color: unit === u ? 'var(--text-primary)' : 'var(--text-secondary)',
                      border: `1px solid ${unit === u ? 'var(--border-strong)' : 'var(--border)'}`,
                      borderRadius: 'var(--radius-control)',
                      padding: 'var(--space-1) var(--space-3)', fontSize: '12px', cursor: 'pointer',
                    }}
                  >
                    {u === 'USD' ? 'USD' : `${symbol} units`}
                  </button>
                ))}
              </div>
            </label>
          </div>

          {!valid && (
            <p className="font-ui text-sm"
               style={{ color: 'var(--warning)', marginTop: 'var(--space-3)', marginBottom: 0 }}>
              {isQuote
                ? <>Amount must be at least the {formatUsd(health.data.minimum_trade_usd)} minimum
                    trade size.</>
                : <>Enter an amount greater than zero.</>}
            </p>
          )}
          {!isQuote && valid && estUsd !== null && estUsd < health.data.minimum_trade_usd && (
            <p className="font-ui text-sm"
               style={{ color: 'var(--warning)', marginTop: 'var(--space-3)', marginBottom: 0 }}>
              Est. {formatUsd(estUsd)} is below the {formatUsd(health.data.minimum_trade_usd)} minimum
              trade size — the exchange may reject it.
            </p>
          )}
        </Panel>

        {valid && (
          <Panel title="Portfolio impact">
            <KpiBand>
              <BandMetric
                emphasis
                label="Allocation after"
                value={afterTotal && afterValue !== null
                  ? formatPercentPlain((afterValue / afterTotal) * 100) : '—'}
                sub={health.data.target_allocation[symbol] !== undefined
                  ? `target ${formatPercentPlain(health.data.target_allocation[symbol] * 100)}`
                  : 'not a core asset'}
              />
                <BandMetric label={isQuote ? 'Est. quantity' : 'Est. value'}
                          value={isQuote
                            ? (price ? formatQty(raw / price) : '—')
                            : formatUsd(estUsd)}
                          sub={price
                            ? (isQuote ? `at ${formatUsd(price)}` : `${formatQty(raw)} ${symbol}`)
                            : 'price unknown'} />
                <BandMetric label={`${symbol} value now`} value={formatUsd(currentValue)} />
                <BandMetric label={`${symbol} value after`} value={formatUsd(afterValue)} />
              </KpiBand>
            {side === 'SELL' && isQuote && raw > currentValue && (
              <p className="font-ui text-sm"
                 style={{ color: 'var(--negative)', marginTop: 'var(--space-4)', marginBottom: 0 }}>
                You hold {formatUsd(currentValue)} of {symbol} — this sell is larger than
                the position.
              </p>
            )}
            {side === 'SELL' && !isQuote && raw > (holding?.total_quantity ?? 0) && (
              <p className="font-ui text-sm"
                 style={{ color: 'var(--negative)', marginTop: 'var(--space-4)', marginBottom: 0 }}>
                You hold {formatQty(holding?.total_quantity ?? 0)} {symbol} — this sell is
                larger than the position.
              </p>
            )}
          </Panel>
        )}

        {valid && (
          <Panel title={isLive ? 'Execute' : 'Execute (simulated)'}>
            <p className="font-ui text-sm"
               style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-3) 0' }}>
              This {isLive ? 'places' : 'simulates'} a market {side} of {isQuote
                ? formatUsd(raw)
                : `${formatQty(raw)} ${symbol}${estUsd !== null ? ` (≈ ${formatUsd(estUsd)})` : ''}`} on
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
