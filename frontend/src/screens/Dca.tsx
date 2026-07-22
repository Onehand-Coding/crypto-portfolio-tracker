import { useState } from 'react';
import { Panel } from '../components/Panel';
import { BandMetric, KpiBand } from '../components/Band';
import { AnalysisBar, Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { ExecutePanel } from '../components/ExecutePanel';
import { TradingStatusBanner } from '../components/TradingStatusBanner';
import { useApi, usePollWhile } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatPercentPlain, formatQty, formatUsd } from '../lib/format';
import type {
  DcaPreviewResponse, DcaResponse, ExecutionStatus, TradeExecuteResponse,
} from '../types';

const STRATEGIES = [
  { id: 'target_weight', label: 'Target weight',
    hint: 'Buys the assets furthest below their target first.' },
  { id: 'proportional', label: 'Proportional',
    hint: 'Splits in proportion to what you already hold.' },
];

export function Dca() {
  const { data, error, reload } = useApi<DcaResponse>('/api/strategy/dca');
  const status = useApi<ExecutionStatus>('/api/execute/status');
  usePollWhile(Boolean(data?.is_running), reload);

  const [amount, setAmount] = useState('50');
  const [strategy, setStrategy] = useState('target_weight');
  const [preview, setPreview] = useState<DcaPreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  async function run() {
    try {
      await apiPost('/api/strategy/dca/run');
    } finally {
      reload();
    }
  }

  async function runPreview() {
    setPreviewError(null);
    try {
      const result = await apiPost<DcaPreviewResponse>('/api/strategy/dca/preview', {
        amount_usd: Number(amount), strategy,
      });
      setPreview(result);
    } catch (e) {
      setPreview(null);
      setPreviewError(e instanceof Error ? e.message : String(e));
    }
  }

  if (error) return <ErrorPanel title="DCA" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="DCA"><Empty>Loading…</Empty></Panel>;

  return (
    <>
      <ScreenHeader title="Dollar cost averaging"
                    subtitle="Preview where new capital would go before committing it" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <TradingStatusBanner status={status.data ?? null} />
        <AnalysisBar state={data} onRun={run} label="Checking your USDT balance" />

        <Panel title="Available to deploy">
          <KpiBand>
            <BandMetric emphasis label="Total USDT" value={formatUsd(data.available_usdt)} />
            <BandMetric label="Spot" value={formatUsd(data.spot_usdt)} />
            <BandMetric label="Earn" value={formatUsd(data.earn_usdt)} />
            <BandMetric label="Minimum trade" value={formatUsd(data.minimum_trade_usd)} />
          </KpiBand>
          {!data.has_data && (
            <p className="font-ui text-sm"
               style={{ color: 'var(--text-secondary)', marginTop: 'var(--space-4)' }}>
              Balances are unknown until you run the check above. The preview below works
              without them — it allocates against your configured target weights.
            </p>
          )}
        </Panel>

        <Panel title="Plan a contribution">
          <div className="flex items-end" style={{ gap: 'var(--space-4)', flexWrap: 'wrap' }}>
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

            <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
              <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                                 letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                Strategy
              </span>
              <div className="flex" style={{ gap: 'var(--space-2)' }}>
                {STRATEGIES.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setStrategy(s.id)}
                    title={s.hint}
                    className="font-ui transition-colors"
                    style={{
                      background: strategy === s.id ? 'var(--surface-2)' : 'transparent',
                      color: strategy === s.id ? 'var(--text-primary)' : 'var(--text-secondary)',
                      border: `1px solid ${strategy === s.id ? 'var(--border-strong)' : 'var(--border)'}`,
                      borderRadius: 'var(--radius-control)',
                      padding: 'var(--space-2) var(--space-3)', fontSize: '13px', cursor: 'pointer',
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>

            <Button onClick={runPreview}>Preview allocation</Button>
          </div>

          <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                          marginTop: 'var(--space-3)', marginBottom: 0 }}>
            {STRATEGIES.find((s) => s.id === strategy)?.hint} This is a preview only —
            it never places an order.
          </p>
        </Panel>

        {previewError && (
          <Panel><p className="font-mono text-sm" style={{ color: 'var(--negative)', margin: 0 }}>
            Preview failed: {previewError}
          </p></Panel>
        )}

        {preview && (
          <Panel title={`Allocation preview — ${formatUsd(preview.amount_usd)}`}>
            {!preview.valid ? (
              <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
                {preview.message}
              </p>
            ) : (
              <div className="table-scroll">
                <table className="data">
                  <thead>
                    <tr>
                      <th className="text-left">Asset</th>
                      <th className="text-right">Amount</th>
                      <th className="text-right">Est. quantity</th>
                      <th className="text-right">Current</th>
                      <th className="text-right">Target</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.allocations.map((a) => (
                      <tr key={a.symbol}>
                        <td className="text-left" style={{ fontWeight: 500 }}>{a.symbol}</td>
                        <td className="text-right">{formatUsd(a.amount_usd)}</td>
                        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                          {formatQty(a.quantity)}
                        </td>
                        <td className="text-right">{formatPercentPlain(a.current_allocation_pct)}</td>
                        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                          {formatPercentPlain(a.target_allocation_pct)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        )}

        {status.data && preview?.valid && preview.allocations.length > 0 && (
          <ExecutePanel
            title={`${status.data.is_live ? 'Execute' : 'Simulate'} DCA`}
            description={`This ${status.data.is_live ? 'deploys' : 'simulates deploying'} ${formatUsd(preview.amount_usd)} across ${preview.allocations.length} asset${preview.allocations.length === 1 ? '' : 's'} as market buys on the Binance ${status.data.testnet ? 'testnet' : 'mainnet'}${status.data.is_live ? '' : ' (live trading is off, so no orders are sent)'}.`}
            execute={() => apiPost<TradeExecuteResponse>('/api/execute/dca', {
              confirm: true,
              strategy,
              trades: preview.allocations.map((a) => ({ asset: a.symbol, amount: a.amount_usd })),
            })}
          />
        )}
      </div>
    </>
  );
}
