import { useState } from 'react';
import { Panel } from '../components/Panel';
import { BandMetric, KpiBand } from '../components/Band';
import { AnalysisBar, Button, Empty, ErrorPanel } from '../components/Screen';
import { ExecutePanel } from '../components/ExecutePanel';
import { ExecutionScreen } from '../components/ExecutionScreen';
import { useApi, usePollWhile } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatPercentPlain, formatQty, formatUsd } from '../lib/format';
import type {
  CompletionResponse, DcaPreviewResponse, DcaResponse, ExecutionStatus, TradeExecuteResponse,
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
  const completion = useApi<CompletionResponse>('/api/strategy/completion');
  const [showCompletion, setShowCompletion] = useState(false);
  usePollWhile(Boolean(data?.is_running), reload);

  const [amount, setAmount] = useState('50');
  const [strategy, setStrategy] = useState('target_weight');
  const [preview, setPreview] = useState<DcaPreviewResponse | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const isChecked = (s: string) => selected[s] ?? true;
  const toggle = (s: string) =>
    setSelected((prev) => ({ ...prev, [s]: !(prev[s] ?? true) }));

  async function run() {
    try {
      await apiPost('/api/strategy/dca/run');
    } finally {
      reload();
    }
  }

  async function runPreview() {
    setPreviewBusy(true);
    setPreviewError(null);
    try {
      const result = await apiPost<DcaPreviewResponse>('/api/strategy/dca/preview', {
        amount_usd: Number(amount), strategy,
      });
      setPreview(result);
    } catch (e) {
      setPreview(null);
      setPreviewError(e instanceof Error ? e.message : String(e));
    } finally {
      setPreviewBusy(false);
    }
  }

  if (error) return <ErrorPanel title="DCA" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="DCA"><Empty>Loading…</Empty></Panel>;

  const actionable = (preview?.valid ? preview.allocations : []).filter(
    (a) => a.amount_usd > 0,
  );
  const checked = actionable.filter((a) => isChecked(a.symbol));
  const checkedTotal = checked.reduce((sum, a) => sum + a.amount_usd, 0);

  return (
    <ExecutionScreen title="Dollar cost averaging"
                     subtitle="Preview where new capital would go before committing it"
                     status={status.data ?? null}>
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
              without them - it allocates against your configured target weights.
            </p>
          )}
        </Panel>

        <Panel title="Completion plan">
          <p className="font-ui" style={{ color: 'var(--text-secondary)', fontSize: '13px', marginTop: 0 }}>
            How much of each asset is still needed to finish your targets without
            selling anything. {completion.data?.valid && completion.data.anchor_symbol
              ? `Anchored by ${completion.data.anchor_symbol} at ${formatUsd(completion.data.implied_total_usd)} implied total.`
              : 'The money needed is the output.'}
          </p>
          <Button onClick={() => setShowCompletion((v) => !v)}>
            {showCompletion ? 'Hide completion plan' : 'Show completion plan'}
          </Button>
          {showCompletion && !completion.data && !completion.error && (
            <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>Loading…</p>
          )}
          {showCompletion && completion.error && (
            <p className="font-mono text-sm" style={{ color: 'var(--negative)', marginBottom: 0 }}>
              Completion plan unavailable: {completion.error}
            </p>
          )}
          {showCompletion && completion.data && !completion.data.valid && (
            <p className="font-ui text-sm" style={{ color: 'var(--warning)', marginBottom: 0 }}>
              {completion.data.message ?? 'No holdings to anchor from yet.'}
            </p>
          )}
          {showCompletion && completion.data?.valid && (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Asset</th>
                    <th className="text-right">Target</th>
                    <th className="text-right">Target value</th>
                    <th className="text-right">Current</th>
                    <th className="text-right">Still to buy</th>
                  </tr>
                </thead>
                <tbody>
                  {completion.data.rows.map((r) => (
                    <tr key={r.symbol}>
                      <td className="text-left" style={{ fontWeight: 500 }}>{r.symbol}</td>
                      <td className="text-right">{formatPercentPlain(r.target_allocation_pct)}</td>
                      <td className="text-right">{formatUsd(r.target_value_usd)}</td>
                      <td className="text-right">{formatUsd(r.current_value_usd)}</td>
                      <td className="text-right">{formatUsd(r.need_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="font-ui text-sm" style={{ marginBottom: 0 }}>
                Additional cash needed: {formatUsd(completion.data.additional_total_usd)}
              </p>
            </div>
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

            <Button onClick={runPreview} disabled={previewBusy}>
              {previewBusy ? 'Previewing…' : 'Preview allocation'}
            </Button>
          </div>

          <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                          marginTop: 'var(--space-3)', marginBottom: 0 }}>
            {STRATEGIES.find((s) => s.id === strategy)?.hint} This is a preview only -
            it never places an order.
          </p>
        </Panel>

        {previewError && (
          <Panel><p className="font-mono text-sm" style={{ color: 'var(--negative)', margin: 0 }}>
            Preview failed: {previewError}
          </p></Panel>
        )}

        {preview && (
          <Panel title={`Allocation preview - ${formatUsd(preview.amount_usd)}`}>
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
            description={`This ${status.data.is_live ? 'deploys' : 'simulates deploying'} ${formatUsd(checkedTotal)} across ${checked.length} asset${checked.length === 1 ? '' : 's'} as market buys on the Binance ${status.data.testnet ? 'testnet' : 'mainnet'}${status.data.is_live ? '' : ' (live trading is off, so no orders are sent)'}.`}
            disabled={checked.length === 0}
            execute={() => apiPost<TradeExecuteResponse>('/api/execute/dca', {
              confirm: true,
              strategy,
              trades: checked.map((a) => ({ asset: a.symbol, amount: a.amount_usd })),
            })}
          >
            <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
              {actionable.map((a) => (
                <label key={a.symbol} className="font-ui"
                       style={{ display: 'flex', alignItems: 'center',
                                gap: 'var(--space-2)', fontSize: '13px',
                                color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    aria-label={`Include ${a.symbol}`}
                    checked={isChecked(a.symbol)}
                    onChange={() => toggle(a.symbol)}
                  />
                  {a.symbol} - {formatUsd(a.amount_usd)}
                </label>
              ))}
              {checked.length === 0 && (
                <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
                  Select at least one trade.
                </p>
              )}
            </div>
          </ExecutePanel>
        )}
    </ExecutionScreen>
  );
}
