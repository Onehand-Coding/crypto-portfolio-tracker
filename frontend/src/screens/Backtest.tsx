import { useState } from 'react';
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { Panel } from '../components/Panel';
import { BandMetric, KpiBand } from '../components/Band';
import { Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { StalenessNote } from '../components/StalenessNote';
import { useApi, usePollWhile } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatPercent, formatUsd } from '../lib/format';
import type { BacktestResponse, SystemHealthResponse } from '../types';

const PERIODS = ['1y', '2y', '3y', '5y', 'max', 'Custom'];
const CUSTOM_PERIOD_RE = /^\d+y$/;
// Defaults mirror config/default_config.json rebalance_technical so an
// untouched advanced section reproduces a plain run.
const DEFAULT_MAJORS_DRIFT = '3.0';
const DEFAULT_ALTS_DRIFT = '3.5';
const DEFAULT_MAJORS_SELL = '0.5';
const DEFAULT_MAJORS_BUY = '0.75';
const DEFAULT_ALTS_SELL = '0.5';
const DEFAULT_ALTS_BUY = '1.0';

/** Backend clamps the same way; clamping here keeps the posted payload honest. */
function clampNum(raw: string, lo: number, hi: number, fallback: number): number {
  // Number inputs sanitise unparseable text to '' — like the backend, that
  // means "no value", not zero (Number('') is 0, which would snap to lo).
  if (raw.trim() === '') return fallback;
  const v = Number(raw);
  if (!Number.isFinite(v)) return fallback;
  return Math.min(hi, Math.max(lo, v));
}
const FREQUENCIES = [
  { id: 'weekly', label: 'Weekly' },
  { id: 'monthly', label: 'Monthly' },
  { id: 'quarterly', label: 'Quarterly' },
];

// The core labels metrics for humans and mixes units in one dict, so each key
// is classified rather than formatted uniformly. A ratio shown as dollars, or
// a dollar shown as a percent, would be actively misleading.
const PERCENT_KEYS = new Set([
  'Strategy Total Return', 'Buy & Hold Return', 'Strategy Outperformance',
  'Maximum Drawdown', 'Annualized Volatility',
]);
const USD_KEYS = new Set(['Initial Capital', 'Final Portfolio Value']);

function formatMetric(key: string, value: number): string {
  if (USD_KEYS.has(key)) return formatUsd(value);
  if (PERCENT_KEYS.has(key)) return formatPercent(value * 100);
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function ConfigButton({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className="font-mono transition-colors"
      style={{
        background: active ? 'var(--surface-2)' : 'transparent',
        color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
        border: `1px solid ${active ? 'var(--border-strong)' : 'var(--border)'}`,
        borderRadius: 'var(--radius-control)',
        padding: 'var(--space-2) var(--space-3)', fontSize: '13px', cursor: 'pointer',
      }}
    >
      {children}
    </button>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
      <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                         letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {label}
      </span>
      {children}
    </div>
  );
}

function NumField({ label, value, onChange, min, max, step }: {
  label: string; value: string; onChange: (v: string) => void;
  min: number; max: number; step: number;
}) {
  return (
    <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
      <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                         letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {label}
      </span>
      <input
        aria-label={label}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(e.target.value)}
        className="font-mono"
        style={{
          background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
          borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
          padding: 'var(--space-2) var(--space-3)', width: '120px', fontSize: '14px',
        }}
      />
    </label>
  );
}

export function Backtest() {
  const { data, error, reload } = useApi<BacktestResponse>('/api/strategy/backtest');
  // Same endpoint Allocation.tsx uses for live target weights.
  const health = useApi<SystemHealthResponse>('/api/system/health');
  usePollWhile(Boolean(data?.is_running), reload);

  const [capital, setCapital] = useState('10000');
  const [period, setPeriod] = useState('2y');
  const [frequency, setFrequency] = useState('monthly');
  const [customPeriod, setCustomPeriod] = useState('6y');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advancedTouched, setAdvancedTouched] = useState(false);
  const [majorsDrift, setMajorsDrift] = useState(DEFAULT_MAJORS_DRIFT);
  const [altsDrift, setAltsDrift] = useState(DEFAULT_ALTS_DRIFT);
  const [majorsSell, setMajorsSell] = useState(DEFAULT_MAJORS_SELL);
  const [majorsBuy, setMajorsBuy] = useState(DEFAULT_MAJORS_BUY);
  const [altsSell, setAltsSell] = useState(DEFAULT_ALTS_SELL);
  const [altsBuy, setAltsBuy] = useState(DEFAULT_ALTS_BUY);
  const [suppressBear, setSuppressBear] = useState(true);
  const [weights, setWeights] = useState<Record<string, string> | null>(null);

  // Mark the advanced section dirty: custom is omitted entirely until then,
  // so an untouched section stays a byte-identical plain run.
  function touch(setter: (v: string) => void) {
    return (v: string) => { setAdvancedTouched(true); setter(v); };
  }

  const customPeriodValid = CUSTOM_PERIOD_RE.test(customPeriod);
  const effectivePeriod = period === 'Custom' ? customPeriod : period;

  const targets = health.data?.target_allocation ?? null;
  const shownWeights: Record<string, string> | null = weights
    ?? (targets
      ? Object.fromEntries(Object.entries(targets).map(([s, w]) => [s, String(Number((w * 100).toFixed(2)))]))
      : null);
  const allocSum = shownWeights
    ? Object.values(shownWeights).reduce((sum, v) => sum + (Number(v) || 0) / 100, 0)
    : 1;
  const allocValid = Math.abs(allocSum - 1) < 0.001;

  function setWeight(symbol: string, v: string) {
    setAdvancedTouched(true);
    const base = shownWeights ?? {};
    setWeights({ ...base, [symbol]: v });
  }

  const runDisabled = Boolean(data?.is_running)
    || (period === 'Custom' && !customPeriodValid)
    || (showAdvanced && shownWeights !== null && !allocValid);

  async function run() {
    const body: Record<string, unknown> = {
      initial_capital: Number(capital), period: effectivePeriod, frequency,
    };
    if (showAdvanced && advancedTouched) {
      const custom: Record<string, unknown> = {
        majors_drift: clampNum(majorsDrift, 1, 20, Number(DEFAULT_MAJORS_DRIFT)),
        alts_drift: clampNum(altsDrift, 1, 20, Number(DEFAULT_ALTS_DRIFT)),
        majors_sell: clampNum(majorsSell, 0.1, 2, Number(DEFAULT_MAJORS_SELL)),
        majors_buy: clampNum(majorsBuy, 0.1, 2, Number(DEFAULT_MAJORS_BUY)),
        alts_sell: clampNum(altsSell, 0.1, 2, Number(DEFAULT_ALTS_SELL)),
        alts_buy: clampNum(altsBuy, 0.1, 2, Number(DEFAULT_ALTS_BUY)),
        suppress_bear: suppressBear,
      };
      if (shownWeights !== null && allocValid) {
        custom.allocation = Object.fromEntries(
          Object.entries(shownWeights).map(([s, v]) => [s, (Number(v) || 0) / 100]),
        );
      }
      body.custom = custom;
    }
    try {
      await apiPost('/api/strategy/backtest/run', body);
    } finally {
      reload();
    }
  }

  if (error) return <ErrorPanel title="Backtesting" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Backtesting"><Empty>Loading…</Empty></Panel>;

  const result = data.result ?? {};
  const metrics = Object.entries(result);
  const history = (data.value_history ?? [])
    .filter((p) => p.value !== null)
    .map((p) => ({ t: p.date, value: p.value as number }));

  const finalValue = result['Final Portfolio Value'] ?? null;
  const stratReturn = result['Strategy Total Return'] ?? null;
  const outperformance = result['Strategy Outperformance'] ?? null;
  const drawdown = result['Maximum Drawdown'] ?? null;

  return (
    <>
      <ScreenHeader
        title="Backtesting"
        subtitle="Simulate the rebalancing strategy over historical data"
        staleness={data.staleness}
      />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel title="Configuration">
          <div className="flex flex-wrap items-end" style={{ gap: 'var(--space-5)' }}>
            <Field label="Starting capital (USD)">
              <input
                value={capital}
                onChange={(e) => setCapital(e.target.value)}
                inputMode="decimal"
                className="font-mono"
                style={{
                  background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
                  borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                  padding: 'var(--space-2) var(--space-3)', width: '150px', fontSize: '14px',
                }}
              />
            </Field>
            <Field label="Period">
              <div className="flex" style={{ gap: 'var(--space-2)', flexWrap: 'wrap' }}>
                {PERIODS.map((p) => (
                  <ConfigButton key={p} active={period === p} onClick={() => setPeriod(p)}>
                    {p}
                  </ConfigButton>
                ))}
              </div>
              {period === 'Custom' && (
                <>
                  <input
                    aria-label="Custom period"
                    value={customPeriod}
                    onChange={(e) => setCustomPeriod(e.target.value)}
                    placeholder="6y"
                    className="font-mono"
                    style={{
                      background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
                      borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                      padding: 'var(--space-2) var(--space-3)', width: '120px', fontSize: '14px',
                      marginTop: 'var(--space-2)',
                    }}
                  />
                  {!customPeriodValid && (
                    <span className="font-ui" style={{ color: 'var(--warning)', fontSize: '12px' }}>
                      Custom period must look like 6y — a number followed by y.
                    </span>
                  )}
                </>
              )}
            </Field>
            <Field label="Rebalance frequency">
              <div className="flex" style={{ gap: 'var(--space-2)' }}>
                {FREQUENCIES.map((f) => (
                  <ConfigButton key={f.id} active={frequency === f.id}
                                onClick={() => setFrequency(f.id)}>
                    {f.label}
                  </ConfigButton>
                ))}
              </div>
            </Field>
            <div className="flex items-center" style={{ gap: 'var(--space-4)' }}>
              <StalenessNote staleness={data.staleness} />
              <Button onClick={run} disabled={runDisabled}>
                {data.is_running ? 'Running…' : 'Run backtest'}
              </Button>
            </div>
          </div>
          {showAdvanced && shownWeights !== null && !allocValid && (
            <p className="font-ui" style={{ color: 'var(--warning)', fontSize: '12px',
                                            marginTop: 'var(--space-2)', marginBottom: 0 }}>
              Custom allocation weights must sum to 100% (now {(allocSum * 100).toFixed(1)}%).
            </p>
          )}
          <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                          marginTop: 'var(--space-3)', marginBottom: 0 }}>
            Fetches historical prices for your target assets and simulates the rebalancing
            strategy. A longer period fetches more data and takes noticeably longer.
          </p>
          {data.error && (
            <p className="font-mono" style={{ color: 'var(--negative)', fontSize: '12px',
                                              marginTop: 'var(--space-2)', marginBottom: 0 }}>
              Last run failed: {data.error}
            </p>
          )}
        </Panel>

        <Panel title="Advanced parameters">
          <Button variant="secondary" onClick={() => setShowAdvanced((v) => !v)}>
            {showAdvanced ? 'Hide advanced parameters' : 'Show advanced parameters'}
          </Button>
          {showAdvanced && (
            <div className="flex flex-col" style={{ gap: 'var(--space-4)', marginTop: 'var(--space-3)' }}>
              <div className="flex flex-wrap" style={{ gap: 'var(--space-4)' }}>
                <NumField label="Majors drift threshold (%)" value={majorsDrift}
                          onChange={touch(setMajorsDrift)} min={1} max={20} step={0.5} />
                <NumField label="Alts drift threshold (%)" value={altsDrift}
                          onChange={touch(setAltsDrift)} min={1} max={20} step={0.5} />
              </div>
              <div className="flex flex-wrap" style={{ gap: 'var(--space-4)' }}>
                <NumField label="Majors sell multiplier" value={majorsSell}
                          onChange={touch(setMajorsSell)} min={0.1} max={2} step={0.1} />
                <NumField label="Majors buy multiplier" value={majorsBuy}
                          onChange={touch(setMajorsBuy)} min={0.1} max={2} step={0.1} />
                <NumField label="Alts sell multiplier" value={altsSell}
                          onChange={touch(setAltsSell)} min={0.1} max={2} step={0.1} />
                <NumField label="Alts buy multiplier" value={altsBuy}
                          onChange={touch(setAltsBuy)} min={0.1} max={2} step={0.1} />
              </div>
              <label className="font-ui"
                     style={{ display: 'flex', alignItems: 'center',
                              gap: 'var(--space-2)', fontSize: '13px',
                              color: 'var(--text-secondary)', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  aria-label="Suppress buys in bear market"
                  checked={suppressBear}
                  onChange={(e) => { setAdvancedTouched(true); setSuppressBear(e.target.checked); }}
                />
                Suppress buys in bear market
              </label>
              <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
                <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                                   letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  Custom allocation (% per asset)
                </span>
                {shownWeights === null ? (
                  <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)', margin: 0 }}>
                    {health.error ? 'Target allocation unavailable.' : 'Loading targets…'}
                  </p>
                ) : (
                  <div className="flex flex-wrap" style={{ gap: 'var(--space-4)' }}>
                    {Object.entries(shownWeights).map(([symbol, v]) => (
                      <NumField key={symbol} label={`${symbol} weight (%)`} value={v}
                                onChange={(nv) => setWeight(symbol, nv)} min={0} max={100} step={0.1} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </Panel>

        {!data.has_data ? (
          <Panel title="Result">
            <Empty>
              No backtest has been run yet. Configure it above and run one — it fetches
              several years of price history and takes longer than the other analyses.
            </Empty>
          </Panel>
        ) : metrics.length === 0 ? (
          <Panel title="Result">
            <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
              The backtest completed but returned no summary figures. That usually means
              price history was unavailable for the configured assets and period.
            </p>
          </Panel>
        ) : (
          <>
            <Panel>
              <KpiBand>
                <BandMetric emphasis label="Final value" value={formatUsd(finalValue)} />
                <BandMetric label="Strategy return"
                            value={formatPercent(stratReturn === null ? null : stratReturn * 100)}
                            signal={stratReturn} />
                <BandMetric label="vs buy & hold"
                            value={formatPercent(outperformance === null ? null : outperformance * 100)}
                            signal={outperformance} />
                <BandMetric label="Max drawdown"
                            value={formatPercent(drawdown === null ? null : drawdown * 100)}
                            signal={drawdown} />
                {data.config && (
                  <BandMetric label="Run"
                              value={`${formatUsd(data.config.initial_capital)} · ${data.config.period}`}
                              sub={data.config.frequency} />
                )}
              </KpiBand>
            </Panel>

            {history.length >= 2 && (
              <Panel title="Equity curve">
                <div style={{ width: '100%', height: 260 }}>
                  <ResponsiveContainer>
                    <AreaChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                      <defs>
                        <linearGradient id="btFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--action)" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="var(--action)" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
                      <XAxis dataKey="t" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                             stroke="var(--border)" minTickGap={48}
                             tickFormatter={(t) => String(t).slice(0, 10)} />
                      <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                             stroke="var(--border)" width={64}
                             tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
                      <Tooltip
                        contentStyle={{ background: 'var(--surface-2)',
                                        border: '1px solid var(--border-strong)',
                                        borderRadius: 'var(--radius-control)', fontSize: '12px' }}
                        labelStyle={{ color: 'var(--text-secondary)' }}
                        labelFormatter={(t) => String(t).slice(0, 10)}
                        formatter={(v) => [formatUsd(typeof v === 'number' ? v : null), 'Value']}
                      />
                      <Area type="monotone" dataKey="value" stroke="var(--action)" strokeWidth={1.5}
                            fill="url(#btFill)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            )}

            <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
                                           gap: 'var(--space-3)' }}>
              <Panel title="Performance metrics">
                <div className="table-scroll">
                  <table className="data">
                    <thead>
                      <tr>
                        <th className="text-left">Metric</th>
                        <th className="text-right">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map(([key, value]) => (
                        <tr key={key}>
                          <td className="text-left">{key}</td>
                          <td className="text-right"
                              style={{ color: PERCENT_KEYS.has(key) && value !== 0
                                ? (value > 0 ? 'var(--positive)' : 'var(--negative)')
                                : undefined }}>
                            {formatMetric(key, value)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>

              <Panel title={`Trade log (${(data.trade_log ?? []).length})`}>
                {(data.trade_log ?? []).length === 0 ? (
                  <Empty>The simulation placed no trades.</Empty>
                ) : (
                  <div className="table-scroll" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                    <ul className="flex flex-col font-mono"
                        style={{ gap: '4px', fontSize: '12px', margin: 0, padding: 0,
                                 listStyle: 'none', color: 'var(--text-secondary)' }}>
                      {(data.trade_log ?? []).map((line, i) => (
                        <li key={i}>{line.replace(/^SIM:\s*/, '')}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </Panel>
            </div>
          </>
        )}
      </div>
    </>
  );
}
