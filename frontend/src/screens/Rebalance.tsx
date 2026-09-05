import { useState } from 'react';
import { Panel } from '../components/Panel';
import { AnalysisBar, Badge, Empty, ErrorPanel } from '../components/Screen';
import { ExecutePanel } from '../components/ExecutePanel';
import { ExecutionScreen } from '../components/ExecutionScreen';
import { useApi, usePollWhile } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatPercent, formatPercentPlain, formatQty, formatUsd, NULL_GLYPH} from '../lib/format';
import type {
  ExecutionStatus, RebalanceResponse, RebalanceSuggestion, TradeExecuteResponse,
} from '../types';

function actionTone(action: string | null) {
  if (!action) return 'neutral' as const;
  const upper = action.toUpperCase();
  if (upper.includes('BUY')) return 'positive' as const;
  if (upper.includes('SELL')) return 'negative' as const;
  return 'neutral' as const;
}

/** Current vs target as a bar, so drift is visible before it is read. */
function DriftBar({ suggestion }: { suggestion: RebalanceSuggestion }) {
  const current = suggestion.current_allocation_pct;
  const target = suggestion.target_allocation_pct;
  if (current === null || target === null) {
    return <span style={{ color: 'var(--text-tertiary)' }}>-</span>;
  }
  const scale = Math.max(current, target, 1) * 1.25;
  return (
    <div style={{ position: 'relative', height: '18px', width: '160px',
                  background: 'var(--surface-2)', borderRadius: '2px' }}>
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0,
                    width: `${(current / scale) * 100}%`,
                    background: 'var(--action)', borderRadius: '2px' }} />
      {/* Target is a marker, not a fill: the question is where the bar sits
          relative to the line, not how long each one is. */}
      <div style={{ position: 'absolute', left: `${(target / scale) * 100}%`, top: '-2px',
                    bottom: '-2px', width: '2px', background: 'var(--text-primary)' }} />
    </div>
  );
}

export function Rebalance() {
  const { data, error, reload } = useApi<RebalanceResponse>('/api/strategy/rebalance');
  const status = useApi<ExecutionStatus>('/api/execute/status');
  usePollWhile(Boolean(data?.is_running), reload);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const isChecked = (s: string) => selected[s] ?? true;
  const toggle = (s: string) =>
    setSelected((prev) => ({ ...prev, [s]: !(prev[s] ?? true) }));

  async function run() {
    try {
      await apiPost('/api/strategy/rebalance/run');
    } finally {
      reload();
    }
  }

  if (error) return <ErrorPanel title="Rebalancing" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Rebalancing"><Empty>Loading…</Empty></Panel>;

  return (
    <ExecutionScreen title="Rebalancing"
                     subtitle="Current vs target allocation, with technical context"
                     status={status.data ?? null}>
      <AnalysisBar state={data} onRun={run} label="Rebalancing analysis" />

        <Panel title="Suggestions">
          {!data.has_data ? (
            <Empty>
              No analysis has been run yet. Run one to see current vs target allocation.
            </Empty>
          ) : data.suggestions.length === 0 ? (
            <Empty>
              The last run produced no suggestions - every core asset is within its
              drift threshold.
            </Empty>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Asset</th>
                    <th className="text-left">Action</th>
                    <th className="text-left">Current vs target</th>
                    <th className="text-right">Current</th>
                    <th className="text-right">Target</th>
                    <th className="text-right">Drift</th>
                    <th className="text-right">Value</th>
                    <th className="text-right">Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {data.suggestions.map((s) => (
                    <tr key={s.symbol}>
                      <td className="text-left" style={{ fontWeight: 500 }}>{s.symbol}</td>
                      <td className="text-left">
                        {s.action ? <Badge text={s.action} tone={actionTone(s.action)} /> : NULL_GLYPH}
                      </td>
                      <td className="text-left"><DriftBar suggestion={s} /></td>
                      <td className="text-right">{formatPercentPlain(s.current_allocation_pct)}</td>
                      <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                        {formatPercentPlain(s.target_allocation_pct)}
                      </td>
                      {/* Drift is not P/L: being over target is not "good" and
                          under target is not "bad". Both are deviations, so
                          colour tracks magnitude, not direction. */}
                      <td className="text-right"
                          style={{ color: s.drift_pct === null ? undefined
                                        : Math.abs(s.drift_pct) < 1 ? 'var(--text-tertiary)'
                                        : 'var(--warning)' }}>
                        {formatPercent(s.drift_pct)}
                      </td>
                      <td className="text-right">{formatUsd(s.current_value_usd)}</td>
                      <td className="text-right">
                        {formatUsd(s.action_amount_usd)}
                        {s.action_quantity !== null && (
                          <span style={{ color: 'var(--text-tertiary)', marginLeft: '8px' }}>
                            {formatQty(s.action_quantity)}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        {data.suggestions.some((s) => s.reason) && (
          <Panel title="Reasoning">
            <ul className="flex flex-col font-ui"
                style={{ gap: 'var(--space-2)', fontSize: '13px', margin: 0, padding: 0,
                         listStyle: 'none' }}>
              {data.suggestions.filter((s) => s.reason).map((s) => (
                <li key={s.symbol} style={{ color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                    {s.symbol}
                  </span>{' '}
                  {s.reason}
                </li>
              ))}
            </ul>
          </Panel>
        )}

        {status.data && (() => {
          const actionable = data.suggestions.filter(
            (s) => s.action && /BUY|SELL/i.test(s.action),
          );
          if (actionable.length === 0) return null;
          const checked = actionable.filter((r) => isChecked(r.symbol));
          const live = status.data.is_live;
          const net = status.data.testnet ? 'testnet' : 'mainnet';
          return (
            <ExecutePanel
              title={`${live ? 'Execute' : 'Simulate'} rebalance`}
              description={`This ${live ? 'places' : 'simulates'} ${checked.length} market order${checked.length === 1 ? '' : 's'} - the selected suggestion${checked.length === 1 ? '' : 's'} above - on the Binance ${net}${live ? '' : ' (live trading is off, so no orders are sent)'}.`}
              disabled={checked.length === 0}
              execute={async () => {
                const res = await apiPost<TradeExecuteResponse>(
                  '/api/execute/rebalance',
                  { confirm: true, symbols: checked.map((r) => r.symbol) });
                reload();
                return res;
              }}
            >
              <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
                {actionable.map((r) => (
                  <label key={r.symbol} className="font-ui"
                         style={{ display: 'flex', alignItems: 'center',
                                  gap: 'var(--space-2)', fontSize: '13px',
                                  color: 'var(--text-secondary)', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      aria-label={`Include ${r.symbol}`}
                      checked={isChecked(r.symbol)}
                      onChange={() => toggle(r.symbol)}
                    />
                    {r.symbol}
                  </label>
                ))}
                {checked.length === 0 && (
                  <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
                    Select at least one trade.
                  </p>
                )}
              </div>
            </ExecutePanel>
          );
        })()}
    </ExecutionScreen>
  );
}
