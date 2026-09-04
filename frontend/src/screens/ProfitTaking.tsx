import { useState } from 'react';
import { Panel } from '../components/Panel';
import { AnalysisBar, Badge, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { ExecutePanel } from '../components/ExecutePanel';
import { TradingStatusBanner } from '../components/TradingStatusBanner';
import { useApi, usePollWhile } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatPercent, formatSigned, formatUsd } from '../lib/format';
import type { ExecutionStatus, ProfitResponse, TradeExecuteResponse } from '../types';

function scoreTone(score: number | null) {
  if (score === null) return 'neutral' as const;
  if (score >= 75) return 'positive' as const;
  if (score >= 60) return 'warning' as const;
  return 'neutral' as const;
}

export function ProfitTaking() {
  const { data, error, reload } = useApi<ProfitResponse>('/api/strategy/profit');
  const status = useApi<ExecutionStatus>('/api/execute/status');
  usePollWhile(Boolean(data?.is_running), reload);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const isChecked = (s: string) => selected[s] ?? true;
  const toggle = (s: string) =>
    setSelected((prev) => ({ ...prev, [s]: !(prev[s] ?? true) }));

  async function run() {
    try {
      await apiPost('/api/strategy/profit/run');
    } finally {
      reload();
    }
  }

  if (error) return <ErrorPanel title="Profit taking" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Profit taking"><Empty>Loading…</Empty></Panel>;

  const actionable = data.opportunities;
  const checked = actionable.filter((o) => isChecked(o.symbol));

  return (
    <>
      <ScreenHeader title="Profit taking"
                    subtitle="Positions scoring high enough to consider trimming" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <TradingStatusBanner status={status.data ?? null} />
        <AnalysisBar state={data} onRun={run} label="Profit-taking analysis" />

        <Panel title="Opportunities">
          {!data.has_data ? (
            <Empty>No analysis has been run yet.</Empty>
          ) : data.opportunities.length === 0 ? (
            <Empty>
              No position currently meets the profit-taking criteria configured in
              your settings.
            </Empty>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Asset</th>
                    <th className="text-right">Score</th>
                    <th className="text-right">Unrealized</th>
                    <th className="text-right">Price</th>
                    <th className="text-right">Support</th>
                    <th className="text-right">Resistance</th>
                  </tr>
                </thead>
                <tbody>
                  {data.opportunities.map((o) => (
                    <tr key={o.symbol}>
                      <td className="text-left" style={{ fontWeight: 500 }}>{o.symbol}</td>
                      <td className="text-right">
                        <Badge
                          text={o.opportunity_score === null
                            ? '—' : o.opportunity_score.toFixed(0)}
                          tone={scoreTone(o.opportunity_score)}
                        />
                      </td>
                      <td className="text-right"
                          style={{ color: (o.unrealized_gain_usd ?? 0) >= 0
                            ? 'var(--positive)' : 'var(--negative)' }}>
                        {formatSigned(o.unrealized_gain_usd)} ({formatPercent(o.unrealized_gain_pct)})
                      </td>
                      <td className="text-right">{formatUsd(o.current_price)}</td>
                      <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                        {formatUsd(o.support_level)}
                      </td>
                      <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                        {formatUsd(o.resistance_level)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        {data.opportunities.some((o) => o.reasons.length > 0) && (
          <Panel title="Why these scored">
            <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
              {data.opportunities.filter((o) => o.reasons.length > 0).map((o) => (
                <div key={o.symbol} className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
                  <span className="font-mono" style={{ fontSize: '13px', fontWeight: 500 }}>
                    {o.symbol}
                  </span>
                  <ul className="flex flex-col font-ui"
                      style={{ gap: '4px', fontSize: '13px', margin: 0,
                               paddingLeft: 'var(--space-4)', color: 'var(--text-secondary)' }}>
                    {o.reasons.map((reason, i) => <li key={i}>{reason}</li>)}
                  </ul>
                </div>
              ))}
            </div>
          </Panel>
        )}

        {status.data && data.has_data && data.opportunities.length > 0 && (
          <ExecutePanel
            title={`${status.data.is_live ? 'Execute' : 'Simulate'} profit-taking`}
            description={`This ${status.data.is_live ? 'trims' : 'simulates trimming'} ${checked.length} scoring position${checked.length === 1 ? '' : 's'} — the selected position${checked.length === 1 ? '' : 's'} above, selling the configured share of each gain — as market sells on the Binance ${status.data.testnet ? 'testnet' : 'mainnet'}${status.data.is_live ? '' : ' (live trading is off, so no orders are sent)'}.`}
            disabled={checked.length === 0}
            execute={async () => {
              const res = await apiPost<TradeExecuteResponse>(
                '/api/execute/profit',
                { confirm: true, symbols: checked.map((o) => o.symbol) });
              reload();
              return res;
            }}
          >
            <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
              {actionable.map((o) => (
                <label key={o.symbol} className="font-ui"
                       style={{ display: 'flex', alignItems: 'center',
                                gap: 'var(--space-2)', fontSize: '13px',
                                color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    aria-label={`Include ${o.symbol}`}
                    checked={isChecked(o.symbol)}
                    onChange={() => toggle(o.symbol)}
                  />
                  {o.symbol}
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
      </div>
    </>
  );
}
