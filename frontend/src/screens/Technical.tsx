import { Panel } from '../components/Panel';
import { AnalysisBar, Badge, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi, usePollWhile } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatUsd } from '../lib/format';
import type { IndicatorRow, TechnicalResponse } from '../types';

const TIMEFRAME_LABEL: Record<string, string> = {
  swing: 'Swing (3 months)',
  long_term: 'Long term (4 years)',
  day: 'Day (60 days)',
};

function rsiTone(rsi: number | null) {
  if (rsi === null) return 'neutral' as const;
  if (rsi < 30) return 'positive' as const;   // oversold: a buy signal here
  if (rsi > 70) return 'negative' as const;   // overbought
  return 'neutral' as const;
}

function IndicatorTable({ rows }: { rows: IndicatorRow[] }) {
  if (rows.length === 0) return <Empty>No indicator data in this timeframe.</Empty>;
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th className="text-left">Asset</th>
            <th className="text-right">Price</th>
            <th className="text-right">RSI</th>
            <th className="text-right">SMA short</th>
            <th className="text-right">SMA long</th>
            <th className="text-right">Support</th>
            <th className="text-right">Resistance</th>
            <th className="text-left">Conditions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.symbol}>
              <td className="text-left" style={{ fontWeight: 500 }}>{row.symbol}</td>
              <td className="text-right">{formatUsd(row.price)}</td>
              <td className="text-right">
                {row.rsi === null ? '—'
                  : <Badge text={row.rsi.toFixed(1)} tone={rsiTone(row.rsi)} />}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(row.sma_short)}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(row.sma_long)}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(row.support)}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(row.resistance)}
              </td>
              <td className="text-left">
                <span className="flex flex-wrap" style={{ gap: '4px' }}>
                  {row.conditions.slice(0, 3).map((c) => (
                    <Badge key={c} text={c.replace(/_/g, ' ').toLowerCase()} />
                  ))}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Technical() {
  const { data, error, reload } = useApi<TechnicalResponse>('/api/strategy/technical');
  usePollWhile(Boolean(data?.is_running), reload);

  async function run() {
    try {
      await apiPost('/api/strategy/technical/run');
    } finally {
      reload();
    }
  }

  if (error) return <ErrorPanel title="Technical analysis" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Technical analysis"><Empty>Loading…</Empty></Panel>;

  const timeframes = Object.entries(data.timeframes);

  return (
    <>
      <ScreenHeader title="Technical analysis"
                    subtitle="RSI, moving averages and support/resistance by timeframe" />

      <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
        <AnalysisBar state={data} onRun={run} label="Technical analysis" />

        {data.bear_market !== null && (
          <Panel>
            <div className="flex items-center" style={{ gap: 'var(--space-3)' }}>
              <Badge
                text={data.bear_market ? 'BEAR MARKET' : 'NOT BEAR MARKET'}
                tone={data.bear_market ? 'negative' : 'positive'}
              />
              <span className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
                Regime is defined by BTC relative to its SMA200. In a bear market your
                config can suppress buy signals.
              </span>
            </div>
          </Panel>
        )}

        {!data.has_data ? (
          <Panel title="Indicators"><Empty>No analysis has been run yet.</Empty></Panel>
        ) : timeframes.length === 0 ? (
          <Panel title="Indicators">
            <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
              The last run returned no timeframe reports. This usually means the market
              data fetch failed rather than that there is nothing to show.
            </p>
          </Panel>
        ) : (
          timeframes.map(([timeframe, rows]) => (
            <Panel key={timeframe} title={TIMEFRAME_LABEL[timeframe] ?? timeframe}>
              <IndicatorTable rows={rows} />
            </Panel>
          ))
        )}
      </div>
    </>
  );
}
