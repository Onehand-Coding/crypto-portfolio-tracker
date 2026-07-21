import { Panel } from '../components/Panel';
import { AnalysisBar, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi, usePollWhile } from '../lib/useApi';
import { apiPost } from '../lib/api';
import type { AnalysisState } from '../types';

interface BacktestResponse extends AnalysisState {
  result: Record<string, unknown> | null;
  report: unknown;
}

function isNumeric(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function label(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function Backtest() {
  const { data, error, reload } = useApi<BacktestResponse>('/api/strategy/backtest');
  usePollWhile(Boolean(data?.is_running), reload);

  async function run() {
    try {
      await apiPost('/api/strategy/backtest/run');
    } finally {
      reload();
    }
  }

  if (error) return <ErrorPanel title="Backtesting" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Backtesting"><Empty>Loading…</Empty></Panel>;

  const result = data.result ?? {};
  const scalars = Object.entries(result).filter(([, v]) => isNumeric(v) || typeof v === 'string');

  return (
    <>
      <ScreenHeader
        title="Backtesting"
        subtitle="Simulates the rebalancing strategy over two years, monthly, from $10,000"
      />

      <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
        <AnalysisBar state={data} onRun={run} label="Backtesting" />

        <Panel title="Result">
          {!data.has_data ? (
            <Empty>
              No backtest has been run yet. A run fetches several years of price history
              and takes noticeably longer than the other analyses.
            </Empty>
          ) : scalars.length === 0 ? (
            <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
              The backtest completed but returned no summary figures. That usually means
              price history was unavailable for the configured assets rather than that
              the strategy produced nothing.
            </p>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Metric</th>
                    <th className="text-right">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {scalars.map(([key, value]) => (
                    <tr key={key}>
                      <td className="text-left">{label(key)}</td>
                      <td className="text-right">
                        {isNumeric(value) ? value.toLocaleString(undefined, {
                          maximumFractionDigits: 2,
                        }) : String(value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
