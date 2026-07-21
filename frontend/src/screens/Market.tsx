import { Link } from 'react-router-dom';
import { Panel } from '../components/Panel';
import { Badge, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { formatPercentPlain, formatUsd } from '../lib/format';
import type { CockpitResponse, SystemHealthResponse, TechnicalResponse } from '../types';

/**
 * Your configured universe with its current prices and trend context.
 *
 * The core has no market-discovery feed — no screener, no movers endpoint — so
 * this deliberately does not pretend to be one. It shows the assets you have
 * chosen to care about, which is what the rest of the app acts on.
 */
export function Market() {
  const cockpit = useApi<CockpitResponse>('/api/portfolio/cockpit');
  const health = useApi<SystemHealthResponse>('/api/system/health');
  const technical = useApi<TechnicalResponse>('/api/strategy/technical');

  const error = cockpit.error ?? health.error ?? technical.error;
  if (error) return <ErrorPanel title="Market" message={`Failed to load market view: ${error}`} />;
  if (!cockpit.data || !health.data || !technical.data) {
    return <Panel title="Market"><Empty>Loading…</Empty></Panel>;
  }

  const holdings = new Map(cockpit.data.holdings.map((h) => [h.symbol, h]));
  const swing = technical.data.timeframes.swing ?? [];
  const indicators = new Map(swing.map((row) => [row.symbol, row]));

  const symbols = Array.from(
    new Set([
      ...Object.keys(health.data.target_allocation),
      ...cockpit.data.holdings.map((h) => h.symbol),
    ]),
  ).sort();

  return (
    <>
      <ScreenHeader
        title="Market"
        subtitle="Your configured universe, with trend context where it has been computed"
        staleness={cockpit.data.staleness}
      />

      <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
        {!technical.data.has_data && (
          <Panel>
            <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)', margin: 0 }}>
              RSI and trend columns are blank because no technical analysis has been run.
              Run one from the Technical screen to fill them in.
            </p>
          </Panel>
        )}

        <Panel title={`Assets (${symbols.length})`}>
          {symbols.length === 0 ? (
            <Empty>No holdings and no target allocation configured.</Empty>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Asset</th>
                    <th className="text-left">Role</th>
                    <th className="text-right">Price</th>
                    <th className="text-right">Held</th>
                    <th className="text-right">RSI</th>
                    <th className="text-right">Support</th>
                    <th className="text-right">Resistance</th>
                  </tr>
                </thead>
                <tbody>
                  {symbols.map((symbol) => {
                    const holding = holdings.get(symbol);
                    const indicator = indicators.get(symbol);
                    const isCore = symbol in health.data!.target_allocation;
                    return (
                      <tr key={symbol}>
                        <td className="text-left" style={{ fontWeight: 500 }}>
                          {/* Every asset links to its own detail screen. */}
                          <Link to={`/assets/${symbol}`}
                                style={{ color: 'var(--text-primary)' }}>
                            {symbol}
                          </Link>
                        </td>
                        <td className="text-left">
                          {isCore
                            ? <Badge text={`core ${formatPercentPlain(
                                health.data!.target_allocation[symbol] * 100)}`} tone="action" />
                            : <Badge text="other" />}
                        </td>
                        <td className="text-right">
                          {formatUsd(holding?.current_price ?? indicator?.price ?? null)}
                        </td>
                        <td className="text-right"
                            style={{ color: holding ? undefined : 'var(--text-tertiary)' }}>
                          {holding ? formatUsd(holding.value_usd) : 'not held'}
                        </td>
                        <td className="text-right">
                          {indicator?.rsi == null ? '—' : indicator.rsi.toFixed(1)}
                        </td>
                        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                          {formatUsd(indicator?.support ?? null)}
                        </td>
                        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                          {formatUsd(indicator?.resistance ?? null)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
