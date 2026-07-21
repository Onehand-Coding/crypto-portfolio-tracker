import { Panel } from '../components/Panel';
import { BandMetric, KpiBand } from '../components/Band';
import { Badge, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { formatPercentPlain, formatUsd } from '../lib/format';
import type { SystemHealthResponse } from '../types';

function humanSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function humanAge(seconds: number | null): string {
  if (seconds === null) return 'never';
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.round(minutes / 60)}h ago`;
}

export function SystemHealth() {
  const { data, error } = useApi<SystemHealthResponse>('/api/system/health');

  if (error) return <ErrorPanel title="System" message={`Failed to load system health: ${error}`} />;
  if (!data) return <Panel title="System"><Empty>Loading…</Empty></Panel>;

  const allocation = Object.entries(data.target_allocation);
  const allocationTotal = allocation.reduce((sum, [, weight]) => sum + weight, 0);

  return (
    <>
      <ScreenHeader title="System & settings"
                    subtitle="Environment, database and configuration" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel title="Environment">
          <div className="flex items-center" style={{ gap: 'var(--space-3)',
                                                      marginBottom: 'var(--space-4)' }}>
            <Badge text={data.environment_label}
                   tone={data.is_testnet ? 'warning' : 'action'} />
            <Badge text={data.binance_configured ? 'API KEYS PRESENT' : 'API KEYS MISSING'}
                   tone={data.binance_configured ? 'positive' : 'negative'} />
            <Badge text={data.live_trading_enabled ? 'LIVE TRADING ENABLED' : 'LIVE TRADING OFF'}
                   tone={data.live_trading_enabled ? 'negative' : 'neutral'} />
          </div>
          <p className="font-mono" style={{ color: 'var(--text-secondary)', fontSize: '12px',
                                            margin: 0, wordBreak: 'break-all' }}>
            {data.database_path}
          </p>
          {!data.database_exists && (
            <p className="font-ui text-sm" style={{ color: 'var(--negative)',
                                                    marginTop: 'var(--space-3)', marginBottom: 0 }}>
              Database file does not exist at this path.
            </p>
          )}
        </Panel>

        <Panel title="Database">
          <KpiBand>
            <BandMetric label="Transactions" value={data.transaction_count.toLocaleString()} />
            <BandMetric label="Assets" value={String(data.asset_count)} />
            <BandMetric label="Snapshots" value={String(data.snapshot_count)} />
            <BandMetric label="Size" value={humanSize(data.database_size_bytes)} />
            <BandMetric label="Metrics cache"
                        value={humanAge(data.metrics_cache_age_seconds)} />
          </KpiBand>
        </Panel>

        <Panel title="Target allocation">
          {allocation.length === 0 ? (
            <Empty>No target allocation configured.</Empty>
          ) : (
            <>
              <div className="table-scroll">
                <table className="data">
                  <thead>
                    <tr>
                      <th className="text-left">Asset</th>
                      <th className="text-right">Target</th>
                      <th className="text-left">Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allocation.map(([symbol, weight]) => (
                      <tr key={symbol}>
                        <td className="text-left" style={{ fontWeight: 500 }}>{symbol}</td>
                        <td className="text-right">{formatPercentPlain(weight * 100)}</td>
                        <td className="text-left">
                          <div style={{ height: '8px', width: '200px',
                                        background: 'var(--surface-2)', borderRadius: '2px' }}>
                            <div style={{ height: '100%', width: `${weight * 100 * 2}%`,
                                          maxWidth: '100%', background: 'var(--action)',
                                          borderRadius: '2px' }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {/* A target allocation that does not sum to 100% silently distorts
                  every rebalance suggestion, so it is called out rather than
                  left for the user to add up. */}
              {Math.abs(allocationTotal - 1) > 0.001 && (
                <p className="font-ui text-sm"
                   style={{ color: 'var(--warning)', marginTop: 'var(--space-4)', marginBottom: 0 }}>
                  Target weights sum to {formatPercentPlain(allocationTotal * 100)}, not 100%.
                  Rebalancing will be skewed until this is corrected.
                </p>
              )}
            </>
          )}
        </Panel>

        <Panel title="Trading limits">
          <KpiBand>
            <BandMetric label="Minimum trade" value={formatUsd(data.minimum_trade_usd)} />
          </KpiBand>
        </Panel>

        <Panel title={`Backups (${data.backups.length})`}>
          {data.backups.length === 0 ? (
            <Empty>No database backups found.</Empty>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Name</th>
                    <th className="text-right">Size</th>
                    <th className="text-left">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {data.backups.map((backup) => (
                    <tr key={backup.name}>
                      <td className="text-left">{backup.name}</td>
                      <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                        {humanSize(backup.size_bytes)}
                      </td>
                      <td className="text-left" style={{ color: 'var(--text-tertiary)' }}>
                        {backup.modified.slice(0, 16).replace('T', ' ')}
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
