import { Panel } from '../components/Panel';
import { Metric } from '../components/Metric';
import { Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { formatQty, formatUsd } from '../lib/format';
import type { WalletBalance, WalletsResponse } from '../types';

function BalanceTable({ rows, emptyText }: { rows: WalletBalance[]; emptyText: string }) {
  if (rows.length === 0) return <Empty>{emptyText}</Empty>;
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th className="text-left">Asset</th>
            <th className="text-right">Quantity</th>
            <th className="text-right">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.symbol}>
              <td className="text-left" style={{ fontWeight: 500 }}>{row.symbol}</td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatQty(row.quantity)}
              </td>
              <td className="text-right"
                  style={{ color: row.value_usd === null ? 'var(--warning)' : undefined }}>
                {/* An unpriced balance shows an em dash, never $0.00. */}
                {formatUsd(row.value_usd)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Wallets() {
  const { data, error } = useApi<WalletsResponse>('/api/wallets');

  if (error) return <ErrorPanel title="Wallets" message={`Failed to load wallets: ${error}`} />;
  if (!data) return <Panel title="Wallets"><Empty>Loading…</Empty></Panel>;

  if (!data.has_data) {
    return (
      <>
        <ScreenHeader title="Wallets" subtitle="Spot & Earn, Futures, Funding" />
        <Panel>
          <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
            No data yet — run a sync to populate wallet balances.
          </p>
        </Panel>
      </>
    );
  }

  return (
    <>
      <ScreenHeader title="Wallets" subtitle="Spot & Earn, Futures, Funding"
                    staleness={data.staleness} />

      <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
        <Panel>
          <div className="grid grid-cols-4" style={{ gap: 'var(--space-5)' }}>
            <Metric label="Spot & Earn" value={formatUsd(data.spot_earn_value_usd)} />
            <Metric label="Futures" value={formatUsd(data.futures_value_usd)} />
            <Metric label="Funding" value={formatUsd(data.funding_value_usd)} />
            <Metric label="Total" value={formatUsd(data.total_value_usd)} />
          </div>
        </Panel>

        <Panel title="Spot & Earn">
          <BalanceTable rows={data.spot_holdings} emptyText="No spot or earn balances." />
        </Panel>
        <Panel title="Futures">
          <BalanceTable rows={data.futures_balances} emptyText="No futures balances." />
        </Panel>
        <Panel title="Funding">
          <BalanceTable rows={data.funding_balances} emptyText="No funding balances." />
        </Panel>
      </div>
    </>
  );
}
