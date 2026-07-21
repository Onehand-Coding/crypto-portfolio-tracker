import { useParams } from 'react-router-dom';
import { Panel } from '../components/Panel';
import { Metric } from '../components/Metric';
import { Badge, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { formatPercent, formatPercentPlain, formatQty, formatSigned, formatUsd } from '../lib/format';
import type { AssetDetailResponse } from '../types';

const BUY_TYPES = ['BUY', 'DEPOSIT', 'P2P_BUY', 'EARN_REWARD', 'DIVIDEND', 'TRANSFER_IN'];

export function AssetDetail() {
  const { symbol = '' } = useParams();
  const { data, error } = useApi<AssetDetailResponse>(
    `/api/assets/${encodeURIComponent(symbol)}`, [symbol],
  );

  if (error) return <ErrorPanel title={symbol} message={`Failed to load ${symbol}: ${error}`} />;
  if (!data) return <Panel title={symbol}><Empty>Loading…</Empty></Panel>;

  if (!data.found) {
    return (
      <>
        <ScreenHeader title={data.symbol} />
        <Panel>
          <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
            No holding or transaction history for {data.symbol}.
          </p>
        </Panel>
      </>
    );
  }

  return (
    <>
      <ScreenHeader
        title={data.symbol}
        subtitle={data.is_core
          ? `Core holding · target ${formatPercentPlain(data.target_allocation_pct)}`
          : 'Non-core holding'}
        staleness={data.staleness}
      />

      <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
        <Panel>
          {data.price_unavailable && (
            <p className="font-ui text-sm"
               style={{ color: 'var(--warning)', margin: '0 0 var(--space-4) 0' }}>
              Price lookup failed for {data.symbol}. Its value and unrealized P/L are
              unknown — not zero.
            </p>
          )}
          <div className="grid grid-cols-4" style={{ gap: 'var(--space-5)' }}>
            <Metric label="Quantity" value={formatQty(data.total_quantity)} />
            <Metric label="Price" value={formatUsd(data.current_price)} />
            <Metric label="Value" value={formatUsd(data.value_usd)} />
            <Metric
              label="Unrealized"
              value={`${formatSigned(data.unrealized_pl_usd)} (${formatPercent(data.unrealized_pl_percent)})`}
              signal={data.unrealized_pl_usd}
            />
          </div>
          <div className="grid grid-cols-4"
               style={{ gap: 'var(--space-5)', marginTop: 'var(--space-5)' }}>
            <Metric label="Avg cost basis" value={formatUsd(data.average_cost_basis)} />
            <Metric label="Cost basis total" value={formatUsd(data.cost_basis_total)} />
          </div>
        </Panel>

        <Panel title={`Transactions (${data.transactions.length})`}>
          {data.transactions.length === 0 ? (
            <Empty>No transactions recorded for {data.symbol}.</Empty>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Date</th>
                    <th className="text-left">Type</th>
                    <th className="text-right">Quantity</th>
                    <th className="text-right">Price</th>
                    <th className="text-right">Value</th>
                    <th className="text-left">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {data.transactions.map((tx, index) => (
                    <tr key={index}>
                      <td className="text-left" style={{ color: 'var(--text-secondary)' }}>
                        {tx.timestamp ? tx.timestamp.slice(0, 16) : '—'}
                      </td>
                      <td className="text-left">
                        <Badge
                          text={tx.type}
                          tone={BUY_TYPES.includes(tx.type) ? 'positive' : 'negative'}
                        />
                      </td>
                      <td className="text-right">{formatQty(tx.quantity)}</td>
                      <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                        {formatUsd(tx.price_usd)}
                      </td>
                      <td className="text-right">{formatUsd(tx.value_usd)}</td>
                      <td className="text-left" style={{ color: 'var(--text-tertiary)' }}>
                        {tx.source ?? '—'}
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
