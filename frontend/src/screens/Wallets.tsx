import { useState } from 'react';
import { Panel } from '../components/Panel';
import { BandMetric, KpiBand } from '../components/Band';
import { Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { ExecutePanel } from '../components/ExecutePanel';
import { TradingStatusBanner } from '../components/TradingStatusBanner';
import { useApi } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatQty, formatUsd } from '../lib/format';
import type {
  ExecutionStatus, TradeExecuteResponse, WalletBalance, WalletsResponse,
} from '../types';

const WALLETS = ['SPOT', 'FUNDING', 'FUTURES'];
const fieldLabel = {
  color: 'var(--text-tertiary)', fontSize: '11px',
  letterSpacing: '0.08em', textTransform: 'uppercase' as const,
};
const control = {
  background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
  borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
  padding: 'var(--space-2) var(--space-3)', fontSize: '14px',
} as const;

/** Move an asset between Spot / Funding / Futures. */
function TransferWidget({ status }: { status: ExecutionStatus }) {
  const [asset, setAsset] = useState('USDT');
  const [amount, setAmount] = useState('10');
  const [from, setFrom] = useState('SPOT');
  const [to, setTo] = useState('FUNDING');
  const amt = Number(amount);
  const valid = Number.isFinite(amt) && amt > 0 && from !== to;
  const net = status.testnet ? 'testnet' : 'mainnet';
  return (
    <ExecutePanel
      title={status.is_live ? 'Transfer' : 'Transfer (simulated)'}
      disabled={!valid}
      description={valid
        ? `This ${status.is_live ? 'moves' : 'simulates moving'} ${amt} ${asset.toUpperCase()} from ${from} to ${to} on the Binance ${net}${status.is_live ? '' : ' (live trading is off — nothing is sent)'}.`
        : 'Choose a positive amount and two different wallets.'}
      execute={() => apiPost<TradeExecuteResponse>('/api/execute/transfer', {
        confirm: true, asset: asset.toUpperCase(), amount: amt,
        from_wallet: from, to_wallet: to,
      })}
    >
      <div className="flex flex-wrap items-end" style={{ gap: 'var(--space-4)' }}>
        <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
          <span className="font-ui" style={fieldLabel}>Asset</span>
          <input value={asset} onChange={(e) => setAsset(e.target.value)}
                 className="font-mono" style={{ ...control, width: '100px', textTransform: 'uppercase' }} />
        </label>
        <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
          <span className="font-ui" style={fieldLabel}>Amount</span>
          <input value={amount} onChange={(e) => setAmount(e.target.value)} inputMode="decimal"
                 className="font-mono" style={{ ...control, width: '120px' }} />
        </label>
        <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
          <span className="font-ui" style={fieldLabel}>From</span>
          <select value={from} onChange={(e) => setFrom(e.target.value)}
                  className="font-mono" style={{ ...control, minWidth: '110px' }}>
            {WALLETS.map((w) => <option key={w} value={w}>{w}</option>)}
          </select>
        </label>
        <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
          <span className="font-ui" style={fieldLabel}>To</span>
          <select value={to} onChange={(e) => setTo(e.target.value)}
                  className="font-mono" style={{ ...control, minWidth: '110px' }}>
            {WALLETS.map((w) => <option key={w} value={w}>{w}</option>)}
          </select>
        </label>
      </div>
    </ExecutePanel>
  );
}

/** Redeem an asset from Simple Earn back to Spot. */
function RedeemWidget({ status }: { status: ExecutionStatus }) {
  const [asset, setAsset] = useState('USDT');
  const [amount, setAmount] = useState('10');
  const amt = Number(amount);
  const valid = Number.isFinite(amt) && amt > 0;
  const net = status.testnet ? 'testnet' : 'mainnet';
  return (
    <ExecutePanel
      title={status.is_live ? 'Redeem from Earn' : 'Redeem from Earn (simulated)'}
      disabled={!valid}
      description={valid
        ? `This ${status.is_live ? 'redeems' : 'simulates redeeming'} ${amt} ${asset.toUpperCase()} from Simple Earn back to Spot on the Binance ${net}${status.is_live ? '' : ' (live trading is off — nothing is sent)'}.`
        : 'Choose a positive amount.'}
      execute={() => apiPost<TradeExecuteResponse>('/api/execute/redeem', {
        confirm: true, asset: asset.toUpperCase(), amount: amt,
      })}
    >
      <div className="flex flex-wrap items-end" style={{ gap: 'var(--space-4)' }}>
        <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
          <span className="font-ui" style={fieldLabel}>Asset</span>
          <input value={asset} onChange={(e) => setAsset(e.target.value)}
                 className="font-mono" style={{ ...control, width: '100px', textTransform: 'uppercase' }} />
        </label>
        <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
          <span className="font-ui" style={fieldLabel}>Amount</span>
          <input value={amount} onChange={(e) => setAmount(e.target.value)} inputMode="decimal"
                 className="font-mono" style={{ ...control, width: '120px' }} />
        </label>
      </div>
    </ExecutePanel>
  );
}

/** Each wallet's share of total value, one decimal. No total (or an
    unpriced balance) renders an em dash, never a confident 0.0%. */
function shareOf(value: number | null, total: number | null): string {
  if (value === null || value === undefined || total === null
      || total === undefined || !(total > 0)) return '—';
  return `${((value / total) * 100).toFixed(1)}%`;
}

function BalanceTable({ rows, total, emptyText }: {
  rows: WalletBalance[]; total: number | null; emptyText: string;
}) {
  if (rows.length === 0) return <Empty>{emptyText}</Empty>;
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th className="text-left">Asset</th>
            <th className="text-right">Quantity</th>
            <th className="text-right">Value</th>
            <th className="text-right">Share</th>
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
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {shareOf(row.value_usd, total)}
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
  const status = useApi<ExecutionStatus>('/api/execute/status');

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

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel>
          <KpiBand>
            <BandMetric emphasis label="Total" value={formatUsd(data.total_value_usd)} />
            <BandMetric label="Spot & Earn" value={formatUsd(data.spot_earn_value_usd)} />
            <BandMetric label="Futures" value={formatUsd(data.futures_value_usd)} />
            <BandMetric label="Funding" value={formatUsd(data.funding_value_usd)} />
          </KpiBand>
        </Panel>

        <Panel title="Spot & Earn">
          <BalanceTable rows={data.spot_holdings} total={data.total_value_usd}
                        emptyText="No spot or earn balances." />
        </Panel>
        <Panel title="Futures">
          <BalanceTable rows={data.futures_balances} total={data.total_value_usd}
                        emptyText="No futures balances." />
        </Panel>
        <Panel title="Funding">
          <BalanceTable rows={data.funding_balances} total={data.total_value_usd}
                        emptyText="No funding balances." />
        </Panel>

        {status.data && (
          <>
            <TradingStatusBanner status={status.data} />
            <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
                                           gap: 'var(--space-3)' }}>
              <TransferWidget status={status.data} />
              <RedeemWidget status={status.data} />
            </div>
          </>
        )}
      </div>
    </>
  );
}
