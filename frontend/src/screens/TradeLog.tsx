import { useMemo, useState } from 'react';
import { Panel } from '../components/Panel';
import { Badge, Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatQty, formatUsd, NULL_GLYPH} from '../lib/format';
import type { GenerateExportResponse, TransactionRow, TransactionsResponse } from '../types';

const BUY_TYPES = new Set(['BUY', 'DEPOSIT', 'TRANSFER_IN', 'EARN_REWARD', 'DIVIDEND']);

function typeTone(type: string) {
  if (BUY_TYPES.has(type)) return 'positive' as const;
  if (type.startsWith('SELL') || type.startsWith('WITHDRAW') || type === 'TRANSFER_OUT') {
    return 'negative' as const;
  }
  return 'neutral' as const;
}

/** Build a CSV from the filtered rows and trigger a download, no server round-trip. */
function downloadCsv(rows: TransactionRow[]) {
  const headers = ['Date', 'Asset', 'Type', 'Quantity', 'Price (USD)',
                   'Value (USD)', 'Fee (USD)', 'Source', 'Notes'];
  const escape = (v: unknown) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [headers.join(',')];
  for (const r of rows) {
    lines.push([r.timestamp ?? '', r.symbol, r.type, r.quantity ?? '', r.price_usd ?? '',
                r.value_usd ?? '', r.fee_usd ?? '', r.source ?? '', r.notes ?? '']
      .map(escape).join(','));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `trade_log_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function Select({ value, onChange, options }: {
  value: string; onChange: (v: string) => void; options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="font-mono"
      style={{
        background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
        borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
        padding: 'var(--space-2) var(--space-3)', fontSize: '13px', minWidth: '130px',
      }}
    >
      {options.map((o) => <option key={o} value={o}>{o === '' ? 'All' : o}</option>)}
    </select>
  );
}

export function TradeLog() {
  const { data, error } = useApi<TransactionsResponse>('/api/transactions');
  const [typeFilter, setTypeFilter] = useState('');
  const [assetFilter, setAssetFilter] = useState('');
  const [excelBusy, setExcelBusy] = useState(false);
  const [excelFile, setExcelFile] = useState<string | null>(null);
  const [excelError, setExcelError] = useState<string | null>(null);

  /** Server-side Excel of the full transaction history (same rows as the CSV). */
  async function exportExcel() {
    setExcelBusy(true);
    setExcelError(null);
    setExcelFile(null);
    try {
      const res = await apiPost<GenerateExportResponse>('/api/reports/generate', {
        data_type: 'transactions', format: 'excel',
      });
      setExcelFile(res.name);
    } catch (e) {
      setExcelError(`Excel export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setExcelBusy(false);
    }
  }

  const { types, assets } = useMemo(() => {
    const t = new Set<string>();
    const a = new Set<string>();
    for (const r of data?.rows ?? []) { t.add(r.type); a.add(r.symbol); }
    return { types: ['', ...[...t].sort()], assets: ['', ...[...a].sort()] };
  }, [data]);

  const filtered = useMemo(() => (data?.rows ?? []).filter((r) =>
    (typeFilter === '' || r.type === typeFilter) &&
    (assetFilter === '' || r.symbol === assetFilter),
  ), [data, typeFilter, assetFilter]);

  if (error) return <ErrorPanel title="Trade log" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Trade log"><Empty>Loading…</Empty></Panel>;

  return (
    <>
      <ScreenHeader
        title="Trade log"
        subtitle={`${data.count} transactions across every asset`}
      />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel>
          <div className="flex flex-wrap items-end justify-between" style={{ gap: 'var(--space-4)' }}>
            <div className="flex flex-wrap items-end" style={{ gap: 'var(--space-4)' }}>
              <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
                <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                                   letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  Type
                </span>
                <Select value={typeFilter} onChange={setTypeFilter} options={types} />
              </label>
              <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
                <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                                   letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  Asset
                </span>
                <Select value={assetFilter} onChange={setAssetFilter} options={assets} />
              </label>
              <span className="font-mono" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                                   paddingBottom: 'var(--space-2)' }}>
                {filtered.length} shown
              </span>
            </div>
            <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-3)' }}>
              <Button variant="secondary" onClick={() => downloadCsv(filtered)}
                      disabled={filtered.length === 0}>
                Export CSV
              </Button>
              <Button variant="secondary" onClick={exportExcel} disabled={excelBusy}>
                {excelBusy ? 'Exporting…' : 'Export Excel'}
              </Button>
              {excelFile && (
                <a href={`/api/reports/download?name=${encodeURIComponent(excelFile)}`}
                   className="font-ui transition-colors"
                   style={{ color: 'var(--text-secondary)',
                            border: '1px solid var(--border-strong)',
                            borderRadius: 'var(--radius-control)',
                            padding: '2px var(--space-3)', fontSize: '12px',
                            textDecoration: 'none' }}>
                  Download {excelFile}
                </a>
              )}
              {excelError && (
                <span className="font-ui" style={{ fontSize: '13px', color: 'var(--negative)' }}>
                  {excelError}
                </span>
              )}
            </div>
          </div>
        </Panel>

        <Panel title="Transactions">
          {filtered.length === 0 ? (
            <Empty>
              {data.count === 0
                ? 'No transactions recorded yet. Run a sync to import them.'
                : 'No transactions match the current filters.'}
            </Empty>
          ) : (
            <div className="table-scroll" style={{ maxHeight: '620px', overflowY: 'auto' }}>
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Date</th>
                    <th className="text-left">Asset</th>
                    <th className="text-left">Type</th>
                    <th className="text-right">Quantity</th>
                    <th className="text-right">Price</th>
                    <th className="text-right">Value</th>
                    <th className="text-right">Fee</th>
                    <th className="text-left">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((r, i) => (
                    <tr key={i}>
                      <td className="text-left" style={{ color: 'var(--text-secondary)' }}>
                        {r.timestamp ? r.timestamp.slice(0, 16).replace('T', ' ') : NULL_GLYPH}
                      </td>
                      <td className="text-left" style={{ fontWeight: 500 }}>{r.symbol}</td>
                      <td className="text-left"><Badge text={r.type} tone={typeTone(r.type)} /></td>
                      <td className="text-right">{formatQty(r.quantity)}</td>
                      <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                        {formatUsd(r.price_usd)}
                      </td>
                      <td className="text-right">{formatUsd(r.value_usd)}</td>
                      <td className="text-right" style={{ color: 'var(--text-tertiary)' }}>
                        {formatUsd(r.fee_usd)}
                      </td>
                      <td className="text-left" style={{ color: 'var(--text-tertiary)' }}>
                        {r.source ?? NULL_GLYPH}
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
