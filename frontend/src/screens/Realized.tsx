import { useState } from 'react';
import { Panel } from '../components/Panel';
import { BandMetric, KpiBand } from '../components/Band';
import { Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatQty, formatSigned, formatUsd } from '../lib/format';
import type { GenerateExportResponse, RealizedResponse } from '../types';

/** Realized P/L in USD, coloured by direction. Zero and unknown are distinct. */
function Gain({ value }: { value: number | null }) {
  const colour = value === null ? undefined
    : value > 0 ? 'var(--positive)' : value < 0 ? 'var(--negative)' : 'var(--text-primary)';
  return <span style={{ color: colour }}>{formatSigned(value)}</span>;
}

export function Realized() {
  const { data, error } = useApi<RealizedResponse>('/api/realized');
  const [exportBusy, setExportBusy] = useState<string | null>(null);
  const [exportFile, setExportFile] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  /** Server-side export of the FIFO realized-gains table. */
  async function exportRealized(format: string) {
    setExportBusy(format);
    setExportError(null);
    setExportFile(null);
    try {
      const res = await apiPost<GenerateExportResponse>('/api/reports/realized', {
        format,
      });
      setExportFile(res.name);
    } catch (e) {
      setExportError(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setExportBusy(null);
    }
  }

  if (error) return <ErrorPanel title="Realized P/L" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Realized P/L"><Empty>Loading…</Empty></Panel>;

  return (
    <>
      <ScreenHeader
        title="Realized P/L"
        subtitle="FIFO realized gains — the closed, taxable half of the accounting"
        staleness={data.staleness}
      />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel>
          <KpiBand>
            <BandMetric emphasis label="Total realized P/L"
                        value={formatSigned(data.total_gain_usd)}
                        signal={data.total_gain_usd} />
            <BandMetric label="Total proceeds" value={formatUsd(data.total_proceeds_usd)} />
            <BandMetric label="Total cost basis" value={formatUsd(data.total_cost_basis_usd)} />
          </KpiBand>
          {/* The cockpit's P/L is unrealized (open positions). This is realized
              (closed lots). Conflating them is the classic accounting error, so
              the distinction is stated rather than left to the label. */}
          <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                          marginTop: 'var(--space-3)', marginBottom: 0 }}>
            Gains locked in by past sells and withdrawals, priced against their FIFO cost
            lots. Distinct from the unrealized P/L on open positions shown in the cockpit.
          </p>
          <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-3)',
                                                                 marginTop: 'var(--space-3)' }}>
            <Button variant="secondary" onClick={() => exportRealized('excel')}
                    disabled={exportBusy !== null}>
              {exportBusy === 'excel' ? 'Exporting…' : 'Export Excel'}
            </Button>
            <Button variant="secondary" onClick={() => exportRealized('csv')}
                    disabled={exportBusy !== null}>
              {exportBusy === 'csv' ? 'Exporting…' : 'Export CSV'}
            </Button>
            {exportFile && (
              <a href={`/api/reports/download?name=${encodeURIComponent(exportFile)}`}
                 className="font-ui transition-colors"
                 style={{ color: 'var(--text-secondary)',
                          border: '1px solid var(--border-strong)',
                          borderRadius: 'var(--radius-control)',
                          padding: '2px var(--space-3)', fontSize: '12px',
                          textDecoration: 'none' }}>
                Download {exportFile}
              </a>
            )}
            {exportError && (
              <span className="font-ui" style={{ fontSize: '13px', color: 'var(--negative)' }}>
                {exportError}
              </span>
            )}
          </div>
        </Panel>

        {!data.has_data ? (
          <Panel title="Realized gains">
            <Empty>No transactions recorded yet. Run a sync to build history.</Empty>
          </Panel>
        ) : data.rows.length === 0 ? (
          <Panel title="Realized gains">
            <Empty>
              No taxable events yet — nothing has been sold or withdrawn, so no gain has
              been realized.
            </Empty>
          </Panel>
        ) : (
          <>
            <Panel title="By asset">
              <div className="table-scroll">
                <table className="data">
                  <thead>
                    <tr>
                      <th className="text-left">Asset</th>
                      <th className="text-right">Realized P/L</th>
                      <th className="text-right">Proceeds</th>
                      <th className="text-right">Cost basis</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.by_asset.map((s) => (
                      <tr key={s.symbol}>
                        <td className="text-left" style={{ fontWeight: 500 }}>{s.symbol}</td>
                        <td className="text-right"><Gain value={s.total_gain_usd} /></td>
                        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                          {formatUsd(s.total_proceeds_usd)}
                        </td>
                        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                          {formatUsd(s.total_cost_basis_usd)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>

            <Panel title={`Taxable events (${data.rows.length})`}>
              <div className="table-scroll" style={{ maxHeight: '520px', overflowY: 'auto' }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th className="text-left">Date</th>
                      <th className="text-right">Year</th>
                      <th className="text-left">Asset</th>
                      <th className="text-right">Quantity</th>
                      <th className="text-right">Proceeds</th>
                      <th className="text-right">Cost basis</th>
                      <th className="text-right">Gain/Loss</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((r, i) => (
                      <tr key={i}>
                        <td className="text-left" style={{ color: 'var(--text-secondary)' }}>
                          {r.date ? r.date.slice(0, 16).replace('T', ' ') : '—'}
                        </td>
                        <td className="text-right" style={{ color: 'var(--text-tertiary)' }}>
                          {r.year ?? '—'}
                        </td>
                        <td className="text-left" style={{ fontWeight: 500 }}>{r.symbol}</td>
                        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                          {formatQty(r.quantity)}
                        </td>
                        <td className="text-right">{formatUsd(r.proceeds_usd)}</td>
                        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                          {formatUsd(r.cost_basis_usd)}
                        </td>
                        <td className="text-right"><Gain value={r.gain_usd} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          </>
        )}
      </div>
    </>
  );
}
