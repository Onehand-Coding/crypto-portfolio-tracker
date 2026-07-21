import { useEffect, useState } from 'react';
import { BandMetric, KpiBand } from '../components/Band';
import { Panel } from '../components/Panel';
import { apiGet } from '../lib/api';
import { formatQty, formatUsd } from '../lib/format';
import type { CapitalFlowResponse } from '../types';

const PROVENANCE_LABEL: Record<string, string> = {
  computed: 'computed',
  usdt_peg_fallback: 'USDT peg fallback',
  failed_lookup: 'failed lookup',
};

export function CapitalFlow() {
  const [data, setData] = useState<CapitalFlowResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<CapitalFlowResponse>('/api/capital/flow')
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  // Checked before the "loading" branch: a fetch failure must always surface
  // as a visible, legible error -- never a blank panel and never a loading
  // state stuck forever.
  if (error) {
    return (
      <Panel title="Capital flow">
        <p className="font-mono text-sm" style={{ color: 'var(--negative)' }}>
          Failed to load capital flow: {error}
        </p>
      </Panel>
    );
  }
  if (!data) {
    return (
      <Panel title="Capital flow">
        <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
          Loading…
        </p>
      </Panel>
    );
  }

  const unpricedInflows = data.rows.filter(
    (row) => row.direction === 'in' && row.is_suspect,
  ).length;

  return (
    <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
      <Panel title="Capital flow">
        <KpiBand>
          <BandMetric emphasis label="Net invested" value={formatUsd(data.net_invested_usd)} />
          <BandMetric
            label="Total in"
            value={formatUsd(data.total_in_usd)}
            sub={unpricedInflows > 0
              ? `excludes ${unpricedInflows} unpriced row${unpricedInflows === 1 ? '' : 's'}`
              : undefined}
          />
          <BandMetric label="Total out" value={formatUsd(data.total_out_usd)} />
        </KpiBand>
        {data.suspect_count > 0 && (
          <p className="mt-3 font-ui text-sm" style={{ color: 'var(--warning)' }}>
            {data.suspect_count} row{data.suspect_count === 1 ? '' : 's'} could not be
            priced from a real exchange rate. Net invested may understate actual inflow.
          </p>
        )}
      </Panel>

      <Panel title="Transactions">
        {data.rows.length === 0 ? (
          <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
            No capital flow recorded yet.
          </p>
        ) : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th className="text-left">Source</th>
                  <th className="text-left">Dir</th>
                  <th className="text-right">Quantity</th>
                  <th className="text-right">Rate</th>
                  <th className="text-right">Value</th>
                  <th className="text-left">Provenance</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, index) => (
                  <tr key={index}>
                    <td className="text-left">{row.source}</td>
                    <td className="text-left"
                        style={{ color: row.direction === 'in' ? 'var(--positive)'
                                                               : 'var(--negative)' }}>
                      {row.direction === 'in' ? '+ in' : '- out'}
                    </td>
                    <td className="text-right">{formatQty(row.quantity)}</td>
                    <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                      {formatQty(row.price_usd)}
                    </td>
                    <td className="text-right">{formatUsd(row.value_usd)}</td>
                    <td className="text-left"
                        style={{ color: row.is_suspect ? 'var(--warning)'
                                                       : 'var(--text-tertiary)' }}>
                      {PROVENANCE_LABEL[row.provenance]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
