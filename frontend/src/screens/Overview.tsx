import {
  Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { Panel } from '../components/Panel';
import { Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { BandMetric, KpiBand } from '../components/Band';
import { useApi } from '../lib/useApi';
import { formatPercent, formatSigned, formatUsd } from '../lib/format';
import type { OverviewResponse } from '../types';

function shortDate(timestamp: string | null): string {
  if (!timestamp) return '';
  return timestamp.slice(0, 10);
}

export function Overview() {
  const { data, error } = useApi<OverviewResponse>('/api/overview');

  if (error) return <ErrorPanel title="Portfolio overview" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Portfolio overview"><Empty>Loading…</Empty></Panel>;

  if (!data.has_data) {
    return (
      <>
        <ScreenHeader title="Portfolio overview" subtitle="Value and cost basis over time" />
        <Panel>
          <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
            No snapshots recorded yet. Snapshots are written by a sync — run one to
            start building history.
          </p>
        </Panel>
      </>
    );
  }

  const points = data.points
    .filter((p) => p.total_value_usd !== null)
    .map((p) => ({
      date: shortDate(p.timestamp),
      value: p.total_value_usd,
      basis: p.total_cost_basis_usd,
    }));

  const latest = data.points[data.points.length - 1];
  const first = data.points[0];
  // Change across the whole recorded history. Both ends must be known --
  // subtracting from an unknown start would invent a figure.
  const change =
    latest?.total_value_usd !== null && latest?.total_value_usd !== undefined &&
    first?.total_value_usd !== null && first?.total_value_usd !== undefined
      ? latest.total_value_usd - first.total_value_usd
      : null;

  return (
    <>
      <ScreenHeader
        title="Portfolio overview"
        subtitle={`${data.points.length} snapshots recorded`}
        staleness={data.staleness}
      />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel>
          <KpiBand>
            <BandMetric emphasis label="Latest value" value={formatUsd(latest?.total_value_usd)} />
            <BandMetric label="Cost basis" value={formatUsd(latest?.total_cost_basis_usd)} />
            <BandMetric
              label="Unrealized"
              value={`${formatSigned(latest?.unrealized_pl_usd)} (${formatPercent(latest?.unrealized_pl_percent)})`}
              signal={latest?.unrealized_pl_usd}
            />
            <BandMetric
              label="Change since first snapshot"
              value={formatSigned(change)}
              signal={change}
              sub={first ? `from ${shortDate(first.timestamp)}` : undefined}
            />
          </KpiBand>
        </Panel>

        <Panel title="Value vs cost basis">
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <AreaChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="valueFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--action)" stopOpacity={0.28} />
                    <stop offset="100%" stopColor="var(--action)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                       stroke="var(--border)" minTickGap={48} />
                <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                       stroke="var(--border)" width={64}
                       tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
                <Tooltip
                  contentStyle={{
                    background: 'var(--surface-2)', border: '1px solid var(--border-strong)',
                    borderRadius: 'var(--radius-control)', fontSize: '12px',
                  }}
                  labelStyle={{ color: 'var(--text-secondary)' }}
                  formatter={(value, name) => [
                    formatUsd(typeof value === 'number' ? value : null),
                    name === 'value' ? 'Value' : 'Cost basis',
                  ]}
                />
                <Area type="monotone" dataKey="value" stroke="var(--action)" strokeWidth={1.5}
                      fill="url(#valueFill)" dot={false} />
                {/* Cost basis as a bare line: the gap between the two is the
                    unrealized P/L, which is the point of showing them together. */}
                <Line type="monotone" dataKey="basis" stroke="var(--text-tertiary)"
                      strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
    </>
  );
}
