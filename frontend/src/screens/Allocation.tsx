import { useMemo } from 'react';
import {
  Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { Panel } from '../components/Panel';
import { Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { formatPercentPlain, formatSigned, formatUsd } from '../lib/format';
import type { CockpitResponse, SystemHealthResponse } from '../types';

// A restrained categorical palette - the accent plus muted supporting hues,
// not a rainbow. Slices beyond this wrap around.
const SLICE_COLOURS = [
  'var(--action)', '#6d8bd0', '#4fa9a0', '#c9a15a', '#a97fc4',
  '#c07a7a', '#7aa9c0', '#9ab060', '#b0709a', '#6fb0a0',
];

/** Y-axis domain for the P/L chart that always contains the data and zero.
    Recharts' auto domain demonstrably clipped a -$34.50 bar at -$24, so the
    domain is owned here and unit-tested with real magnitudes.
    Input contract: values are finite numbers; non-finite input is an upstream serialization bug, not handled here. */
export function plDomain(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo;
  if (span === 0) return [Math.min(0, lo - 1), Math.max(0, hi + 1)];
  const pad = span * 0.05;
  return [Math.min(0, lo - pad), Math.max(0, hi + pad)];
}

// Ring slices below this share of total value merge into one "Others (n)"
// slice. The drift table still lists every asset: grouping is ring-only.
export const DUST_THRESHOLD_PCT = 1;

export interface PieSlice { name: string; value: number; }

export function groupDust(slices: PieSlice[], total: number): PieSlice[] {
  const isDust = (s: PieSlice) => total > 0 && (s.value / total) * 100 < DUST_THRESHOLD_PCT;
  const dust = slices.filter(isDust);
  if (dust.length === 0) return slices;
  const kept = slices.filter((s) => !isDust(s));
  const sum = dust.reduce((acc, s) => acc + s.value, 0);
  return [...kept, { name: `Others (${dust.length})`, value: sum }];
}

export function Allocation() {
  const cockpit = useApi<CockpitResponse>('/api/portfolio/cockpit');
  const health = useApi<SystemHealthResponse>('/api/system/health');

  const error = cockpit.error ?? health.error;

  const pie = useMemo(() => {
    const holdings = cockpit.data?.holdings ?? [];
    const total = cockpit.data?.total_value_usd ?? 0;
    const slices = holdings
      .filter((h) => (h.value_usd ?? 0) > 0)
      .map((h) => ({ name: h.symbol, value: h.value_usd as number }))
      .sort((a, b) => b.value - a.value);
    return groupDust(slices, total);
  }, [cockpit.data]);

  const pl = useMemo(() => {
    const holdings = cockpit.data?.holdings ?? [];
    return holdings
      .filter((h) => h.unrealized_pl_usd !== null)
      .map((h) => ({ name: h.symbol, value: h.unrealized_pl_usd as number }))
      .sort((a, b) => b.value - a.value);
  }, [cockpit.data]);

  const drift = useMemo(() => {
    const data = cockpit.data;
    const targets = health.data?.target_allocation ?? {};
    const total = data?.total_value_usd ?? 0;
    if (!data || !total) return [];
    return Object.entries(targets).map(([symbol, weight]) => {
      const holding = data.holdings.find((h) => h.symbol === symbol);
      return {
        name: symbol,
        current: ((holding?.value_usd ?? 0) / total) * 100,
        target: weight * 100,
      };
    }).sort((a, b) => b.target - a.target);
  }, [cockpit.data, health.data]);

  const colourOf = new Map(
    pie.map((s, i) => [s.name, SLICE_COLOURS[i % SLICE_COLOURS.length]]),
  );

  if (error) return <ErrorPanel title="Allocation" message={`Failed to load: ${error}`} />;
  if (!cockpit.data || !health.data) {
    return <Panel title="Allocation"><Empty>Loading…</Empty></Panel>;
  }

  return (
    <>
      <ScreenHeader title="Allocation"
                    subtitle="Portfolio composition, drift from target, and contribution to P/L" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.3fr)',
                                       gap: 'var(--space-3)' }}>
          <Panel title="Current allocation">
            {pie.length === 0 ? (
              <Empty>No priced holdings to chart.</Empty>
            ) : (
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={pie} dataKey="value" nameKey="name" innerRadius={60}
                         outerRadius={110} paddingAngle={1} stroke="var(--surface-1)"
                         strokeWidth={1}>
                      {pie.map((s) => (
                        <Cell key={s.name} fill={colourOf.get(s.name)} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ background: 'var(--surface-2)',
                                      border: '1px solid var(--border-strong)',
                                      borderRadius: 'var(--radius-control)', fontSize: '12px' }}
                      formatter={(v) => [formatUsd(typeof v === 'number' ? v : null), 'Value']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </Panel>

          <Panel title="Current vs target">
            {drift.length === 0 ? (
              <Empty>No target allocation configured.</Empty>
            ) : (
              <div className="table-scroll">
                <table className="data">
                  <thead>
                    <tr>
                      <th className="text-left">Asset</th>
                      <th className="text-right">Current</th>
                      <th className="text-right">Target</th>
                      <th className="text-right">Drift</th>
                    </tr>
                  </thead>
                  <tbody>
                    {drift.map((d) => {
                      const delta = d.current - d.target;
                      return (
                        <tr key={d.name}>
                          <td className="text-left" style={{ fontWeight: 500 }}>
                          <span data-testid="drift-dot" style={{
                            display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                            background: colourOf.get(d.name) ?? 'var(--text-tertiary)',
                            marginRight: 8,
                          }} />
                          {d.name}
                        </td>
                          <td className="text-right">{formatPercentPlain(d.current)}</td>
                          <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                            {formatPercentPlain(d.target)}
                          </td>
                          <td className="text-right"
                              style={{ color: Math.abs(delta) < 1 ? 'var(--text-tertiary)'
                                            : delta > 0 ? 'var(--positive)' : 'var(--negative)' }}>
                            {formatSigned(delta).replace('$', '')}%
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

        <Panel title="Unrealized P/L by asset">
          {pl.length === 0 ? (
            <Empty>No priced holdings to chart.</Empty>
          ) : (
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <BarChart data={pl} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                  <XAxis dataKey="name" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                         stroke="var(--border)" interval={0} angle={-35} textAnchor="end"
                         height={56} />
                  <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                         stroke="var(--border)" width={64}
                         domain={plDomain(pl.map((d) => d.value))}
                         tickFormatter={(v) => `$${Number(v).toFixed(0)}`} />
                  <Tooltip
                    cursor={{ fill: 'var(--surface-2)' }}
                    contentStyle={{ background: 'var(--surface-2)',
                                    border: '1px solid var(--border-strong)',
                                    borderRadius: 'var(--radius-control)', fontSize: '12px' }}
                    labelStyle={{ color: 'var(--text-primary)' }}
                    itemStyle={{ color: 'var(--text-secondary)' }}
                    formatter={(v) => [formatSigned(typeof v === 'number' ? v : null), 'Unrealized']}
                  />
                  <Bar dataKey="value" radius={[2, 2, 0, 0]} minPointSize={3}>
                    {pl.map((d, i) => (
                      <Cell key={i} fill={d.value >= 0 ? 'var(--positive)' : 'var(--negative)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
