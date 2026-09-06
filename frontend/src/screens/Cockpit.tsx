import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { BandMetric } from '../components/Band';
import { HoldingsTable } from '../components/HoldingsTable';
import { Panel } from '../components/Panel';
import { Empty, ErrorPanel } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { formatPercent, formatPercentPlain, formatSigned, formatUsd } from '../lib/format';
import type {
  AccountingBasis, CockpitResponse, OverviewResponse, ProfitResponse, SystemHealthResponse,
} from '../types';

const RANGES = [
  { id: '1m', label: '1M', days: 30 },
  { id: '3m', label: '3M', days: 90 },
  { id: '1y', label: '1Y', days: 365 },
  { id: 'all', label: 'ALL', days: Infinity },
];

function basisValue(basis: AccountingBasis): string {
  return `${formatSigned(basis.pl_usd)} (${formatPercent(basis.pl_percent)})`;
}

/** One row of the allocation drift rail: current fill against a target marker. */
function DriftRow({
  symbol, currentPct, targetPct,
}: { symbol: string; currentPct: number; targetPct: number }) {
  const drift = currentPct - targetPct;
  const scale = Math.max(currentPct, targetPct, 1) * 1.3;
  return (
    <div className="flex flex-col" style={{ gap: '5px' }}>
      <div className="flex items-baseline justify-between" style={{ gap: 'var(--space-2)' }}>
        <span className="font-mono" style={{ fontSize: '12px', fontWeight: 500 }}>{symbol}</span>
        <span className="font-mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
          {formatPercentPlain(currentPct)}
          <span style={{ color: 'var(--text-tertiary)' }}> / {formatPercentPlain(targetPct)}</span>
          <span style={{ marginLeft: '6px',
                         color: Math.abs(drift) < 1 ? 'var(--text-tertiary)'
                              : drift > 0 ? 'var(--positive)' : 'var(--negative)' }}>
            {formatPercent(drift)}
          </span>
        </span>
      </div>
      <div style={{ position: 'relative', height: '6px', background: 'var(--surface-2)',
                    borderRadius: '2px' }}>
        <div style={{ position: 'absolute', inset: '0 auto 0 0',
                      width: `${Math.min((currentPct / scale) * 100, 100)}%`,
                      background: 'var(--action)', borderRadius: '2px' }} />
        <div style={{ position: 'absolute', top: '-2px', bottom: '-2px',
                      left: `${Math.min((targetPct / scale) * 100, 100)}%`,
                      width: '2px', background: 'var(--text-primary)' }} />
      </div>
    </div>
  );
}

interface Alert {
  kind: string;
  tone: 'warning' | 'negative' | 'action';
  title: string;
  body: string;
  action?: { to: string; label: string };
}

/** An alert card: a stated problem and the screen that resolves it. */
function AlertCard({ alert }: { alert: Alert }) {
  const colour = `var(--${alert.tone === 'action' ? 'action' : alert.tone})`;
  return (
    <div style={{ background: 'var(--surface-2)', border: '1px solid var(--border)',
                  borderLeft: `2px solid ${colour}`, borderRadius: 'var(--radius-control)',
                  padding: 'var(--space-3)' }}>
      <div className="font-ui" style={{ color: colour, fontSize: '10px', fontWeight: 700,
                                        letterSpacing: '0.06em', textTransform: 'uppercase',
                                        marginBottom: '5px' }}>
        {alert.kind}
      </div>
      <div className="font-ui" style={{ fontSize: '13px', marginBottom: '3px' }}>{alert.title}</div>
      <div className="font-ui" style={{ color: 'var(--text-secondary)', fontSize: '12px',
                                        lineHeight: 1.45 }}>
        {alert.body}
      </div>
      {alert.action && (
        <Link
          to={alert.action.to}
          className="font-ui"
          style={{ display: 'inline-block', marginTop: 'var(--space-3)', fontSize: '11px',
                   fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase',
                   color: 'var(--text-secondary)', border: '1px solid var(--border-strong)',
                   borderRadius: 'var(--radius-control)', padding: '4px var(--space-3)' }}
        >
          {alert.action.label}
        </Link>
      )}
    </div>
  );
}

function PerformancePanel() {
  const overview = useApi<OverviewResponse>('/api/overview');
  const [range, setRange] = useState('3m');
  const series = useMemo(() => {
    const points = overview.data?.points ?? [];
    const days = RANGES.find((r) => r.id === range)?.days ?? Infinity;
    const cutoff = days === Infinity ? 0 : Date.now() - days * 86_400_000;
    return points
      .filter((p) => p.timestamp !== null && p.total_value_usd !== null)
      .filter((p) => new Date(p.timestamp!).getTime() >= cutoff)
      .map((p) => ({
        t: new Date(p.timestamp!).getTime(),
        value: p.total_value_usd as number,
        basis: p.total_cost_basis_usd,
      }));
  }, [overview.data, range]);
  const historyChange = series.length >= 2
    ? series.at(-1)!.value - series[0].value
    : null;

  return (
    <section className="flex flex-col"
             style={{ background: 'var(--surface-1)', border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-panel)', padding: 'var(--space-4)' }}>
      <div className="flex shrink-0 items-center justify-between" style={{ marginBottom: 'var(--space-3)' }}>
        <h2 className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px', fontWeight: 500,
                                         letterSpacing: '0.08em', textTransform: 'uppercase', margin: 0 }}>
          Performance
        </h2>
        <div className="flex" style={{ gap: '2px' }}>
          {RANGES.map((r) => (
            <button key={r.id} onClick={() => setRange(r.id)} className="font-mono transition-colors"
                    style={{ background: range === r.id ? 'var(--surface-2)' : 'transparent',
                             color: range === r.id ? 'var(--text-primary)' : 'var(--text-tertiary)',
                             border: '1px solid transparent', borderRadius: 'var(--radius-control)',
                             padding: '2px var(--space-2)', fontSize: '10px', cursor: 'pointer' }}>
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {historyChange !== null && (
        <p className="font-mono" style={{ color: 'var(--text-secondary)', fontSize: '11px', margin: '0 0 var(--space-3)' }}>
          Change since first snapshot: {formatSigned(historyChange)} from {new Date(series[0].t).toISOString().slice(0, 10)}
        </p>
      )}

      {series.length < 2 ? (
        <div className="flex flex-1 items-center" style={{ minHeight: '208px' }}>
          {overview.error ? (
            <p className="font-mono" style={{ color: 'var(--negative)', fontSize: '12px' }}>
              Could not load history: {overview.error}
            </p>
          ) : (
            <Empty>{overview.data ? 'Not enough snapshots in this range to draw a line.' : 'Loading history…'}</Empty>
          )}
        </div>
      ) : (
        <div className="flex-1" style={{ minHeight: '208px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={series} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <defs><linearGradient id="cockpitFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--action)" stopOpacity={0.28} />
                <stop offset="100%" stopColor="var(--action)" stopOpacity={0} />
              </linearGradient></defs>
              <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="t" type="number" scale="time" domain={['dataMin', 'dataMax']}
                     tickFormatter={(t) => new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                     stroke="var(--text-tertiary)" tickLine={false} axisLine={false}
                     tick={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }} minTickGap={48} />
              <YAxis stroke="var(--text-tertiary)" tickLine={false} axisLine={false} width={54}
                     tick={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                     tickFormatter={(v) => `$${Math.round(v)}`} domain={['dataMin', 'dataMax']} />
              <Tooltip contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border-strong)',
                                       borderRadius: 'var(--radius-control)', fontSize: '12px', fontFamily: 'JetBrains Mono, monospace' }}
                       labelStyle={{ color: 'var(--text-tertiary)' }} labelFormatter={(t) => new Date(t as number).toLocaleDateString()}
                       formatter={(v, name) => [formatUsd(typeof v === 'number' ? v : null), name === 'FIFO cost basis at snapshot' ? name : 'Value']} />
              <Area type="monotone" dataKey="value" stroke="var(--action)" strokeWidth={1.5} fill="url(#cockpitFill)" dot={false} />
              <Line type="monotone" dataKey="basis" name="FIFO cost basis at snapshot" stroke="var(--text-secondary)"
                    strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

export function Cockpit() {
  const cockpit = useApi<CockpitResponse>('/api/portfolio/cockpit');
  const health = useApi<SystemHealthResponse>('/api/system/health');
  // Fail-soft by construction: a rejected profit fetch leaves data null and
  // this section simply contributes no card, so the Dashboard never depends
  // on the analysis pipeline being up.
  const profit = useApi<ProfitResponse>('/api/strategy/profit');

  const data = cockpit.data;

  // Current vs target allocation, computed here rather than fetched: both
  // halves are already on screen, and a network call for arithmetic would
  // break the rule that reads never touch the wire.
  const drift = useMemo(() => {
    if (!data || !health.data) return [];
    // A health response without a target allocation is a valid state, not a
    // crash: the drift rail simply has nothing to compare against.
    const targets = health.data.target_allocation ?? {};
    const total = data.total_value_usd;
    if (!total) return [];
    return Object.entries(targets)
      .map(([symbol, weight]) => {
        const holding = data.holdings.find((h) => h.symbol === symbol);
        return {
          symbol,
          currentPct: ((holding?.value_usd ?? 0) / total) * 100,
          targetPct: weight * 100,
        };
      })
      .sort((a, b) => b.targetPct - a.targetPct);
  }, [data, health.data]);

  const alerts = useMemo<Alert[]>(() => {
    if (!data) return [];
    const out: Alert[] = [];

    if (data.unpriced_count > 0) {
      out.push({
        // Worded as the action to take, not as a restatement of the caveat
        // already sitting next to the total it qualifies.
        kind: 'Data quality', tone: 'warning',
        title: `Prices missing for ${data.unpriced_count} holding${data.unpriced_count === 1 ? '' : 's'}`,
        body: 'A fresh sync usually resolves this. Until it does, the total above is understated.',
        action: { to: '/sync', label: 'Re-sync' },
      });
    }
    if (data.staleness.is_stale) {
      out.push({
        kind: 'Staleness', tone: 'warning',
        title: 'Portfolio data is stale',
        body: 'Prices and balances are older than the freshness threshold.',
        action: { to: '/sync', label: 'Synchronize' },
      });
    }
    // Worst drift only: a rail of twelve identical warnings is noise, and the
    // full picture is already in the drift panel above.
    const worst = [...drift].sort(
      (a, b) => Math.abs(b.currentPct - b.targetPct) - Math.abs(a.currentPct - a.targetPct),
    )[0];
    if (worst && Math.abs(worst.currentPct - worst.targetPct) >= 5) {
      const over = worst.currentPct > worst.targetPct;
      out.push({
        kind: over ? 'Over-exposure' : 'Opportunity', tone: over ? 'negative' : 'action',
        title: `${worst.symbol} is ${formatPercentPlain(Math.abs(worst.currentPct - worst.targetPct))} ${over ? 'above' : 'below'} target`,
        body: `Holding ${formatPercentPlain(worst.currentPct)} against a ${formatPercentPlain(worst.targetPct)} target allocation.`,
        action: { to: '/rebalance', label: 'Review rebalance' },
      });
    }
    // Profit-taking only: the core returns opportunities solely when every
    // rebalance signal is HOLD, so this card and the drift card above are
    // near-mutually exclusive by construction — no dedup rule needed. The
    // freshness gate matters: a trim nudge from a stale run is a real-money
    // action on dead signal, so absent/running/stale analysis stays silent.
    // The backend sorts highest score first; 75 is the screen's own
    // "positive" boundary for "high enough to consider trimming".
    const analysis = profit.data;
    const fresh = analysis !== null && analysis.has_data && !analysis.is_running
      && !analysis.staleness.is_stale;
    const top = fresh ? (analysis.opportunities ?? [])[0] : undefined;
    const score = top?.opportunity_score;
    if (top && score !== null && score !== undefined && score >= 75) {
      out.push({
        kind: 'Profit-taking', tone: 'action',
        title: `${top.symbol} scored ${score.toFixed(0)} — trim candidate`,
        body: `Up ${formatSigned(top.unrealized_gain_usd)} (${formatPercent(top.unrealized_gain_pct)}) at ${formatUsd(top.current_price)}.`,
        action: { to: '/profit', label: 'Review profit-taking' },
      });
    }
    return out;
  }, [data, drift, profit.data]);

  if (cockpit.error) {
    return <ErrorPanel title="Dashboard" message={`Failed to load dashboard data: ${cockpit.error}`} />;
  }
  if (!data) return <Panel title="Dashboard"><Empty>Loading…</Empty></Panel>;
  if (!data.has_data) {
    return (
      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel title="Dashboard">
          <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
            No data yet - run a sync to populate the portfolio.
          </p>
        </Panel>
        <PerformancePanel />
      </div>
    );
  }

  return (
    <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
      {/* KPI band: the total anchored left, both accounting bases inline beside
          it. Previously these were three stacked cards that filled the screen
          and pushed the holdings below the fold. */}
      <Panel>
        <div className="flex flex-wrap items-start" style={{ gap: 'var(--space-6)' }}>
          <div className="flex flex-col" style={{ gap: '3px' }}>
            <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '10px',
                                               fontWeight: 700, letterSpacing: '0.06em',
                                               textTransform: 'uppercase' }}>
              Total portfolio value
            </span>
            <span className="font-mono" style={{ fontSize: '30px', lineHeight: '36px',
                                                 letterSpacing: '-0.02em' }}>
              {formatUsd(data.total_value_usd)}
            </span>
          </div>
          <BandMetric label={data.net_invested.label} value={basisValue(data.net_invested)}
                      signal={data.net_invested.pl_usd}
                      sub={`on ${formatUsd(data.net_invested.basis_usd)} net in`}
                      note={data.net_invested.question} />
          <BandMetric label={data.fifo.label} value={basisValue(data.fifo)}
                      signal={data.fifo.pl_usd}
                      sub={`on ${formatUsd(data.fifo.basis_usd)} cost basis`}
                      note={data.fifo.question} />
        </div>

        {data.unpriced_count > 0 && (
          <p className="font-ui" style={{ color: 'var(--warning)', fontSize: '12px',
                                          marginTop: 'var(--space-3)', marginBottom: 0 }}>
            Understated: {data.unpriced_count} holding
            {data.unpriced_count === 1 ? '' : 's'} could not be priced and
            {data.unpriced_count === 1 ? ' is' : ' are'} excluded from this total.
          </p>
        )}
      </Panel>

      {/* Chart and drift side by side: value over time is the question the main
          screen exists to answer, and drift is what you act on. */}
      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 2.1fr) minmax(0, 1fr)',
                                     gap: 'var(--space-3)' }}>
        <PerformancePanel />

        <Panel title="Allocation drift">
          {drift.length === 0 ? (
            <Empty>No target allocation configured.</Empty>
          ) : (
            <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
              {drift.map((row) => <DriftRow key={row.symbol} {...row} />)}
            </div>
          )}
        </Panel>
      </div>

      {alerts.length > 0 && (
        <Panel title={`Alerts & review items (${alerts.length})`}>
          {/* Fixed track width, not auto-fit: a single alert stretched to the
              full 1520px reads as a banner rather than as one item of several. */}
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 340px))',
                                         gap: 'var(--space-3)' }}>
            {alerts.map((alert) => <AlertCard key={alert.kind + alert.title} alert={alert} />)}
          </div>
        </Panel>
      )}

      <Panel title="Holdings">
        <HoldingsTable holdings={data.holdings} />
      </Panel>
    </div>
  );
}
