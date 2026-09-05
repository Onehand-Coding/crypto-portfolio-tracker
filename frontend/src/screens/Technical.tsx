import { useState } from 'react';
import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { Panel } from '../components/Panel';
import { AnalysisBar, Badge, Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { StalenessNote } from '../components/StalenessNote';
import { useApi, usePollWhile } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatUsd, NULL_GLYPH } from '../lib/format';
import type { IndicatorRow, IndicatorsResponse, TechnicalResponse } from '../types';

const TIMEFRAME_LABEL: Record<string, string> = {
  swing: 'Swing (3 months)',
  long_term: 'Long term (4 years)',
  day: 'Day (60 days)',
};

function rsiTone(rsi: number | null) {
  if (rsi === null) return 'neutral' as const;
  if (rsi < 30) return 'positive' as const;   // oversold: a buy signal here
  if (rsi > 70) return 'negative' as const;   // overbought
  return 'neutral' as const;
}

function IndicatorTable({ rows }: { rows: IndicatorRow[] }) {
  if (rows.length === 0) return <Empty>No indicator data in this timeframe.</Empty>;
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th className="text-left">Asset</th>
            <th className="text-right">Price</th>
            <th className="text-right">RSI</th>
            <th className="text-right">SMA short</th>
            <th className="text-right">SMA long</th>
            <th className="text-right">Support</th>
            <th className="text-right">Resistance</th>
            <th className="text-left">Conditions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.symbol}>
              <td className="text-left" style={{ fontWeight: 500 }}>{row.symbol}</td>
              <td className="text-right">{formatUsd(row.price)}</td>
              <td className="text-right">
                {row.rsi === null ? NULL_GLYPH
                  : <Badge text={row.rsi.toFixed(1)} tone={rsiTone(row.rsi)} />}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(row.sma_short)}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(row.sma_long)}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(row.support)}
              </td>
              <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                {formatUsd(row.resistance)}
              </td>
              <td className="text-left">
                <span className="flex flex-wrap" style={{ gap: '4px' }}>
                  {row.conditions.slice(0, 3).map((c) => (
                    <Badge key={c} text={c.replace(/_/g, ' ').toLowerCase()} />
                  ))}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const INDICATOR_TIMEFRAMES = ['long_term', 'swing', 'day'];

/**
 * The history endpoint only accepts bare uppercase tickers (^[A-Z0-9]{2,10}$),
 * while technical symbols may carry a quote suffix (e.g. BTC-USD) - so strip
 * anything from the first dash on before sending. Picker options still show
 * the raw symbol, exactly as the table above renders it.
 */
function apiSymbol(symbol: string): string {
  return symbol.split('-')[0].toUpperCase();
}

function SelectField({ label, value, onChange, children }: {
  label: string; value: string; onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
      <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                         letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {label}
      </span>
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="font-mono"
        style={{
          background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
          borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
          padding: 'var(--space-2) var(--space-3)', fontSize: '14px',
        }}
      >
        {children}
      </select>
    </label>
  );
}

const TOOLTIP_STYLE = {
  background: 'var(--surface-2)',
  border: '1px solid var(--border-strong)',
  borderRadius: 'var(--radius-control)',
  fontSize: '12px',
};

function IndicatorViewer({ symbols }: { symbols: string[] }) {
  const [coin, setCoin] = useState(symbols[0] ?? '');
  const [timeframe, setTimeframe] = useState('swing');
  const symbol = apiSymbol(coin);
  const { data, error, reload } = useApi<IndicatorsResponse>(
    `/api/strategy/indicators?symbol=${symbol}&timeframe=${timeframe}`,
    [symbol, timeframe],
  );
  usePollWhile(Boolean(data?.is_running), reload);

  async function run() {
    try {
      await apiPost('/api/strategy/indicators/run', { symbol, timeframe });
    } finally {
      reload();
    }
  }

  // Points pass through to recharts unconverted: a null stays a null so the
  // line gaps there. Zero-filling would invent a plausible, wrong data point.
  const points = data?.points ?? [];
  const latest = points.length > 0 ? points[points.length - 1] : null;
  const label = TIMEFRAME_LABEL[timeframe] ?? timeframe;

  return (
    <>
      <Panel title="Per-coin indicator history">
        <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
          <div className="flex flex-wrap items-end" style={{ gap: 'var(--space-4)' }}>
            <SelectField label="Coin" value={coin} onChange={setCoin}>
              {symbols.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </SelectField>
            <SelectField label="Timeframe" value={timeframe} onChange={setTimeframe}>
              {INDICATOR_TIMEFRAMES.map((t) => (
                <option key={t} value={t}>{TIMEFRAME_LABEL[t] ?? t}</option>
              ))}
            </SelectField>
            <div className="flex items-center" style={{ gap: 'var(--space-4)' }}>
              {data && <StalenessNote staleness={data.staleness} />}
              <Button variant="secondary" onClick={reload}>Refresh</Button>
              <Button onClick={run} disabled={Boolean(data?.is_running)}>
                {data?.is_running ? 'Running…' : 'Run indicators'}
              </Button>
            </div>
          </div>

          {error && (
            <p className="font-mono" style={{ color: 'var(--negative)', fontSize: '12px', margin: 0 }}>
              Failed to load indicator history: {error}
            </p>
          )}
          {data?.error && (
            <p className="font-mono" style={{ color: 'var(--negative)', fontSize: '12px', margin: 0 }}>
              Last run failed: {data.error}
            </p>
          )}

          {data && !data.has_data ? (
            <Empty>
              No indicator history for {symbol} ({label}) yet. Run the viewer above -
              fetching price history needs network access to the price feed.
            </Empty>
          ) : data && points.length > 0 ? (
            <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)', margin: 0 }}>
              {data.symbol} · {label} · {points.length} points · latest close {formatUsd(latest?.close ?? null)}
            </p>
          ) : null}
        </div>
      </Panel>

      {data?.has_data && points.length > 0 && (
        <>
          <Panel title="Price and moving averages">
            <div style={{ width: '100%', height: 240 }}>
              <ResponsiveContainer>
                <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                         stroke="var(--border)" minTickGap={48}
                         tickFormatter={(t) => String(t).slice(0, 10)} />
                  <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                         stroke="var(--border)" width={72}
                         tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
                         domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={TOOLTIP_STYLE}
                    labelStyle={{ color: 'var(--text-secondary)' }}
                    labelFormatter={(t) => String(t).slice(0, 10)}
                    formatter={(v) => formatUsd(typeof v === 'number' ? v : null)}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="close" name="Close" stroke="var(--action)"
                        strokeWidth={1.5} dot={false} connectNulls={false} />
                  <Line type="monotone" dataKey="sma_short" name="SMA short"
                        stroke="var(--positive)" strokeWidth={1.5} dot={false}
                        connectNulls={false} />
                  <Line type="monotone" dataKey="sma_long" name="SMA long"
                        stroke="var(--text-secondary)" strokeWidth={1.5} dot={false}
                        connectNulls={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          <Panel title="RSI">
            <div style={{ width: '100%', height: 180 }}>
              <ResponsiveContainer>
                <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                         stroke="var(--border)" minTickGap={48}
                         tickFormatter={(t) => String(t).slice(0, 10)} />
                  <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                         stroke="var(--border)" width={48}
                         tickFormatter={(v) => Number(v).toFixed(0)}
                         domain={[0, 100]} />
                  <Tooltip
                    contentStyle={TOOLTIP_STYLE}
                    labelStyle={{ color: 'var(--text-secondary)' }}
                    labelFormatter={(t) => String(t).slice(0, 10)}
                    formatter={(v) => (typeof v === 'number' ? v.toFixed(1) : NULL_GLYPH)}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="rsi" name="RSI" stroke="var(--action)"
                        strokeWidth={1.5} dot={false} connectNulls={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          <Panel title="MACD">
            <div style={{ width: '100%', height: 220 }}>
              <ResponsiveContainer>
                <ComposedChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                         stroke="var(--border)" minTickGap={48}
                         tickFormatter={(t) => String(t).slice(0, 10)} />
                  <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                         stroke="var(--border)" width={64}
                         tickFormatter={(v) => Number(v).toFixed(0)} />
                  <Tooltip
                    contentStyle={TOOLTIP_STYLE}
                    labelStyle={{ color: 'var(--text-secondary)' }}
                    labelFormatter={(t) => String(t).slice(0, 10)}
                    formatter={(v) => formatUsd(typeof v === 'number' ? v : null)}
                  />
                  <Legend />
                  <Bar dataKey="macd_hist" name="Histogram" fill="var(--text-secondary)"
                       fillOpacity={0.45} />
                  <Line type="monotone" dataKey="macd" name="MACD" stroke="var(--action)"
                        strokeWidth={1.5} dot={false} connectNulls={false} />
                  <Line type="monotone" dataKey="macd_signal" name="Signal"
                        stroke="var(--positive)" strokeWidth={1.5} dot={false}
                        connectNulls={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </>
      )}
    </>
  );
}

export function Technical() {
  const { data, error, reload } = useApi<TechnicalResponse>('/api/strategy/technical');
  usePollWhile(Boolean(data?.is_running), reload);
  async function run() {
    try {
      await apiPost('/api/strategy/technical/run');
    } finally {
      reload();
    }
  }

  if (error) return <ErrorPanel title="Technical analysis" message={`Failed to load: ${error}`} />;
  if (!data) return <Panel title="Technical analysis"><Empty>Loading…</Empty></Panel>;

  const timeframes = Object.entries(data.timeframes);
  // Coin context for the per-coin viewer comes from this screen's own
  // payload: every symbol the indicator tables already report on.
  const symbols = Array.from(
    new Set(Object.values(data.timeframes).flatMap((rows) => rows.map((r) => r.symbol))),
  ).sort();

  return (
    <>
      <ScreenHeader title="Technical analysis"
                    subtitle="RSI, moving averages and support/resistance by timeframe" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <AnalysisBar state={data} onRun={run} label="Technical analysis" />

        {data.bear_market !== null && (
          <Panel>
            <div className="flex items-center" style={{ gap: 'var(--space-3)' }}>
              <Badge
                text={data.bear_market ? 'BEAR MARKET' : 'NOT BEAR MARKET'}
                tone={data.bear_market ? 'negative' : 'positive'}
              />
              <span className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
                Regime is defined by BTC relative to its SMA200. In a bear market your
                config can suppress buy signals.
              </span>
            </div>
          </Panel>
        )}

        {!data.has_data ? (
          <Panel title="Indicators"><Empty>No analysis has been run yet.</Empty></Panel>
        ) : timeframes.length === 0 ? (
          <Panel title="Indicators">
            <p className="font-ui text-sm" style={{ color: 'var(--warning)', margin: 0 }}>
              The last run returned no timeframe reports. This usually means the market
              data fetch failed rather than that there is nothing to show.
            </p>
          </Panel>
        ) : (
          timeframes.map(([timeframe, rows]) => (
            <Panel key={timeframe} title={TIMEFRAME_LABEL[timeframe] ?? timeframe}>
              <IndicatorTable rows={rows} />
            </Panel>
          ))
        )}

        {symbols.length === 0 ? (
          <Panel title="Per-coin indicator history">
            <Empty>No coins in the technical payload yet - run the analysis above first.</Empty>
          </Panel>
        ) : (
          <IndicatorViewer key={symbols.join(',')} symbols={symbols} />
        )}
      </div>
    </>
  );
}
