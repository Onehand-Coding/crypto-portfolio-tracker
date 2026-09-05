import { useEffect, useRef, useState } from 'react';
import { Panel } from '../components/Panel';
import { Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { apiGet, apiPut } from '../lib/api';
import type { LogPreviewResponse, SettingsResponse } from '../types';

function Field({ label, hint, children }: {
  label: string; hint?: string; children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
      <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px',
                                         letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {label}
      </span>
      {children}
      {hint && (
        <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px' }}>
          {hint}
        </span>
      )}
    </label>
  );
}

/** A labelled on/off switch. The track colours to `accent` when on. */
function Toggle({ checked, onChange, label, hint, accent }: {
  checked: boolean; onChange: (v: boolean) => void;
  label: string; hint?: string; accent: string;
}) {
  return (
    <label className="flex items-start" style={{ gap: 'var(--space-3)', cursor: 'pointer' }}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className="shrink-0 transition-colors"
        style={{
          position: 'relative', width: '38px', height: '22px', marginTop: '1px',
          borderRadius: '999px', border: '1px solid var(--border-strong)',
          background: checked ? accent : 'var(--surface-0)',
        }}
      >
        <span style={{
          position: 'absolute', top: '2px', left: checked ? '18px' : '2px',
          width: '16px', height: '16px', borderRadius: '50%',
          background: '#fff', transition: 'left 120ms ease',
        }} />
      </button>
      <span className="flex flex-col" style={{ gap: '2px' }}>
        <span className="font-ui text-sm" style={{ color: 'var(--text-primary)' }}>{label}</span>
        {hint && (
          <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '11px' }}>
            {hint}
          </span>
        )}
      </span>
    </label>
  );
}

const inputStyle = {
  background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
  borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
  padding: 'var(--space-2) var(--space-3)', fontSize: '14px', width: '160px',
} as const;

function NumberInput({ value, onChange, width }: {
  value: string; onChange: (v: string) => void; width?: number;
}) {
  return (
    <input value={value} onChange={(e) => onChange(e.target.value)} inputMode="decimal"
           className="font-mono" style={{ ...inputStyle, width: width ?? 160 }} />
  );
}

const FREQUENCIES = ['daily', 'weekly', 'biweekly', 'monthly', 'quarterly'] as const;
const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] as const;
const LOOKBACK_KEYS = [
  'trades', 'deposits', 'withdrawals', 'p2p_buys',
  'internal_transfers', 'spot_futures_transfers', 'spot_convert_history',
  'simple_earn_rewards', 'simple_earn_subscriptions', 'simple_earn_redemptions',
  'dividend_history', 'staking_history',
] as const;
const TIMEFRAMES = ['long_term', 'swing', 'day'] as const;

export function Settings() {
  const { data, error, reload } = useApi<SettingsResponse>('/api/system/settings');
  const [form, setForm] = useState<SettingsResponse | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [previewCount, setPreviewCount] = useState('50');
  const [preview, setPreview] = useState<LogPreviewResponse | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewMessage, setPreviewMessage] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [transferMessage, setTransferMessage] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [confirmImport, setConfirmImport] = useState(false);

  useEffect(() => { if (data) setForm(structuredClone(data)); }, [data]);

  if (error) return <ErrorPanel title="Settings" message={`Failed to load: ${error}`} />;
  if (!data || !form) return <Panel title="Settings"><Empty>Loading…</Empty></Panel>;

  const pt = form.profit_taking;
  function setPt<K extends keyof SettingsResponse['profit_taking']>(
    key: K, value: SettingsResponse['profit_taking'][K],
  ) {
    setForm((f) => f && { ...f, profit_taking: { ...f.profit_taking, [key]: value } });
  }

  const ta = form.trend_analyzer;
  function setTa<K extends keyof SettingsResponse['trend_analyzer']>(
    key: K, value: SettingsResponse['trend_analyzer'][K],
  ) {
    setForm((f) => f && { ...f, trend_analyzer: { ...f.trend_analyzer, [key]: value } });
  }

  const auto = form.automation;
  function setAuto<K extends keyof SettingsResponse['automation']>(
    key: K, value: SettingsResponse['automation'][K],
  ) {
    setForm((f) => f && { ...f, automation: { ...f.automation, [key]: value } });
  }

  const apis = form.apis;
  function setApis<K extends keyof SettingsResponse['apis']>(
    key: K, value: SettingsResponse['apis'][K],
  ) {
    setForm((f) => f && { ...f, apis: { ...f.apis, [key]: value } });
  }

  function setLookback(key: string, value: string) {
    setForm((f) => f && {
      ...f, history_lookback_days: { ...f.history_lookback_days, [key]: value as unknown as number },
    });
  }

  const lg = form.logging;
  function setLg<K extends keyof SettingsResponse['logging']>(
    key: K, value: SettingsResponse['logging'][K],
  ) {
    setForm((f) => f && { ...f, logging: { ...f.logging, [key]: value } });
  }

  function setTf(
    name: keyof SettingsResponse['trend_timeframes'],
    key: 'period' | 'sma_short_window' | 'sma_long_window',
    value: string,
  ) {
    setForm((f) => f && {
      ...f,
      trend_timeframes: {
        ...f.trend_timeframes,
        [name]: { ...f.trend_timeframes[name], [key]: value as unknown as number },
      },
    });
  }

  async function save() {
    setSaving(true);
    setMessage(null);
    try {
      const lookbacks: Record<string, number> = {};
      for (const key of LOOKBACK_KEYS) {
        lookbacks[key] = Number(form!.history_lookback_days[key]);
      }
      const result = await apiPut<SettingsResponse>('/api/system/settings', {
        minimum_trade_usd: Number(form!.minimum_trade_usd),
        testnet_mode: form!.testnet_mode,
        live_trading_enabled: form!.live_trading_enabled,
        profit_taking: {
          enabled: form!.profit_taking.enabled,
          min_opportunity_score: Number(form!.profit_taking.min_opportunity_score),
          min_unrealized_gain_pct: Number(form!.profit_taking.min_unrealized_gain_pct),
          min_unrealized_gain_usd: Number(form!.profit_taking.min_unrealized_gain_usd),
          max_gain_take_pct: Number(form!.profit_taking.max_gain_take_pct),
          default_take_percentage: Number(form!.profit_taking.default_take_percentage),
        },
        p2p_fiat_currency: form!.p2p_fiat_currency,
        crypto_quotes: form!.crypto_quotes,
        stablecoin_symbols: form!.stablecoin_symbols,
        trend_analyzer: {
          rsi_period: Number(form!.trend_analyzer.rsi_period),
          rsi_oversold: Number(form!.trend_analyzer.rsi_oversold),
          rsi_overbought: Number(form!.trend_analyzer.rsi_overbought),
          cryptocurrencies: form!.trend_analyzer.cryptocurrencies,
        },
        cleanup_days: Number(form!.cleanup_days),
        automation: {
          dca_frequency: form!.automation.dca_frequency,
          rebalancing_frequency: form!.automation.rebalancing_frequency,
        },
        apis: {
          coingecko_timeout: Number(form!.apis.coingecko_timeout),
          binance_timeout: Number(form!.apis.binance_timeout),
          binance_recv_window: Number(form!.apis.binance_recv_window),
          binance_delay_ms: Number(form!.apis.binance_delay_ms),
          coingecko_delay_ms: Number(form!.apis.coingecko_delay_ms),
        },
        history_lookback_days: lookbacks,
        logging: {
          level: form!.logging.level,
          file_enabled: form!.logging.file_enabled,
          file_path: form!.logging.file_path,
          console_enabled: form!.logging.console_enabled,
        },
        trend_timeframes: {
          long_term: {
            period: form!.trend_timeframes.long_term.period,
            sma_short_window: Number(form!.trend_timeframes.long_term.sma_short_window),
            sma_long_window: Number(form!.trend_timeframes.long_term.sma_long_window),
          },
          swing: {
            period: form!.trend_timeframes.swing.period,
            sma_short_window: Number(form!.trend_timeframes.swing.sma_short_window),
            sma_long_window: Number(form!.trend_timeframes.swing.sma_long_window),
          },
          day: {
            period: form!.trend_timeframes.day.period,
            sma_short_window: Number(form!.trend_timeframes.day.sma_short_window),
            sma_long_window: Number(form!.trend_timeframes.day.sma_long_window),
          },
        },
      });
      setForm(structuredClone(result));
      setMessage('Settings saved.');
      reload();
    } catch (e) {
      setMessage(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSaving(false);
    }
  }

  async function loadPreview() {
    setPreviewBusy(true);
    setPreviewMessage(null);
    try {
      const n = Math.max(1, Math.min(500, Number(previewCount) || 50));
      const result = await apiGet<LogPreviewResponse>(`/api/system/logs/preview?lines=${n}`);
      setPreview(result);
    } catch (e) {
      setPreviewMessage(`Preview failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setPreviewBusy(false);
    }
  }

  async function importConfig() {
    const file = fileRef.current?.files?.[0];
    if (!file) { setTransferMessage('Choose a file first.'); return; }
    setImporting(true);
    setTransferMessage(null);
    try {
      const body = new FormData();
      body.append('file', file);
      const res = await fetch('/api/system/config/import', { method: 'POST', body });
      if (!res.ok) {
        const detail = await res.text();
        setTransferMessage(`Import failed: ${detail || res.statusText}`);
        return;
      }
      const result = (await res.json()) as SettingsResponse;
      setForm(structuredClone(result));
      setTransferMessage('Config imported.');
      if (fileRef.current) fileRef.current.value = '';
      setConfirmImport(false);
      reload();
    } catch (e) {
      setTransferMessage(`Import failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setImporting(false);
    }
  }

  const dirty = JSON.stringify(form) !== JSON.stringify(data);

  return (
    <>
      <ScreenHeader title="Settings" subtitle="Trading, profit-taking and currency configuration" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel title="Trading mode">
          <p className="font-ui text-sm"
             style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-4) 0' }}>
            Two independent switches, the same the CLI and Streamlit use. Testnet
            selects the exchange endpoint; live trading arms real orders. With
            live trading off, every screen still works but orders are simulated.
          </p>
          <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
            <Toggle
              checked={form.testnet_mode}
              onChange={(v) => setForm((f) => f && { ...f, testnet_mode: v })}
              label="🧪 Binance testnet mode"
              hint="Switches between mainnet and testnet. Takes full effect after a server restart."
              accent="var(--warning)"
            />
            <Toggle
              checked={form.live_trading_enabled}
              onChange={(v) => setForm((f) => f && { ...f, live_trading_enabled: v })}
              label="🔴 Enable live trading"
              hint="On: real orders are placed. Off: trades are simulated (dry run)."
              accent="var(--negative)"
            />
          </div>
        </Panel>

        <Panel title="Trading">
          <Field label="Minimum trade (USD)"
                 hint="Trades below this size are suppressed across rebalancing and DCA.">
            <NumberInput value={String(form.minimum_trade_usd)}
                         onChange={(v) => setForm((f) => f && { ...f, minimum_trade_usd: v as unknown as number })} />
          </Field>
        </Panel>

        <Panel title="Schedules">
          <div className="grid grid-cols-3" style={{ gap: 'var(--space-5)' }}>
            <Field label="DCA frequency">
              <select value={auto.dca_frequency}
                      onChange={(e) => setAuto('dca_frequency', e.target.value)}
                      className="font-mono" style={inputStyle}>
                {FREQUENCIES.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
            </Field>
            <Field label="Rebalancing frequency">
              <select value={auto.rebalancing_frequency}
                      onChange={(e) => setAuto('rebalancing_frequency', e.target.value)}
                      className="font-mono" style={inputStyle}>
                {FREQUENCIES.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
            </Field>
          </div>
        </Panel>

        <Panel title="API">
          <div className="grid grid-cols-3" style={{ gap: 'var(--space-5)' }}>
            <Field label="CoinGecko timeout" hint="Seconds.">
              <NumberInput width={120} value={String(apis.coingecko_timeout)}
                           onChange={(v) => setApis('coingecko_timeout', v as unknown as number)} />
            </Field>
            <Field label="Binance timeout" hint="Seconds.">
              <NumberInput width={120} value={String(apis.binance_timeout)}
                           onChange={(v) => setApis('binance_timeout', v as unknown as number)} />
            </Field>
            <Field label="Binance recv window" hint="Milliseconds.">
              <NumberInput width={120} value={String(apis.binance_recv_window)}
                           onChange={(v) => setApis('binance_recv_window', v as unknown as number)} />
            </Field>
            <Field label="Binance delay" hint="Milliseconds between requests.">
              <NumberInput width={120} value={String(apis.binance_delay_ms)}
                           onChange={(v) => setApis('binance_delay_ms', v as unknown as number)} />
            </Field>
            <Field label="CoinGecko delay" hint="Milliseconds between requests.">
              <NumberInput width={120} value={String(apis.coingecko_delay_ms)}
                           onChange={(v) => setApis('coingecko_delay_ms', v as unknown as number)} />
            </Field>
          </div>
        </Panel>

        <Panel title="Profit taking">
          <label className="flex items-center" style={{ gap: 'var(--space-2)',
                                                        marginBottom: 'var(--space-4)', cursor: 'pointer' }}>
            <input type="checkbox" checked={pt.enabled}
                   onChange={(e) => setPt('enabled', e.target.checked)} />
            <span className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
              Profit-taking analysis enabled
            </span>
          </label>
          <div className="grid grid-cols-3" style={{ gap: 'var(--space-5)' }}>
            <Field label="Min opportunity score" hint="0–100">
              <NumberInput width={120} value={String(pt.min_opportunity_score)}
                           onChange={(v) => setPt('min_opportunity_score', v as unknown as number)} />
            </Field>
            <Field label="Min unrealized gain %">
              <NumberInput width={120} value={String(pt.min_unrealized_gain_pct)}
                           onChange={(v) => setPt('min_unrealized_gain_pct', v as unknown as number)} />
            </Field>
            <Field label="Min unrealized gain USD">
              <NumberInput width={120} value={String(pt.min_unrealized_gain_usd)}
                           onChange={(v) => setPt('min_unrealized_gain_usd', v as unknown as number)} />
            </Field>
            <Field label="Max gain take %" hint="Cap on how much of a gain to take.">
              <NumberInput width={120} value={String(pt.max_gain_take_pct)}
                           onChange={(v) => setPt('max_gain_take_pct', v as unknown as number)} />
            </Field>
            <Field label="Default take %" hint="Default portion of a position to trim.">
              <NumberInput width={120} value={String(pt.default_take_percentage)}
                           onChange={(v) => setPt('default_take_percentage', v as unknown as number)} />
            </Field>
          </div>
        </Panel>

        <Panel title="Currency">
          <div className="grid grid-cols-3" style={{ gap: 'var(--space-5)' }}>
            <Field label="P2P fiat currency" hint="e.g. USD, PHP.">
              <input value={form.p2p_fiat_currency}
                     onChange={(e) => setForm((f) => f && { ...f, p2p_fiat_currency: e.target.value })}
                     className="font-mono" style={{ ...inputStyle, width: 120, textTransform: 'uppercase' }} />
            </Field>
            <Field label="Crypto quotes" hint="Comma-separated, e.g. USDT, BTC.">
              <input value={form.crypto_quotes.join(', ')}
                     onChange={(e) => setForm((f) => f && { ...f, crypto_quotes: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })}
                     className="font-mono" style={inputStyle} />
            </Field>
            <Field label="Stablecoin symbols" hint="Comma-separated, e.g. USDT, USDC.">
              <input value={form.stablecoin_symbols.join(', ')}
                     onChange={(e) => setForm((f) => f && { ...f, stablecoin_symbols: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })}
                     className="font-mono" style={inputStyle} />
            </Field>
          </div>
        </Panel>

        <Panel title="Lookbacks">
          <p className="font-ui text-sm"
             style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-4) 0' }}>
            History window in days for each transaction source. Minimum 1 day.
          </p>
          <div className="grid grid-cols-3" style={{ gap: 'var(--space-5)' }}>
            {LOOKBACK_KEYS.map((key) => (
              <Field key={key} label={key}>
                <NumberInput width={120} value={String(form.history_lookback_days[key])}
                             onChange={(v) => setLookback(key, v)} />
              </Field>
            ))}
          </div>
        </Panel>

        <Panel title="Logging">
          <div className="grid grid-cols-3" style={{ gap: 'var(--space-5)',
                                                     marginBottom: 'var(--space-4)' }}>
            <Field label="Level">
              <select value={lg.level}
                      onChange={(e) => setLg('level', e.target.value)}
                      className="font-mono" style={inputStyle}>
                {LOG_LEVELS.map((l) => <option key={l} value={l}>{l}</option>)}
              </select>
            </Field>
            <Field label="Log file path" hint="Relative or absolute path.">
              <input value={lg.file_path}
                     onChange={(e) => setLg('file_path', e.target.value)}
                     className="font-mono" style={{ ...inputStyle, width: '100%' }} />
            </Field>
          </div>
          <div className="flex flex-col" style={{ gap: 'var(--space-4)',
                                                  marginBottom: 'var(--space-4)' }}>
            <Toggle
              checked={lg.file_enabled}
              onChange={(v) => setLg('file_enabled', v)}
              label="Log to file"
              hint="Writes to the path above."
              accent="var(--action)"
            />
            <Toggle
              checked={lg.console_enabled}
              onChange={(v) => setLg('console_enabled', v)}
              label="Log to console"
              hint="Streams log records to the server console."
              accent="var(--action)"
            />
          </div>
          <div className="flex flex-wrap items-end" style={{ gap: 'var(--space-3)' }}>
            <Field label="Lines" hint="1–500.">
              <NumberInput width={120} value={previewCount} onChange={setPreviewCount} />
            </Field>
            <Button onClick={loadPreview} disabled={previewBusy}>
              {previewBusy ? 'Loading…' : 'Preview'}
            </Button>
            {previewMessage && (
              <span className="font-ui" style={{ fontSize: '13px', color: 'var(--negative)' }}>
                {previewMessage}
              </span>
            )}
          </div>
          {preview && (
            <div style={{ marginTop: 'var(--space-3)' }}>
              <p className="font-mono" style={{ color: 'var(--text-tertiary)',
                                               fontSize: '11px', margin: '0 0 var(--space-2) 0' }}>
                {preview.path}
              </p>
              <pre className="font-mono" style={{ background: 'var(--surface-0)',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-control)',
                    padding: 'var(--space-3)', fontSize: '12px', color: 'var(--text-secondary)',
                    maxHeight: '240px', overflowY: 'auto', whiteSpace: 'pre-wrap', margin: 0 }}>
                {preview.lines.join('\n')}
              </pre>
              {preview.truncated && (
                <p className="font-ui" style={{ color: 'var(--text-tertiary)',
                                               fontSize: '11px', margin: 'var(--space-2) 0 0 0' }}>
                  Showing the last {preview.lines.length} of {preview.total_lines} lines - truncated.
                </p>
              )}
            </div>
          )}
        </Panel>

        <Panel title="Trend analyzer">
          <div className="grid grid-cols-3" style={{ gap: 'var(--space-5)',
                                                     marginBottom: 'var(--space-4)' }}>
            <Field label="RSI period">
              <NumberInput width={120} value={String(ta.rsi_period)}
                           onChange={(v) => setTa('rsi_period', v as unknown as number)} />
            </Field>
            <Field label="RSI oversold" hint="0–100">
              <NumberInput width={120} value={String(ta.rsi_oversold)}
                           onChange={(v) => setTa('rsi_oversold', v as unknown as number)} />
            </Field>
            <Field label="RSI overbought" hint="0–100">
              <NumberInput width={120} value={String(ta.rsi_overbought)}
                           onChange={(v) => setTa('rsi_overbought', v as unknown as number)} />
            </Field>
          </div>
          <Field label="Cryptocurrencies" hint="Comma-separated tickers analysed for trend, e.g. BTC-USD, ETH-USD.">
            <input value={ta.cryptocurrencies.join(', ')}
                   onChange={(e) => setTa('cryptocurrencies',
                     e.target.value.split(',').map((s) => s.trim()).filter(Boolean) as unknown as string[])}
                   className="font-mono" style={{ ...inputStyle, width: '100%' }} />
          </Field>
        </Panel>

        <Panel title="Trend timeframes">
          <p className="font-ui text-sm"
             style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-4) 0' }}>
            SMA window pairs per timeframe. Short must stay below long (1–200).
            Periods use Xy/Xd/Xmo fetch windows (e.g. 4y, 90d, 7d).
          </p>
          <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
            {TIMEFRAMES.map((name) => (
              <div key={name} className="grid grid-cols-3" style={{ gap: 'var(--space-5)' }}>
                <Field label={`${name} period`}>
                  <input value={form.trend_timeframes[name].period}
                         onChange={(e) => setTf(name, 'period', e.target.value)}
                         className="font-mono" style={{ ...inputStyle, width: 120 }} />
                </Field>
                <Field label={`${name} short window`}>
                  <NumberInput width={120}
                               value={String(form.trend_timeframes[name].sma_short_window)}
                               onChange={(v) => setTf(name, 'sma_short_window', v)} />
                </Field>
                <Field label={`${name} long window`}>
                  <NumberInput width={120}
                               value={String(form.trend_timeframes[name].sma_long_window)}
                               onChange={(v) => setTf(name, 'sma_long_window', v)} />
                </Field>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Data retention">
          <Field label="Cleanup days"
                 hint="Age beyond which snapshots are eligible for cleanup. 0 disables it.">
            <NumberInput value={String(form.cleanup_days)}
                         onChange={(v) => setForm((f) => f && { ...f, cleanup_days: v as unknown as number })} />
          </Field>
        </Panel>

        <Panel title="Config transfer">
          <p className="font-ui text-sm"
             style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-3) 0' }}>
            Export the sanitized config as JSON, or import a previously exported
            file. Secrets are preserved on import and a backup is written first,
            so a bad import is reversible.
          </p>
          <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-3)' }}>
            <a href="/api/system/config/export" download
               className="font-ui" style={{ color: 'var(--action)', fontSize: '13px' }}>
              Export config
            </a>
            <Field label="Config file">
              <input ref={fileRef} type="file" accept=".json,application/json"
                     className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }} />
            </Field>
            <Button onClick={() => setConfirmImport(true)} disabled={importing || confirmImport}>
              Import
            </Button>
            {confirmImport && (
              <>
                <span className="font-ui" style={{ fontSize: '13px', color: 'var(--warning)' }}>
                  Overwrite current settings with this file?
                </span>
                <Button onClick={importConfig} disabled={importing}>
                  {importing ? 'Importing…' : 'Confirm import'}
                </Button>
                <Button onClick={() => setConfirmImport(false)} disabled={importing}>
                  Cancel
                </Button>
              </>
            )}
            {transferMessage && (
              <span className="font-ui" style={{ fontSize: '13px',
                       color: transferMessage.startsWith('Import failed')
                         ? 'var(--negative)' : 'var(--text-secondary)' }}>
                {transferMessage}
              </span>
            )}
          </div>
        </Panel>

        <div className="flex items-center" style={{ gap: 'var(--space-4)' }}>
          <Button onClick={save} disabled={saving || !dirty}>
            {saving ? 'Saving…' : 'Save settings'}
          </Button>
          {message && (
            <span className="font-ui" style={{ fontSize: '13px',
                     color: message.startsWith('Save failed') ? 'var(--negative)' : 'var(--text-secondary)' }}>
              {message}
            </span>
          )}
        </div>
      </div>
    </>
  );
}
