import { useEffect, useState } from 'react';
import { Panel } from '../components/Panel';
import { Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { apiPut } from '../lib/api';
import type { SettingsResponse } from '../types';

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

export function Settings() {
  const { data, error, reload } = useApi<SettingsResponse>('/api/system/settings');
  const [form, setForm] = useState<SettingsResponse | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => { if (data) setForm(structuredClone(data)); }, [data]);

  if (error) return <ErrorPanel title="Settings" message={`Failed to load: ${error}`} />;
  if (!data || !form) return <Panel title="Settings"><Empty>Loading…</Empty></Panel>;

  const pt = form.profit_taking;
  function setPt<K extends keyof SettingsResponse['profit_taking']>(
    key: K, value: SettingsResponse['profit_taking'][K],
  ) {
    setForm((f) => f && { ...f, profit_taking: { ...f.profit_taking, [key]: value } });
  }

  async function save() {
    setSaving(true);
    setMessage(null);
    try {
      const result = await apiPut<SettingsResponse>('/api/system/settings', {
        minimum_trade_usd: Number(form!.minimum_trade_usd),
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

  const dirty = JSON.stringify(form) !== JSON.stringify(data);

  return (
    <>
      <ScreenHeader title="Settings" subtitle="Trading, profit-taking and currency configuration" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel title="Trading">
          <Field label="Minimum trade (USD)"
                 hint="Trades below this size are suppressed across rebalancing and DCA.">
            <NumberInput value={String(form.minimum_trade_usd)}
                         onChange={(v) => setForm((f) => f && { ...f, minimum_trade_usd: v as unknown as number })} />
          </Field>
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
