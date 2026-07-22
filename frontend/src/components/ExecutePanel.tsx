import { useState, type ReactNode } from 'react';
import { Panel } from './Panel';
import { Badge, Button } from './Screen';
import type { TradeExecuteResponse } from '../types';

/**
 * Shared execution gate. Every order-placing action funnels through this: a
 * plain-language description, a typed EXECUTE confirmation, and the raw result.
 *
 * The confirmation is deliberate friction. `execute` is only called once the
 * word is typed exactly, and the caller supplies the actual API request.
 */
export function ExecutePanel({
  title, description, execute, disabled, children,
}: {
  title: string;
  description: string;
  execute: () => Promise<TradeExecuteResponse>;
  disabled?: boolean;
  children?: ReactNode;
}) {
  const [confirmText, setConfirmText] = useState('');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<TradeExecuteResponse | null>(null);

  async function run() {
    setRunning(true);
    setResult(null);
    try {
      setResult(await execute());
      setConfirmText('');
    } catch (e) {
      setResult({ success: false, testnet: true, messages: [],
                  errors: [e instanceof Error ? e.message : String(e)] });
    } finally {
      setRunning(false);
    }
  }

  return (
    <Panel title={title}>
      {children && <div style={{ marginBottom: 'var(--space-4)' }}>{children}</div>}
      <p className="font-ui text-sm"
         style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-3) 0' }}>
        {description} To confirm, type <span className="font-mono"
        style={{ color: 'var(--text-primary)' }}>EXECUTE</span> below.
      </p>
      <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-3)' }}>
        <input
          value={confirmText}
          onChange={(e) => setConfirmText(e.target.value)}
          placeholder="EXECUTE"
          disabled={disabled}
          className="font-mono"
          style={{ background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
                   borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                   padding: 'var(--space-2) var(--space-3)', width: '160px', fontSize: '14px' }}
        />
        <Button onClick={run} disabled={disabled || running || confirmText.trim() !== 'EXECUTE'}>
          {running ? 'Executing…' : title}
        </Button>
      </div>

      {result && (
        <div style={{ marginTop: 'var(--space-4)' }}>
          <Badge text={result.success ? 'EXECUTED' : 'FAILED'}
                 tone={result.success ? 'positive' : 'negative'} />
          <ul className="flex flex-col font-mono"
              style={{ gap: '4px', fontSize: '12px', margin: 'var(--space-3) 0 0 0',
                       padding: 0, listStyle: 'none' }}>
            {result.messages.map((m, i) => (
              <li key={`m${i}`} style={{ color: 'var(--text-secondary)', wordBreak: 'break-all' }}>
                {m}
              </li>
            ))}
            {result.errors.map((m, i) => (
              <li key={`e${i}`} style={{ color: 'var(--negative)', wordBreak: 'break-all' }}>{m}</li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}
