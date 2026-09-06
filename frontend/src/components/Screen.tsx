import { useState, type ReactNode } from 'react';
import { Panel } from './Panel';
import { StalenessNote } from './StalenessNote';
import type { AnalysisState, Staleness } from '../types';
import { ApiError, NetworkError } from '../lib/api';

/** Page title row. Gives every screen the same anchor and staleness position. */
export function ScreenHeader({
  title, subtitle, staleness, actions,
}: {
  title: string;
  subtitle?: string;
  staleness?: Staleness;
  actions?: ReactNode;
}) {
  return (
    // Compact: the screen name is already in the top-bar tabs, so this is
    // context for it, not a second announcement of where you are.
    <div className="flex items-baseline justify-between"
         style={{ marginBottom: 'var(--space-3)', gap: 'var(--space-4)' }}>
      <div className="flex flex-wrap items-baseline" style={{ gap: 'var(--space-3)' }}>
        <h1 className="font-ui"
            style={{ fontSize: '15px', fontWeight: 600, letterSpacing: '-0.01em' }}>
          {title}
        </h1>
        {subtitle && (
          <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px' }}>
            {subtitle}
          </p>
        )}
      </div>
      <div className="flex shrink-0 items-center" style={{ gap: 'var(--space-4)' }}>
        {staleness && <StalenessNote staleness={staleness} />}
        {actions}
      </div>
    </div>
  );
}

export function Button({
  onClick, disabled, children, variant = 'primary',
}: {
  onClick?: () => void;
  disabled?: boolean;
  children: ReactNode;
  variant?: 'primary' | 'secondary';
}) {
  const primary = variant === 'primary';
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="font-ui shrink-0 transition-colors"
      style={{
        background: disabled ? 'var(--surface-2)'
                  : primary ? 'var(--action)' : 'transparent',
        color: disabled ? 'var(--text-tertiary)'
             : primary ? '#FFFFFF' : 'var(--text-secondary)',
        border: primary ? 'none' : '1px solid var(--border-strong)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        borderRadius: 'var(--radius-control)',
        padding: 'var(--space-2) var(--space-4)',
        fontSize: '13px',
        fontWeight: 500,
      }}
    >
      {children}
    </button>
  );
}

/** Semantic pill. Colour plus the word, never colour alone. */
export function Badge({ text, tone = 'neutral' }: {
  text: string;
  tone?: 'positive' | 'negative' | 'warning' | 'neutral' | 'action';
}) {
  const colour = {
    positive: 'var(--positive)',
    negative: 'var(--negative)',
    warning: 'var(--warning)',
    action: 'var(--action)',
    neutral: 'var(--text-secondary)',
  }[tone];
  return (
    <span
      className="font-ui"
      style={{
        color: colour,
        // 18% tint of the same hue rather than a solid fill, so a row of these
        // does not overpower the numbers they annotate.
        background: `color-mix(in srgb, ${colour} 18%, transparent)`,
        border: `1px solid color-mix(in srgb, ${colour} 35%, transparent)`,
        borderRadius: 'var(--radius-control)',
        padding: '2px 8px',
        fontSize: '11px',
        fontWeight: 600,
        letterSpacing: '0.03em',
        whiteSpace: 'nowrap',
      }}
    >
      {text}
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)', margin: 0 }}>
      {children}
    </p>
  );
}

export function ErrorPanel({ title, message }: { title: string; message: string }) {
  return (
    <Panel title={title}>
      <p className="font-mono text-sm" style={{ color: 'var(--negative)', margin: 0 }}>
        {message}
      </p>
    </Panel>
  );
}

/**
 * Shared shell for the four live-analysis screens. Each needs an explicit run,
 * and each must distinguish four states that are easy to conflate: never run,
 * running, failed (showing the previous result), and current.
 */
export function AnalysisBar({
  state, onRun, label,
}: {
  state: AnalysisState;
  onRun: () => void | Promise<unknown>;
  label: string;
}) {
  // Optimistic "Starting" state: the run adapters do blocking network I/O
  // before their first await, so the reload that would flip is_running can
  // queue seconds behind the click. Without this the button sits dead, then
  // flashes "Running…". Every screen's onRun is try/finally with no catch,
  // so a rejected start propagates here -- and would otherwise be silent.
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  function startErrorMessage(error: unknown): string {
    if (error instanceof ApiError && error.status === 409) {
      return 'This analysis is already running. Please wait for it to finish.';
    }
    if (error instanceof NetworkError) {
      return 'The server could not be reached. Check that it is running, then try again.';
    }
    if (error instanceof ApiError) return error.message;
    return 'This analysis could not be started. Please try again.';
  }

  async function handleRun() {
    setStarting(true);
    setStartError(null);
    try {
      await onRun();
    } catch (e) {
      setStartError(startErrorMessage(e));
    } finally {
      setStarting(false);
    }
  }

  const running = state.is_running || starting;
  return (
    <Panel>
      <div className="flex items-center justify-between" style={{ gap: 'var(--space-5)' }}>
        <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
          <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)', margin: 0 }}>
            {label} contacts Binance. Results are cached, so this page opens instantly
            and always shows how old its figures are.
          </p>
          {state.error && (
            <p className="font-mono" style={{ color: 'var(--negative)', fontSize: '12px', margin: 0 }}>
              Last run failed: {state.error}
              {state.has_data && ' - the figures below are from the previous run.'}
            </p>
          )}
          {startError && !state.is_running && (
            <p className="font-mono" style={{ color: 'var(--negative)', fontSize: '12px', margin: 0 }}>
              {startError}
            </p>
          )}
        </div>
        <div className="flex shrink-0 items-center" style={{ gap: 'var(--space-4)' }}>
          <StalenessNote staleness={state.staleness} verb="run" />
          <Button onClick={handleRun} disabled={running}>
            {state.is_running ? 'Running…' : starting ? 'Starting…' : 'Run analysis'}
          </Button>
        </div>
      </div>
    </Panel>
  );
}
