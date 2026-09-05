import { useEffect, useRef, useState } from 'react';
import { Panel } from '../components/Panel';
import { apiPost, ApiError, NetworkError } from '../lib/api';

interface SyncEvent {
  event: 'progress' | 'complete' | 'error';
  message: string;
  /** Log level of the underlying core record. Present on progress events. */
  level?: string;
  /** On the terminal `complete` event: failures logged during the run. */
  error_count?: number;
  warning_count?: number;
}

/**
 * A core ERROR arrives as an ordinary progress event -- it must not close the
 * stream, since the sync keeps running. Colouring by level is what stops it
 * scrolling past in muted grey among dozens of chunk lines.
 */
function lineColour(event: SyncEvent): string {
  if (event.event === 'error' || event.level === 'ERROR' || event.level === 'CRITICAL') {
    return 'var(--negative)';
  }
  if (event.level === 'WARNING') {
    return 'var(--warning)';
  }
  if (event.event === 'complete') {
    return (event.error_count ?? 0) > 0 ? 'var(--warning)' : 'var(--positive)';
  }
  return 'var(--text-secondary)';
}

function lineText(event: SyncEvent): string {
  if (event.event === 'error') {
    return `error: ${event.message}`;
  }
  // Prefixed rather than relying on colour alone: colour is never the sole
  // carrier of meaning anywhere else in this UI.
  if (event.level === 'ERROR' || event.level === 'CRITICAL') {
    return `error: ${event.message}`;
  }
  if (event.level === 'WARNING') {
    return `warning: ${event.message}`;
  }
  if (event.event === 'complete') {
    const errors = event.error_count ?? 0;
    const warnings = event.warning_count ?? 0;
    if (errors > 0) {
      return `${event.message} - but ${errors} error${errors === 1 ? '' : 's'} occurred. `
           + 'Some data may be missing or mispriced.';
    }
    if (warnings > 0) {
      return `${event.message} - ${warnings} warning${warnings === 1 ? '' : 's'} logged.`;
    }
  }
  return event.message;
}

function startErrorMessage(e: unknown): string {
  // A 409 means the backend already has a sync in flight -- present that
  // plainly rather than the raw "A sync is already running" HTTP body text.
  if (e instanceof ApiError && e.status === 409) {
    return 'A sync is already running. Wait for it to finish before starting another.';
  }
  if (e instanceof NetworkError || e instanceof ApiError) {
    return e.message;
  }
  return String(e);
}

export function Sync() {
  const [events, setEvents] = useState<SyncEvent[]>([]);
  const [running, setRunning] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  // Set when a `complete` or `error` event has been handled and the stream
  // was closed deliberately. The backend closes the stream after a terminal
  // event, and closing from this side also fires the browser's onerror --
  // without this flag that would be reported as a spurious lost connection
  // right after every successful sync.
  const terminatedRef = useRef(false);

  useEffect(() => {
    return () => {
      sourceRef.current?.close();
    };
  }, []);

  async function start() {
    setEvents([]);
    setRunning(true);
    terminatedRef.current = false;
    try {
      await apiPost('/api/sync');
    } catch (e) {
      setEvents([{ event: 'error', message: startErrorMessage(e) }]);
      setRunning(false);
      return;
    }

    const source = new EventSource('/api/sync/stream');
    sourceRef.current = source;
    source.onmessage = (message) => {
      let parsed: SyncEvent;
      try {
        parsed = JSON.parse(message.data);
      } catch {
        terminatedRef.current = true;
        source.close();
        setRunning(false);
        setEvents((previous) => [
          ...previous,
          { event: 'error', message: 'Received an unreadable event from the server.' },
        ]);
        return;
      }
      setEvents((previous) => [...previous, parsed]);
      if (parsed.event === 'complete' || parsed.event === 'error') {
        terminatedRef.current = true;
        source.close();
        setRunning(false);
      }
    };
    source.onerror = () => {
      if (terminatedRef.current) {
        return;
      }
      terminatedRef.current = true;
      source.close();
      setRunning(false);
      setEvents((previous) => [
        ...previous,
        {
          event: 'error',
          message:
            'Lost connection to the sync stream. The sync may still be running on the server -- reload to check.',
        },
      ]);
    };
  }

  return (
    <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
      <Panel title="Sync">
        <div className="flex items-center justify-between" style={{ gap: 'var(--space-5)' }}>
          <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)', margin: 0 }}>
            Sync is the only action that contacts Binance. Everything else reads
            local data.
          </p>
          <button
            onClick={start}
            disabled={running}
            className="font-ui shrink-0 transition-colors"
            style={{
              background: running ? 'var(--surface-2)' : 'var(--action)',
              color: running ? 'var(--text-secondary)' : '#FFFFFF',
              cursor: running ? 'not-allowed' : 'pointer',
              borderRadius: 'var(--radius-control)',
              padding: 'var(--space-3) var(--space-5)',
              fontSize: '14px',
              fontWeight: 500,
            }}
          >
            {running ? 'Syncing…' : 'Start sync'}
          </button>
        </div>
      </Panel>

      {events.length > 0 && (
        <Panel title="Progress">
          <ul
            className="flex flex-col font-mono"
            style={{
              gap: 'var(--space-2)',
              fontSize: '12px',
              lineHeight: 1.5,
              maxHeight: '480px',
              overflowY: 'auto',
              margin: 0,
              padding: 0,
              listStyle: 'none',
            }}
          >
            {events.map((event, index) => (
              <li key={index} style={{ color: lineColour(event) }}>
                {lineText(event)}
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
