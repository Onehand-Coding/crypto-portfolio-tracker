import { useEffect, useRef, useState } from 'react';
import { Panel } from '../components/Panel';
import { apiPost, ApiError, NetworkError } from '../lib/api';

interface SyncEvent {
  event: 'progress' | 'complete' | 'error';
  message: string;
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
    <div className="flex flex-col gap-4">
      <Panel title="Sync">
        <p className="mb-3 font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
          Sync is the only action that contacts Binance. Everything else reads
          local data.
        </p>
        <button
          onClick={start}
          disabled={running}
          className="rounded-control px-3 py-1 font-ui text-sm"
          style={{
            background: running ? 'var(--surface-2)' : 'var(--action)',
            color: 'var(--text-primary)',
            cursor: running ? 'not-allowed' : 'pointer',
          }}
        >
          {running ? 'Syncing…' : 'Start sync'}
        </button>
      </Panel>

      {events.length > 0 && (
        <Panel title="Progress">
          <ul className="flex flex-col gap-1 font-mono text-xs">
            {events.map((event, index) => (
              <li
                key={index}
                style={{
                  color: event.event === 'error' ? 'var(--negative)'
                       : event.event === 'complete' ? 'var(--positive)'
                       : 'var(--text-secondary)',
                }}
              >
                {event.event === 'error' ? `error: ${event.message}` : event.message}
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
