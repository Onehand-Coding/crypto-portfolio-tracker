import { useEffect, useState } from 'react';
import { apiGet } from './api';
import type { SyncStatus } from '../types';

/**
 * The single source of truth for "how old is the synced data", polled so it
 * cannot freeze. Consumed by the app shell's top bar; screens must not fetch
 * their own metrics-cache age (their per-endpoint staleness payloads stay for
 * analysis-run ages only, via StalenessNote with a verb).
 *
 * A failed poll keeps the last known value rather than blanking: the
 * EnvBanner already covers an unreachable API, and a flickering age would
 * read as activity.
 */
export function useSyncStatus(pollMs = 30000): SyncStatus | null {
  const [data, setData] = useState<SyncStatus | null>(null);

  useEffect(() => {
    let stop = false;
    const load = () => {
      apiGet<SyncStatus>('/api/sync/status')
        .then((result) => {
          if (!stop) setData(result);
        })
        .catch(() => {
          // Keep the last known age; see docstring.
        });
    };
    load();
    const id = setInterval(load, pollMs);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [pollMs]);

  return data;
}
