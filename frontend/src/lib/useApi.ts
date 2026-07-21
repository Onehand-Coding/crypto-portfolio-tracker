import { useCallback, useEffect, useState } from 'react';
import { apiGet } from './api';

/**
 * Fetch-on-mount with explicit error state.
 *
 * Every screen needs the same three states and the branch order matters: an
 * error must be checked before "no data yet", or a failed fetch renders as a
 * loading state that never resolves.
 */
export function useApi<T>(path: string, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    apiGet<T>(path)
      .then((result) => {
        setData(result);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);

  useEffect(reload, [reload]);

  return { data, error, reload };
}

/** Poll while a condition holds — used to follow a running analysis. */
export function usePollWhile(active: boolean, reload: () => void, intervalMs = 2000) {
  useEffect(() => {
    if (!active) return;
    const id = setInterval(reload, intervalMs);
    return () => clearInterval(id);
  }, [active, reload, intervalMs]);
}
