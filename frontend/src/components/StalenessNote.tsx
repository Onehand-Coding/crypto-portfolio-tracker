import type { Staleness } from '../types';

/** Staleness is displayed, never hidden behind a spinner. */
export function StalenessNote({ staleness }: { staleness: Staleness }) {
  if (staleness.age_seconds === null) {
    return (
      <span className="font-mono text-xs" style={{ color: 'var(--warning)' }}>
        never synced
      </span>
    );
  }
  const minutes = Math.round(staleness.age_seconds / 60);
  const text = minutes < 1 ? 'synced just now' : `synced ${minutes}m ago`;
  return (
    <span
      className="font-mono text-xs"
      style={{ color: staleness.is_stale ? 'var(--warning)' : 'var(--text-secondary)' }}
    >
      {text}
    </span>
  );
}
