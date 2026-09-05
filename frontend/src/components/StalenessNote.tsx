import type { Staleness } from '../types';

/** Staleness is displayed, never hidden behind a spinner.
    The verb defaults to sync age (the top bar). Analysis-run ages pass
    verb="run" so two numbers on one screen never read as the same fact. */
export function StalenessNote({ staleness, verb = 'synced' }: {
  staleness: Staleness;
  verb?: string;
}) {
  if (staleness.age_seconds === null) {
    return (
      <span className="font-mono text-xs" style={{ color: 'var(--warning)' }}>
        never synced
      </span>
    );
  }
  const minutes = Math.round(staleness.age_seconds / 60);
  const text = minutes < 1 ? `${verb} just now` : `${verb} ${minutes}m ago`;
  return (
    <span
      className="font-mono text-xs"
      style={{ color: staleness.is_stale ? 'var(--warning)' : 'var(--text-secondary)' }}
    >
      {text}
    </span>
  );
}
