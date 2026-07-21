import type { Environment } from '../types';

/**
 * Always rendered, in both environments. Testnet uses the warning colour AND
 * the word TESTNET AND the database filename -- three independent signals,
 * because presenting testnet figures as live is the worst failure this UI has.
 *
 * When environment is null (still loading, or the API could not be reached),
 * the banner must never simply be absent -- absence reads as "nothing
 * unusual". Instead it renders an explicit unknown state using the negative
 * colour, since an unverifiable environment is a hazard, not a warning.
 */
export function EnvBanner({ environment }: { environment: Environment | null }) {
  if (!environment) {
    return (
      <div
        className="flex shrink-0 items-center gap-3 border-t font-mono"
      // shrink-0: inside the fixed-height shell a flex child will otherwise
      // compress, and the environment label is the last thing that should.
        style={{
          borderColor: 'var(--border)',
          background: 'var(--negative)',
          color: 'var(--surface-0)',
          padding: '4px var(--space-4)',
          fontSize: '10px',
        }}
      >
        <span className="font-bold tracking-wider">ENVIRONMENT UNKNOWN</span>
        <span>cannot reach API — do not trust displayed figures</span>
      </div>
    );
  }

  const isTestnet = environment.is_testnet;
  return (
    <div
      className="flex shrink-0 items-center gap-3 border-t font-mono"
      // shrink-0: inside the fixed-height shell a flex child will otherwise
      // compress, and the environment label is the last thing that should.
      style={{
        borderColor: 'var(--border)',
        background: isTestnet ? 'var(--warning)' : 'var(--surface-1)',
        color: isTestnet ? 'var(--surface-0)' : 'var(--text-tertiary)',
        padding: '4px var(--space-4)',
        fontSize: '10px',
      }}
    >
      <span className="font-bold tracking-wider">{environment.label}</span>
      <span>{environment.database_path}</span>
    </div>
  );
}
