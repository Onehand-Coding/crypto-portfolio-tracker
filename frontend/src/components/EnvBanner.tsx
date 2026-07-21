import type { Environment } from '../types';

/**
 * Always rendered, in both environments. Testnet uses the warning colour AND
 * the word TESTNET AND the database filename -- three independent signals,
 * because presenting testnet figures as live is the worst failure this UI has.
 */
export function EnvBanner({ environment }: { environment: Environment }) {
  const isTestnet = environment.is_testnet;
  return (
    <div
      className="flex items-center gap-3 border-b px-3 py-1 font-mono text-xs"
      style={{
        borderColor: 'var(--border)',
        background: isTestnet ? 'var(--warning)' : 'var(--surface-1)',
        color: isTestnet ? 'var(--surface-0)' : 'var(--text-secondary)',
      }}
    >
      <span className="font-bold tracking-wider">{environment.label}</span>
      <span>{environment.database_path}</span>
    </div>
  );
}
