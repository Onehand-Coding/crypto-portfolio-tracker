import type { ReactNode } from 'react';

/** Depth is tonal layering plus a 1px border. No shadows. */
export function Panel({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section
      className="rounded-panel border p-3"
      style={{ background: 'var(--surface-1)', borderColor: 'var(--border)' }}
    >
      {title && (
        <h2
          className="mb-2 font-ui text-xs uppercase tracking-wider"
          style={{ color: 'var(--text-secondary)' }}
        >
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}
