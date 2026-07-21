import type { ReactNode } from 'react';

/** Depth is tonal layering plus a 1px border. No shadows. */
export function Panel({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section
      style={{
        background: 'var(--surface-1)',
        borderColor: 'var(--border)',
        borderWidth: '1px',
        borderRadius: 'var(--radius-panel)',
        padding: 'var(--space-5)',
      }}
    >
      {title && (
        <h2
          className="font-ui"
          style={{
            color: 'var(--text-tertiary)',
            fontSize: '11px',
            fontWeight: 500,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            marginBottom: 'var(--space-4)',
          }}
        >
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}
