import { NavLink, Route, Routes } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { EnvBanner } from './components/EnvBanner';
import { apiGet } from './lib/api';
import type { CockpitResponse, Environment } from './types';
import { Cockpit } from './screens/Cockpit';
import { CapitalFlow } from './screens/CapitalFlow';
import { Sync } from './screens/Sync';

const NAV = [
  { to: '/', label: 'Cockpit' },
  { to: '/capital', label: 'Capital Flow' },
  { to: '/sync', label: 'Sync' },
];

export default function App() {
  const [environment, setEnvironment] = useState<Environment | null>(null);

  useEffect(() => {
    apiGet<CockpitResponse>('/api/portfolio/cockpit')
      .then((data) => setEnvironment(data.environment))
      .catch(() => setEnvironment(null));
  }, []);

  return (
    // h-full + overflow-hidden makes this a fixed shell: the sidebar runs the
    // full window height and <main> is the only thing that scrolls.
    <div className="flex h-full flex-col overflow-hidden"
         style={{ background: 'var(--surface-0)' }}>
      <EnvBanner environment={environment} />
      <div className="flex min-h-0 flex-1">
        <nav
          className="flex w-56 shrink-0 flex-col gap-1 border-r"
          style={{
            borderColor: 'var(--border)',
            background: 'var(--surface-1)',
            padding: 'var(--space-4) var(--space-3)',
          }}
        >
          <span
            className="font-ui"
            style={{
              color: 'var(--text-tertiary)',
              fontSize: '11px',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              padding: `0 var(--space-3) var(--space-3)`,
            }}
          >
            Portfolio
          </span>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className="font-ui text-sm transition-colors"
              style={({ isActive }) => ({
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                // The selected item is a filled row, not a 2px hairline that
                // was invisible next to the browser's focus rectangle.
                background: isActive ? 'var(--surface-2)' : 'transparent',
                borderRadius: 'var(--radius-control)',
                padding: 'var(--space-2) var(--space-3)',
                fontWeight: isActive ? 500 : 400,
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <main className="min-w-0 flex-1 overflow-y-auto"
              style={{ padding: 'var(--space-6)' }}>
          {/* Capped and centred: a table row stretched across 1920px forces the
              eye to travel from the label to its own number. */}
          <div className="mx-auto w-full" style={{ maxWidth: 'var(--content-max)' }}>
            <Routes>
              <Route path="/" element={<Cockpit />} />
              <Route path="/capital" element={<CapitalFlow />} />
              <Route path="/sync" element={<Sync />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
