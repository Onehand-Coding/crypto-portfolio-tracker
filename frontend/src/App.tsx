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
    <div className="min-h-screen" style={{ background: 'var(--surface-0)' }}>
      {environment && <EnvBanner environment={environment} />}
      <div className="flex">
        <nav className="flex w-48 shrink-0 flex-col gap-1 border-r p-3"
             style={{ borderColor: 'var(--border)' }}>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className="px-2 py-1 font-ui text-sm"
              style={({ isActive }) => ({
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                borderLeft: `2px solid ${isActive ? 'var(--action)' : 'transparent'}`,
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <main className="flex-1 p-4">
          <Routes>
            <Route path="/" element={<Cockpit />} />
            <Route path="/capital" element={<CapitalFlow />} />
            <Route path="/sync" element={<Sync />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
