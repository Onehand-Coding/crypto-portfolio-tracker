import { NavLink, Route, Routes } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { EnvBanner } from './components/EnvBanner';
import { apiGet } from './lib/api';
import type { CockpitResponse, Environment } from './types';
import { Cockpit } from './screens/Cockpit';
import { CapitalFlow } from './screens/CapitalFlow';
import { Sync } from './screens/Sync';
import { Overview } from './screens/Overview';
import { Wallets } from './screens/Wallets';
import { AssetDetail } from './screens/AssetDetail';
import { Rebalance } from './screens/Rebalance';
import { Dca } from './screens/Dca';
import { ProfitTaking } from './screens/ProfitTaking';
import { Technical } from './screens/Technical';
import { Market } from './screens/Market';
import { Backtest } from './screens/Backtest';
import { Trading } from './screens/Trading';
import { Reports } from './screens/Reports';
import { SystemHealth } from './screens/SystemHealth';

/**
 * Grouped so related screens sit together rather than as one flat list. The
 * group order follows how the tool is used: look at where you are, understand
 * the market, decide what to do, then act.
 */
const NAV_GROUPS = [
  {
    label: 'Portfolio',
    items: [
      { to: '/', label: 'Cockpit' },
      { to: '/overview', label: 'Overview' },
      { to: '/wallets', label: 'Wallets' },
      { to: '/capital', label: 'Capital Flow' },
    ],
  },
  {
    label: 'Analyze',
    items: [
      { to: '/market', label: 'Market' },
      { to: '/technical', label: 'Technical' },
    ],
  },
  {
    label: 'Strategies',
    items: [
      { to: '/rebalance', label: 'Rebalancing' },
      { to: '/dca', label: 'DCA' },
      { to: '/profit', label: 'Profit Taking' },
      { to: '/backtest', label: 'Backtesting' },
    ],
  },
  {
    label: 'Execute',
    items: [{ to: '/trade', label: 'Trading' }],
  },
  {
    label: 'Data',
    items: [
      { to: '/sync', label: 'Sync' },
      { to: '/reports', label: 'Reports' },
      { to: '/system', label: 'System' },
    ],
  },
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
          className="flex w-60 shrink-0 flex-col overflow-y-auto border-r"
          style={{
            borderColor: 'var(--border)',
            background: 'var(--surface-1)',
            padding: 'var(--space-4) var(--space-3)',
          }}
        >
          <div style={{ padding: `0 var(--space-3) var(--space-5)` }}>
            <div className="font-ui" style={{ fontSize: '14px', fontWeight: 600 }}>
              Portfolio Tracker
            </div>
            <div className="font-mono"
                 style={{ color: 'var(--text-tertiary)', fontSize: '11px' }}>
              {environment ? environment.label.toLowerCase() : 'connecting…'}
            </div>
          </div>

          {NAV_GROUPS.map((group) => (
            <div key={group.label} className="flex flex-col"
                 style={{ gap: '2px', marginBottom: 'var(--space-4)' }}>
              <span
                className="font-ui"
                style={{
                  color: 'var(--text-tertiary)', fontSize: '10px', fontWeight: 600,
                  letterSpacing: '0.1em', textTransform: 'uppercase',
                  padding: `0 var(--space-3) var(--space-2)`,
                }}
              >
                {group.label}
              </span>
              {group.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className="font-ui text-sm transition-colors"
                  style={({ isActive }) => ({
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    background: isActive ? 'var(--surface-2)' : 'transparent',
                    borderRadius: 'var(--radius-control)',
                    padding: 'var(--space-2) var(--space-3)',
                    fontWeight: isActive ? 500 : 400,
                  })}
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        <main className="min-w-0 flex-1 overflow-y-auto"
              style={{ padding: 'var(--space-6)' }}>
          <div className="mx-auto w-full" style={{ maxWidth: 'var(--content-max)' }}>
            <Routes>
              <Route path="/" element={<Cockpit />} />
              <Route path="/overview" element={<Overview />} />
              <Route path="/wallets" element={<Wallets />} />
              <Route path="/capital" element={<CapitalFlow />} />
              <Route path="/assets/:symbol" element={<AssetDetail />} />
              <Route path="/market" element={<Market />} />
              <Route path="/technical" element={<Technical />} />
              <Route path="/rebalance" element={<Rebalance />} />
              <Route path="/dca" element={<Dca />} />
              <Route path="/profit" element={<ProfitTaking />} />
              <Route path="/backtest" element={<Backtest />} />
              <Route path="/trade" element={<Trading />} />
              <Route path="/sync" element={<Sync />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/system" element={<SystemHealth />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
