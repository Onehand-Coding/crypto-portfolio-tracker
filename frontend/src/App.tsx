import { NavLink, Route, Routes, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { EnvBanner } from './components/EnvBanner';
import { StalenessNote } from './components/StalenessNote';
import { NAV_SECTIONS, sectionForPath } from './nav';
import { apiGet } from './lib/api';
import type { CockpitResponse } from './types';
import { Cockpit } from './screens/Cockpit';
import { CapitalFlow } from './screens/CapitalFlow';
import { Sync } from './screens/Sync';
import { Overview } from './screens/Overview';
import { Realized } from './screens/Realized';
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
import { TradeLog } from './screens/TradeLog';
import { SystemHealth } from './screens/SystemHealth';
import { Settings } from './screens/Settings';

/** The rail: six sections, fixed, never scrolls at any window height. */
function Rail() {
  const { pathname } = useLocation();
  const active = sectionForPath(pathname);

  return (
    <nav
      className="flex shrink-0 flex-col"
      style={{
        width: '188px',
        borderRight: '1px solid var(--border)',
        background: 'var(--surface-1)',
        padding: 'var(--space-3) var(--space-2)',
      }}
    >
      <div className="flex items-center"
           style={{ gap: 'var(--space-3)', padding: `var(--space-2) var(--space-3) var(--space-5)` }}>
        <div className="flex items-center justify-center shrink-0"
             style={{ width: '26px', height: '26px', borderRadius: 'var(--radius-control)',
                      background: 'var(--action)', color: '#fff', fontSize: '13px',
                      fontWeight: 700, letterSpacing: '-0.03em' }}>
          ₿
        </div>
        <div className="flex flex-col" style={{ lineHeight: 1.25 }}>
          <span className="font-ui" style={{ fontSize: '13px', fontWeight: 600 }}>
            Portfolio Tracker
          </span>
          <span className="font-mono" style={{ color: 'var(--text-tertiary)', fontSize: '10px' }}>
            control room
          </span>
        </div>
      </div>

      <div className="flex flex-col" style={{ gap: '2px' }}>
        {NAV_SECTIONS.map((section) => {
          const isActive = section.id === active.id;
          return (
            <NavLink
              key={section.id}
              to={section.items[0].to}
              className="flex items-center font-ui transition-colors"
              style={{
                position: 'relative',
                gap: 'var(--space-3)',
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                background: isActive ? 'var(--surface-2)' : 'transparent',
                borderRadius: 'var(--radius-control)',
                padding: 'var(--space-2) var(--space-3)',
                fontSize: '13px',
                fontWeight: isActive ? 500 : 400,
              }}
            >
              {/* Sharp 0px selection bar on the leading edge, per the design system. */}
              {isActive && (
                <span style={{ position: 'absolute', left: 0, top: '6px', bottom: '6px',
                               width: '2px', background: 'var(--action)' }} />
              )}
              <span style={{ color: isActive ? 'var(--action)' : 'var(--text-tertiary)',
                             display: 'flex' }}>
                {section.icon}
              </span>
              {section.label}
            </NavLink>
          );
        })}
      </div>
    </nav>
  );
}

/** Section tabs plus global status. Replaces the per-screen page header. */
function TopBar({ cockpit }: { cockpit: CockpitResponse | null }) {
  const { pathname } = useLocation();
  const section = sectionForPath(pathname);

  return (
    <header
      className="flex shrink-0 items-center justify-between"
      style={{
        height: '44px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--surface-1)',
        padding: '0 var(--space-4)',
        gap: 'var(--space-4)',
      }}
    >
      <div className="flex items-center" style={{ gap: '2px', minWidth: 0, overflow: 'hidden' }}>
        {section.items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className="font-ui transition-colors"
            style={({ isActive }) => ({
              color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
              background: isActive ? 'var(--surface-2)' : 'transparent',
              borderRadius: 'var(--radius-control)',
              padding: 'var(--space-1) var(--space-3)',
              fontSize: '13px',
              fontWeight: isActive ? 500 : 400,
              whiteSpace: 'nowrap',
            })}
          >
            {item.label}
          </NavLink>
        ))}
      </div>

      <div className="flex shrink-0 items-center" style={{ gap: 'var(--space-3)' }}>
        {cockpit && (
          <span
            className="font-mono"
            style={{
              fontSize: '10px', fontWeight: 700, letterSpacing: '0.08em',
              padding: '3px var(--space-2)', borderRadius: 'var(--radius-control)',
              color: cockpit.environment.is_testnet ? 'var(--warning)' : 'var(--positive)',
              background: cockpit.environment.is_testnet
                ? 'color-mix(in srgb, var(--warning) 14%, transparent)'
                : 'color-mix(in srgb, var(--positive) 14%, transparent)',
            }}
          >
            {cockpit.environment.label}
          </span>
        )}
        {cockpit && <StalenessNote staleness={cockpit.staleness} />}
      </div>
    </header>
  );
}

export default function App() {
  const [cockpit, setCockpit] = useState<CockpitResponse | null>(null);

  useEffect(() => {
    apiGet<CockpitResponse>('/api/portfolio/cockpit')
      .then(setCockpit)
      .catch(() => setCockpit(null));
  }, []);

  return (
    // h-full + overflow-hidden makes this a fixed shell: the rail runs the full
    // window height and <main> is the only thing that scrolls.
    <div className="flex h-full flex-col overflow-hidden"
         style={{ background: 'var(--surface-0)' }}>
      <div className="flex min-h-0 flex-1">
        <Rail />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar cockpit={cockpit} />
          <main className="min-w-0 flex-1 overflow-y-auto"
                style={{ padding: 'var(--space-4)' }}>
            <div className="mx-auto w-full" style={{ maxWidth: 'var(--content-max)' }}>
              <Routes>
                <Route path="/" element={<Cockpit />} />
                <Route path="/overview" element={<Overview />} />
                <Route path="/realized" element={<Realized />} />
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
                <Route path="/transactions" element={<TradeLog />} />
                <Route path="/sync" element={<Sync />} />
                <Route path="/reports" element={<Reports />} />
                <Route path="/system" element={<SystemHealth />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>
      {/* Environment lives at the bottom as a persistent status strip: always on
          screen, never competing with the data for the top of the viewport. */}
      <EnvBanner environment={cockpit?.environment ?? null} />
    </div>
  );
}
