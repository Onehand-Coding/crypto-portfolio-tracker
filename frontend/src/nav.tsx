import type { ReactNode } from 'react';

/**
 * Navigation is two levels, not one flat list.
 *
 * Fifteen links in a single column overflowed the viewport at 1080p and put a
 * scrollbar in the sidebar -- the nav could not show you where you were without
 * scrolling. Sections live in the rail; the screens inside a section become
 * tabs in the top bar. The rail is then a fixed six items at any window height.
 */
export interface NavItem {
  to: string;
  label: string;
}

export interface NavSection {
  id: string;
  label: string;
  icon: ReactNode;
  items: NavItem[];
}

const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" {...stroke}>
      {children}
    </svg>
  );
}

export const NAV_SECTIONS: NavSection[] = [
  {
    id: 'portfolio',
    label: 'Portfolio',
    icon: <Icon><rect x="2" y="2" width="5" height="5" /><rect x="9" y="2" width="5" height="5" />
      <rect x="2" y="9" width="5" height="5" /><rect x="9" y="9" width="5" height="5" /></Icon>,
    items: [
      { to: '/', label: 'Cockpit' },
      { to: '/overview', label: 'Overview' },
      { to: '/realized', label: 'Realized P/L' },
      { to: '/wallets', label: 'Wallets' },
      { to: '/capital', label: 'Capital Flow' },
    ],
  },
  {
    id: 'analyze',
    label: 'Analyze',
    icon: <Icon><path d="M2 13V8M6 13V4M10 13v-6M14 13V2" /></Icon>,
    items: [
      { to: '/market', label: 'Market' },
      { to: '/technical', label: 'Technical' },
    ],
  },
  {
    id: 'strategies',
    label: 'Strategies',
    icon: <Icon><circle cx="8" cy="8" r="6" /><path d="M8 4v4l2.5 2.5" /></Icon>,
    items: [
      { to: '/rebalance', label: 'Rebalancing' },
      { to: '/dca', label: 'DCA' },
      { to: '/profit', label: 'Profit Taking' },
      { to: '/backtest', label: 'Backtesting' },
    ],
  },
  {
    id: 'execute',
    label: 'Execute',
    icon: <Icon><path d="M2 8h9M8 5l3 3-3 3M13 3v10" /></Icon>,
    items: [{ to: '/trade', label: 'Trading' }],
  },
  {
    id: 'data',
    label: 'Data',
    icon: <Icon><ellipse cx="8" cy="4" rx="5.5" ry="2" /><path d="M2.5 4v8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2V4" />
      <path d="M2.5 8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2" /></Icon>,
    items: [
      { to: '/transactions', label: 'Trade Log' },
      { to: '/sync', label: 'Sync' },
      { to: '/reports', label: 'Reports' },
    ],
  },
  {
    id: 'system',
    label: 'System',
    icon: <Icon><circle cx="8" cy="8" r="2.5" />
      <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.1 3.1l1.4 1.4M11.5 11.5l1.4 1.4M12.9 3.1l-1.4 1.4M4.5 11.5l-1.4 1.4" /></Icon>,
    items: [{ to: '/system', label: 'System' }],
  },
];

/**
 * The asset detail screen is reachable from several tables but is not itself a
 * nav destination, so it maps to the section it conceptually belongs to rather
 * than leaving the rail with nothing highlighted.
 */
export function sectionForPath(pathname: string): NavSection {
  if (pathname.startsWith('/assets/')) return NAV_SECTIONS[0];
  const match = NAV_SECTIONS.find((section) =>
    section.items.some((item) => (item.to === '/' ? pathname === '/' : pathname.startsWith(item.to))),
  );
  return match ?? NAV_SECTIONS[0];
}
