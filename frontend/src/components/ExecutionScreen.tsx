import type { ReactNode } from 'react';
import { ScreenHeader } from './Screen';
import { TradingStatusBanner } from './TradingStatusBanner';
import type { ExecutionStatus, Staleness } from '../types';

/**
 * Shared shell for every screen that can place orders. It owns the vertical
 * order -- ScreenHeader, then the trading posture strip, then the screen's
 * own content -- so no screen can drift the banner below the figures it
 * qualifies. Screens keep their early returns (loading / error / empty);
 * only the main return wraps in this.
 */
export function ExecutionScreen({
  title, subtitle, status, staleness, children,
}: {
  title: string;
  subtitle?: string;
  status: ExecutionStatus | null;
  staleness?: Staleness;
  children: ReactNode;
}) {
  return (
    <>
      <ScreenHeader title={title} subtitle={subtitle} staleness={staleness} />
      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <TradingStatusBanner status={status} />
        {children}
      </div>
    </>
  );
}
