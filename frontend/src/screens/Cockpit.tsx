import { useEffect, useState } from 'react';
import { HoldingsTable } from '../components/HoldingsTable';
import { Metric } from '../components/Metric';
import { Panel } from '../components/Panel';
import { StalenessNote } from '../components/StalenessNote';
import { apiGet } from '../lib/api';
import { formatPercent, formatSigned, formatUsd } from '../lib/format';
import type { AccountingBasis, CockpitResponse } from '../types';

function BasisBlock({ basis, denominator }: { basis: AccountingBasis; denominator: string }) {
  return (
    <div
      className="flex flex-col"
      style={{
        gap: 'var(--space-2)',
        background: 'var(--surface-2)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-panel)',
        padding: 'var(--space-4)',
      }}
    >
      <Metric
        label={basis.label}
        value={`${formatSigned(basis.pl_usd)}  (${formatPercent(basis.pl_percent)})`}
        signal={basis.pl_usd}
        sub={denominator}
      />
      <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px' }}>
        {basis.question}
      </span>
    </div>
  );
}

export function Cockpit() {
  const [data, setData] = useState<CockpitResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<CockpitResponse>('/api/portfolio/cockpit')
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  // Checked before the "no data yet" loading branch: a fetch failure must
  // always surface as a visible, legible error -- never a blank panel and
  // never a loading state stuck forever.
  if (error) {
    return (
      <Panel title="Cockpit">
        <p className="font-mono text-sm" style={{ color: 'var(--negative)' }}>
          Failed to load cockpit data: {error}
        </p>
      </Panel>
    );
  }
  if (!data) {
    return (
      <Panel title="Cockpit">
        <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }}>
          Loading…
        </p>
      </Panel>
    );
  }

  if (!data.has_data) {
    return (
      <Panel title="Cockpit">
        <p className="font-ui text-sm" style={{ color: 'var(--warning)' }}>
          No data yet — run a sync to populate the portfolio.
        </p>
      </Panel>
    );
  }

  return (
    <div className="flex flex-col" style={{ gap: 'var(--space-4)' }}>
      <Panel>
        <div className="flex items-start justify-between">
          <div className="flex flex-col" style={{ gap: 'var(--space-1)' }}>
            <span
              className="font-ui"
              style={{
                color: 'var(--text-tertiary)',
                fontSize: '11px',
                fontWeight: 500,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}
            >
              Portfolio value
            </span>
            <span
              className="font-mono"
              style={{ fontSize: '44px', lineHeight: 1.1, letterSpacing: '-0.02em' }}
            >
              {formatUsd(data.total_value_usd)}
            </span>
          </div>
          <StalenessNote staleness={data.staleness} />
        </div>

        {/*
          The total sums only the holdings that could be priced, so an unpriced
          position makes it an understatement of unknown size. Saying so beats
          presenting a confident figure that is quietly missing a holding.
        */}
        {data.unpriced_count > 0 && (
          <p className="mt-2 font-ui text-sm" style={{ color: 'var(--warning)' }}>
            Understated: {data.unpriced_count} holding
            {data.unpriced_count === 1 ? '' : 's'} could not be priced and
            {data.unpriced_count === 1 ? ' is' : ' are'} excluded from this total.
          </p>
        )}

        {/*
          Both bases, side by side, each with its denominator and its question.
          They are computed from different sources and routinely differ several
          fold; rendering them as one number would be a lie.
        */}
        <div className="grid grid-cols-2"
             style={{ gap: 'var(--space-4)', marginTop: 'var(--space-5)' }}>
          <BasisBlock
            basis={data.net_invested}
            denominator={`on ${formatUsd(data.net_invested.basis_usd)} net in`}
          />
          <BasisBlock
            basis={data.fifo}
            denominator={`on ${formatUsd(data.fifo.basis_usd)} cost basis`}
          />
        </div>
      </Panel>

      <Panel title="Holdings">
        <HoldingsTable holdings={data.holdings} />
      </Panel>
    </div>
  );
}
