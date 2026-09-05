# Dashboard and Overview Consolidation

## Goal

Make the root portfolio screen the Dashboard, combining Cockpit's actionable
portfolio view with Overview's historical performance information. Remove the
separate Overview navigation destination without breaking existing deep links.

## Dashboard Composition

The Dashboard keeps the current Cockpit content and order:

1. Current portfolio value and the two accounting models: net invested basis
   and FIFO basis.
2. Performance chart beside allocation drift.
3. Alerts and review items.
4. Holdings table.

The Performance panel absorbs Overview's history-specific information:

- Keep the 1M, 3M, 1Y, and All range selector.
- Plot portfolio value as the existing area series.
- Plot the historical cost-basis series as a dashed line labelled "FIFO cost
  basis at snapshot". It is the stored basis at each sync, not a current
  holdings-table calculation.
- Show only the unique history KPI, "Change since first snapshot", next to the
  chart. Do not duplicate current portfolio value or current FIFO basis from
  the Dashboard's main KPI band.

The selected range applies to both series and the chart's history KPI. Missing
or insufficient snapshots retain the current explicit loading, error, and
not-enough-data states; missing values are never rendered as zero.

## Navigation and Routing

- Rename the Portfolio section's "Cockpit" navigation label to "Dashboard".
- Remove the visible "Overview" tab.
- The root route continues to render the consolidated Dashboard.
- `/overview` remains a client-side redirect to `/` for saved links and browser
  history.

## Implementation Boundaries

No API, schema, or database changes are required. The existing
`/api/portfolio/cockpit` and `/api/overview` responses provide all required
data. The screen component may keep its existing `Cockpit` export initially to
avoid unrelated churn; user-facing labels and navigation use "Dashboard".

## Tests

- Update Cockpit screen tests to cover the historical cost-basis line, the
  unique history-change KPI, and an explicit failed overview request.
- Update route/navigation tests to prove Dashboard is visible, Overview is not
  a tab, and `/overview` redirects to the consolidated root screen.
- Keep a failure-fetch screen test so the Dashboard does not mask a failed API
  request as empty or zero data.
