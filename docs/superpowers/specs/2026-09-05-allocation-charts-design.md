# Allocation Chart Readability Design

**Status:** Approved by owner (2026-09-05). Sections 1 (P/L chart) and 2 (donut)
accepted as presented, including the 1% "Others" grouping.

**Scope:** `frontend/src/screens/Allocation.tsx` only, plus its missing test
   file. No backend, no schema, no other screen. The figures do not change anywhere
   — same data, honest scales.

**Problem (grounded in live data 2026-09-05, total $175.02):** BTC $149.47
(85.4%) + ETH $24.58 (14.0%) plus ten dust holdings (SOL $0.03 down to ONDO
$0.00). Every visual assumed a balanced portfolio:
- P/L bar chart: BTC -$34.50 overflowed an axis ending at -$24 (clipped bar =
  wrong number); ten assets within ±$0.15 rendered sub-pixel; hover tooltip
  used the default dark-grey label on a near-black surface (unreadable).
- Donut: dust slices rendered as near-dots detached from the ring
  (`paddingAngle` on 12 slices); ring sorted by value while the drift table is
  sorted by target, with no colour dots in the table, so ring and rows cannot
  be matched.

## Section 1 — Unrealized P/L chart

1. **Fitting domain.** New pure helper `plDomain(values: number[]): [number, number]`
   returns `[Math.min(0, min - pad), Math.max(0, max + pad)]` with `pad` = 5% of
   the data span; when the span is 0 (all values equal, e.g. all zero) the
   domain is `[min(0, v - 1), max(0, v + 1)]` so it never degenerates to a
   zero-width `[0, 0]`. Zero baseline always present, data always
   inside. YAxis gets `domain={plDomain(...)}` (keep `allowDataOverflow` default
   false). Rationale for a helper over recharts auto: the default demonstrably
   misfit this data; an owned function is unit-testable with real magnitudes
   (min -$34.50, max +$0.15 must both lie inside, zero included).
2. **Minimum visible bar.** `minPointSize={3}` on the Bar. Exact-zero values
   stay zero-height (recharts skips min-size for 0 — pin in test).
3. **Legible tooltip and axis.** Tooltip gains `labelStyle={{ color:
   'var(--text-primary)' }}` and `itemStyle={{ color: 'var(--text-secondary)' }}`;
   keep the existing `contentStyle`, cursor, and `Unrealized` formatter text
   unchanged. XAxis: `interval={0}`, `angle={-35}`, `textAnchor="end"`,
   `height={56}`; BarChart bottom margin 0 → 8. Ticks stay 11px tertiary.

## Section 2 — Donut and drift table

1. **Shared symbol→colour map.** Build `colourOf: Map<string, string>` once from
   pie order (post-grouping, so "Others" owns a colour too) using the existing
   `SLICE_COLOURS` wrap-around. Drift-table Asset cell renders a 8px dot in the
   symbol's colour before the name. Ring and table agree by construction;
   table keeps its target-desc sort (the rebalancing view).
2. **"Others" grouping.** Holdings below `DUST_THRESHOLD_PCT = 1` (percent of
   total portfolio value, named constant next to `SLICE_COLOURS`) merge into
   one slice named `Others (n)` with summed value. Tooltip formatter shows the
   summed value (existing `Value` formatter, unchanged). The drift table still
   lists every asset individually — grouping affects the ring only. Edge cases:
   no holdings below threshold → no Others slice, colours unchanged; ALL
   holdings below threshold (empty-ish portfolio) → single "Others (n)" slice,
   still truthful; threshold measured against `total_value_usd`, and unpriced
   (null value) holdings are excluded from the ring exactly as today.
3. **Missing fetch-fail test.** New `frontend/src/screens/Allocation.test.tsx`
   with the convention-required failure test (fetch rejects → visible error,
   no permanent loading), plus: `plDomain` unit cases (mixed signs, all
   positive, all negative, all zero, real magnitudes), "Others" grouping cases
   (dust grouped with count in name, no-dust passthrough, all-dust single
   slice), and a render test that every drift-row asset shows its colour dot.

## Non-goals

- No change to figures, ordering semantics, or the drift math.
- No backend/API/schema/type changes (`CockpitResponse` already carries
  everything).
- No other screen (Backtest/Technical chart treatment is a separate decision).
- No new dependencies, no chart-library swap.
