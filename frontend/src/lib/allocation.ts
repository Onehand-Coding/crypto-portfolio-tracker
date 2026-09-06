/**
 * Targets are shares of the core sleeve, not of the whole portfolio.
 *
 * The rebalance engine filters live balances to target-allocation members
 * before dividing (portfolio_analyzer.py:243, rebalancing_logic.py:170),
 * and DCA divides by the same core subtotal (strategy.py:271). Any screen
 * comparing a holding against its target must use this denominator too, or
 * the same "35% BTC" target reads two ways at once -- 33% on Strategies,
 * 3% on the Dashboard -- whenever non-core positions are large.
 */
export function coreTotalUsd(
  holdings: { symbol: string; value_usd: number | null }[],
  targets: Record<string, number>,
): number {
  return holdings
    .filter((h) => Object.prototype.hasOwnProperty.call(targets, h.symbol))
    .reduce((sum, h) => sum + (h.value_usd ?? 0), 0);
}
