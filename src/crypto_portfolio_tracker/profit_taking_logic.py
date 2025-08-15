"""
Profit Taking Logic Module

This module implements a professional-grade profit-taking strategy that:
1. Only operates when the portfolio is balanced (all assets show HOLD signals)
2. Takes profits from unrealized gains only (not principal)
3. Uses multi-factor analysis to identify optimal profit-taking opportunities
4. Maintains portfolio balance after profit-taking
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import logging
from dataclasses import dataclass
from .crypto_trend_analyzer import CryptoTrendAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ProfitOpportunity:
    """Represents a profit-taking opportunity for a specific asset."""

    symbol: str
    unrealized_gain_usd: float
    unrealized_gain_pct: float
    opportunity_score: float
    rsi_score: float
    pl_score: float
    resistance_score: float
    market_context_score: float
    current_price: float
    support_level: float
    resistance_level: float
    reasons: List[str]


class ProfitTakingAnalyzer:
    """
    Analyzes portfolio holdings to identify optimal profit-taking opportunities.

    Uses a multi-factor scoring system:
    - Unrealized P/L (40%): Primary driver - asset must be significantly in profit
    - RSI Analysis (25%): Momentum and overbought conditions
    - Resistance Proximity (25%): Technical analysis - near resistance levels
    - Market Context (10%): Broader market conditions (Bitcoin sentiment)
    """

    def __init__(self, config: Dict[str, Any], analyzer: CryptoTrendAnalyzer):
        self.config = config
        self.analyzer = analyzer
        self.profit_config = config.get("profit_taking", {})

        # Scoring weights (must sum to 1.0)
        self.weights = {
            "unrealized_pl": 0.40,  # Primary driver
            "rsi": 0.25,  # Momentum/overbought
            "resistance": 0.25,  # Technical resistance
            "market_context": 0.10,  # Broader market
        }

        # Thresholds
        self.min_opportunity_score = self.profit_config.get("min_opportunity_score", 60)
        self.min_unrealized_gain_pct = self.profit_config.get(
            "min_unrealized_gain_pct", 15.0
        )
        self.min_unrealized_gain_usd = self.profit_config.get(
            "min_unrealized_gain_usd", 10.0
        )
        self.max_gain_take_pct = self.profit_config.get("max_gain_take_pct", 50)

        # Cache for reused data
        self._cached_swing_report = None
        self._cached_long_term_report = None
        self._cached_market_context = None

    async def analyze_profit_opportunities(
        self, core_holdings_df: pd.DataFrame, rebalance_suggestions_df: pd.DataFrame
    ) -> List[ProfitOpportunity]:
        """
        Analyzes core holdings to identify profit-taking opportunities.

        Args:
            core_holdings_df: DataFrame with current holdings and cost basis
            rebalance_suggestions_df: Current rebalancing suggestions

        Returns:
            List of ProfitOpportunity objects for assets that meet criteria
        """
        # Validate pre-conditions
        if not self._validate_preconditions(rebalance_suggestions_df):
            return []

        opportunities = []

        # Generate trend reports for analysis
        swing_report = await self.analyzer.generate_report("swing")
        long_term_report = await self.analyzer.generate_report("long_term")

        if not swing_report or not long_term_report:
            logger.error("Could not generate trend reports for profit-taking analysis")
            return []

        # Get Bitcoin context for market scoring
        btc_analysis = swing_report.get("benchmark_analysis", {})
        market_context_factor = self._calculate_market_context_score(btc_analysis)

        # Analyze each core holding
        for _, holding in core_holdings_df.iterrows():
            symbol = holding.get("symbol", "")
            if symbol in ["USDT", "BUSD", "USDC"]:  # Skip stablecoins
                continue

            opportunity = await self._analyze_single_asset(
                holding, swing_report, long_term_report, market_context_factor
            )

            if (
                opportunity
                and opportunity.opportunity_score >= self.min_opportunity_score
            ):
                opportunities.append(opportunity)

        # Sort by opportunity score (highest first)
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)

        return opportunities

    async def analyze_profit_opportunities_optimized(
        self,
        core_holdings_df: pd.DataFrame,
        rebalance_suggestions_df: pd.DataFrame,
        cached_data: Optional[Dict[str, Any]] = None,
    ) -> List[ProfitOpportunity]:
        """
        Optimized version that can reuse trend analysis data from rebalancing workflow.

        Args:
            core_holdings_df: DataFrame with current holdings and cost basis
            rebalance_suggestions_df: Current rebalancing suggestions
            cached_data: Optional dict with pre-computed trend analysis data:
                        {"swing_report": dict, "long_term_report": dict}

        Returns:
            List of ProfitOpportunity objects for assets that meet criteria
        """
        # Validate pre-conditions
        if not self._validate_preconditions(rebalance_suggestions_df):
            return []

        opportunities = []

        # Use cached data if available, otherwise generate fresh reports
        if (
            cached_data
            and "swing_report" in cached_data
            and "long_term_report" in cached_data
        ):
            logger.info(
                "Using cached trend analysis data for profit-taking optimization"
            )
            swing_report = cached_data["swing_report"]
            long_term_report = cached_data["long_term_report"]

            # Cache the reports for potential reuse
            self._cached_swing_report = swing_report
            self._cached_long_term_report = long_term_report
        else:
            # Check if we have cached reports from a recent analysis
            if self._cached_swing_report and self._cached_long_term_report:
                logger.info("Reusing previously cached trend analysis data")
                swing_report = self._cached_swing_report
                long_term_report = self._cached_long_term_report
            else:
                # Generate fresh reports
                logger.info("Generating fresh trend analysis data for profit-taking")
                swing_report = await self.analyzer.generate_report("swing")
                long_term_report = await self.analyzer.generate_report("long_term")

                # Cache for future use
                self._cached_swing_report = swing_report
                self._cached_long_term_report = long_term_report

        if not swing_report or not long_term_report:
            logger.error(
                "Could not generate or retrieve trend reports for profit-taking analysis"
            )
            return []

        # Get Bitcoin context for market scoring (cache this too)
        if self._cached_market_context is None:
            btc_analysis = swing_report.get("benchmark_analysis", {})
            self._cached_market_context = self._calculate_market_context_score(
                btc_analysis
            )

        market_context_factor = self._cached_market_context

        # Analyze each core holding
        for _, holding in core_holdings_df.iterrows():
            symbol = holding.get("symbol", "")
            if symbol in ["USDT", "BUSD", "USDC"]:  # Skip stablecoins
                continue

            opportunity = await self._analyze_single_asset(
                holding, swing_report, long_term_report, market_context_factor
            )

            if (
                opportunity
                and opportunity.opportunity_score >= self.min_opportunity_score
            ):
                opportunities.append(opportunity)

        # Sort by opportunity score (highest first)
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)

        logger.info(
            f"Optimized analysis found {len(opportunities)} profit-taking opportunities"
        )
        return opportunities

    def clear_cache(self):
        """
        Clears cached analysis data. Call this when you want to force fresh analysis.
        """
        logger.debug("Clearing profit-taking analyzer cache")
        self._cached_swing_report = None
        self._cached_long_term_report = None
        self._cached_market_context = None

    def _validate_preconditions(self, rebalance_suggestions_df: pd.DataFrame) -> bool:
        """
        Validates that all preconditions are met for profit-taking.

        Returns True only if ALL assets show HOLD signals (portfolio is balanced).
        """
        if rebalance_suggestions_df.empty:
            logger.info(
                "No rebalancing suggestions available - cannot proceed with profit-taking"
            )
            return False

        # Check if ALL signals are HOLD
        non_hold_signals = rebalance_suggestions_df[
            rebalance_suggestions_df["Signal"] != "HOLD"
        ]

        if not non_hold_signals.empty:
            logger.info(
                f"Portfolio rebalancing needed for {len(non_hold_signals)} assets. "
                "Profit-taking is only available when all assets show HOLD signals."
            )
            return False

        logger.info("✅ Portfolio is balanced - profit-taking analysis can proceed")
        return True

    async def _analyze_single_asset(
        self,
        holding: pd.Series,
        swing_report: Dict[str, Any],
        long_term_report: Dict[str, Any],
        market_context_factor: float,
    ) -> Optional[ProfitOpportunity]:
        """Analyzes a single asset for profit-taking opportunity."""
        symbol = holding.get("symbol", "")
        current_value = holding.get("value_usd", 0)
        cost_basis = holding.get("cost_basis_total", 0)
        unrealized_pl_usd = holding.get("unrealized_pl_usd", 0)

        # Must have unrealized gains
        if unrealized_pl_usd <= 0:
            return None

        # Calculate unrealized gain percentage
        unrealized_gain_pct = (
            (unrealized_pl_usd / cost_basis * 100) if cost_basis > 0 else 0
        )

        # Must meet minimum thresholds
        if (
            unrealized_gain_pct < self.min_unrealized_gain_pct
            or unrealized_pl_usd < self.min_unrealized_gain_usd
        ):
            return None

        # Get technical analysis data
        yf_symbol = f"{symbol}-USD"
        swing_analysis = swing_report.get("coin_analyses", {}).get(yf_symbol, {})
        long_term_analysis = long_term_report.get("coin_analyses", {}).get(
            yf_symbol, {}
        )

        if not swing_analysis or not long_term_analysis:
            logger.warning(f"No technical analysis data available for {symbol}")
            return None

        # Calculate individual scores
        pl_score = self._calculate_pl_score(unrealized_gain_pct)
        rsi_score = self._calculate_rsi_score(swing_analysis, long_term_analysis)
        resistance_score = self._calculate_resistance_score(swing_analysis)

        # Calculate composite opportunity score
        opportunity_score = (
            pl_score * self.weights["unrealized_pl"]
            + rsi_score * self.weights["rsi"]
            + resistance_score * self.weights["resistance"]
            + market_context_factor * self.weights["market_context"]
        ) * 100  # Convert to 0-100 scale

        # Build reasons list
        reasons = self._build_reasons_list(
            unrealized_gain_pct,
            pl_score,
            rsi_score,
            resistance_score,
            market_context_factor,
            swing_analysis,
        )

        return ProfitOpportunity(
            symbol=symbol,
            unrealized_gain_usd=unrealized_pl_usd,
            unrealized_gain_pct=unrealized_gain_pct,
            opportunity_score=opportunity_score,
            rsi_score=rsi_score * 100,
            pl_score=pl_score * 100,
            resistance_score=resistance_score * 100,
            market_context_score=market_context_factor * 100,
            current_price=swing_analysis.get("current_price", 0),
            support_level=swing_analysis.get("support_level", 0),
            resistance_level=swing_analysis.get("resistance_level", 0),
            reasons=reasons,
        )

    def _calculate_pl_score(self, unrealized_gain_pct: float) -> float:
        """
        Calculates P/L score based on unrealized gain percentage.

        Returns score between 0.0 and 1.0
        """
        if unrealized_gain_pct < 15:
            return 0.0
        elif unrealized_gain_pct < 25:
            return 0.3
        elif unrealized_gain_pct < 50:
            return 0.6
        elif unrealized_gain_pct < 100:
            return 0.8
        else:
            return 1.0

    def _calculate_rsi_score(
        self, swing_analysis: Dict, long_term_analysis: Dict
    ) -> float:
        """
        Calculates RSI-based score considering both swing and long-term RSI.

        Returns score between 0.0 and 1.0
        """
        swing_rsi = swing_analysis.get("rsi", 50)
        long_term_rsi = long_term_analysis.get("rsi", 50)

        # Weight swing RSI more heavily (70% vs 30%)
        combined_rsi = swing_rsi * 0.7 + long_term_rsi * 0.3

        if combined_rsi > 80:
            return 1.0  # Extremely overbought
        elif combined_rsi > 70:
            return 0.8  # Overbought
        elif combined_rsi > 60:
            return 0.5  # Moderately overbought
        elif combined_rsi > 50:
            return 0.2  # Neutral-bullish
        else:
            return 0.0  # Not overbought

    def _calculate_resistance_score(self, swing_analysis: Dict) -> float:
        """
        Calculates resistance proximity score.

        Returns score between 0.0 and 1.0
        """
        current_price = swing_analysis.get("current_price", 0)
        resistance_level = swing_analysis.get("resistance_level", 0)

        if current_price == 0 or resistance_level == 0:
            return 0.0

        # Calculate how close we are to resistance
        distance_to_resistance = (resistance_level - current_price) / resistance_level

        if distance_to_resistance < 0:
            # Already above resistance
            return 1.0
        elif distance_to_resistance < 0.02:  # Within 2%
            return 0.9
        elif distance_to_resistance < 0.05:  # Within 5%
            return 0.7
        elif distance_to_resistance < 0.10:  # Within 10%
            return 0.4
        elif distance_to_resistance < 0.15:  # Within 15%
            return 0.2
        else:
            return 0.0

    def _calculate_market_context_score(self, btc_analysis: Dict) -> float:
        """
        Calculates market context score based on Bitcoin analysis.

        Returns score between 0.0 and 1.0
        """
        if not btc_analysis:
            return 0.5  # Neutral if no data

        btc_rsi = btc_analysis.get("rsi", 50)
        btc_conditions = btc_analysis.get("active_conditions", [])

        # Check for overbought conditions in Bitcoin
        if "RSI Overbought" in btc_conditions:
            return 1.0
        elif btc_rsi > 70:
            return 0.8
        elif btc_rsi > 60:
            return 0.6
        elif btc_rsi > 50:
            return 0.5
        else:
            return 0.3  # Bearish market context

    def _build_reasons_list(
        self,
        unrealized_gain_pct: float,
        pl_score: float,
        rsi_score: float,
        resistance_score: float,
        market_context_factor: float,
        swing_analysis: Dict,
    ) -> List[str]:
        """Builds a human-readable list of reasons for the profit-taking suggestion."""
        reasons = []

        # P/L reasons
        if unrealized_gain_pct > 100:
            reasons.append(f"Exceptional gains (+{unrealized_gain_pct:.1f}%)")
        elif unrealized_gain_pct > 50:
            reasons.append(f"Strong gains (+{unrealized_gain_pct:.1f}%)")
        elif unrealized_gain_pct > 25:
            reasons.append(f"Good gains (+{unrealized_gain_pct:.1f}%)")

        # RSI reasons
        swing_rsi = swing_analysis.get("rsi", 50)
        if swing_rsi > 80:
            reasons.append("RSI Extremely Overbought")
        elif swing_rsi > 70:
            reasons.append("RSI Overbought")
        elif swing_rsi > 60:
            reasons.append("RSI Moderately High")

        # Resistance reasons
        if resistance_score > 0.8:
            reasons.append("Near/Above Resistance")
        elif resistance_score > 0.6:
            reasons.append("Close to Resistance")

        # Market context reasons
        if market_context_factor > 0.7:
            reasons.append("Market Overbought")
        elif market_context_factor < 0.4:
            reasons.append("Market Weakness")

        return reasons

    def calculate_profit_take_amount(
        self, opportunity: ProfitOpportunity, take_percentage: float
    ) -> Tuple[float, float]:
        """
        Calculates the exact USD amount and coin quantity to sell for profit-taking.

        Args:
            opportunity: The profit opportunity
            take_percentage: Percentage of gains to take (e.g., 30 for 30%)

        Returns:
            Tuple of (usd_amount_to_sell, coin_quantity_to_sell)
        """
        # Validate percentage
        take_percentage = max(1, min(take_percentage, self.max_gain_take_pct))

        # Calculate USD amount (percentage of gains only)
        usd_amount_to_sell = opportunity.unrealized_gain_usd * (take_percentage / 100)

        # Calculate coin quantity
        coin_quantity_to_sell = (
            usd_amount_to_sell / opportunity.current_price
            if opportunity.current_price > 0
            else 0
        )

        return usd_amount_to_sell, coin_quantity_to_sell

    def validate_profit_take_percentage(self, percentage: float) -> Tuple[bool, str]:
        """
        Validates that the profit-taking percentage is within allowed limits.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if percentage <= 0:
            return False, "Percentage must be greater than 0"

        if percentage > self.max_gain_take_pct:
            return (
                False,
                f"Percentage cannot exceed {self.max_gain_take_pct}% (configured limit)",
            )

        return True, ""


def format_profit_opportunities_table(opportunities: List[ProfitOpportunity]) -> str:
    """
    Formats profit opportunities into a readable table string.
    """
    if not opportunities:
        return "No profit-taking opportunities found."

    lines = []
    lines.append("=" * 95)
    lines.append("💰 PROFIT-TAKING OPPORTUNITIES")
    lines.append("=" * 95)

    header = f"{'Symbol':<8} {'Gain':<12} {'Score':<6} {'RSI':<5} {'Reasons':<50}"
    lines.append(header)
    lines.append("-" * 95)

    for opp in opportunities:
        gain_str = f"${opp.unrealized_gain_usd:,.0f} (+{opp.unrealized_gain_pct:.1f}%)"
        score_str = f"{opp.opportunity_score:.0f}/100"
        rsi_str = f"{opp.rsi_score:.0f}"
        reasons_str = ", ".join(opp.reasons[:2])  # Show first 2 reasons
        if len(opp.reasons) > 2:
            reasons_str += "..."

        line = f"{opp.symbol:<8} {gain_str:<12} {score_str:<6} {rsi_str:<5} {reasons_str:<50}"
        lines.append(line)

    lines.append("=" * 95)
    return "\n".join(lines)
