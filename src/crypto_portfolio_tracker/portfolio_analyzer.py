"""
Portfolio Analyzer - Handles all portfolio analysis and metrics calculation
Moved from CryptoPortfolioTracker to separate concerns.
"""

import logging
import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from .crypto_trend_analyzer import CryptoTrendAnalyzer
from .rebalancing_logic import get_live_rebalance_suggestions
from .profit_taking_logic import ProfitTakingAnalyzer, ProfitOpportunity


class PortfolioAnalyzer:
    """
    Handles all portfolio analysis operations including:
    - Portfolio metrics calculation
    - Profit-taking opportunity analysis
    - Core portfolio rebalancing suggestions
    """

    def __init__(self, config: Dict[str, Any], db_manager, binance_client=None,
                 fetcher=None, enricher=None, offline_mode: bool = False, config_manager=None,
                 data_synchronizer=None):
        """
        Initialize PortfolioAnalyzer with necessary dependencies.

        Args:
            config: Configuration dictionary
            db_manager: Database manager instance
            binance_client: Optional Binance client instance
            fetcher: Optional BinanceFetcher instance
            enricher: Optional PriceEnricher instance
            offline_mode: Whether operating in offline mode
            config_manager: Optional ConfigManager instance
            data_synchronizer: Optional DataSynchronizer instance
        """
        self.config = config
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.binance_client = binance_client
        self.fetcher = fetcher
        self.enricher = enricher
        self.offline_mode = offline_mode
        self.data_synchronizer = data_synchronizer

    async def get_profit_taking_opportunities(
        self,
        core_holdings_df: Optional[pd.DataFrame] = None,
        rebalance_suggestions_df: Optional[pd.DataFrame] = None,
        cached_trend_data: Optional[Dict[str, Any]] = None,
    ) -> List[ProfitOpportunity]:
        """
        Analyzes portfolio for profit-taking opportunities.

        Args:
            core_holdings_df: Optional pre-computed core holdings DataFrame
            rebalance_suggestions_df: Optional pre-computed rebalancing suggestions DataFrame
            cached_trend_data: Optional cached trend analysis data from rebalancing workflow

        Returns:
            List of ProfitOpportunity objects for assets that meet criteria
        """
        self.logger.info("Analyzing profit-taking opportunities...")

        try:
            # Get rebalancing suggestions if not provided
            if rebalance_suggestions_df is None:
                suggestions_df = (
                    await self.get_core_portfolio_rebalance_suggestions_technical()
                )
                if suggestions_df is None or suggestions_df.empty:
                    self.logger.info(
                        "No rebalancing suggestions available - cannot analyze profit opportunities"
                    )
                    return []
            else:
                suggestions_df = rebalance_suggestions_df
                self.logger.info(
                    "Using provided rebalancing suggestions for profit-taking analysis"
                )

            # Get core holdings if not provided
            if core_holdings_df is None:
                metrics = await self.calculate_portfolio_metrics()
                core_holdings_df = metrics.get("core_holdings_df", pd.DataFrame())

                if core_holdings_df.empty:
                    self.logger.info(
                        "No core holdings found - cannot analyze profit opportunities"
                    )
                    return []
            else:
                self.logger.info(
                    "Using provided core holdings for profit-taking analysis"
                )

            # Initialize profit-taking analyzer
            analyzer = CryptoTrendAnalyzer(
                config=self.config, binance_client=self.binance_client
            )
            profit_analyzer = ProfitTakingAnalyzer(self.config, analyzer)

            # Use optimized analysis if we have cached data, otherwise use regular analysis
            if cached_trend_data:
                self.logger.info(
                    "Using optimized profit-taking analysis with cached trend data"
                )
                opportunities = (
                    await profit_analyzer.analyze_profit_opportunities_optimized(
                        core_holdings_df, suggestions_df, cached_trend_data
                    )
                )
            else:
                opportunities = await profit_analyzer.analyze_profit_opportunities(
                    core_holdings_df, suggestions_df
                )

            self.logger.info(
                f"Found {len(opportunities)} profit-taking opportunities with minimum score threshold"
            )

            return opportunities

        except Exception as e:
            self.logger.error(
                f"Error analyzing profit-taking opportunities: {e}", exc_info=True
            )
            return []

    async def get_core_portfolio_rebalance_suggestions_technical(
        self,
    ) -> Optional[pd.DataFrame]:
        """
        Generates rebalancing suggestions by fetching live portfolio data and passing
        it to the central rebalancing logic orchestrator.
        """
        self.logger.info("Calculating Core Portfolio rebalance suggestions...")

        # 1. Prepare live data
        analyzer = CryptoTrendAnalyzer(
            config=self.config, binance_client=self.binance_client
        )
        live_balances_df = self.fetcher.fetch_binance_balances()
        if live_balances_df.empty:
            self.logger.error("Could not fetch live balances. Cannot rebalance.")
            return None

        # Use the data synchronizer to get current prices
        prices = {}
        if self.data_synchronizer:
            try:
                prices = self.data_synchronizer._get_current_prices(list(live_balances_df["symbol"].unique()))
            except Exception as e:
                self.logger.error(f"Error fetching prices with data synchronizer: {e}")
                # Fallback to the enricher if data synchronizer fails
                if self.enricher:
                    try:
                        prices = await self.enricher.get_current_prices(list(live_balances_df["symbol"].unique()))
                        # Ensure USDT has a price of $1.0 if it's in our symbols
                        if "USDT" in prices:
                            prices["USDT"] = 1.0
                    except Exception as e2:
                        self.logger.error(f"Error fetching prices with enricher: {e2}")
                        # Final fallback to the stub method
                        prices = self._get_current_prices(list(live_balances_df["symbol"].unique()))
                else:
                    # Fallback to the stub method if enricher is not available
                    prices = self._get_current_prices(list(live_balances_df["symbol"].unique()))
        else:
            # Fallback to the enricher if data synchronizer is not available
            if self.enricher:
                try:
                    prices = await self.enricher.get_current_prices(list(live_balances_df["symbol"].unique()))
                    # Ensure USDT has a price of $1.0 if it's in our symbols
                    if "USDT" in prices:
                        prices["USDT"] = 1.0
                except Exception as e:
                    self.logger.error(f"Error fetching prices with enricher: {e}")
                    # Final fallback to the stub method
                    prices = self._get_current_prices(list(live_balances_df["symbol"].unique()))
            else:
                # Fallback to the stub method if neither data synchronizer nor enricher is available
                prices = self._get_current_prices(list(live_balances_df["symbol"].unique()))

        live_balances_df["value_usd"] = (
            live_balances_df["symbol"].map(prices).fillna(0.0)
            * live_balances_df["quantity"]
        )

        # Get the list of coins we actually want to rebalance from the config
        target_symbols = list(self.config.get("target_allocation", {}).keys())

        # Filter the DataFrame to create a "core" portfolio
        core_portfolio_df = live_balances_df[
            live_balances_df["symbol"].isin(target_symbols)
        ].copy()

        if core_portfolio_df.empty:
            self.logger.warning(
                "No assets from target_allocation found in live balances. Cannot generate rebalancing suggestions."
            )
            return pd.DataFrame()  # Return an empty DataFrame

        self.logger.info(
            f"Rebalancing logic will be based on {len(core_portfolio_df)} core assets."
        )

        # 2. Call the new central orchestrator with the live data
        suggestions_df = await get_live_rebalance_suggestions(
            analyzer=analyzer, portfolio_df=core_portfolio_df, config=self.config
        )

        return suggestions_df

    async def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculates key portfolio metrics using a consolidated view of holdings,
        correctly summing values from Spot, Earn, Futures, and Funding wallets.
        """
        self.logger.info(
            "Calculating consolidated portfolio metrics (Spot, Earn, Futures, Funding)..."
        )
        cost_basis_df = self.db_manager.get_holdings()

        # In offline mode, avoid all network calls and build metrics from DB only
        if self.offline_mode:
            holdings_df = cost_basis_df.copy()
            if not holdings_df.empty:
                holdings_df = holdings_df.rename(columns={"quantity": "total_quantity"})
                # Without network prices, set current_price to 0 and derive others to 0
                holdings_df["spot_quantity"] = holdings_df.get("spot_quantity", 0.0)
                holdings_df["earn_quantity"] = holdings_df.get("earn_quantity", 0.0)
                holdings_df["current_price"] = 0.0
                holdings_df["value_usd"] = 0.0
                holdings_df["cost_basis_total"] = holdings_df[
                    "total_quantity"
                ] * holdings_df.get("average_cost_basis", 0.0)
                holdings_df["unrealized_pl_usd"] = 0.0
                holdings_df["unrealized_pl_percent"] = 0.0
                target_symbols = list(self.config.get("target_allocation", {}).keys())
                holdings_df["is_core"] = holdings_df["symbol"].isin(target_symbols)
                core_holdings_df = holdings_df[holdings_df["is_core"]].copy()
                other_holdings_df = holdings_df[~holdings_df["is_core"]].copy()
            else:
                holdings_df = pd.DataFrame(
                    columns=[
                        "symbol",
                        "total_quantity",
                        "spot_quantity",
                        "earn_quantity",
                        "value_usd",
                        "average_cost_basis",
                        "cost_basis_total",
                    ]
                )
                core_holdings_df = holdings_df.copy()
                other_holdings_df = holdings_df.copy()

            metrics = {
                "total_value_usd": 0.0,
                "spot_earn_value_usd": 0.0,
                "futures_value_usd": 0.0,
                "funding_value_usd": 0.0,
                "total_cost_basis_usd": holdings_df.get(
                    "cost_basis_total", pd.Series(dtype=float)
                ).sum()
                if not holdings_df.empty
                else 0.0,
                "unrealized_pl_usd": 0.0,
                "unrealized_pl_percent": 0.0,
                "total_invested_capital": self.db_manager.calculate_total_invested_capital(),
                "overall_pl_usd": 0.0,
                "overall_pl_percent": 0.0,
                "holdings_df": holdings_df,
                "core_holdings_df": core_holdings_df,
                "other_holdings_df": other_holdings_df,
                "futures_balances": [],
                "funding_balances": [],
                "timestamp": datetime.datetime.now(),
            }
            self.logger.info("Calculated offline metrics from database only.")
            return metrics

        # 1. Calculate Spot + Earn Value
        total_balances_api_df = self.fetcher.fetch_binance_balances().rename(
            columns={"quantity": "total_quantity_api"}
        )
        earn_balances_df = pd.DataFrame(columns=["symbol", "earn_quantity"])
        if not getattr(self, 'config_manager', None) or not self.config_manager.is_testnet_mode:
            earn_dict = self.fetcher.fetch_simple_earn_balances(total_balances_api_df)
            if earn_dict:
                earn_balances_df = pd.DataFrame(
                    list(earn_dict.items()), columns=["symbol", "earn_quantity"]
                )

        holdings_df = pd.merge(
            total_balances_api_df, earn_balances_df, on="symbol", how="outer"
        )
        holdings_df["total_quantity_api"] = pd.to_numeric(
            holdings_df["total_quantity_api"], errors="coerce"
        ).fillna(0)
        holdings_df["earn_quantity"] = pd.to_numeric(
            holdings_df["earn_quantity"], errors="coerce"
        ).fillna(0)
        holdings_df["total_quantity"] = holdings_df[
            ["total_quantity_api", "earn_quantity"]
        ].max(axis=1)
        holdings_df["spot_quantity"] = (
            holdings_df["total_quantity"] - holdings_df["earn_quantity"]
        ).clip(lower=0)
        holdings_df = holdings_df[holdings_df["total_quantity"] > 1e-8].reset_index(
            drop=True
        )

        spot_earn_value = 0
        if not holdings_df.empty:
            if not cost_basis_df.empty:
                holdings_df = pd.merge(
                    holdings_df,
                    cost_basis_df[["symbol", "average_cost_basis"]],
                    on="symbol",
                    how="left",
                )
            else:
                holdings_df["average_cost_basis"] = 0.0
            holdings_df["average_cost_basis"] = holdings_df[
                "average_cost_basis"
            ].fillna(0.0)

            prices = await self.enricher.get_current_prices(
                holdings_df["symbol"].tolist()
            )
            holdings_df["current_price"] = holdings_df["symbol"].map(prices).fillna(0.0)
            holdings_df["value_usd"] = (
                holdings_df["total_quantity"] * holdings_df["current_price"]
            )
            holdings_df["cost_basis_total"] = (
                holdings_df["total_quantity"] * holdings_df["average_cost_basis"]
            )
            holdings_df["unrealized_pl_usd"] = (
                holdings_df["value_usd"] - holdings_df["cost_basis_total"]
            )
            holdings_df.loc[
                holdings_df["cost_basis_total"] > 0, "unrealized_pl_percent"
            ] = (
                holdings_df["unrealized_pl_usd"] / holdings_df["cost_basis_total"]
            ) * 100

            spot_earn_value = holdings_df["value_usd"].sum()

        else:
            holdings_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "total_quantity",
                    "spot_quantity",
                    "earn_quantity",
                    "value_usd",
                    "average_cost_basis",
                    "cost_basis_total",
                ]
            )

        # 2. Calculate Futures Value
        futures_value = 0
        futures_balances = []
        if not getattr(self, 'config_manager', None) or not self.config_manager.is_testnet_mode:
            futures_balances = self.fetcher.fetch_futures_balance()
        for item in futures_balances:
            if item.get("asset") == "USDT":
                futures_value += float(item.get("balance", 0.0))

        # 3. Calculate Funding Wallet Value
        funding_value = 0
        funding_balances_raw = []
        if not getattr(self, 'config_manager', None) or not self.config_manager.is_testnet_mode:
            funding_balances_raw = self.fetcher.fetch_funding_balance()
        funding_balances = [
            b for b in funding_balances_raw if float(b.get("free", 0.0)) > 1e-8
        ]

        if funding_balances:
            funding_assets = [b["asset"] for b in funding_balances]
            funding_prices = await self.enricher.get_current_prices(funding_assets)
            for item in funding_balances:
                asset = item["asset"]
                price = funding_prices.get(asset, 0.0)
                quantity = float(item.get("free", 0.0))
                funding_value += quantity * price

        # 4. Calculate Grand Total
        total_portfolio_value = spot_earn_value + futures_value + funding_value

        # 5. Consolidate Metrics
        # Calculate allocation based on spot/earn value only, not total portfolio value
        spot_earn_total = holdings_df["value_usd"].sum()
        if spot_earn_total > 0:
            holdings_df["allocation"] = holdings_df["value_usd"] / spot_earn_total
        else:
            holdings_df["allocation"] = 0

        target_symbols = list(self.config.get("target_allocation", {}).keys())
        holdings_df["is_core"] = holdings_df["symbol"].isin(target_symbols)
        core_holdings_df = holdings_df[holdings_df["is_core"]].copy()
        other_holdings_df = holdings_df[~holdings_df["is_core"]].copy()

        total_core_value = core_holdings_df["value_usd"].sum()
        if total_core_value > 0:
            core_holdings_df["core_allocation"] = (
                core_holdings_df["value_usd"] / total_core_value
            )
        else:
            core_holdings_df["core_allocation"] = 0

        # Get stablecoin symbols from configuration
        stablecoin_symbols = self.config.get("portfolio", {}).get("stablecoin_symbols", [])
        
        # Calculate crypto-only value (excluding stablecoins) for more intuitive P/L calculation
        crypto_holdings_df = holdings_df[~holdings_df["symbol"].isin(stablecoin_symbols)]
        crypto_only_value = crypto_holdings_df["value_usd"].sum() if not crypto_holdings_df.empty else 0.0

        # Calculate wallet-agnostic total cost basis (all holdings regardless of wallet)
        all_holdings_cost_basis = self.db_manager.get_holdings()
        if not all_holdings_cost_basis.empty:
            total_cost_basis = (all_holdings_cost_basis["quantity"] * all_holdings_cost_basis["average_cost_basis"]).sum()
        else:
            total_cost_basis = 0.0

        # Calculate traditional unrealized P/L against total portfolio value
        total_pl_usd = total_portfolio_value - total_cost_basis
        total_pl_percent = (
            (total_pl_usd / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        )
        
        # Calculate crypto-only unrealized P/L (excluding stablecoins) for more intuitive performance metric
        crypto_only_pl_usd = crypto_only_value - total_cost_basis
        crypto_only_pl_percent = (
            (crypto_only_pl_usd / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        )
        total_invested = self.db_manager.calculate_total_invested_capital()

        overall_pl_usd = total_portfolio_value - total_invested
        overall_pl_percent = (
            (overall_pl_usd / total_invested * 100) if total_invested > 0 else 0.0
        )

        metrics = {
            "total_value_usd": total_portfolio_value,
            "spot_earn_value_usd": spot_earn_value,
            "futures_value_usd": futures_value,
            "funding_value_usd": funding_value,
            "total_cost_basis_usd": total_cost_basis,
            "unrealized_pl_usd": total_pl_usd,
            "unrealized_pl_percent": total_pl_percent,
            "crypto_only_unrealized_pl_usd": crypto_only_pl_usd,
            "crypto_only_unrealized_pl_percent": crypto_only_pl_percent,
            "total_invested_capital": total_invested,
            "overall_pl_usd": overall_pl_usd,
            "overall_pl_percent": overall_pl_percent,
            "holdings_df": holdings_df,
            "core_holdings_df": core_holdings_df,
            "other_holdings_df": other_holdings_df,
            "futures_balances": futures_balances,
            "funding_balances": funding_balances,
            "timestamp": datetime.datetime.now(),
        }
        self.logger.info("Successfully calculated consolidated portfolio metrics.")
        return metrics
