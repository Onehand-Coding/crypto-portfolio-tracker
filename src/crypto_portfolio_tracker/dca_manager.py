"""
DCA Manager - Handles all Dollar Cost Averaging operations
Moved from CryptoPortfolioTracker to separate concerns.
"""

import uuid
import logging
from typing import Dict, Any, List, Tuple

import pandas as pd

from .models import TradeResult


class DCAManager:
    """
    Handles all DCA (Dollar Cost Averaging) operations including:
    - Proportional DCA calculations
    - Target weight DCA calculations
    - USDT balance validation
    - DCA trade execution
    - DCA execution validation
    """

    def __init__(self, config: Dict[str, Any], binance_client=None,
                 config_manager=None, fetcher=None, trade_executor=None,
                 portfolio_analyzer=None, data_manager=None):
        """
        Initialize DCAManager with necessary dependencies.

        Args:
            config: Configuration dictionary
            binance_client: Optional Binance client instance
            config_manager: Configuration manager instance
            fetcher: Optional BinanceFetcher instance
            trade_executor: TradeExecutor instance for executing trades
            portfolio_analyzer: PortfolioAnalyzer instance for metrics
            data_manager: DataManager instance for recording transactions
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.binance_client = binance_client
        self.config_manager = config_manager
        self.fetcher = fetcher
        self.trade_executor = trade_executor
        self.portfolio_analyzer = portfolio_analyzer
        self.data_manager = data_manager

    def calculate_proportional_dca(
        self, new_funds: float, target_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate Proportional DCA buy amounts.
        Formula: Buy Amount = New Funds × Target Allocation %

        Args:
            new_funds: Amount of USDT to invest
            target_allocation: Target allocation percentages (e.g., {"BTC": 0.35, "ETH": 0.20})

        Returns:
            Dictionary mapping asset to buy amount
        """
        if new_funds <= 0:
            self.logger.warning("New funds must be greater than 0 for DCA calculation")
            return {}

        trades = {}
        for asset, target_pct in target_allocation.items():
            buy_amount = new_funds * target_pct
            if buy_amount > 0:
                trades[asset] = buy_amount

        self.logger.info(
            f"Calculated Proportional DCA for {len(trades)} assets with {new_funds} USDT"
        )
        return trades

    def calculate_target_weight_dca(
        self,
        new_funds: float,
        current_portfolio: pd.DataFrame,
        target_allocation: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate Target-Weight Rebalancing DCA buy amounts.
        Formula: Buy/Sell Amount = (New Total × Target %) - Current Value

        Args:
            new_funds: Amount of USDT to invest
            current_portfolio: DataFrame with current holdings (must have 'symbol' and 'value_usd' columns)
            target_allocation: Target allocation percentages

        Returns:
            Dictionary mapping asset to buy/sell amount (positive = buy, negative = sell)
        """
        if new_funds <= 0:
            self.logger.warning("New funds must be greater than 0 for DCA calculation")
            return {}

        if current_portfolio.empty:
            self.logger.warning(
                "Current portfolio is empty, cannot calculate Target-Weight DCA"
            )
            return {}

        # Get current portfolio value
        current_portfolio_value = current_portfolio["value_usd"].sum()
        new_total = current_portfolio_value + new_funds

        self.logger.info(
            f"Target-Weight DCA: Current portfolio ${current_portfolio_value:,.2f}, New funds ${new_funds:,.2f}, New total ${new_total:,.2f}"
        )

        trades = {}
        for asset, target_pct in target_allocation.items():
            # Find current value of this asset
            asset_row = current_portfolio[current_portfolio["symbol"] == asset]
            current_value = (
                asset_row["value_usd"].iloc[0] if not asset_row.empty else 0.0
            )

            # Calculate target value
            target_value = new_total * target_pct

            # Calculate buy/sell amount
            trade_amount = target_value - current_value

            # Only include if there's a meaningful trade (minimum threshold)
            min_trade_threshold = 0.01  # $0.01 minimum
            if abs(trade_amount) >= min_trade_threshold:
                trades[asset] = trade_amount
                self.logger.debug(
                    f"{asset}: Current ${current_value:,.2f}, Target ${target_value:,.2f}, Trade ${trade_amount:,.2f}"
                )

        self.logger.info(f"Calculated Target-Weight DCA for {len(trades)} assets")
        return trades

    def get_available_usdt_balance(self) -> Dict[str, float]:
        """
        Get USDT balances from all wallets.

        Returns:
            Dictionary with spot_earn, funding, and total USDT balances
        """
        spot_earn_balance = 0.0
        funding_balance = 0.0

        try:
            # Get Spot + Earn balance (consolidated)
            if self.binance_client:
                spot_balance = float(
                    self.binance_client.get_asset_balance(asset="USDT").get("free", 0.0)
                )
                spot_earn_balance += spot_balance

                # Add Earn balance if not in testnet
                if self.config_manager and not self.config_manager.is_testnet_mode and self.fetcher:
                    earn_positions = self.fetcher.fetch_simple_earn_balances(
                        pd.DataFrame([{"symbol": "USDT"}])
                    )
                    earn_balance = earn_positions.get("USDT", 0.0)
                    spot_earn_balance += earn_balance

                # Get Funding balance (mainnet only)
                if self.config_manager and not self.config_manager.is_testnet_mode and self.fetcher:
                    funding_balances = self.fetcher.fetch_funding_balance()
                    for balance in funding_balances:
                        if balance.get("asset") == "USDT":
                            funding_balance = float(balance.get("free", 0.0))
                            break

        except Exception as e:
            self.logger.error(f"Error fetching USDT balances: {e}")

        total_balance = spot_earn_balance + funding_balance

        # Apply consistent rounding to prevent precision issues
        spot_earn_balance = round(spot_earn_balance, 8)
        funding_balance = round(funding_balance, 8)
        total_balance = round(total_balance, 8)

        self.logger.info(
            f"USDT Balances - Spot+Earn: ${spot_earn_balance:,.2f}, Funding: ${funding_balance:,.2f}, Total: ${total_balance:,.2f}"
        )

        return {
            "spot_earn": spot_earn_balance,
            "funding": funding_balance,
            "total": total_balance,
        }

    def validate_dca_amount(self, amount: float) -> Tuple[bool, str]:
        """
        Validate if user can afford the DCA amount.

        Args:
            amount: USDT amount to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if amount <= 0:
            return False, "Amount must be greater than 0"

        available_balance = self.get_available_usdt_balance()
        total_available = available_balance["total"]

        if amount > total_available:
            return False, f"Insufficient USDT. Available: ${total_available:,.2f}"

        return True, f"✅ Sufficient funds available (${total_available:,.2f})"

    async def get_dca_suggestions(
        self,
        new_funds: float,
        method: str = "both",  # "proportional", "target_weight", "both"
    ) -> Dict[str, Any]:
        """
        Get comprehensive DCA suggestions for both methods.

        Args:
            new_funds: Amount of USDT to invest
            method: Which DCA method to calculate ("proportional", "target_weight", "both")

        Returns:
            Dictionary with DCA suggestions and portfolio data
        """
        # Validate amount first
        is_valid, message = self.validate_dca_amount(new_funds)
        if not is_valid:
            self.logger.warning(f"DCA validation failed: {message}")
            return {
                "error": message,
                "proportional": {},
                "target_weight": {},
                "current_portfolio": pd.DataFrame(),
                "target_allocation": {},
                "available_usdt": {"total": 0},
                "new_funds": new_funds,
            }

        # Get current portfolio metrics
        if self.portfolio_analyzer:
            metrics = await self.portfolio_analyzer.calculate_portfolio_metrics()
        else:
            self.logger.error("Portfolio analyzer not available for DCA suggestions")
            return {
                "error": "Portfolio analyzer not available",
                "proportional": {},
                "target_weight": {},
                "current_portfolio": pd.DataFrame(),
                "target_allocation": {},
                "available_usdt": {"total": 0},
                "new_funds": new_funds,
            }

        # Get target allocation from config
        target_allocation = self.config.get("target_allocation", {})

        # Get available USDT
        available_usdt = self.get_available_usdt_balance()

        # Get core portfolio (assets in target allocation)
        core_portfolio = metrics.get("core_holdings_df", pd.DataFrame())

        # Calculate DCA methods
        proportional_trades = {}
        target_weight_trades = {}

        if method in ["proportional", "both"]:
            proportional_trades = self.calculate_proportional_dca(
                new_funds, target_allocation
            )

        if method in ["target_weight", "both"]:
            target_weight_trades = self.calculate_target_weight_dca(
                new_funds, core_portfolio, target_allocation
            )

        return {
            "proportional": proportional_trades,
            "target_weight": target_weight_trades,
            "current_portfolio": core_portfolio,
            "target_allocation": target_allocation,
            "available_usdt": available_usdt,
            "new_funds": new_funds,
            "current_portfolio_value": core_portfolio["value_usd"].sum()
            if not core_portfolio.empty
            else 0.0,
        }

    async def execute_dca_trades(
        self, selected_trades: List[Dict[str, Any]], method: str, is_live: bool = False
    ) -> TradeResult:
        """
        Execute selected DCA trades using existing trade execution infrastructure.

        Args:
            selected_trades: List of trade dictionaries [{"asset": str, "amount": float, "method": str}]
            method: DCA method used ("proportional" or "target_weight")
            is_live: Whether to execute live trades

        Returns:
            TradeResult with execution results
        """
        result = TradeResult(success=True)  # Initialize with success=True
        batch_id = str(uuid.uuid4())
        mode = (
            "TESTNET"
            if self.config_manager and self.config_manager.is_testnet_mode
            else ("LIVE" if is_live else "SIM")
        )

        if not selected_trades:
            result.messages.append("No trades selected for execution")
            result.success = False  # Set to False if no trades
            return result

        self.logger.info(
            f"Executing {len(selected_trades)} DCA trades using {method} method"
        )

        if not self.trade_executor:
            result.messages.append("Trade executor not available for DCA execution")
            result.success = False
            return result

        trades_executed_count = 0

        for trade in selected_trades:
            asset = trade["asset"]
            amount = trade["amount"]

            # Determine trade type based on amount
            trade_type = "BUY" if amount > 0 else "SELL"
            trade_amount = abs(amount)

            self.logger.info(
                f"Executing DCA {trade_type} for {asset}: ${trade_amount:,.2f}"
            )

            # Use trade executor's manual trade execution method
            try:
                trade_result = await self.trade_executor.execute_manual_trade_core(
                    trade_type=trade_type,
                    symbol=asset,
                    trade_ticker=f"{asset}USDT",
                    amount=trade_amount,
                    is_quote_qty=True,  # DCA amounts are in USD
                    is_live=is_live,
                )

                # Aggregate results
                result.success &= trade_result.success
                result.messages.extend(trade_result.messages)
                result.errors.extend(trade_result.errors)

                if trade_result.success:
                    trades_executed_count += 1

                    # Record trade using actual order data if available
                    order_data = trade_result.data.get("order", {})
                    if order_data and "price" in order_data:
                        price_usd = float(order_data.get("price", 0))
                        qty = float(order_data.get("executedQty", 0))
                    else:
                        # Calculate approximate values for simulation
                        price_usd = 1.0  # Placeholder, would need actual price
                        qty = trade_amount  # Approximate, would need price conversion

                    if self.data_manager:
                        self.data_manager._record_trade_transaction(
                            symbol=asset,
                            side=trade_type,
                            quantity=qty,
                            price_usd=price_usd,
                            source="DCA",
                            mode=mode,
                            batch_id=batch_id,
                            order=order_data,
                            error="\n".join(trade_result.errors) if trade_result.errors else None,
                        )

            except Exception as e:
                error_msg = f"Error executing DCA {trade_type} for {asset}: {e}"
                result.messages.append(f"❌ {error_msg}")
                result.errors.append(error_msg)
                self.logger.error(error_msg, exc_info=True)
                result.success = False

        result.messages.append("\n" + "=" * 80)
        result.messages.append(
            f"✅ {trades_executed_count} DCA trade(s) executed successfully!"
        )
        result.data["batch_id"] = batch_id
        result.data["trades_executed"] = trades_executed_count
        result.success = trades_executed_count > 0 and result.success
        return result

    def validate_dca_execution(
        self, selected_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate DCA execution before proceeding.

        Args:
            selected_trades: List of trade dictionaries [{"asset": str, "amount": float, "method": str}]

        Returns:
            Dictionary with validation results
        """
        # Get configuration
        is_live = self.config.get("portfolio", {}).get("live_trading_enabled", False)
        is_testnet = self.config.get("portfolio", {}).get("testnet_mode", True)
        min_trade_usd = self.config.get("portfolio", {}).get("minimum_trade_usd", 5.0)

        # Calculate total USDT needed
        total_usdt_needed = sum(
            abs(trade["amount"]) for trade in selected_trades if trade["amount"] > 0
        )

        # Get available USDT balances
        available_balance = self.get_available_usdt_balance()

        # Validation results
        validation = {
            "can_execute": True,
            "messages": [],
            "warnings": [],
            "errors": [],
            "summary": {
                "total_needed": total_usdt_needed,
                "total_available": available_balance["total"],
                "spot_earn_available": available_balance["spot_earn"],
                "funding_available": available_balance["funding"],
                "num_trades": len(selected_trades),
                "is_live": is_live,
                "is_testnet": is_testnet,
                "min_trade_usd": min_trade_usd,
            },
        }

        # 1. Check total USDT requirement
        if total_usdt_needed > available_balance["total"]:
            validation["can_execute"] = False
            validation["errors"].append(
                f"Insufficient USDT. Need ${total_usdt_needed:,.2f}, Available ${available_balance['total']:,.2f}"
            )

        # 2. Check minimum trade amounts
        small_trades = [
            trade for trade in selected_trades if abs(trade["amount"]) < min_trade_usd
        ]
        if small_trades:
            small_assets = [t["asset"] for t in small_trades]
            validation["warnings"].append(
                f"Some trades below minimum (${min_trade_usd}): {small_assets}"
            )

        # 3. Check if Earn redemption needed - IMPROVED LOGIC
        # Only warn if we need more than what's in Spot+Earn AND we have USDT in Earn AND no USDT in funding
        if (
            total_usdt_needed > available_balance["spot_earn"]
            and available_balance["spot_earn"] > 0
            and available_balance["funding"] == 0
        ):
            validation["warnings"].append(
                f"USDT in Earn wallet may need redemption. Spot+Earn: ${available_balance['spot_earn']:,.2f}, Needed: ${total_usdt_needed:,.2f}"
            )

        # Note: Live trading status is already shown in the UI banner, so no need to repeat here

        self.logger.info(
            f"DCA validation completed. Can execute: {validation['can_execute']}"
        )
        return validation
