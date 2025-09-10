"""
Trade Executor - Handles all trade execution operations
Moved from CryptoPortfolioTracker to separate concerns.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException
from diskcache import Cache

from .models import ExecutionMode, TradeResult


class TradeExecutor:
    """
    Handles all trade execution operations including:
    - Symbol filter management
    - Quantity adjustment for lot sizes
    - Earn redemption checks
    - Directional trade execution
    - Profit-taking trade execution
    - Rebalancing trade execution
    - Manual trade execution
    """

    def __init__(
        self,
        config: Dict[str, Any],
        binance_client: Optional[Client] = None,
        config_manager=None,
        yfinance_disk_cache: Optional[Cache] = None,
        data_synchronizer=None,
        data_manager=None,
    ):
        """
        Initialize TradeExecutor with necessary dependencies.

        Args:
            config: Configuration dictionary
            binance_client: Optional Binance client instance
            config_manager: Configuration manager instance
            yfinance_disk_cache: Cache for symbol filters
            data_synchronizer: Data synchronizer for current prices
            data_manager: Data manager for recording transactions
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.binance_client = binance_client
        self.config_manager = config_manager
        self.yfinance_disk_cache = yfinance_disk_cache
        self.data_synchronizer = data_synchronizer
        self.data_manager = data_manager

    def _get_symbol_filters(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches and caches trading filters for a given symbol."""
        if not self.yfinance_disk_cache:
            self.logger.warning("No cache available for symbol filters")
            return None

        cache_key = f"filters_{symbol}"
        cached_filters = self.yfinance_disk_cache.get(cache_key)
        if cached_filters:
            return cached_filters

        try:
            self.logger.info(f"Fetching trading rules for {symbol}...")
            symbol_info = self.binance_client.get_symbol_info(symbol)
            if not symbol_info:
                return None

            filters = {f["filterType"]: f for f in symbol_info["filters"]}
            self.yfinance_disk_cache.set(
                cache_key, filters, expire=3600 * 24
            )  # Cache for 24 hours
            return filters
        except Exception as e:
            self.logger.error(f"Could not fetch symbol info for {symbol}: {e}")
            return None

    def _adjust_quantity_to_lot_size(
        self, symbol: str, quantity: float
    ) -> Optional[float]:
        """Rounds the quantity down to the nearest valid step size for the LOT_SIZE filter."""
        filters = self._get_symbol_filters(symbol)
        if not filters or "LOT_SIZE" not in filters:
            self.logger.warning(
                f"Could not get LOT_SIZE filter for {symbol}. Cannot adjust quantity."
            )
            return None

        step_size_str = filters["LOT_SIZE"].get("stepSize")
        if not step_size_str:
            return None

        # Calculate the number of decimal places from the stepSize (e.g., "0.0001" -> 4)
        if "." in step_size_str:
            precision = len(step_size_str.split(".")[1].rstrip("0"))
        else:
            precision = 0

        # Floor the quantity to the required precision
        factor = 10**precision
        adjusted_quantity = (int(quantity * factor)) / factor

        self.logger.info(
            f"Adjusted quantity for {symbol} from {quantity} to {adjusted_quantity} (precision: {precision})"
        )
        return adjusted_quantity

    def redeem_from_earn(
        self, asset: str, amount: float, is_live: bool = False
    ) -> TradeResult:
        """
        Directly redeems a specified amount of an asset from Binance Simple Earn Flexible products.

        This method allows users to directly redeem assets from their Earn positions without
        first checking spot wallet balances.

        Args:
            asset: The asset symbol to redeem (e.g., 'BTC', 'ETH')
            amount: The amount to redeem
            is_live: Whether to actually execute the redemption (True) or simulate it (False)

        Returns:
            TradeResult with success status, messages, and any errors
        """
        result = TradeResult(success=False)

        if not self.binance_client:
            result.messages.append("⚠️ No Binance client available for earn redemption")
            result.errors.append("No Binance client available")
            return result

        if self.config_manager and self.config_manager.is_testnet_mode:
            result.messages.append(
                "⚠️ TESTNET MODE: Simple Earn operations not supported."
            )
            result.messages.append(
                f"✅ [DRY RUN] Would redeem {amount:.8f} {asset} from Simple Earn"
            )
            result.success = True
            return result

        try:
            # First, we need to find the product ID for this asset
            # Get all flexible products
            products = self.binance_client.get_simple_earn_flexible_product_list(
                asset=asset
            )

            # Find the product ID for the requested asset
            product_id = None
            for product in products.get("rows", []):
                if product.get("asset") == asset:
                    product_id = product.get("productId")
                    break

            if not product_id:
                result.messages.append(
                    f"❌ Could not find Simple Earn product for {asset}"
                )
                result.errors.append(f"No Simple Earn product found for {asset}")
                return result

            if is_live:
                # Execute actual redemption
                redemption_result = (
                    self.binance_client.redeem_simple_earn_flexible_product(
                        productId=product_id, amount=str(amount)
                    )
                )

                if redemption_result.get("success", False):
                    result.success = True
                    result.messages.append(
                        f"✅ Successfully redeemed {amount:.8f} {asset} from Simple Earn"
                    )
                    result.messages.append(
                        f"Redemption ID: {redemption_result.get('redeemId')}"
                    )
                else:
                    result.messages.append(
                        f"❌ Failed to redeem {amount:.8f} {asset} from Simple Earn"
                    )
                    result.errors.append(f"Redemption failed: {redemption_result}")
            else:
                # Dry run - just show what would happen
                result.success = True
                result.messages.append(
                    f"✅ [DRY RUN] Would redeem {amount:.8f} {asset} from Simple Earn product {product_id}"
                )

        except Exception as e:
            result.messages.append(f"❌ Error during redemption: {str(e)}")
            result.errors.append(str(e))
            self.logger.error(
                f"Error during earn redemption for {asset}: {e}", exc_info=True
            )

        return result

    def _execute_directional_trade(
        self, trade: Dict[str, Any], client: Client
    ) -> TradeResult:
        """Helper function to execute a single directional BUY or SELL trade."""
        result = TradeResult(success=False)
        symbol = trade["Symbol"]
        signal = trade["Signal"]
        size = trade.get("Size", 1.0)  # Default to 100% if not provided
        trade_ticker = f"{symbol}USDT"
        min_trade_usd = self.config.get("portfolio", {}).get("minimum_trade_usd", 10.0)
        is_live = self.config.get("portfolio", {}).get("live_trading_enabled", False)

        self.logger.info(
            f"Executing {signal} for {symbol} (Size: {size:.2%}) on account via directional strategy. LIVE: {is_live}"
        )

        try:
            if signal == "BUY":
                usdt_balance = float(
                    client.get_asset_balance(asset="USDT").get("free", 0.0)
                )
                trade_amount_usd = usdt_balance * size

                if trade_amount_usd < min_trade_usd:
                    result.messages.append(
                        f"⚠️ SKIPPING BUY for {symbol}: Calculated trade size (${trade_amount_usd:,.2f}) is below minimum of ${min_trade_usd:,.2f}."
                    )
                    return result

                result.messages.append(
                    f"\nPreparing MARKET BUY for ${trade_amount_usd:,.2f} of {symbol}..."
                )

                if is_live:
                    order = client.order_market_buy(
                        symbol=trade_ticker, quoteOrderQty=f"{trade_amount_usd:.2f}"
                    )
                    result.messages.append(f"✅ LIVE BUY ORDER PLACED: {order}")
                    self.logger.info(f"LIVE BUY ORDER PLACED: {order}")
                    result.success = True
                else:
                    result.messages.append(
                        f"✅ [DRY RUN] Market BUY order for ${trade_amount_usd:,.2f} of {trade_ticker} was not placed."
                    )
                    self.logger.info(
                        f"[DRY RUN] Market BUY order for {trade_ticker} was not placed."
                    )
                    result.success = True

            elif signal == "SELL":
                asset_balance = float(
                    client.get_asset_balance(asset=symbol).get("free", 0.0)
                )
                trade_quantity = asset_balance * size

                # Get current price from data synchronizer
                current_price = 0
                if self.data_synchronizer:
                    prices = self.data_synchronizer._get_current_prices([symbol])
                    current_price = prices.get(symbol, 0)

                if (trade_quantity * current_price) < min_trade_usd:
                    result.messages.append(
                        f"⚠️ SKIPPING SELL for {symbol}: Position value is below minimum trade size."
                    )
                    return result

                result.messages.append(
                    f"\nPreparing MARKET SELL for {trade_quantity:.8f} {symbol} ({size:.0%} of holding)..."
                )

                if is_live:
                    order = client.order_market_sell(
                        symbol=trade_ticker, quantity=f"{trade_quantity:.8f}"
                    )
                    result.messages.append(f"✅ LIVE SELL ORDER PLACED: {order}")
                    self.logger.info(f"LIVE SELL ORDER PLACED: {order}")
                    result.success = True
                else:
                    result.messages.append(
                        f"✅ [DRY RUN] Market SELL order for {trade_quantity:.8g} {symbol} was not placed."
                    )
                    self.logger.info(
                        f"[DRY RUN] Market SELL order for {trade_quantity:.8g} {symbol} was not placed."
                    )
                    result.success = True

        except BinanceAPIException as e:
            result.messages.append(
                f"❌ {('LIVE' if is_live else 'DRY RUN')} {signal} FAILED for {symbol}: {e}"
            )
            self.logger.error(
                f"{('LIVE' if is_live else 'DRY RUN')} {signal} FAILED for {symbol}: {e}"
            )
            result.errors.append(str(e))
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred executing directional trade for {symbol}: {e}",
                exc_info=True,
            )
            result.messages.append(f"❌ Unexpected Error for {symbol}: {e}")
            result.errors.append(str(e))

        return result

    async def execute_profit_taking_trades(
        self, profit_trades: List[Dict[str, Any]], is_live: bool = False
    ) -> TradeResult:
        """
        Execute selected profit-taking trades.

        Args:
            profit_trades: List of profit-taking trade dictionaries
                          [{"symbol": str, "usd_amount": float, "coin_quantity": float, "take_percentage": float}]
            is_live: Whether to execute live trades

        Returns:
            TradeResult with execution results
        """
        result = TradeResult(success=True)
        batch_id = str(uuid.uuid4())
        mode = (
            "TESTNET"
            if self.config_manager and self.config_manager.is_testnet_mode
            else ("LIVE" if is_live else "SIM")
        )

        if not profit_trades:
            result.messages.append("No profit-taking trades selected for execution")
            result.success = False
            return result

        self.logger.info(f"Executing {len(profit_trades)} profit-taking trades")

        trades_executed_count = 0

        for trade in profit_trades:
            symbol = trade["symbol"]
            coin_quantity = trade["coin_quantity"]
            usd_amount = trade["usd_amount"]
            take_percentage = trade.get("take_percentage", 0)

            trade_ticker = f"{symbol}USDT"

            self.logger.info(
                f"Executing profit-taking SELL for {symbol}: {coin_quantity:.8f} coins (~${usd_amount:,.2f})"
            )

            try:
                # Adjust quantity to exchange lot size rules
                adjusted_quantity = self._adjust_quantity_to_lot_size(
                    trade_ticker, coin_quantity
                )

                if adjusted_quantity is None or adjusted_quantity <= 0:
                    result.messages.append(
                        f"⚠️ SKIPPING profit-taking for {symbol}: Adjusted quantity is zero or invalid after applying lot size rules."
                    )
                    continue

                # No automatic redemption from Earn - let the trade fail naturally if insufficient balance
                # Users can manually redeem using the direct redemption functionality if needed

                result.messages.append(
                    f"\n💰 Executing profit-taking SELL for {adjusted_quantity:.8f} {symbol} ({take_percentage:.0f}% of gains)..."
                )

                if is_live:
                    order = self.binance_client.order_market_sell(
                        symbol=trade_ticker, quantity=f"{adjusted_quantity:.8f}"
                    )
                    result.messages.append(
                        f"✅ LIVE profit-taking SELL ORDER PLACED: {order}"
                    )
                    self.logger.info(f"LIVE profit-taking SELL ORDER PLACED: {order}")
                    trades_executed_count += 1

                    # Record trade using data manager
                    if self.data_synchronizer:
                        current_price = self.data_synchronizer._get_current_prices(
                            [symbol]
                        ).get(symbol, 0)
                    else:
                        current_price = 0

                    if self.data_manager:
                        self.data_manager._record_trade_transaction(
                            symbol=symbol,
                            side="SELL",
                            quantity=adjusted_quantity,
                            price_usd=current_price or 0.0,
                            source="PROFIT_TAKING",
                            mode=mode,
                            batch_id=batch_id,
                            order=order,
                        )
                else:
                    result.messages.append(
                        f"✅ [DRY RUN] Profit-taking SELL order for {adjusted_quantity:.8f} {symbol} was not placed."
                    )
                    self.logger.info(
                        f"[DRY RUN] Profit-taking SELL order for {adjusted_quantity:.8f} {symbol} was not placed."
                    )
                    trades_executed_count += 1

                    # Record simulated trade
                    if self.data_synchronizer:
                        current_price = self.data_synchronizer._get_current_prices(
                            [symbol]
                        ).get(symbol, 0)
                    else:
                        current_price = 0

                    if self.data_manager:
                        self.data_manager._record_trade_transaction(
                            symbol=symbol,
                            side="SELL",
                            quantity=adjusted_quantity,
                            price_usd=current_price or 0.0,
                            source="PROFIT_TAKING",
                            mode=mode,
                            batch_id=batch_id,
                            order=None,
                        )

            except BinanceAPIException as e:
                result.messages.append(
                    f"❌ LIVE profit-taking SELL FAILED for {symbol}: {e}"
                )
                self.logger.error(f"LIVE profit-taking SELL FAILED for {symbol}: {e}")
                result.errors.append(f"SELL {symbol}: {e}")
                result.success = False
            except Exception as e:
                result.messages.append(
                    f"❌ Unexpected error during profit-taking for {symbol}: {e}"
                )
                self.logger.error(
                    f"Unexpected error during profit-taking for {symbol}: {e}",
                    exc_info=True,
                )
                result.errors.append(f"{symbol}: {e}")
                result.success = False

        result.messages.append("\n" + "=" * 80)
        result.messages.append(
            f"✅ {trades_executed_count} profit-taking trade(s) executed successfully!"
        )
        result.data["trades_executed"] = trades_executed_count
        result.data["batch_id"] = batch_id
        result.success = trades_executed_count > 0 and result.success

        return result

    async def execute_rebalancing_trades_core(
        self,
        suggestions_df,
        earn_balances,
        confirmation_callback: Optional[Callable[[str], bool]] = None,
        execution_mode: ExecutionMode = ExecutionMode.CONFIRM,
    ) -> TradeResult:
        """
        Execute rebalancing trades with UI-agnostic confirmation handling.

        Args:
            suggestions_df: DataFrame with rebalancing suggestions
            earn_balances: Dictionary of earn balances
            confirmation_callback: Optional callback for user confirmation (UI-specific)
            execution_mode: Mode of execution (auto, bulk, interactive, confirm)
        """
        result = TradeResult(success=False)
        trades_executed_count = 0

        if suggestions_df.empty or "Signal" not in suggestions_df.columns:
            result.messages.append("No rebalancing suggestions to execute.")
            return result

        trades_to_execute = suggestions_df[
            suggestions_df["Signal"].isin(["BUY", "SELL"])
        ]
        trades_to_execute = trades_to_execute.sort_values(
            by=["Signal", "Drift (pts)"], ascending=[False, True]
        )

        if trades_to_execute.empty:
            result.messages.append(
                "\n✅ No BUY or SELL actions suggested. Nothing to execute."
            )
            return result

        portfolio_config = self.config.get("portfolio", {})
        min_trade_usd = portfolio_config.get("minimum_trade_usd", 10.0)
        is_live = portfolio_config.get("live_trading_enabled", False)
        mode = (
            "TESTNET"
            if self.config_manager and self.config_manager.is_testnet_mode
            else ("LIVE" if is_live else "SIM")
        )

        # Initialize simulated balances
        simulated_balances = {}
        all_trade_symbols = set(trades_to_execute["Symbol"].unique()) | {"USDT"}
        for symbol in all_trade_symbols:
            try:
                if self.binance_client:
                    spot_bal = float(
                        self.binance_client.get_asset_balance(asset=symbol).get(
                            "free", 0.0
                        )
                    )
                else:
                    spot_bal = 0.0
                earn_bal = earn_balances.get(symbol, 0.0)
                simulated_balances[symbol] = spot_bal + earn_bal
                self.logger.info(
                    f"Initialized simulated balance for {symbol}: {simulated_balances[symbol]:.8f}"
                )
            except Exception as e:
                self.logger.error(f"Could not fetch balance for {symbol}: {e}")
                simulated_balances[symbol] = 0.0

        result.messages.append("\n" + "=" * 80)
        if is_live:
            result.messages.append("🔴🔴🔴 WARNING: Live Trading is ENABLED. 🔴🔴🔴")
        else:
            result.messages.append("🟡🟡🟡 NOTE: Live Trading is DISABLED. 🟡🟡🟡")
        result.messages.append("=" * 80)

        items_to_process = []

        # Handle confirmation based on execution mode
        if execution_mode == ExecutionMode.AUTO:
            # WebUI mode - auto confirm all trades
            items_to_process = list(trades_to_execute.iterrows())

        elif execution_mode == ExecutionMode.BULK:
            # Bulk execution mode - execute all trades immediately
            result.messages.append("🚨 PROPOSED TRADES - PLEASE REVIEW CAREFULLY 🚨")
            result.messages.append(
                trades_to_execute[
                    ["Symbol", "Signal", "Suggested Action Detail"]
                ].to_string(index=False)
            )
            result.messages.append("=" * 80)
            self.logger.info("User confirmed bulk trade execution.")
            items_to_process = list(trades_to_execute.iterrows())

        elif execution_mode == ExecutionMode.INTERACTIVE and confirmation_callback:
            # Interactive mode - ask for each trade individually
            result.messages.append(
                "👀 Entering interactive confirmation mode. You will be prompted for each trade."
            )
            result.messages.append("")

            for _, row in trades_to_execute.iterrows():
                symbol = row["Symbol"]
                signal = row["Signal"]

                # Ask for individual trade approval
                approval_prompt = (
                    f"Approve this trade for {symbol}? Type YES to confirm: "
                )
                approved = confirmation_callback(approval_prompt)

                if approved:
                    self.logger.info(f"User approved trade for {symbol}.")
                    items_to_process.append((_, row))
                else:
                    self.logger.info(f"User rejected trade for {symbol}.")

        elif execution_mode == ExecutionMode.CONFIRM and confirmation_callback:
            # Legacy confirmation mode - use callback for bulk confirmation
            result.messages.append("🚨 PROPOSED TRADES - PLEASE REVIEW CAREFULLY 🚨")
            result.messages.append(
                trades_to_execute[
                    ["Symbol", "Signal", "Suggested Action Detail"]
                ].to_string(index=False)
            )
            result.messages.append("=" * 80)

            # Use callback for confirmation
            confirmed = confirmation_callback(
                "Type EXECUTE ALL to proceed with all trades listed above: "
            )
            if confirmed:
                self.logger.info("User confirmed bulk trade execution.")
                items_to_process = list(trades_to_execute.iterrows())
            else:
                result.messages.append("🛑 Bulk trade execution cancelled by user.")
                return result
        else:
            # No confirmation mechanism available - skip execution
            result.messages.append(
                "🛑 No confirmation mechanism available. Skipping trade execution."
            )
            return result

        if not items_to_process:
            result.messages.append("No trades were executed.")
            return result

        result.messages.append(
            f"Executing {len(items_to_process)} approved trade(s)..."
        )

        # Execute trades
        for _, row in items_to_process:
            symbol = row["Symbol"]
            signal = row["Signal"]
            trade_ticker = f"{symbol}USDT"
            usd_value = row.get("action_usd_value", 0.0)
            coin_quantity = row.get("action_coin_quantity", 0.0)

            if usd_value < min_trade_usd:
                result.messages.append(
                    f"\n⚠️ SKIPPING {signal} for {symbol}: Suggested trade value (~${usd_value:,.2f}) is below the minimum of ${min_trade_usd:,.2f}."
                )
                continue

            try:
                if signal == "SELL":
                    if coin_quantity > simulated_balances.get(symbol, 0.0):
                        result.messages.append(
                            f"⚠️ SKIPPING SELL for {symbol}: Required quantity ({coin_quantity:.8f}) exceeds simulated available balance ({simulated_balances.get(symbol, 0.0):.8f})."
                        )
                        continue

                    adjusted_quantity = self._adjust_quantity_to_lot_size(
                        trade_ticker, coin_quantity
                    )
                    if adjusted_quantity is None or adjusted_quantity <= 0:
                        result.messages.append(
                            f"⚠️ SKIPPING SELL for {symbol}: Adjusted quantity is zero or invalid after applying lot size rules."
                        )
                        continue

                    # Get current price
                    current_price = 0
                    if self.data_synchronizer:
                        prices = self.data_synchronizer._get_current_prices([symbol])
                        current_price = prices.get(symbol, 0)

                    final_notional_value = adjusted_quantity * current_price
                    if final_notional_value < min_trade_usd:
                        result.messages.append(
                            f"⚠️ SKIPPING SELL for {symbol}: Final trade value (~${final_notional_value:,.2f}) is below minimum of ${min_trade_usd:,.2f} after applying exchange rules."
                        )
                        continue

                    # No automatic redemption from Earn - let the trade fail naturally if insufficient balance
                    # Users can manually redeem using the direct redemption functionality if needed

                    result.messages.append(
                        f"\nPreparing MARKET SELL for {adjusted_quantity:.8f} {symbol}..."
                    )

                    if is_live and self.binance_client:
                        try:
                            order = self.binance_client.order_market_sell(
                                symbol=trade_ticker, quantity=f"{adjusted_quantity:.8f}"
                            )
                            result.messages.append(
                                f"✅ LIVE SELL ORDER PLACED: {order}"
                            )
                            self.logger.info(f"LIVE SELL ORDER PLACED: {order}")
                            trades_executed_count += 1
                            simulated_balances[symbol] -= adjusted_quantity

                            # Record trade
                            if self.data_manager:
                                self.data_manager._record_trade_transaction(
                                    symbol=symbol,
                                    side="SELL",
                                    quantity=adjusted_quantity,
                                    price_usd=current_price or 0.0,
                                    source="REBALANCE",
                                    mode=mode,
                                    batch_id=str(uuid.uuid4()),
                                    order=order,
                                )
                        except Exception as e:
                            result.messages.append(
                                f"❌ LIVE SELL FAILED for {symbol}: {e}"
                            )
                            self.logger.error(f"LIVE SELL FAILED for {symbol}: {e}")
                            result.errors.append(f"SELL {symbol}: {e}")
                    else:
                        result.messages.append(
                            f"✅ [DRY RUN] Market SELL order for {adjusted_quantity:.8f} {symbol} was not placed."
                        )
                        self.logger.info(
                            f"[DRY RUN] Market SELL order for {adjusted_quantity:.8f} {symbol} was not placed."
                        )
                        trades_executed_count += 1
                        simulated_balances[symbol] -= adjusted_quantity

                        # Record simulated trade
                        if self.data_manager:
                            self.data_manager._record_trade_transaction(
                                symbol=symbol,
                                side="SELL",
                                quantity=adjusted_quantity,
                                price_usd=current_price or 0.0,
                                source="REBALANCE",
                                mode=mode,
                                batch_id=str(uuid.uuid4()),
                                order=None,
                            )

                elif signal == "BUY":
                    if usd_value > simulated_balances.get("USDT", 0.0):
                        result.messages.append(
                            f"⚠️ SKIPPING BUY for {symbol}: Required USDT ({usd_value:.2f}) exceeds simulated available balance ({simulated_balances.get('USDT', 0.0):.2f})."
                        )
                        continue

                    result.messages.append(
                        f"\nPreparing MARKET BUY for ${usd_value:.2f} of {symbol}..."
                    )

                    if is_live and self.binance_client:
                        try:
                            order = self.binance_client.order_market_buy(
                                symbol=trade_ticker, quoteOrderQty=f"{usd_value:.2f}"
                            )
                            result.messages.append(f"✅ LIVE BUY ORDER PLACED: {order}")
                            self.logger.info(f"LIVE BUY ORDER PLACED: {order}")
                            trades_executed_count += 1
                            simulated_balances["USDT"] -= usd_value

                            # Record trade (approx quantity using current price)
                            price_now = 0
                            if self.data_synchronizer:
                                prices = self.data_synchronizer._get_current_prices(
                                    [symbol]
                                )
                                price_now = prices.get(symbol, 0.0) or 0.0
                            qty = (usd_value / price_now) if price_now > 0 else 0.0

                            if self.data_manager:
                                self.data_manager._record_trade_transaction(
                                    symbol=symbol,
                                    side="BUY",
                                    quantity=qty,
                                    price_usd=price_now,
                                    source="REBALANCE",
                                    mode=mode,
                                    batch_id=str(uuid.uuid4()),
                                    order=order,
                                )
                        except Exception as e:
                            result.messages.append(
                                f"❌ LIVE BUY FAILED for {symbol}: {e}"
                            )
                            self.logger.error(f"LIVE BUY FAILED for {symbol}: {e}")
                            result.errors.append(f"BUY {symbol}: {e}")
                    else:
                        result.messages.append(
                            f"✅ [DRY RUN] Market BUY order for ${usd_value:.2f} of {symbol} was not placed."
                        )
                        self.logger.info(
                            f"[DRY RUN] Market BUY order for ${usd_value:.2f} of {symbol} was not placed."
                        )
                        trades_executed_count += 1
                        simulated_balances["USDT"] -= usd_value

                        price_now = 0
                        if self.data_synchronizer:
                            prices = self.data_synchronizer._get_current_prices(
                                [symbol]
                            )
                            price_now = prices.get(symbol, 0.0) or 0.0
                        qty = (usd_value / price_now) if price_now > 0 else 0.0

                        if self.data_manager:
                            self.data_manager._record_trade_transaction(
                                symbol=symbol,
                                side="BUY",
                                quantity=qty,
                                price_usd=price_now,
                                source="REBALANCE",
                                mode=mode,
                                batch_id=str(uuid.uuid4()),
                                order=None,
                            )

            except Exception as e:
                result.messages.append(
                    f"❌ Unexpected error executing trade for {symbol}: {e}"
                )
                self.logger.error(f"Unexpected error executing trade for {symbol}: {e}")
                result.errors.append(f"{symbol}: {e}")

        result.messages.append("\n" + "=" * 80)
        result.messages.append(
            f"✅ {trades_executed_count} trade(s) executed successfully!"
        )
        result.data["trades_executed"] = trades_executed_count
        result.data["batch_id"] = str(uuid.uuid4())
        result.success = trades_executed_count > 0
        return result

    async def execute_manual_trade_core(
        self, trade_type, symbol, trade_ticker, amount, is_quote_qty, is_live
    ) -> TradeResult:
        """Manual trade execution core logic.
        Args:
            trade_type: "BUY" or "SELL",
            symbol: Symbol to trade,
            trade_ticker: "BTCUSDT" or "ETHUSDT",
            amount: Amount to trade,
            is_quote_qty: True or False,
            is_live: True or False.
        Returns:
            TradeResult object
        """
        result = TradeResult(success=False)
        min_trade_usd = self.config.get("portfolio", {}).get("minimum_trade_usd", 5.0)

        try:
            if trade_type == "BUY":
                usdt_to_spend = amount if is_quote_qty else 0
                if not is_quote_qty:
                    # Get current price
                    prices = {}
                    if self.data_synchronizer:
                        prices = self.data_synchronizer._get_current_prices([symbol])

                    if not prices.get(symbol):
                        result.errors.append(
                            f"Could not fetch price for {symbol} to calculate trade value."
                        )
                        return result
                    usdt_to_spend = amount * prices[symbol]

                if usdt_to_spend < min_trade_usd:
                    result.errors.append(
                        f"SKIPPING BUY for {symbol}: Required value (~${usdt_to_spend:,.2f}) is below the minimum of ${min_trade_usd:,.2f}."
                    )
                    return result

                result.messages.append(f"Preparing MARKET BUY for {symbol}...")

                if is_live and self.binance_client:
                    order = self.binance_client.order_market_buy(
                        symbol=trade_ticker, quoteOrderQty=f"{usdt_to_spend:.2f}"
                    )
                    result.messages.append(f"LIVE BUY ORDER PLACED: {order}")
                    result.data["order"] = order
                else:
                    result.messages.append("(Dry Run) BUY Trade was not placed.")
                result.success = True

            elif trade_type == "SELL":
                coin_quantity_to_sell = amount if not is_quote_qty else 0
                if is_quote_qty:
                    # Get current price
                    prices = {}
                    if self.data_synchronizer:
                        prices = self.data_synchronizer._get_current_prices([symbol])

                    if not prices.get(symbol) or prices[symbol] == 0:
                        result.errors.append(
                            f"Could not fetch price for {symbol} to calculate quantity."
                        )
                        return result
                    coin_quantity_to_sell = amount / prices[symbol]

                adjusted_quantity = self._adjust_quantity_to_lot_size(
                    trade_ticker, coin_quantity_to_sell
                )
                if adjusted_quantity is None or adjusted_quantity <= 0:
                    result.errors.append(
                        f"SKIPPING SELL for {symbol}: Quantity is zero or invalid after applying exchange lot size rules."
                    )
                    return result

                result.messages.append(
                    f"Preparing MARKET SELL for {adjusted_quantity:.8f} {symbol}..."
                )

                if is_live and self.binance_client:
                    order = self.binance_client.order_market_sell(
                        symbol=trade_ticker, quantity=f"{adjusted_quantity:.8f}"
                    )
                    result.messages.append(f"LIVE SELL ORDER PLACED: {order}")
                    result.data["order"] = order
                else:
                    result.messages.append("(Dry Run) SELL Trade was not placed.")
                result.success = True

            result.messages.append(
                "Recommendation: Run 'Full Sync & Analysis' to update your portfolio with this trade."
            )

        except Exception as e:
            result.errors.append(f"Unexpected Error for {symbol}: {e}")

        return result
