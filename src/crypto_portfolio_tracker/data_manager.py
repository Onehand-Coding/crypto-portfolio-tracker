"""
Data Manager - Handles all data persistence and transaction management
Moved from CryptoPortfolioTracker to separate concerns.
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from .utils import calculate_fifo_cost_basis


class DataManager:
    """
    Handles all data persistence and transaction management operations including:
    - Strategy state loading and saving
    - Transaction recording
    - Holdings updates with FIFO cost basis
    - Portfolio snapshots
    - Data cleanup operations
    """

    def __init__(self, config: Dict[str, Any], db_manager=None,
                 strategy_state_path: Optional[Path] = None):
        """
        Initialize DataManager with necessary dependencies.

        Args:
            config: Configuration dictionary
            db_manager: Database manager instance
            strategy_state_path: Path to strategy state JSON file
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.strategy_state_path = strategy_state_path
        self.strategy_states = {}

        # Load strategy states if path is provided
        if self.strategy_state_path:
            self.strategy_states = self._load_strategy_state()

    def _load_strategy_state(self) -> Dict[str, Any]:
        """Loads the state of all strategies from a JSON file."""
        if not self.strategy_state_path or not self.strategy_state_path.exists():
            return {}

        try:
            with open(self.strategy_state_path, "r") as f:
                states = json.load(f)
                self.logger.info(
                    f"Loaded strategy states from {self.strategy_state_path}"
                )
                return states
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(
                f"Error loading strategy state file: {e}. Starting fresh."
            )
            return {}

    def _save_strategy_state(self):
        """Saves the current state of all strategies to a JSON file."""
        if not self.strategy_state_path:
            self.logger.warning("No strategy state path configured, cannot save")
            return

        try:
            with open(self.strategy_state_path, "w") as f:
                json.dump(self.strategy_states, f, indent=4)
                self.logger.info(f"Saved strategy states to {self.strategy_state_path}")
        except IOError as e:
            self.logger.error(f"Error saving strategy state file: {e}")

    def _make_tx_hash(
        self, source: str, batch_id: str, symbol: str, side: str,
        ts: datetime.datetime, order: Optional[dict] = None
    ) -> str:
        """Generate a unique transaction hash for deduplication."""
        symbol_upper = symbol.upper()
        order_id = ""
        if order and isinstance(order, dict):
            order_id = str(order.get("orderId", ""))

        # Create hash from source:batch_id:symbol:side:timestamp_ms[:order_id]
        base_hash = f"{source}:{batch_id}:{symbol_upper}:{side}:{int(ts.timestamp() * 1000)}"
        if order_id:
            base_hash += f":{order_id}"

        return base_hash

    def _record_trade_transaction(
        self,
        *,
        symbol: str,
        side: str,  # "BUY" | "SELL"
        quantity: float,
        price_usd: float,
        source: str,  # "REBALANCE" | "DCA" | "PROFIT_TAKING" | "MANUAL"
        mode: str,  # "LIVE" | "TESTNET" | "SIM"
        batch_id: str,
        order: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a trade transaction to the database.

        Args:
            symbol: Asset symbol (e.g., "BTC")
            side: Trade side ("BUY" or "SELL")
            quantity: Quantity of the asset traded
            price_usd: Price in USD
            source: Source of the trade
            mode: Execution mode
            batch_id: Batch identifier for grouping trades
            order: Optional order data from exchange
            error: Optional error message if trade failed
        """
        if not self.db_manager:
            self.logger.error("Database manager not available for recording transaction")
            return

        try:
            ts = datetime.datetime.now(datetime.timezone.utc)
            notes = {
                "batch_id": batch_id,
                "mode": mode,
                "source": source,
                "status": "FILLED" if not error else "ERROR",
                "error": error,
            }

            if order and isinstance(order, dict):
                # Keep only lightweight identifiers
                notes["order"] = {
                    k: order.get(k)
                    for k in (
                        "orderId",
                        "clientOrderId",
                        "transactTime",
                        "symbol",
                        "side",
                    )
                }

            tx = {
                "symbol": symbol,
                "timestamp": ts,
                "type": side,  # Must be BUY/SELL per schema
                "quantity": float(quantity) if quantity is not None else 0.0,
                "price_usd": float(price_usd) if price_usd is not None else 0.0,
                "fee_quantity": None,
                "fee_currency": None,
                "fee_usd": None,
                "source": source,
                "notes": json.dumps(notes, default=str),
                "transaction_hash": self._make_tx_hash(
                    source, batch_id, symbol, side, ts, order
                ),
            }

            self.db_manager.bulk_insert_transactions([tx])
            self.logger.debug(f"Recorded {side} transaction for {symbol}: {quantity} @ ${price_usd}")

        except Exception as e:
            self.logger.error(f"Failed to record trade transaction: {e}", exc_info=True)

    def update_holdings_from_transactions(self):
        """
        Processes all transactions and updates holdings table with FIFO cost basis.
        """
        if not self.db_manager:
            self.logger.error("Database manager not available for updating holdings")
            return

        self.logger.info("Updating holdings from transaction history using FIFO...")
        all_txs = self.db_manager.get_all_transactions()

        if all_txs.empty:
            self.logger.warning("No transactions found in DB. Cannot update holdings.")
            return

        updated_holdings = []
        for symbol, group_df in all_txs.groupby("symbol"):
            self.logger.debug(f"Calculating FIFO for {symbol}...")

            group_df_copy = group_df.copy()
            group_df_copy["price_usd"] = pd.to_numeric(
                group_df_copy["price_usd"], errors="coerce"
            ).fillna(0.0)
            group_df_copy["quantity"] = pd.to_numeric(
                group_df_copy["quantity"], errors="coerce"
            ).fillna(0.0)
            group_df_copy["timestamp"] = pd.to_datetime(
                group_df_copy["timestamp"], errors="coerce"
            )
            group_df_copy.dropna(subset=["timestamp"], inplace=True)

            # 1. Isolate real trades for cost basis calculation
            # Exclude transfer transactions, Simple Earn activities, and Staking activities
            cost_basis_tx_df = group_df_copy[
                ~group_df_copy["source"].str.contains(
                    "Simple Earn|Binance Transfer|Staking", case=False, na=False
                )
            ]
            self.logger.debug(
                f"Calculating cost basis for {symbol} using {len(cost_basis_tx_df)} non-transfer transactions."
            )

            if cost_basis_tx_df.empty:
                self.logger.debug(
                    f"No non-transfer transactions for {symbol}. Cannot calculate cost basis."
                )
                continue  # Skip to the next symbol

            # 2. Calculate cost basis and the remaining quantity FROM THAT BASIS
            cost_basis_qty, avg_cost = calculate_fifo_cost_basis(cost_basis_tx_df)

            # 3. If there's a valid average cost, save it.
            # The quantity saved here is just a placeholder; the final report uses the live wallet balance.
            if avg_cost > 0:
                self.logger.debug(
                    f"Calculated for {symbol}: Qty_from_basis={cost_basis_qty:.8f}, AvgCost={avg_cost:.8f}. Storing avg_cost."
                )
                updated_holdings.append(
                    {
                        "symbol": symbol,
                        "quantity": cost_basis_qty,
                        "average_cost_basis": avg_cost,
                    }
                )
            else:
                self.logger.debug(
                    f"No cost basis calculated for {symbol} (likely no 'BUY' transactions in history)."
                )

        if updated_holdings:
            holdings_df = pd.DataFrame(updated_holdings)
            self.db_manager.update_holdings(holdings_df)
            self.logger.info(
                f"Successfully updated/inserted {len(holdings_df)} asset holdings in the database with new cost basis."
            )
        else:
            self.logger.warning(
                "No holdings with valid cost basis to update in the database."
            )

    def save_snapshot(self, metrics: Dict[str, Any]):
        """Save a portfolio snapshot using data from calculated metrics."""
        if not self.db_manager:
            self.logger.error("Database manager not available for saving snapshot")
            return

        if "error" in metrics or "total_value_usd" not in metrics:
            self.logger.warning(
                "Skipping snapshot save due to missing data in metrics."
            )
            return

        timestamp = metrics.get(
            "timestamp", datetime.datetime.now(datetime.timezone.utc)
        )
        total_value = metrics.get("total_value_usd", 0)
        total_cost_basis = metrics.get("total_cost_basis_usd", 0)
        unrealized_pl_usd = metrics.get("unrealized_pl_usd", 0)
        unrealized_pl_percent = metrics.get("unrealized_pl_percent", 0)

        # Pass all relevant metrics to the database manager
        self.db_manager.save_portfolio_snapshot(
            timestamp=timestamp,
            total_value=total_value,
            total_cost_basis=total_cost_basis,
            unrealized_pl=unrealized_pl_usd,
            unrealized_pl_percent=unrealized_pl_percent,
        )

        self.logger.info(f"Saved portfolio snapshot: ${total_value:,.2f} total value")

    def cleanup_old_data(self):
        """Clean up old data using the database manager."""
        if self.db_manager:
            self.db_manager.cleanup_old_data()
            self.logger.info("Completed data cleanup")
        else:
            self.logger.warning("Database manager not available for cleanup")

    def get_strategy_state(self, strategy_name: str) -> Dict[str, Any]:
        """Get the state for a specific strategy."""
        return self.strategy_states.get(strategy_name, {})

    def set_strategy_state(self, strategy_name: str, state: Dict[str, Any]):
        """Set the state for a specific strategy."""
        self.strategy_states[strategy_name] = state
        self._save_strategy_state()

    def update_strategy_state(self, strategy_name: str, updates: Dict[str, Any]):
        """Update specific fields in a strategy's state."""
        if strategy_name not in self.strategy_states:
            self.strategy_states[strategy_name] = {}

        self.strategy_states[strategy_name].update(updates)
        self._save_strategy_state()

    def clear_strategy_state(self, strategy_name: str):
        """Clear the state for a specific strategy."""
        if strategy_name in self.strategy_states:
            del self.strategy_states[strategy_name]
            self._save_strategy_state()

    def get_all_strategy_states(self) -> Dict[str, Any]:
        """Get all strategy states."""
        return self.strategy_states.copy()

    def backup_data(self) -> Dict[str, Any]:
        """Create a backup of important data."""
        if not self.db_manager:
            self.logger.error("Database manager not available for backup")
            return {}

        backup_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy_states": self.strategy_states,
            "transactions_count": len(self.db_manager.get_all_transactions()),
            "holdings_count": len(self.db_manager.get_holdings()),
            "snapshots_count": len(self.db_manager.get_all_snapshots()),
        }

        self.logger.info(f"Created data backup summary: {backup_data}")
        return backup_data

    def restore_strategy_states(self, states: Dict[str, Any]):
        """Restore strategy states from backup."""
        self.strategy_states = states.copy()
        self._save_strategy_state()
        self.logger.info(f"Restored {len(states)} strategy states")
