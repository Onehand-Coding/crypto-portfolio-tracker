# src/binance_fetcher.py

import time
import logging
import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from .symbol_mapper import SymbolMapper
from .exceptions import NetworkOperationError


class BinanceFetcher:
    """
    A class dedicated to fetching all types of raw data from the Binance API.
    
    This class handles fetching various types of data from Binance including:
    - Account balances (spot, futures, funding)
    - Transaction history (trades, deposits, withdrawals)
    - Earn product balances and history
    - Dividend history
    - Staking history
    - P2P buy history
    - Spot convert history
    
    It also provides functionality for transferring funds between wallets.
    """

    def __init__(
        self,
        client: Optional[Client],
        symbol_mapper: SymbolMapper,
        config: Dict[str, Any],
    ):
        """
        Initialize the Binance fetcher.
        
        Args:
            client: Initialized Binance client
            symbol_mapper: SymbolMapper instance for asset symbol normalization
            config: Configuration dictionary
            
        Raises:
            ValueError: If client is not provided
        """
        if not client:
            raise ValueError("BinanceFetcher requires an initialized Binance client.")
        self.logger = logging.getLogger(__name__)
        self.binance_client = client
        self.symbol_mappings = symbol_mapper
        self.config = config
        self.target_assets_for_sync = set(
            self.config.get("target_allocation", {}).keys()
        )
        self.target_assets_for_sync.add("USDT")
        self.logger.debug("BinanceFetcher initialized.")

    def _handle_binance_api_exception(self, e: BinanceAPIException, context: str = "") -> bool:
        """
        Centralized error handling for Binance API exceptions.
        
        Args:
            e: The BinanceAPIException that was raised
            context: Additional context about where the error occurred
            
        Returns:
            bool: True if the error is recoverable and retry might help, False otherwise
        """
        error_code = getattr(e, 'code', None)
        error_msg = str(e)
        
        # Log the error with context
        self.logger.error(f"Binance API Error {context}: {error_msg} (Code: {error_code})")
        
        # Handle specific error codes
        if error_code == -1021:
            # Timestamp out of recvWindow - try to resync time
            self.logger.warning(f"Timestamp sync issue {context}. Attempting to resync...")
            self._resync_time()
            return True  # Indicate we can retry
            
        elif error_code == -2015:
            # Invalid API key/permissions
            self.logger.error(f"API key/permission error {context}. Check your API credentials.")
            print("❌ Binance API error: Invalid API-key, IP, or permissions for action. Please check your API credentials and permissions.")
            return False  # Don't retry
            
        elif error_code == -1121:
            # Invalid symbol
            self.logger.debug(f"Invalid symbol {context}. This might be a temporary issue.")
            return False  # Don't retry
            
        elif error_code in [-6001, -11001]:
            # Invalid asset for Simple Earn
            self.logger.debug(f"Asset not supported for Simple Earn {context}.")
            return False  # Don't retry
            
        elif "not supported" in error_msg.lower() or "invalid asset" in error_msg.lower():
            # Handle string-based error messages for unsupported assets
            self.logger.debug(f"Asset not supported {context}: {error_msg}")
            return False  # Don't retry
            
        else:
            # Unknown error - log and don't retry by default
            self.logger.error(f"Unhandled Binance API error {context}: {error_msg} (Code: {error_code})")
            return False  # Don't retry by default

    def _get_start_end_timestamps(
        self,
        source_name: str,
        days_back: int,
        latest_known_ts: Optional[datetime.datetime],
    ) -> Optional[tuple[int, int]]:
        """Helper to calculate the start and end timestamps for an API call."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        end_ts = int(now_utc.timestamp() * 1000)

        start_dt: datetime.datetime
        if latest_known_ts:
            effective_start_from_db = latest_known_ts - datetime.timedelta(
                minutes=60
            )  # 1 hour buffer
            fallback_start_from_days = now_utc - datetime.timedelta(days=days_back)
            start_dt = max(effective_start_from_db, fallback_start_from_days)
        else:
            start_dt = now_utc - datetime.timedelta(days=days_back)

        if start_dt >= now_utc:
            self.logger.debug(f"[{source_name}] History is up-to-date. Skipping fetch.")
            return None

        start_ts = int(start_dt.timestamp() * 1000)
        self.logger.debug(
            f"Fetching [{source_name}] data from {start_dt.strftime('%Y-%m-%d %H:%M')} to {now_utc.strftime('%Y-%m-%d %H:%M')}"
        )
        return start_ts, end_ts

    def _fetch_paginated_history(
        self, endpoint_path: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generic helper to handle paginated endpoints that use 'current' and 'size'."""
        all_rows = []
        params["current"] = 1
        params["size"] = 100  # Max for many of these endpoints

        while True:
            try:
                history = self.binance_client._request_margin_api(
                    "get", endpoint_path, True, data=params
                )
                rows = (
                    history.get("rows", [])
                    if isinstance(history, dict)
                    else history
                    if isinstance(history, list)
                    else []
                )
                if not rows:
                    break
                all_rows.extend(rows)
                if len(rows) < params["size"]:
                    break  # Reached the last page
                params["current"] += 1
            except Exception as e:
                self.logger.error(
                    f"Error fetching paginated data from {endpoint_path}: {e}"
                )
                break
        return all_rows

    def fetch_binance_balances(self) -> pd.DataFrame:
        """Fetch current balances from Binance Spot wallet."""
        if not self.binance_client:
            self.logger.warning(
                "Binance client not initialized. Cannot fetch Spot balances."
            )
            return pd.DataFrame(columns=["symbol", "quantity"])

        try:
            self.logger.info(
                "Re-synchronizing time with Binance server for fresh data..."
            )
            server_time = self.binance_client.get_server_time()["serverTime"]
            local_time = int(time.time() * 1000)
            time_offset = server_time - local_time
            self.binance_client._server_time_offset = time_offset
            self.logger.info(
                f"Time re-synchronized for balance fetch. Offset: {time_offset}ms."
            )
        except Exception as e:
            self.logger.error(
                f"Could not re-sync time for balance fetch: {e}. Proceeding with old offset."
            )

        self.logger.info("Fetching current spot wallet balances...")
        
        # Add retry logic for balance fetching
        retries = 3
        wait_time_seconds = 15
        api_timeout = self.config.get("apis", {}).get("binance", {}).get("timeout", 180)
        
        while retries > 0:
            try:
                account_info = self.binance_client.get_account()

                balances_raw = account_info.get("balances", [])
                if not balances_raw:
                    self.logger.info("Fetched 0 balances from Binance Spot wallet.")
                    return pd.DataFrame(columns=["symbol", "quantity"])

                processed_balances = []
                for b in balances_raw:
                    quantity = float(b.get("free", 0.0)) + float(b.get("locked", 0.0))
                    if quantity > 1e-8:
                        final_symbol = self.symbol_mappings.normalize_symbol(
                            b.get("asset", "")
                        )

                        # Skip known fiat currencies to prevent unnecessary discovery attempts
                        if final_symbol in [
                            "EUR",
                            "JPY",
                            "BRL",
                            "ARS",
                            "CZK",
                            "MXN",
                            "UAH",
                            "ZAR",
                            "TRY",
                        ]:
                            self.logger.debug(
                                f"Skipping known fiat currency: {final_symbol}"
                            )
                            continue

                        if final_symbol:
                            processed_balances.append(
                                {"symbol": final_symbol, "quantity": quantity}
                            )

                if not processed_balances:
                    return pd.DataFrame(columns=["symbol", "quantity"])

                df = pd.DataFrame(processed_balances)
                # Consolidate any duplicate symbols (e.g. from LD-prefixed assets)
                df = df.groupby("symbol", as_index=False)["quantity"].sum()

                self.logger.debug(f"Fetched and consolidated {len(df)} non-zero balances.")
                return df
                
            except BinanceAPIException as e:
                # Use our centralized error handling
                if self._handle_binance_api_exception(e, "fetching spot balances"):
                    # For recoverable errors, retry
                    retries -= 1
                    if retries > 0:
                        self.logger.info(f"Retrying balance fetch in {wait_time_seconds}s... ({retries} retries left)")
                        time.sleep(wait_time_seconds)
                    else:
                        self.logger.error("Failed to fetch Spot balances after multiple retries.")
                        break
                else:
                    # For non-recoverable errors, don't retry
                    self.logger.error(f"Non-recoverable Binance API Error fetching Spot balances: {e}")
                    break
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ) as e:
                retries -= 1
                self.logger.error(
                    f"Network error fetching Binance Spot balances: {e}. Retries left: {retries}."
                )
                if retries > 0:
                    self.logger.info(f"Waiting {wait_time_seconds}s before retrying...")
                    time.sleep(wait_time_seconds)
                else:
                    self.logger.error(
                        "Failed to fetch Spot balances after multiple retries due to network errors."
                    )
            except Exception as e:
                self.logger.error(
                    f"Unexpected error fetching Spot balances: {e}", exc_info=True
                )
                break
                
        return pd.DataFrame(columns=["symbol", "quantity"])

    def fetch_binance_transactions(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ts, _ = time_window
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_chunk_start_dt = pd.to_datetime(
            start_ts, unit="ms", utc=True
        ).to_pydatetime()
        raw_transactions = []
        stablecoin_quotes = self.config.get("portfolio", {}).get(
            "stablecoin_symbols", ["USDT"]
        )
        crypto_quotes = [
            s
            for s in self.config.get("portfolio", {}).get("crypto_quotes", ["BTC"])
            if s in self.target_assets_for_sync
        ]
        all_quotes_to_check = sorted(list(set(stablecoin_quotes + crypto_quotes)))
        potential_base_assets = [
            s for s in self.target_assets_for_sync if s not in stablecoin_quotes
        ]
        processed_pairs = set()

        try:
            for base_asset in potential_base_assets:
                for quote_asset in all_quotes_to_check:
                    if base_asset == quote_asset:
                        continue
                    pair = f"{base_asset}{quote_asset}"
                    if pair in processed_pairs:
                        continue
                    processed_pairs.add(pair)

                    chunk_start = current_chunk_start_dt
                    while chunk_start < now_utc:
                        chunk_end = min(
                            chunk_start + datetime.timedelta(hours=23), now_utc
                        )
                        try:
                            trades = self.binance_client.get_my_trades(
                                symbol=pair,
                                startTime=int(chunk_start.timestamp() * 1000),
                                endTime=int(chunk_end.timestamp() * 1000),
                                limit=1000,
                            )
                            for trade in trades:
                                timestamp = pd.to_datetime(
                                    trade.get("time"), unit="ms", utc=True
                                )
                                if pd.isna(timestamp):
                                    self.logger.warning(
                                        f"Skipping trade with invalid timestamp: {trade}"
                                    )
                                    continue
                                raw_transactions.append(
                                    {
                                        "tx_type": "TRADE",
                                        "timestamp": timestamp,
                                        "source": "Binance Trade",
                                        "transaction_hash": f"binance_trade_{trade['id']}",
                                        "raw_data": {
                                            "base_asset": self.symbol_mappings.normalize_symbol(
                                                base_asset
                                            ),
                                            "quote_asset": self.symbol_mappings.normalize_symbol(
                                                quote_asset
                                            ),
                                            "is_buyer": trade["isBuyer"],
                                            "quantity": float(trade["qty"]),
                                            "price": float(trade["price"]),
                                            "fee_quantity": float(trade["commission"]),
                                            "fee_currency": self.symbol_mappings.normalize_symbol(
                                                trade["commissionAsset"]
                                            ),
                                        },
                                    }
                                )
                        except BinanceAPIException as e:
                            # Use our centralized error handling for trade fetching
                            if e.code == -1121:
                                self.logger.debug(f"Invalid pair: {pair}. Skipping.")
                                break  # Invalid pair, no point retrying
                            else:
                                # For other API errors, use our centralized handling
                                if self._handle_binance_api_exception(e, f"fetching trades for {pair} in {source_name}"):
                                    # If recoverable, we could retry, but for now just log and continue
                                    self.logger.warning(
                                        f"[{source_name}] Recoverable API Error for {pair}, continuing with other pairs: {e}"
                                    )
                                else:
                                    # Non-recoverable error, log and break this pair
                                    self.logger.error(
                                        f"[{source_name}] Non-recoverable API Error fetching trades for {pair}: {e}"
                                    )
                                    break
                        except Exception as e:
                            self.logger.error(
                                f"[{source_name}] Unexpected error fetching trades for {pair}: {e}"
                            )
                        chunk_start = chunk_end
            return raw_transactions
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching Binance transactions: {e}", exc_info=True
            )
            raise NetworkOperationError(f"Failed to fetch Binance transactions: {e}")

    def fetch_deposit_history(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ms, end_ms = time_window
        raw_transactions = []
        try:
            for deposit in self.binance_client.get_deposit_history(
                startTime=start_ms, endTime=end_ms, status=1
            ):
                timestamp = pd.to_datetime(
                    deposit.get("insertTime"), unit="ms", utc=True
                )
                if pd.isna(timestamp):
                    continue
                normalized_symbol = self.symbol_mappings.normalize_symbol(
                    deposit.get("coin").upper()
                )
                if normalized_symbol not in self.target_assets_for_sync:
                    continue
                raw_transactions.append(
                    {
                        "tx_type": "DEPOSIT",
                        "timestamp": timestamp,
                        "source": "Binance Deposit",
                        "transaction_hash": deposit.get("txId"),
                        "raw_data": {
                            "symbol": normalized_symbol,
                            "quantity": float(deposit.get("amount", 0)),
                            "notes": f"Network: {deposit.get('network')}",
                        },
                    }
                )
            return raw_transactions
        except BinanceAPIException as e:
            # Use our centralized error handling
            if self._handle_binance_api_exception(e, f"fetching deposit history for {source_name}"):
                # If error is recoverable, try once more
                try:
                    for deposit in self.binance_client.get_deposit_history(
                        startTime=start_ms, endTime=end_ms, status=1
                    ):
                        timestamp = pd.to_datetime(
                            deposit.get("insertTime"), unit="ms", utc=True
                        )
                        if pd.isna(timestamp):
                            continue
                        normalized_symbol = self.symbol_mappings.normalize_symbol(
                            deposit.get("coin").upper()
                        )
                        if normalized_symbol not in self.target_assets_for_sync:
                            continue
                        raw_transactions.append(
                            {
                                "tx_type": "DEPOSIT",
                                "timestamp": timestamp,
                                "source": "Binance Deposit",
                                "transaction_hash": deposit.get("txId"),
                                "raw_data": {
                                    "symbol": normalized_symbol,
                                    "quantity": float(deposit.get("amount", 0)),
                                    "notes": f"Network: {deposit.get('network')}",
                                },
                            }
                        )
                    return raw_transactions
                except BinanceAPIException as e2:
                    self.logger.error(
                        f"Error fetching deposit history after retry: {e2}",
                        exc_info=True,
                    )
                    print(f"❌ Binance API error: {e2}")
            # Non-recoverable error or retry failed
            return raw_transactions
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching deposit history: {e}", exc_info=True
            )
            print(f"❌ Unexpected error: {e}")

    def fetch_withdrawal_history(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ms, end_ms = time_window
        raw_transactions = []
        try:
            for withdrawal in self.binance_client.get_withdraw_history(
                startTime=start_ms, endTime=end_ms, status=6
            ):
                timestamp = pd.to_datetime(withdrawal.get("applyTime"), utc=True)
                if pd.isna(timestamp):
                    continue
                normalized_symbol = self.symbol_mappings.normalize_symbol(
                    withdrawal.get("coin").upper()
                )
                if normalized_symbol not in self.target_assets_for_sync:
                    continue
                raw_transactions.append(
                    {
                        "tx_type": "WITHDRAWAL",
                        "timestamp": timestamp,
                        "source": "Binance Withdrawal",
                        "transaction_hash": withdrawal.get("txId")
                        or f"binance_withdraw_{withdrawal.get('id')}",
                        "raw_data": {
                            "symbol": normalized_symbol,
                            "quantity": float(withdrawal.get("amount", 0)),
                            "fee_quantity": float(
                                withdrawal.get("transactionFee", 0.0)
                            ),
                            "fee_currency": normalized_symbol,
                        },
                    }
                )
            return raw_transactions
        except BinanceAPIException as e:
            # Use our centralized error handling
            if self._handle_binance_api_exception(e, f"fetching withdrawal history for {source_name}"):
                # If error is recoverable, try once more
                try:
                    for withdrawal in self.binance_client.get_withdraw_history(
                        startTime=start_ms, endTime=end_ms, status=6
                    ):
                        timestamp = pd.to_datetime(
                            withdrawal.get("applyTime"), utc=True
                        )
                        if pd.isna(timestamp):
                            continue
                        normalized_symbol = self.symbol_mappings.normalize_symbol(
                            withdrawal.get("coin").upper()
                        )
                        if normalized_symbol not in self.target_assets_for_sync:
                            continue
                        raw_transactions.append(
                            {
                                "tx_type": "WITHDRAWAL",
                                "timestamp": timestamp,
                                "source": "Binance Withdrawal",
                                "transaction_hash": withdrawal.get("txId")
                                or f"binance_withdraw_{withdrawal.get('id')}",
                                "raw_data": {
                                    "symbol": normalized_symbol,
                                    "quantity": float(withdrawal.get("amount", 0)),
                                    "fee_quantity": float(
                                        withdrawal.get("transactionFee", 0.0)
                                    ),
                                    "fee_currency": normalized_symbol,
                                },
                            }
                        )
                    return raw_transactions
                except BinanceAPIException as e2:
                    self.logger.error(
                        f"Error fetching withdrawal history after retry: {e2}",
                        exc_info=True,
                    )
                    print(f"❌ Binance API error: {e2}")
            # Non-recoverable error or retry failed
            return raw_transactions
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching withdrawal history: {e}", exc_info=True
            )
            print(f"❌ Unexpected error: {e}")

    def fetch_p2p_usdt_buys(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetches P2P buy history, filtering for COMPLETED trades first and then
        de-duplicating by orderNumber to ensure absolute accuracy.
        """
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ms, end_ms = time_window

        p2p_fiat_currency = (
            self.config.get("portfolio", {}).get("p2p_fiat_currency", "PHP").upper()
        )

        all_completed_trades = []
        current_page = 1
        PAGE_SIZE = 100

        try:
            while True:
                history = self.binance_client.get_c2c_trade_history(
                    tradeType="BUY",
                    page=current_page,
                    rows=PAGE_SIZE,
                    startTimestamp=start_ms,
                    endTimestamp=end_ms,
                )
                trades_in_page = history.get("data", [])
                if not trades_in_page:
                    break

                for trade in trades_in_page:
                    # Step 1: Filter for only completed trades. The status is 'COMPLETED'.
                    if trade.get("orderStatus") == "COMPLETED":
                        all_completed_trades.append(trade)

                if len(trades_in_page) < PAGE_SIZE:
                    break
                current_page += 1
                time.sleep(0.5)
            # Step 2: De-duplicate the COMPLETED trades by orderNumber to be 100% safe.
            # This handles the case where the API might send the same completed order twice.
            unique_trades = {
                trade["orderNumber"]: trade for trade in all_completed_trades
            }
            final_trades = list(unique_trades.values())

            raw_transactions = []
            for trade in final_trades:
                raw_transactions.append(
                    {
                        "tx_type": "P2P_BUY",
                        "timestamp": pd.to_datetime(
                            trade.get("createTime"), unit="ms", utc=True
                        ),
                        "source": "Binance P2P Buy",
                        "transaction_hash": trade.get("orderNumber"),
                        "raw_data": {
                            "asset": "USDT",
                            "quantity": float(trade.get("amount", 0)),
                            "fiat_currency": p2p_fiat_currency,
                            "fiat_amount": float(trade.get("totalPrice", 0)),
                        },
                    }
                )

            self.logger.debug(
                f"Found {len(raw_transactions)} COMPLETED and de-duplicated P2P buy transactions."
            )
            return raw_transactions
        except BinanceAPIException as e:
            # Use our centralized error handling
            if self._handle_binance_api_exception(e, f"fetching P2P USDT buys for {source_name}"):
                # If error is recoverable, try once more
                try:
                    current_page = 1
                    all_completed_trades = []
                    while True:
                        history = self.binance_client.get_c2c_trade_history(
                            tradeType="BUY",
                            page=current_page,
                            rows=PAGE_SIZE,
                            startTimestamp=start_ms,
                            endTimestamp=end_ms,
                        )
                        trades_in_page = history.get("data", [])
                        if not trades_in_page:
                            break
                        for trade in trades_in_page:
                            if trade.get("orderStatus") == "COMPLETED":
                                all_completed_trades.append(trade)
                        if len(trades_in_page) < PAGE_SIZE:
                            break
                        current_page += 1
                    # Step 2: De-duplicate the COMPLETED trades by orderNumber to be 100% safe.
                    # This handles the case where the API might send the same completed order twice.
                    unique_trades = {
                        trade["orderNumber"]: trade for trade in all_completed_trades
                    }
                    final_trades = list(unique_trades.values())

                    raw_transactions = []
                    for trade in final_trades:
                        raw_transactions.append(
                            {
                                "tx_type": "P2P_BUY",
                                "timestamp": pd.to_datetime(
                                    trade.get("createTime"), unit="ms", utc=True
                                ),
                                "source": "Binance P2P Buy",
                                "transaction_hash": trade.get("orderNumber"),
                                "raw_data": {
                                    "asset": "USDT",
                                    "quantity": float(trade.get("amount", 0)),
                                    "fiat_currency": p2p_fiat_currency,
                                    "fiat_amount": float(trade.get("totalPrice", 0)),
                                },
                            }
                        )

                    self.logger.debug(
                        f"Found {len(raw_transactions)} COMPLETED and de-duplicated P2P buy transactions."
                    )
                    return raw_transactions
                except BinanceAPIException as e2:
                    self.logger.error(
                        f"Error fetching P2P USDT buys after retry: {e2}",
                        exc_info=True,
                    )
                    print(f"❌ Binance API error: {e2}")
            # Non-recoverable error or retry failed
            return raw_transactions
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching P2P USDT buys: {e}", exc_info=True
            )
            print(f"❌ Unexpected error: {e}")

    def fetch_spot_convert_history(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ms, end_ms = time_window
        raw_transactions = []
        try:
            history = self.binance_client.get_convert_trade_history(
                startTime=start_ms, endTime=end_ms, limit=1000
            )
            for trade in history.get("list", []):
                if trade.get("orderStatus") == "SUCCESS":
                    timestamp = pd.to_datetime(
                        trade.get("createTime"), unit="ms", utc=True
                    )
                    if pd.isna(timestamp):
                        continue
                    from_asset = self.symbol_mappings.normalize_symbol(
                        trade["fromAsset"].upper()
                    )
                    to_asset = self.symbol_mappings.normalize_symbol(
                        trade["toAsset"].upper()
                    )
                    if not (
                        from_asset in self.target_assets_for_sync
                        or to_asset in self.target_assets_for_sync
                    ):
                        continue
                    raw_transactions.append(
                        {
                            "tx_type": "CONVERT",
                            "timestamp": timestamp,
                            "source": "Binance Convert",
                            "transaction_hash": f"convert_{trade.get('quoteId')}",
                            "raw_data": {
                                "from_asset": from_asset,
                                "from_quantity": float(trade["fromAmount"]),
                                "to_asset": to_asset,
                                "to_quantity": float(trade["toAmount"]),
                            },
                        }
                    )
            return raw_transactions
        except BinanceAPIException as e:
            # Use our centralized error handling
            if self._handle_binance_api_exception(e, f"fetching spot convert history for {source_name}"):
                # If error is recoverable, try once more
                try:
                    history = self.binance_client.get_convert_trade_history(
                        startTime=start_ms, endTime=end_ms, limit=1000
                    )
                    for trade in history.get("list", []):
                        if trade.get("orderStatus") == "SUCCESS":
                            timestamp = pd.to_datetime(
                                trade.get("createTime"), unit="ms", utc=True
                            )
                            if pd.isna(timestamp):
                                continue
                            from_asset = self.symbol_mappings.normalize_symbol(
                                trade["fromAsset"].upper()
                            )
                            to_asset = self.symbol_mappings.normalize_symbol(
                                trade["toAsset"].upper()
                            )
                            if not (
                                from_asset in self.target_assets_for_sync
                                or to_asset in self.target_assets_for_sync
                            ):
                                continue
                            raw_transactions.append(
                                {
                                    "tx_type": "CONVERT",
                                    "timestamp": timestamp,
                                    "source": "Binance Convert",
                                    "transaction_hash": f"convert_{trade.get('quoteId')}",
                                    "raw_data": {
                                        "from_asset": from_asset,
                                        "from_quantity": float(trade["fromAmount"]),
                                        "to_asset": to_asset,
                                        "to_quantity": float(trade["toAmount"]),
                                    },
                                }
                            )
                    return raw_transactions
                except BinanceAPIException as e2:
                    self.logger.error(
                        f"Error fetching spot convert history after retry: {e2}",
                        exc_info=True,
                    )
                    print(f"❌ Binance API error: {e2}")
            # Non-recoverable error or retry failed
            return raw_transactions
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching spot convert history: {e}", exc_info=True
            )
            print(f"❌ Unexpected error: {e}")

    def fetch_simple_earn_balances(
        self, spot_balances_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Efficiently fetch balances from Binance Simple Earn Flexible products
        by only checking for assets that exist in the provided spot balances dataframe.
        """
        if not self.binance_client or spot_balances_df.empty:
            self.logger.info(
                "Binance client not available or no spot balances to check for Earn positions."
            )
            return {}

        earn_balances_aggregated: Dict[str, float] = {}
        assets_to_check_api = spot_balances_df["symbol"].unique().tolist()
        self.logger.info(
            f"Fetching Simple Earn Flexible balances for the {len(assets_to_check_api)} assets found in your spot wallet..."
        )

        for asset_api_name in assets_to_check_api:
            try:
                if not asset_api_name:
                    continue
                positions = (
                    self.binance_client.get_simple_earn_flexible_product_position(
                        asset=asset_api_name
                    )
                )
                if (
                    positions
                    and isinstance(positions.get("rows"), list)
                    and positions["rows"]
                ):
                    total_amount_for_asset = sum(
                        float(pos.get("totalAmount", 0.0)) for pos in positions["rows"]
                    )
                    if total_amount_for_asset > 0:
                        earn_balances_aggregated[asset_api_name] = (
                            total_amount_for_asset
                        )

                # A small delay to be polite to the API
                time.sleep(0.2)
            except BinanceAPIException as e:
                # Use our centralized error handling
                if not self._handle_binance_api_exception(e, f"checking Simple Earn for {asset_api_name}"):
                    # For non-recoverable errors, just log and continue with other assets
                    self.logger.error(
                        f"Non-recoverable API Error checking Simple Earn for {asset_api_name}: {e}"
                    )
                # Continue with other assets regardless of error type
            except Exception as e:
                self.logger.error(
                    f"Unexpected error checking Simple Earn for {asset_api_name}: {e}"
                )

        self.logger.info(
            f"Finished checking Earn balances. Found holdings for {len(earn_balances_aggregated)} asset(s)."
        )
        return earn_balances_aggregated

    def fetch_simple_earn_rewards(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetches raw Simple Earn flexible reward data with robust pagination."""
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ms, end_ms = time_window

        raw_transactions = []
        current_page = 1
        endpoint_path = "simple-earn/flexible/history/rewardRecord"

        while True:
            try:
                params = {
                    "type": "FLEXIBLE",
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "current": current_page,
                    "size": 100,
                }
                response = self.binance_client._request_margin_api(
                    "get", endpoint_path, True, data=params
                )
                rows = response.get("rows", [])
                if not rows:
                    break

                for item in rows:
                    timestamp = pd.to_datetime(item.get("time"), unit="ms", utc=True)
                    if pd.isna(timestamp):
                        continue

                    normalized_symbol = self.symbol_mappings.normalize_symbol(
                        item.get("asset", "").upper()
                    )
                    if normalized_symbol not in self.target_assets_for_sync:
                        continue

                    raw_transactions.append(
                        {
                            "tx_type": "EARN_REWARD",
                            "timestamp": timestamp,
                            "source": f"Binance Simple Earn Reward ({item.get('type')})",
                            "transaction_hash": str(item.get("tranId")),
                            "raw_data": {
                                "symbol": normalized_symbol,
                                "quantity": float(item.get("rewards", 0.0)),
                            },
                        }
                    )

                if len(rows) < 100:
                    break
                current_page += 1
            except Exception as e:
                self.logger.error(
                    f"Error fetching Simple Earn rewards (Page {current_page}): {e}"
                )
                break
        return raw_transactions

    def fetch_simple_earn_subscriptions(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetches raw Simple Earn flexible subscription data with robust pagination."""
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ms, end_ms = time_window

        raw_transactions = []
        current_page = 1
        endpoint_path = "simple-earn/flexible/history/subscriptionRecord"

        while True:
            try:
                params = {
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "current": current_page,
                    "size": 100,
                }
                response = self.binance_client._request_margin_api(
                    "get", endpoint_path, True, data=params
                )
                rows = response.get("rows", [])
                if not rows:
                    break

                for item in rows:
                    timestamp = pd.to_datetime(item.get("time"), unit="ms", utc=True)
                    if pd.isna(timestamp):
                        continue
                    normalized_symbol = self.symbol_mappings.normalize_symbol(
                        item.get("asset", "").upper()
                    )
                    if normalized_symbol not in self.target_assets_for_sync:
                        continue
                    raw_transactions.append(
                        {
                            "tx_type": "EARN_SUBSCRIPTION",
                            "timestamp": timestamp,
                            "source": "Binance Simple Earn Subscription",
                            "transaction_hash": str(item.get("purchaseId")),
                            "raw_data": {
                                "symbol": normalized_symbol,
                                "quantity": float(item.get("amount", 0.0)),
                            },
                        }
                    )

                if len(rows) < 100:
                    break
                current_page += 1
            except Exception as e:
                self.logger.error(
                    f"Error fetching Simple Earn subscriptions (Page {current_page}): {e}"
                )
                break
        return raw_transactions

    def fetch_simple_earn_redemptions(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetches raw Simple Earn flexible redemption data with robust pagination."""
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ms, end_ms = time_window

        raw_transactions = []
        current_page = 1
        endpoint_path = "simple-earn/flexible/history/redemptionRecord"

        while True:
            try:
                params = {
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "current": current_page,
                    "size": 100,
                }
                response = self.binance_client._request_margin_api(
                    "get", endpoint_path, True, data=params
                )
                rows = response.get("rows", [])
                if not rows:
                    break

                for item in rows:
                    timestamp = pd.to_datetime(item.get("time"), unit="ms", utc=True)
                    if pd.isna(timestamp):
                        continue
                    normalized_symbol = self.symbol_mappings.normalize_symbol(
                        item.get("asset", "").upper()
                    )
                    if normalized_symbol not in self.target_assets_for_sync:
                        continue
                    raw_transactions.append(
                        {
                            "tx_type": "EARN_REDEMPTION",
                            "timestamp": timestamp,
                            "source": "Binance Simple Earn Redemption",
                            "transaction_hash": str(item.get("redeemId")),
                            "raw_data": {
                                "symbol": normalized_symbol,
                                "quantity": float(item.get("amount", 0.0)),
                            },
                        }
                    )

                if len(rows) < 100:
                    break
                current_page += 1
            except Exception as e:
                self.logger.error(
                    f"Error fetching Simple Earn redemptions (Page {current_page}): {e}"
                )
                break
        return raw_transactions

    def fetch_dividend_history(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(
            source_name, days_back, latest_known_ts
        )
        if not time_window:
            return []
        start_ms, end_ms = time_window
        raw_transactions = []
        try:
            for item in self.binance_client.get_asset_dividend_history(
                startTime=start_ms, endTime=end_ms, limit=500
            ).get("rows", []):
                timestamp = pd.to_datetime(item.get("divTime"), unit="ms", utc=True)
                if pd.isna(timestamp):
                    continue
                normalized_symbol = self.symbol_mappings.normalize_symbol(
                    item.get("asset", "").upper()
                )
                if normalized_symbol not in self.target_assets_for_sync:
                    continue
                raw_transactions.append(
                    {
                        "tx_type": "DIVIDEND",
                        "timestamp": timestamp,
                        "source": "Binance Dividend",
                        "transaction_hash": str(item.get("tranId")),
                        "raw_data": {
                            "symbol": normalized_symbol,
                            "quantity": float(item.get("amount", 0.0)),
                            "notes": item.get("enInfo", ""),
                        },
                    }
                )
        except BinanceAPIException as e:
            # Use our centralized error handling
            if self._handle_binance_api_exception(e, f"fetching dividend history for {source_name}"):
                # Log that we're skipping due to recoverable error and continue
                self.logger.warning(f"Skipping dividend history for {source_name} due to recoverable API error")
            # For both recoverable and non-recoverable errors, return what we have
        except Exception as e:
            self.logger.error(f"Error fetching dividend history: {e}")
        return raw_transactions

    def fetch_staking_history(
        self,
        source_name: str,
        days_back: int = 90,
        latest_known_ts_map: Optional[Dict[str, datetime.datetime]] = None,
    ) -> List[Dict[str, Any]]:
        if latest_known_ts_map is None:
            latest_known_ts_map = {}
        all_txs = []
        transaction_types = {
            "SUBSCRIPTION": {
                "source": "Binance Staking Subscription",
                "tx_type": "STAKING_SUBSCRIBE",
            },
            "REDEMPTION": {
                "source": "Binance Staking Redemption",
                "tx_type": "STAKING_REDEMPTION",
            },
            "INTEREST": {
                "source": "Binance Staking Interest",
                "tx_type": "STAKING_INTEREST",
            },
        }
        for txn_type, details in transaction_types.items():
            source_name = details["source"]
            time_window = self._get_start_end_timestamps(
                source_name, days_back, latest_known_ts_map.get(source_name)
            )
            if not time_window:
                continue
            start_ms, end_ms = time_window
            try:
                for item in self._fetch_paginated_history(
                    "staking/history",
                    {
                        "product": "STAKING",
                        "txnType": txn_type,
                        "startTime": start_ms,
                        "endTime": end_ms,
                    },
                ):
                    timestamp = pd.to_datetime(item.get("time"), unit="ms", utc=True)
                    if pd.isna(timestamp):
                        continue
                    normalized_symbol = self.symbol_mappings.normalize_symbol(
                        item.get("asset", "").upper()
                    )
                    if normalized_symbol not in self.target_assets_for_sync:
                        continue
                    all_txs.append(
                        {
                            "tx_type": details["tx_type"],
                            "timestamp": timestamp,
                            "source": source_name,
                            "transaction_hash": str(item.get("txnId")),
                            "raw_data": {
                                "symbol": normalized_symbol,
                                "quantity": float(item.get("amount", 0.0)),
                            },
                        }
                    )
            except Exception as e:
                self.logger.error(f"Error fetching staking history for {txn_type}: {e}")
        return all_txs

    def fetch_futures_balance(self) -> List[Dict[str, Any]]:
        """Fetches futures account balance."""
        self.logger.info("Fetching futures account balances...")
        try:
            futures_account = self.binance_client.futures_account_balance()
            return futures_account
        except BinanceAPIException as e:
            # Use our centralized error handling
            if self._handle_binance_api_exception(e, "fetching futures account balance"):
                # For recoverable errors, try once more
                try:
                    futures_account = self.binance_client.futures_account_balance()
                    return futures_account
                except BinanceAPIException as e2:
                    self.logger.error(f"Error fetching futures account balance after retry: {e2}")
                    return []
            # For non-recoverable errors, return empty list
            return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred fetching futures account balance: {e}")
            return []

    def fetch_funding_balance(self) -> List[Dict[str, Any]]:
        """Fetches funding wallet balance by calling the SAPI endpoint directly."""
        self.logger.info("Fetching funding wallet balances...")
        try:
            funding_wallet = self.binance_client._request_margin_api(
                "post", "asset/get-funding-asset", signed=True, data={}
            )
            return funding_wallet
        except BinanceAPIException as e:
            # Use our centralized error handling
            if self._handle_binance_api_exception(e, "fetching funding wallet"):
                # For recoverable errors, try once more
                try:
                    funding_wallet = self.binance_client._request_margin_api(
                        "post", "asset/get-funding-asset", signed=True, data={}
                    )
                    return funding_wallet
                except BinanceAPIException as e2:
                    self.logger.error(f"Binance API error fetching funding wallet after retry: {e2}")
                    return []
            # For non-recoverable errors, return empty list
            return []
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred fetching funding wallet: {e}"
            )
            return []

    def transfer_funding_to_spot(
        self, asset: str, amount: float, is_live: bool = False
    ) -> Dict[str, Any]:
        """
        Transfer funds from funding wallet to spot wallet using Binance API.

        Args:
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)

        Returns:
            Dictionary with success status, messages, and any errors
        """
        result = {"success": False, "messages": [], "errors": [], "transfer_id": None}

        try:
            # Validate inputs
            if not asset or amount <= 0:
                result["errors"].append("Invalid asset or amount")
                return result

            # Check funding balance first
            funding_balances = self.fetch_funding_balance()
            funding_balance = 0.0

            for balance in funding_balances:
                if balance.get("asset") == asset:
                    funding_balance = float(balance.get("free", 0.0))
                    break

            if funding_balance < amount:
                result["errors"].append(
                    f"Insufficient {asset} in funding wallet. Available: {funding_balance:.8f}, Required: {amount:.8f}"
                )
                return result

            result["messages"].append(
                f"✅ Funding wallet has sufficient {asset} balance: {funding_balance:.8f}"
            )

            if not is_live:
                result["messages"].append(
                    f"(Dry Run) Would transfer {amount:.8f} {asset} from funding to spot wallet"
                )
                result["success"] = True
                return result

            # Execute the transfer using Binance Asset Transfer API
            # The correct type for funding to spot is "FUNDING_MAIN" (from funding to main)
            transfer_params = {
                "type": "FUNDING_MAIN",  # This transfers FROM funding TO main (spot)
                "asset": asset,
                "amount": f"{amount:.8f}",
            }

            self.logger.info(f"Executing transfer: {transfer_params}")

            # Use the asset transfer endpoint
            transfer_result = self.binance_client._request_margin_api(
                "post", "asset/transfer", signed=True, data=transfer_params
            )

            self.logger.info(f"Transfer response: {transfer_result}")

            if transfer_result and transfer_result.get("tranId"):
                result["transfer_id"] = transfer_result["tranId"]
                result["messages"].append(
                    f"✅ Successfully transferred {amount:.8f} {asset} from funding to spot wallet"
                )
                result["messages"].append(f"Transfer ID: {transfer_result['tranId']}")
                result["success"] = True
            else:
                result["errors"].append(
                    f"Transfer failed - no transfer ID returned. Response: {transfer_result}"
                )

        except BinanceAPIException as e:
            error_msg = f"Binance API error during transfer: {e}"
            self.logger.error(error_msg)
            result["errors"].append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during transfer: {e}"
            self.logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)

        return result

    def _resync_time(self):
        """Synchronize local time with Binance server."""
        try:
            self.logger.info(
                "Re-synchronizing time with Binance server for fresh data..."
            )
            server_time = self.binance_client.get_server_time()["serverTime"]
            local_time = int(time.time() * 1000)
            time_offset = server_time - local_time
            self.binance_client._server_time_offset = time_offset
            self.logger.info(
                f"Time re-synchronized for balance fetch. Offset: {time_offset}ms."
            )
        except Exception as e:
            self.logger.error(f"Failed to sync time with Binance: {e}", exc_info=True)
