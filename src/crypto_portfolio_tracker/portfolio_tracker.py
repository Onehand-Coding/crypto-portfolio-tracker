"""
Crypto Portfolio Tracker - Main Class
Handles API interactions, data processing, analysis, and orchestration.
"""
import os
import re
import json
import time
import copy
import inspect
import asyncio
import logging
import datetime
import functools
from pathlib import Path
from diskcache import Cache
from collections import deque
from urllib3.util.retry import Retry
from datetime import timezone, timedelta
from typing import Dict, Any, Optional, List

import requests
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from binance.client import Client
from requests.adapters import HTTPAdapter
from binance.exceptions import BinanceAPIException

from . config import ConfigManager
from . database import DatabaseManager
from . visualizations import Visualizer
from . price_enricher import PriceEnricher
from . binance_fetcher import BinanceFetcher
from . import trading_strategies
from . strategy_backtester import StrategyBacktester
from . rebalancing_backtester import RebalancingBacktester
from . exporters import ExcelExporter, HtmlExporter, CsvExporter
from . rebalancing_logic import get_live_rebalance_suggestions
from . crypto_trend_analyzer import CryptoTrendAnalyzer

logger = logging.getLogger(__name__)


def calculate_fifo_cost_basis(transactions_df: pd.DataFrame):
    """
    Calculates current quantity and average cost basis using FIFO.
    Assumes transactions_df is for a single asset, sorted by timestamp.
    Incorporates fee_usd into the cost of BUY/DEPOSIT transactions.
    """
    buy_lots = deque()  # Queue to store {"qty": quantity, "price": effective_price_usd_per_unit}
    current_quantity = 0.0
    total_cost_basis = 0.0 # This will track the cost basis of *remaining* lots

    if transactions_df.empty:
        symbol = transactions_df["symbol"].iloc[0] if not transactions_df.empty and "symbol" in transactions_df.columns else "Unknown"
        logger.debug(f"No transactions provided to calculate_fifo_cost_basis for symbol {symbol}. Returning 0, 0.")
        return 0.0, 0.0

    # Ensure correct dtypes and sorting
    transactions_df = transactions_df.copy()
    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"], errors="coerce")
    transactions_df = transactions_df.sort_values(by="timestamp").reset_index(drop=True)

    if transactions_df["timestamp"].isna().any():
        logger.warning(f"Found NaT timestamps for symbol {transactions_df["symbol"].iloc[0] if not transactions_df.empty else "Unknown"} after coercion. Dropping these rows for FIFO.")
        transactions_df.dropna(subset=["timestamp"], inplace=True)
        if transactions_df.empty:
            logger.warning(f"All transactions dropped for symbol due to NaT timestamps. Returning 0, 0.")
            return 0.0,0.0

    for _, row in transactions_df.iterrows():
        tx_type = row.get("type")

        try:
            quantity = float(row.get("quantity", 0.0))
        except (ValueError, TypeError):
            logger.warning(f"Invalid quantity '{row.get("quantity")}' for tx {row.get("transaction_hash")}. Using 0.0.")
            quantity = 0.0

        if quantity == 0.0 and tx_type in ["BUY", "SELL", "DEPOSIT", "WITHDRAWAL"]:
            logger.debug(f"Skipping zero quantity {tx_type} transaction: {row.get("transaction_hash")}")
            continue

        price = row.get("price_usd")

        fee_usd = row.get("fee_usd", 0.0)
        if pd.isna(fee_usd) or not isinstance(fee_usd, (int, float)):
            logger.debug(f"Invalid or missing fee_usd ('{row.get("fee_usd")}') for tx {row.get("transaction_hash")}. Using 0.0 for fee_usd.")
            fee_usd = 0.0
        else:
            try:
                fee_usd = float(fee_usd)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert fee_usd '{row.get("fee_usd")}' to float for tx {row.get("transaction_hash")}. Using 0.0 for fee_usd.")
                fee_usd = 0.0

        if tx_type == "BUY" or tx_type == "DEPOSIT":
            actual_tx_price_usd = 0.0
            if price is not None and isinstance(price, (int, float)) and price > 0:
                actual_tx_price_usd = float(price)
            elif price == 0.0 and tx_type == "DEPOSIT":
                actual_tx_price_usd = 0.0
            elif price is None or not isinstance(price, (int, float)) or price <=0 :
                logger.warning(f"{tx_type} tx for {row.get("symbol", "N/A")} (ID: {row.get("transaction_hash", "N/A")}) "
                               f"has invalid/missing price ({price}). Using $0 price component for this lot.")
                actual_tx_price_usd = 0.0

            # <<< FEE INCORPORATION FOR BUYS/DEPOSITS >>>
            cost_for_this_transaction_lot = (quantity * actual_tx_price_usd) + fee_usd
            effective_price_per_unit = cost_for_this_transaction_lot / quantity if quantity > 0 else 0.0 # Price per unit *including* fee

            buy_lots.append({"qty": quantity, "price": effective_price_per_unit})
            current_quantity += quantity
            total_cost_basis += cost_for_this_transaction_lot # Add full cost (value + fee)

        elif tx_type == "SELL" or tx_type == "WITHDRAWAL":
            sell_qty = quantity
            current_quantity -= sell_qty

            while sell_qty > 0 and buy_lots:
                oldest_buy = buy_lots[0]
                # oldest_buy["price"] is the effective_price_per_unit from its acquisition (already includes its buy fee)

                if oldest_buy["qty"] <= sell_qty:
                    qty_removed_from_lot = oldest_buy["qty"]
                    cost_basis_removed = qty_removed_from_lot * oldest_buy["price"]
                    total_cost_basis -= cost_basis_removed
                    sell_qty -= qty_removed_from_lot
                    buy_lots.popleft()
                else:
                    cost_basis_removed = sell_qty * oldest_buy["price"]
                    total_cost_basis -= cost_basis_removed
                    oldest_buy["qty"] -= sell_qty
                    sell_qty = 0

            if sell_qty > 0:
                logger.warning(f"{tx_type} of {sell_qty} {row.get("symbol", "N/A")} more than available from history. Cost basis may be affected.")

        total_cost_basis = max(0, total_cost_basis)

    average_cost_basis = total_cost_basis / current_quantity if current_quantity > 0 else 0.0
    current_quantity = max(0, current_quantity)

    logger.debug(f"FIFO Result for {transactions_df["symbol"].iloc[0] if not transactions_df.empty else "Unknown"}: Qty={current_quantity:.8f}, AvgCost={average_cost_basis:.8f}, TotalCostBasisForAsset={total_cost_basis:.8f}")
    return current_quantity, average_cost_basis


class CryptoPortfolioTracker:
    """Main class for the crypto portfolio tracker."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize the tracker and its specialist components."""
        self.config_manager = config_manager
        self.config = self.config_manager.config
        self.logger = logging.getLogger(__name__)

        # --- Initialize Core Components ---
        self.db_manager = DatabaseManager(self.config)
        self.symbol_mappings = self.config_manager.symbol_mapper
        self.binance_client = self._init_binance_client()

        # --- Initialize Caches ---
        self.cache_dir = Path(self.config.get("cache", {}).get("path", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.coingecko_price_cache = Cache(str(self.cache_dir / "coingecko_historical"))
        self.yfinance_disk_cache = Cache(str(self.cache_dir / "yfinance_historical"))
        self.fiat_exchange_rate_cache = Cache(str(self.cache_dir / "fiat_exchange_rates"))

        # --- Instantiate our new specialist classes ---
        self.fetcher = BinanceFetcher(self.binance_client, self.symbol_mappings, self.config)
        self.enricher = PriceEnricher(self.symbol_mappings, self.config, self.coingecko_price_cache)

        # --- Initialize other components ---
        self.excel_exporter = ExcelExporter(self.config)
        self.html_exporter = HtmlExporter(self.config)
        self.csv_exporter = CsvExporter(self.config)
        self.visualizer = Visualizer(self.config)

        # --- Initialize strategy state ---
        self.strategy_state_path = self.cache_dir.parent / "strategy_state.json"
        self.strategy_states = self._load_strategy_state()

        self.logger.info("Tracker and all components initialized.")

    def _init_binance_client(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> Optional[Client]:
        """Initialize and return Binance client with robust session, retries, and time sync."""

        if not api_key and not api_secret:
            keys = self.config_manager.get_binance_keys()
            api_key = keys.get("api_key")
            api_secret = keys.get("api_secret")

        if not api_key or not api_secret:
            self.logger.warning("Binance API key/secret not found. Binance client disabled.")
            return None

        try:
            # --- Read all Binance settings from config ---
            binance_config = self.config.get("apis", {}).get("binance", {})
            client_timeout = binance_config.get("timeout", 60)
            recv_window = binance_config.get("recv_window", 20000)

            self.logger.info(f"Initializing Binance client with timeout: {client_timeout}s.")

            client = Client(api_key, api_secret, requests_params={"timeout": client_timeout})

            client.RECV_WINDOW = recv_window
            self.logger.info(f"Set Binance client recvWindow to {client.RECV_WINDOW}ms.")

            # Actively sync time with the server to prevent all recvWindow errors.
            self._sync_binance_client_time(client, context="initialization")

            retry_strategy = Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            client.session.mount("https://", adapter)
            client.session.mount("http://", adapter)

            if self.config_manager.is_testnet_mode:
                self.logger.info("Connecting to Binance Testnet endpoint.")
                client.API_URL = "https://testnet.binance.vision/api"

            client.ping()
            self.logger.info("Binance client initialized successfully.")
            return client
        except Exception as e:
            self.logger.error(f"Failed to init Binance client: {e}", exc_info=True)
            return None

    def _get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Get current prices for a list of symbols, with a simple retry for rate limiting.
        """
        if not symbols:
            return {}

        prices: Dict[str, Optional[float]] = {symbol: 0.0 for symbol in symbols}
        coingecko_config = self.config.get("apis", {}).get("coingecko", {})

        ids_to_fetch = {self.symbol_mappings.get(s.upper()) for s in symbols if self.symbol_mappings.get(s.upper())}
        if not ids_to_fetch:
            return prices

        ids_string = ",".join(list(ids_to_fetch))
        url = f"{coingecko_config.get("base_url")}/simple/price"
        params = {"ids": ids_string, "vs_currencies": "usd"}

        for attempt in range(3): # Try up to 3 times
            try:
                self.logger.info(f"Fetching current prices for {len(ids_to_fetch)} unique assets...")
                response = requests.get(url, params=params, timeout=coingecko_config.get("timeout", 30))
                response.raise_for_status()
                fetched_price_data = response.json()

                for symbol in symbols:
                    coin_id = self.symbol_mappings.get(symbol.upper())
                    if coin_id in fetched_price_data:
                        prices[symbol] = fetched_price_data[coin_id].get("usd")

                return prices # Return on success

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < 2:
                    self.logger.warning("Rate limited fetching current prices. Waiting 10 seconds before retrying...")
                    time.sleep(10)
                    continue # Go to the next attempt
                else:
                    self.logger.error(f"Fatal error fetching batch prices from CoinGecko: {e}")
                    return prices
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Fatal error fetching batch prices from CoinGecko: {e}")
                return prices

        return prices

    def _get_coingecko_historical_price(self, coin_id: str, date_str: str) -> Optional[float]:
        """Fetch historical price from CoinGecko for a specific date, using disk cache."""
        if not coin_id or not date_str:
            self.logger.error("Coin ID or date string is missing for CoinGecko historical price fetch.")
            return None

        cache_key = f"{coin_id}_{date_str}"
        cached_price = self.coingecko_historical_price_disk_cache.get(cache_key)
        if cached_price is not None:
            self.logger.debug(f"Disk Cache HIT for CoinGecko: {coin_id} on {date_str} -> ${cached_price:.6f}")
            return cached_price

        self.logger.debug(f"Disk Cache MISS for CoinGecko: {coin_id} on {date_str}. Fetching from API.")

        base_url = self.coingecko_api.get("base_url")
        timeout = self.coingecko_api.get("timeout", 30)

        try:
            parsed_date = pd.to_datetime(date_str, errors="coerce", dayfirst=True).date()
            today = datetime.datetime.now(timezone.utc).date()

            if parsed_date >= today:
                self.logger.warning(f"Requested date {date_str} is today or in the future. Fetching previous day's data instead.")
                parsed_date = today - timedelta(days=1)

            api_date_str = parsed_date.strftime("%d-%m-%Y")

        except (ValueError, TypeError):
            self.logger.error(f"Could not parse date '{date_str}' for CoinGecko API.")
            self.coingecko_historical_price_disk_cache.set(cache_key, None, expire=3600)
            return None

        url = f"{base_url}/coins/{coin_id}/history?date={api_date_str}&localization=false"

        max_retries = 3
        initial_wait_seconds = 60
        server_error_codes_to_retry = [500, 502, 503, 504]
        retries_left = max_retries

        while retries_left >= 0:
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                if "market_data" in data and "current_price" in data["market_data"] and "usd" in data["market_data"]["current_price"]:
                    price = float(data["market_data"]["current_price"]["usd"])
                    self.logger.info(f"Fetched price for {coin_id} on {api_date_str}: ${price:.6f}")
                    self.coingecko_historical_price_disk_cache.set(cache_key, price)
                    return price
                else:
                    self.logger.warning(f"No price data in 'usd' found for {coin_id} on {api_date_str}.")
                    self.coingecko_historical_price_disk_cache.set(cache_key, None, expire=3600 * 24)
                    return None
            except requests.exceptions.HTTPError as e:
                should_retry = False
                if e.response.status_code == 429:
                    should_retry = True
                    self.logger.error(f"Rate limited (429) by CoinGecko for {coin_id} on {api_date_str}.")
                elif e.response.status_code in server_error_codes_to_retry:
                    should_retry = True
                    self.logger.error(f"Server error ({e.response.status_code}) from CoinGecko for {coin_id} on {api_date_str}.")

                if should_retry and retries_left > 0:
                    wait_time = initial_wait_seconds * (max_retries - retries_left + 1)
                    self.logger.info(f"Waiting {wait_time}s before retry... ({retries_left} retries left)")
                    time.sleep(wait_time)
                    retries_left -= 1
                else:
                    if e.response.status_code == 404:
                        self.logger.warning(f"No historical data found on CoinGecko (404) for {coin_id} on {api_date_str}.")
                    else:
                        self.logger.error(f"Failed to fetch CoinGecko historical price for {coin_id} on {api_date_str}: {e}", exc_info=True)
                    self.coingecko_historical_price_disk_cache.set(cache_key, None, expire=3600*24)
                    return None
            except Exception as e:
                self.logger.error(f"Unexpected error fetching CoinGecko historical price for {coin_id}: {e}", exc_info=True)
                self.coingecko_historical_price_disk_cache.set(cache_key, None, expire=3600)
                return None

        self.logger.error(f"Failed to fetch historical price for {coin_id} after exhausted retries.")
        self.coingecko_historical_price_disk_cache.set(cache_key, None, expire=3600*24)
        return None

    def _get_historical_fiat_exchange_rate(self, date_str_orig: str, from_currency: str, to_currency: str) -> Optional[float]:
        """
        Fetches the historical exchange rate using yfinance.
        date_str_orig should be in "YYYY-MM-DD" or "DD-MM-YYYY".
        Returns how many "to_currency" 1 unit of "from_currency" is worth.
        """
        if not date_str_orig or not from_currency or not to_currency:
            self.logger.error("Date, from_currency, or to_currency missing for fiat exchange rate lookup.")
            return None

        if from_currency.upper() == to_currency.upper():
            return 1.0

        try:
            target_dt = pd.to_datetime(date_str_orig)
            api_date_str = target_dt.strftime("%Y-%m-%d")
        except ValueError:
            self.logger.error(f"Invalid date format '{date_str_orig}'. Could not parse for yfinance.")
            return None

        cache_key = f"yfinance_{api_date_str}_{from_currency}_{to_currency}"
        if cache_key in self.fiat_exchange_rate_cache:
            self.logger.debug(f"Cache HIT for yfinance fiat rate: {from_currency}->{to_currency} on {api_date_str}.")
            return self.fiat_exchange_rate_cache[cache_key]

        self.logger.debug(f"Cache MISS. Fetching yfinance fiat rate: {from_currency}->{to_currency} on {api_date_str}.")

        ticker_symbol = f"{from_currency.upper()}{to_currency.upper()}=X"
        rate = None

        start_date_fetch = (target_dt - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        end_date_fetch = (target_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        try:
            self.logger.debug(f"yfinance download: Ticker='{ticker_symbol}', Start='{start_date_fetch}', End='{end_date_fetch}'")
            data_window = yf.download(ticker_symbol, start=start_date_fetch, end=end_date_fetch, progress=False, auto_adjust=True)

            if data_window.empty:
                self.logger.warning(f"No data returned by yfinance for {ticker_symbol} in window {start_date_fetch} to {end_date_fetch}.")
            else:
                data_window = data_window.sort_index()
                closest_price_intermediate = data_window["Close"].asof(target_dt)

                self.logger.debug(f"yfinance .asof() result type: {type(closest_price_intermediate)}, value: {closest_price_intermediate}")

                closest_price_scalar = None
                if isinstance(closest_price_intermediate, pd.Series):
                    if not closest_price_intermediate.empty:
                        closest_price_scalar = closest_price_intermediate.iloc[0]
                    else:
                        closest_price_scalar = pd.NA
                else:
                    closest_price_scalar = closest_price_intermediate

                if pd.notna(closest_price_scalar):
                    actual_price_date_idx = data_window.index.get_indexer([target_dt], method="ffill")
                    if actual_price_date_idx[0] != -1 :
                        actual_price_date = data_window.index[actual_price_date_idx[0]]
                        rate = float(closest_price_scalar)
                        self.logger.debug(f"Fetched yfinance rate for {from_currency}->{to_currency} (for {api_date_str}, using data from {actual_price_date.strftime("%Y-%m-%d")}): {rate}")
                    else:
                        self.logger.warning(f"Could not determine actual date for 'asof' price for {ticker_symbol} on {api_date_str}, but got a value. Using asof value directly.")
                        rate = float(closest_price_scalar)
                        self.logger.debug(f"Fetched yfinance rate for {from_currency}->{to_currency} (for {api_date_str}, using .asof() value directly): {rate}")
                else:
                    self.logger.warning(f"No valid data found for {ticker_symbol} on or before {api_date_str} in the fetched window using .asof() (result was NaN or pd.NA).")

        except Exception as e:
            self.logger.error(f"Error during yfinance download or processing for {ticker_symbol} on {api_date_str}: {e}", exc_info=True)
            rate = None

        finally:
            self.fiat_exchange_rate_cache[cache_key] = rate
            yf_delay_ms = self.config.get("apis", {}).get("yfinance", {}).get("request_delay_ms", 200)
            if yf_delay_ms > 0:
                time.sleep(yf_delay_ms / 1000.0)

        return rate

    def _get_yfinance_ticker(self, symbol: str) -> Optional[str]:
        """
        Converts common crypto symbols to yfinance ticker format (e.g., BTC -> BTC-USD).
        Returns None if no suitable ticker is found or for stablecoins.
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self.stablecoin_symbols:
            self.logger.debug(f"Skipping yfinance ticker for stablecoin: {symbol_upper}")
            return None

        # Add more specific mappings if needed, e.g., RENDER might be RNDR-USD
        # This is a common source of issues, ensure tickers are correct for yfinance.
        # Example: self.config.get("yfinance_ticker_map", {}).get(symbol_upper, f"{symbol_upper}-USD")
        return f"{symbol_upper}-USD"

    def _load_strategy_state(self) -> Dict[str, Any]:
        """Loads the state of all strategies from a JSON file."""
        if not self.strategy_state_path.exists():
            return {}
        try:
            with open(self.strategy_state_path, "r") as f:
                states = json.load(f)
                self.logger.info(f"Loaded strategy states from {self.strategy_state_path}")
                return states
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading strategy state file: {e}. Starting fresh.")
            return {}

    def _save_strategy_state(self):
        """Saves the current state of all strategies to a JSON file."""
        try:
            with open(self.strategy_state_path, "w") as f:
                json.dump(self.strategy_states, f, indent=4)
                self.logger.info(f"Saved strategy states to {self.strategy_state_path}")
        except IOError as e:
            self.logger.error(f"Error saving strategy state file: {e}")

    def _get_symbol_filters(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches and caches trading filters for a given symbol."""
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
            self.yfinance_disk_cache.set(cache_key, filters, expire=3600 * 24) # Cache for 24 hours
            return filters
        except Exception as e:
            self.logger.error(f"Could not fetch symbol info for {symbol}: {e}")
            return None

    def _restore_database_interactive(self):
        """Handles the interactive process of restoring a database."""
        print("\n--- Restoring Database from Backup ---")
        backups = self.db_manager.list_backups()

        if not backups:
            print("❌ No backup files found in the 'data/db_backups/' directory.")
            return

        print("Available backups (newest first):")
        for i, backup_file in enumerate(backups):
            # Show relative path for cleaner output
            try:
                relative_path = backup_file.relative_to(self.config_manager.project_root)
            except ValueError:
                relative_path = backup_file # Fallback to absolute path if not relative
            print(f"  {i + 1}. {relative_path}")

        try:
            selection_str = input(f"\nEnter the number of the backup to restore (or press Enter to cancel): ").strip()
            if not selection_str:
                print("Restore cancelled.")
                return

            selection_idx = int(selection_str) - 1
            if not 0 <= selection_idx < len(backups):
                print("❌ Invalid selection.")
                return

            selected_backup = backups[selection_idx]
            print("\n" + "="*50)
            print("🚨 WARNING: THIS IS A DESTRUCTIVE ACTION 🚨")
            print("The current database will be permanently overwritten.")
            print(f"You are about to restore from:\n  -> {selected_backup.name}")
            print("="*50)

            confirm = input("Type 'RESTORE' to proceed: ")
            if confirm == "RESTORE":
                print("Restoring database...")
                success = self.db_manager.restore_from_backup(selected_backup)
                if success:
                    print("\n✅ Restore successful.")
                    print("‼️ PLEASE RESTART THE APPLICATION to load the restored database.")
                else:
                    print("❌ Restore failed. The original database was not modified. Check logs.")
            else:
                print("🛑 Restore cancelled. No changes were made.")

        except (ValueError, IndexError):
            print("❌ Invalid input.")

    def _adjust_quantity_to_lot_size(self, symbol: str, quantity: float) -> Optional[float]:
        """Rounds the quantity down to the nearest valid step size for the LOT_SIZE filter."""
        filters = self._get_symbol_filters(symbol)
        if not filters or "LOT_SIZE" not in filters:
            self.logger.warning(f"Could not get LOT_SIZE filter for {symbol}. Cannot adjust quantity.")
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
        factor = 10 ** precision
        adjusted_quantity = (int(quantity * factor)) / factor

        self.logger.info(f"Adjusted quantity for {symbol} from {quantity} to {adjusted_quantity} (precision: {precision})")
        return adjusted_quantity

    def _redeem_from_earn_if_needed(self, asset: str, required_amount: float, is_live: bool) -> bool:
        """
        Checks for sufficient spot balance. If needed, it finds the correct productId
        and redeems from Simple Earn only if in live mode.
        Returns True if funds are sufficient, False otherwise.
        """
        try:
            spot_balance_info = self.binance_client.get_asset_balance(asset=asset)
            free_spot_balance = float(spot_balance_info.get("free", 0.0))

            if free_spot_balance >= required_amount:
                self.logger.info(f"Sufficient {asset} balance in Spot wallet. No action needed.")
                return True

            shortfall = required_amount - free_spot_balance
            self.logger.warning(f"Shortfall of {shortfall:.8f} {asset}. Checking Simple Earn.")

            positions = self.binance_client.get_simple_earn_flexible_product_position(asset=asset)
            if not positions or not positions.get("rows"):
                print(f"❌ No active Simple Earn Flexible positions found for {asset} to redeem from.")
                return False

            product_to_redeem_from = positions["rows"][0]
            product_id = product_to_redeem_from.get("productId")
            available_to_redeem = float(product_to_redeem_from.get("totalAmount", 0.0))

            if shortfall > available_to_redeem:
                print(f"❌ Cannot redeem required amount. Need {shortfall:.8f} {asset}, but only {available_to_redeem:.8f} is available in Simple Earn.")
                return False

            if is_live:
                print(f"⚠️ Insufficient Spot balance. Attempting to redeem {shortfall:.8f} {asset} from product {product_id}.")
                self.logger.info(f"LIVE MODE: Attempting to redeem {shortfall:.8f} {asset} from productId {product_id}.")
                redemption_result = self.binance_client.redeem_simple_earn_flexible_product(
                    productId=product_id,
                    amount=f"{shortfall:.8f}"
                )
                if redemption_result and redemption_result.get("success"):
                    self.logger.info(f"Successfully initiated LIVE redemption. Waiting 5 seconds...")
                    print(f"✅ Redemption initiated for {shortfall:.8f} {asset}. Waiting 5 seconds...")
                    time.sleep(5)
                    return True # Assume success and let the trade proceed
                else:
                    self.logger.error(f"Failed to redeem {asset} from {product_id}: {redemption_result}")
                    print(f"❌ LIVE redemption FAILED for {asset}. See logs.")
                    return False
            else:
                # In Dry Run mode, we only simulate the check.
                print(f"DRY RUN: Would redeem {shortfall:.8f} {asset} from Simple Earn product {product_id}.")
                self.logger.info(f"DRY RUN: Would redeem {shortfall:.8f} {asset} from {product_id}.")
                return True # Return True to allow the simulated trade to continue

        except BinanceAPIException as e:
            self.logger.error(f"API Error during redemption process for {asset}: {e}", exc_info=True)
            print(f"❌ An API Error occurred when trying to process redemption for {asset}: {e.message}")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during redemption check for {asset}: {e}", exc_info=True)
            return False

    def _execute_directional_trade(self, trade: Dict[str, Any], client: Client):
        """Helper function to execute a single directional BUY or SELL trade."""
        symbol = trade["Symbol"]
        signal = trade["Signal"]
        size = trade.get("Size", 1.0) # Default to 100% if not provided
        trade_ticker = f"{symbol}USDT"
        min_trade_usd = self.config.get("portfolio", {}).get("minimum_trade_usd", 10.0)
        is_live = self.config.get("portfolio", {}).get("live_trading_enabled", False)

        self.logger.info(f"Executing {signal} for {symbol} (Size: {size:.2%}) on account via directional strategy. LIVE: {is_live}")

        try:
            if signal == "BUY":
                usdt_balance = float(client.get_asset_balance(asset="USDT").get("free", 0.0))
                trade_amount_usd = usdt_balance * size

                if trade_amount_usd < min_trade_usd:
                    print(f"⚠️ SKIPPING BUY for {symbol}: Calculated trade size (${trade_amount_usd:,.2f}) is below minimum of ${min_trade_usd:,.2f}.")
                    return

                print(f"\nPreparing MARKET BUY for ${trade_amount_usd:,.2f} of {symbol}...")

                if is_live:
                    order = client.order_market_buy(symbol=trade_ticker, quoteOrderQty=f"{trade_amount_usd:.2f}")
                    print(f"✅ LIVE BUY ORDER PLACED: {order}")
                    self.logger.info(f"LIVE BUY ORDER PLACED: {order}")
                else:
                    print(f"✅ [DRY RUN] Market BUY order for ${trade_amount_usd:,.2f} of {trade_ticker} was not placed.")
                    self.logger.info(f"[DRY RUN] Market BUY order for {trade_ticker} was not placed.")

            elif signal == "SELL":
                asset_balance = float(client.get_asset_balance(asset=symbol).get("free", 0.0))
                trade_quantity = asset_balance * size
                current_price = self._get_current_prices([symbol]).get(symbol, 0)

                if (trade_quantity * current_price) < min_trade_usd:
                    print(f"⚠️ SKIPPING SELL for {symbol}: Position value is below minimum trade size.")
                    return

                print(f"\nPreparing MARKET SELL for {trade_quantity:.8g} {symbol} ({size:.0%} of holding)...")

                if is_live:
                    order = client.order_market_sell(symbol=trade_ticker, quantity=f"{trade_quantity:.8g}")
                    print(f"✅ LIVE SELL ORDER PLACED: {order}")
                    self.logger.info(f"LIVE SELL ORDER PLACED: {order}")
                else:
                    print(f"✅ [DRY RUN] Market SELL order for {trade_quantity:.8g} {symbol} was not placed.")
                    self.logger.info(f"[DRY RUN] Market SELL order for {trade_quantity:.8g} {symbol} was not placed.")

        except BinanceAPIException as e:
            print(f"❌ {("LIVE" if is_live else "DRY RUN")} {signal} FAILED for {symbol}: {e}")
            self.logger.error(f"{("LIVE" if is_live else "DRY RUN")} {signal} FAILED for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred executing directional trade for {symbol}: {e}", exc_info=True)
            print(f"❌ Unexpected Error for {symbol}: {e}")

    def _sync_binance_client_time(self, client: Client, context: str = "general"):
        """
        Reusable utility to synchronize the given Binance client's clock with the server.
        """
        if not client:
            self.logger.warning(f"Cannot sync time: Binance client for '{context}' is not available.")
            return
        try:
            self.logger.info(f"Synchronizing time for {context}...")
            server_time = client.get_server_time()['serverTime']
            local_time = int(time.time() * 1000)
            time_offset = server_time - local_time
            client._server_time_offset = time_offset
            self.logger.info(f"Time synchronized for {context}. New offset: {time_offset}ms.")
        except Exception as e:
            self.logger.error(f"Could not sync time for {context}: {e}. Proceeding with old offset.")

    async def _execute_rebalancing_trades(self, suggestions_df: pd.DataFrame, earn_balances: Dict[str, float], interactive: bool):
        """
        Executes rebalancing trades, enforcing minimum trade size and updating
        simulated balances after each trade to enable sequential funding.
        Can run in bulk or interactive (one-by-one) mode.
        """
        if suggestions_df.empty or "Signal" not in suggestions_df.columns:
            print("No rebalancing suggestions to execute.")
            return

        trades_to_execute = suggestions_df[suggestions_df["Signal"].isin(["BUY", "SELL"])]
        trades_to_execute = trades_to_execute.sort_values(by=["Signal", "Drift (pts)"], ascending=[False, True])

        if trades_to_execute.empty:
            print("\n✅ No BUY or SELL actions suggested. Nothing to execute.")
            return

        portfolio_config = self.config.get("portfolio", {})
        min_trade_usd = portfolio_config.get("minimum_trade_usd", 10.0)
        is_live = portfolio_config.get("live_trading_enabled", False)

        simulated_balances = {}
        all_trade_symbols = set(trades_to_execute["Symbol"].unique()) | {"USDT"}
        for symbol in all_trade_symbols:
            try:
                spot_bal = float(self.binance_client.get_asset_balance(asset=symbol).get("free", 0.0))
                earn_bal = earn_balances.get(symbol, 0.0)
                simulated_balances[symbol] = spot_bal + earn_bal
                self.logger.info(f"Initialized simulated balance for {symbol}: {simulated_balances[symbol]:.8f}")
            except Exception as e:
                self.logger.error(f"Could not fetch balance for {symbol}: {e}")
                simulated_balances[symbol] = 0.0

        print("\n" + "="*80)
        if is_live:
            print("🔴🔴🔴 WARNING: Live Trading is ENABLED. 🔴🔴🔴")
        else:
            print("🟡🟡🟡 NOTE: Live Trading is DISABLED. 🟡🟡🟡")
        print("="*80)

        items_to_process = []
        if not interactive:
            print("🚨 PROPOSED TRADES - PLEASE REVIEW CAREFULLY 🚨")
            print(trades_to_execute[["Symbol", "Signal", "Suggested Action Detail"]].to_string(index=False))
            print("="*80)
            try:
                confirm = input("Type EXECUTE ALL to proceed with all trades listed above: ")
                if confirm == "EXECUTE ALL":
                    self.logger.info("User confirmed bulk trade execution.")
                    items_to_process = list(trades_to_execute.iterrows())
                else:
                    print("🛑 Bulk trade execution cancelled by user."); return
            except KeyboardInterrupt:
                print("\n🛑 Trade execution cancelled by user."); return
        else:
            print("👀 Entering interactive confirmation mode. You will be prompted for each trade.")
            for index, row in trades_to_execute.iterrows():
                print("\n" + "-"*80)
                print("🚨 PROPOSED TRADE - PLEASE REVIEW CAREFULLY 🚨")
                print(pd.DataFrame([row])[["Symbol", "Signal", "Suggested Action Detail"]].to_string(index=False))
                try:
                    confirm_one = input(f"Approve this trade for {row["Symbol"]}? Type YES to confirm: ").upper().strip()
                    if confirm_one == "YES":
                        self.logger.info(f"User approved trade for {row["Symbol"]}.")
                        items_to_process.append((index, row))
                    else:
                        self.logger.info(f"User skipped trade for {row["Symbol"]}.")
                        print(f"Skipping trade for {row["Symbol"]}.")
                except KeyboardInterrupt:
                    print("\n🛑 Trade execution cancelled by user."); return
            print("="*80)
            print(f"Executing {len(items_to_process)} approved trade(s)...")

        trades_executed_count = 0
        for _, row in items_to_process:
            symbol = row["Symbol"]
            signal = row["Signal"]
            trade_ticker = f"{symbol}USDT"
            usd_value = row.get("action_usd_value", 0.0)
            coin_quantity = row.get("action_coin_quantity", 0.0)

            if usd_value < min_trade_usd:
                print(f"\n⚠️ SKIPPING {signal} for {symbol}: Suggested trade value (~${usd_value:,.2f}) is below the minimum of ${min_trade_usd:,.2f}.")
                continue

            try:
                if signal == "SELL":
                    if coin_quantity > simulated_balances.get(symbol, 0.0):
                        print(f"⚠️ SKIPPING SELL for {symbol}: Required quantity ({coin_quantity:.8f}) exceeds simulated available balance ({simulated_balances.get(symbol, 0.0):.8f}).")
                        continue

                    adjusted_quantity = self._adjust_quantity_to_lot_size(trade_ticker, coin_quantity)
                    if adjusted_quantity is None or adjusted_quantity <= 0:
                        print(f"⚠️ SKIPPING SELL for {symbol}: Adjusted quantity is zero or invalid after applying lot size rules.")
                        continue

                    if not self._redeem_from_earn_if_needed(asset=symbol, required_amount=adjusted_quantity, is_live=is_live):
                        print(f"⚠️ SKIPPING SELL for {symbol} due to redemption check failure.")
                        continue

                    print(f"\nPreparing MARKET SELL for {adjusted_quantity:.8g} {symbol}...")
                    if is_live:
                        try:
                            print("🚀 PLACING LIVE ORDER...")
                            order = self.binance_client.order_market_sell(symbol=trade_ticker, quantity=f"{adjusted_quantity:.8g}")
                            print(f"✅ LIVE SELL ORDER PLACED.")
                            self.logger.info(f"LIVE SELL ORDER PLACED: {order}")
                            simulated_balances[symbol] -= adjusted_quantity
                            simulated_balances["USDT"] += float(order.get("cummulativeQuoteQty", 0.0))
                            self.logger.info(f"Updated simulated balances: {symbol}={simulated_balances[symbol]:.8f}, USDT={simulated_balances["USDT"]:.2f}")
                            trades_executed_count += 1
                        except BinanceAPIException as e:
                            print(f"❌ LIVE SELL FAILED for {symbol}: {e}")
                            self.logger.error(f"LIVE SELL FAILED for {symbol}: {e}")
                    else:
                        print("✅ (Dry Run) Trade was not placed.")

                elif signal == "BUY":
                    if usd_value > simulated_balances.get("USDT", 0.0):
                        print(f"\n⚠️ SKIPPING BUY for {symbol}: Required USDT (${usd_value:,.2f}) exceeds simulated available USDT balance (${simulated_balances.get("USDT", 0.0):,.2f}).")
                        continue

                    if not self._redeem_from_earn_if_needed(asset="USDT", required_amount=usd_value, is_live=is_live):
                        print(f"⚠️ SKIPPING BUY for {symbol} due to insufficient USDT after redemption check.")
                        continue

                    print(f"\nPreparing MARKET BUY for ${usd_value:,.2f} of {symbol}...")
                    if is_live:
                        try:
                            print("🚀 PLACING LIVE ORDER...")
                            order = self.binance_client.order_market_buy(symbol=trade_ticker, quoteOrderQty=f"{usd_value:.2f}")
                            print(f"✅ LIVE BUY ORDER PLACED.")
                            self.logger.info(f"LIVE BUY ORDER PLACED: {order}")
                            simulated_balances["USDT"] -= float(order.get("cummulativeQuoteQty", 0.0))
                            simulated_balances[symbol] = simulated_balances.get(symbol, 0.0) + float(order.get("executedQty", 0.0))
                            self.logger.info(f"Updated simulated balances: {symbol}={simulated_balances[symbol]:.8f}, USDT={simulated_balances["USDT"]:.2f}")
                            trades_executed_count += 1
                        except BinanceAPIException as e:
                            print(f"❌ LIVE BUY FAILED for {symbol}: {e}")
                            self.logger.error(f"LIVE BUY FAILED for {symbol}: {e}")
                    else:
                        print("✅ (Dry Run) Trade was not placed.")

            except Exception as e:
                self.logger.error(f"An unexpected error occurred executing trade for {symbol}: {e}", exc_info=True)
                print(f"❌ Unexpected Error for {symbol}: {e}")

        if trades_executed_count > 0:
            print(f"\n🎉 Rebalancing execution complete. {trades_executed_count} trade(s) processed.")
        else:
            print("\nNo trades were executed.")

    async def _execute_manual_trade(self, trade_type, symbol, trade_ticker, amount, is_quote_qty, is_live):
        """Internal method to execute a validated manual trade."""
        min_trade_usd = self.config.get("portfolio", {}).get("minimum_trade_usd", 10.0)

        try:
            if trade_type == "BUY":
                usdt_to_spend = amount if is_quote_qty else 0
                if not is_quote_qty:
                    # If buying a coin quantity, we need to estimate the USDT value
                    prices = self._get_current_prices([symbol])
                    if not prices.get(symbol):
                        print(f"❌ Could not fetch price for {symbol} to calculate trade value.")
                        return
                    usdt_to_spend = amount * prices[symbol]

                if usdt_to_spend < min_trade_usd:
                    print(f"⚠️ SKIPPING BUY for {symbol}: Required value (~${usdt_to_spend:,.2f}) is below the minimum of ${min_trade_usd:,.2f}.")
                    return

                print(f"\nPreparing MARKET BUY for {symbol}...")
                if is_live:
                    print("🚀 PLACING LIVE ORDER...")
                    order = self.binance_client.order_market_buy(symbol=trade_ticker, quoteOrderQty=f"{usdt_to_spend:.2f}")
                    print(f"✅ LIVE BUY ORDER PLACED.")
                    self.logger.info(f"LIVE MANUAL BUY ORDER PLACED: {order}")
                else:
                    print("✅ (Dry Run) BUY Trade was not placed.")

            elif trade_type == "SELL":
                coin_quantity_to_sell = amount if not is_quote_qty else 0
                if is_quote_qty:
                    # If selling for a USDT amount, we need to estimate the coin quantity
                    prices = self._get_current_prices([symbol])
                    if not prices.get(symbol) or prices[symbol] == 0:
                        print(f"❌ Could not fetch price for {symbol} to calculate quantity.")
                        return
                    coin_quantity_to_sell = amount / prices[symbol]

                # We must adjust the final coin quantity to match exchange rules (LOT_SIZE)
                adjusted_quantity = self._adjust_quantity_to_lot_size(trade_ticker, coin_quantity_to_sell)
                if adjusted_quantity is None or adjusted_quantity <= 0:
                    print(f"⚠️ SKIPPING SELL for {symbol}: Quantity is zero or invalid after applying exchange lot size rules.")
                    return

                print(f"\nPreparing MARKET SELL for {adjusted_quantity:.8g} {symbol}...")
                if is_live:
                    print("🚀 PLACING LIVE ORDER...")
                    order = self.binance_client.order_market_sell(symbol=trade_ticker, quantity=f"{adjusted_quantity:.8g}")
                    print(f"✅ LIVE SELL ORDER PLACED.")
                    self.logger.info(f"LIVE MANUAL SELL ORDER PLACED: {order}")
                else:
                    print("✅ (Dry Run) SELL Trade was not placed.")

            print("\n💡 Recommendation: Run 'Full Sync & Analysis' (Option 1) to update your portfolio with this trade.")

        except BinanceAPIException as e:
            print(f"❌ LIVE TRADE FAILED for {symbol}: {e}")
            self.logger.error(f"LIVE MANUAL TRADE FAILED for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during manual trade for {symbol}: {e}", exc_info=True)
            print(f"❌ Unexpected Error for {symbol}: {e}")

    async def get_core_portfolio_rebalance_suggestions_technical(self) -> Optional[pd.DataFrame]:
        """
        Generates rebalancing suggestions by fetching live portfolio data and passing
        it to the central rebalancing logic orchestrator.
        """
        self.logger.info("Calculating Core Portfolio rebalance suggestions...")

        # 1. Prepare live data
        analyzer = CryptoTrendAnalyzer(config=self.config, binance_client=self.binance_client)
        live_balances_df = self.fetch_binance_balances()
        if live_balances_df.empty:
            self.logger.error("Could not fetch live balances. Cannot rebalance.")
            return None

        prices = self._get_current_prices(list(live_balances_df["symbol"].unique()))
        live_balances_df["value_usd"] = live_balances_df["symbol"].map(prices).fillna(0.0) * live_balances_df["quantity"]

        # Get the list of coins we actually want to rebalance from the config
        target_symbols = list(self.config.get("target_allocation", {}).keys())

        # Filter the DataFrame to create a "core" portfolio
        core_portfolio_df = live_balances_df[live_balances_df["symbol"].isin(target_symbols)].copy()

        if core_portfolio_df.empty:
            self.logger.warning("No assets from target_allocation found in live balances. Cannot generate rebalancing suggestions.")
            return pd.DataFrame() # Return an empty DataFrame

        self.logger.info(f"Rebalancing logic will be based on {len(core_portfolio_df)} core assets.")

        # 2. Call the new central orchestrator with the live data
        suggestions_df = await get_live_rebalance_suggestions(
            analyzer=analyzer,
            portfolio_df=core_portfolio_df,
            config=self.config
        )

        return suggestions_df

    async def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculates key portfolio metrics using a consolidated view of holdings,
        correctly handling the relationship between Spot and Earn balances.
        """
        self.logger.info("Calculating consolidated portfolio metrics (Spot + Earn)...")
        cost_basis_df = self.db_manager.get_holdings()

        # 1. Fetch balances and merge with cost basis and price data
        total_balances_api_df = self.fetcher.fetch_binance_balances().rename(columns={"quantity": "total_quantity_api"})
        earn_balances_df = pd.DataFrame(columns=["symbol", "earn_quantity"])
        if not self.config_manager.is_testnet_mode:
            earn_dict = self.fetcher.fetch_simple_earn_balances(total_balances_api_df)
            if earn_dict:
                 earn_balances_df = pd.DataFrame(list(earn_dict.items()), columns=["symbol", "earn_quantity"])

        holdings_df = pd.merge(total_balances_api_df, earn_balances_df, on="symbol", how="outer")
        holdings_df["total_quantity_api"] = pd.to_numeric(holdings_df["total_quantity_api"], errors="coerce").fillna(0)
        holdings_df["earn_quantity"] = pd.to_numeric(holdings_df["earn_quantity"], errors="coerce").fillna(0)
        holdings_df["total_quantity"] = holdings_df[["total_quantity_api", "earn_quantity"]].max(axis=1)
        holdings_df["spot_quantity"] = (holdings_df["total_quantity"] - holdings_df["earn_quantity"]).clip(lower=0)
        holdings_df = holdings_df[holdings_df["total_quantity"] > 1e-8].reset_index(drop=True)

        if holdings_df.empty:
            return {"total_value_usd": 0, "holdings_df": pd.DataFrame()}

        if not cost_basis_df.empty:
            holdings_df = pd.merge(holdings_df, cost_basis_df[["symbol", "average_cost_basis"]], on="symbol", how="left")
        else:
            holdings_df["average_cost_basis"] = 0.0
        holdings_df["average_cost_basis"] = holdings_df["average_cost_basis"].fillna(0.0)

        prices = await self.enricher.get_current_prices(holdings_df["symbol"].tolist())
        holdings_df["current_price"] = holdings_df["symbol"].map(prices).fillna(0.0)

        # 2. Calculate existing metrics
        holdings_df["value_usd"] = holdings_df["total_quantity"] * holdings_df["current_price"]
        holdings_df["cost_basis_total"] = holdings_df["total_quantity"] * holdings_df["average_cost_basis"]
        holdings_df["unrealized_pl_usd"] = holdings_df["value_usd"] - holdings_df["cost_basis_total"]
        holdings_df.loc[holdings_df["cost_basis_total"] > 0, "unrealized_pl_percent"] = (holdings_df["unrealized_pl_usd"] / holdings_df["cost_basis_total"]) * 100

        total_value = holdings_df["value_usd"].sum()
        if total_value > 0:
            holdings_df["allocation"] = holdings_df["value_usd"] / total_value
        else:
            holdings_df["allocation"] = 0

        # 3. Separate Core vs. Other assets and calculate core-specific allocation
        target_symbols = list(self.config.get("target_allocation", {}).keys())
        holdings_df["is_core"] = holdings_df["symbol"].isin(target_symbols)

        core_holdings_df = holdings_df[holdings_df["is_core"]].copy()
        other_holdings_df = holdings_df[~holdings_df["is_core"]].copy()

        total_core_value = core_holdings_df["value_usd"].sum()
        if total_core_value > 0:
            core_holdings_df["core_allocation"] = core_holdings_df["value_usd"] / total_core_value
        else:
            core_holdings_df["core_allocation"] = 0

        total_cost_basis = holdings_df["cost_basis_total"].sum()
        total_pl_usd = total_value - total_cost_basis
        total_pl_percent = (total_pl_usd / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        total_invested = self.db_manager.calculate_total_invested_capital()
        overall_pl_usd = total_value - total_invested
        overall_pl_percent = (overall_pl_usd / total_invested * 100) if total_invested > 0 else 0.0

        metrics = {
            "total_value_usd": total_value,
            "total_cost_basis_usd": total_cost_basis,
            "unrealized_pl_usd": total_pl_usd,
            "unrealized_pl_percent": total_pl_percent,
            "total_invested_capital": total_invested,
            "overall_pl_usd": overall_pl_usd,
            "overall_pl_percent": overall_pl_percent,
            "holdings_df": holdings_df,
            "core_holdings_df": core_holdings_df,
            "other_holdings_df": other_holdings_df,
            "timestamp": datetime.datetime.now()
        }
        self.logger.info(f"Successfully calculated consolidated portfolio metrics.")
        return metrics

    async def view_trends(self):
        """
        Provides an interactive menu to view cryptocurrency trend analysis reports.
        """
        try:
            analyzer = CryptoTrendAnalyzer(config=self.config, binance_client=self.binance_client)
            print("✅ Trend Analyzer initialized with centralized config.")
        except Exception as e:
            self.logger.error(f"Failed to initialize CryptoTrendAnalyzer: {e}", exc_info=True)
            print("❌ Error: Could not initialize the Trend Analyzer. Please check the logs.")
            return

        while True:
            print("\n--- 📈 Crypto Trend Analysis ---\n")
            print("Select the timeframe for the analysis:")
            print("1. Long-term (4 Years)")
            print("2. Swing (3 Months)")
            print("3. Day (1 Month)")

            try:
                choice_str = input("Select option (1-3): ").strip()
                if not choice_str:
                    return
                if not choice_str.isdigit() or int(choice_str) not in range(1, 4):
                    print("❌ Invalid input. Please enter a number between 1 and 3.")
                    continue
                choice = int(choice_str)
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
                continue

            timeframe = None
            if choice == 1: timeframe = "long_term"
            elif choice == 2: timeframe = "swing"
            elif choice == 3: timeframe = "day"
            elif choice == 4: break

            if timeframe:
                print(f"\n🔄 Generating {timeframe.replace("_", " ")} trend report...")
                try:
                    report = await analyzer.generate_report(timeframe)

                    if report:
                        self.print_trend_report(report)
                        export_choice = input("\nDo you want to export this report? (y/n): ").lower()
                        if export_choice == "y":
                            self.export_trend_report(report, timeframe)
                    else:
                        print(f"❌ Could not generate the {timeframe} trend report. See logs for details.")

                except Exception as e:
                    self.logger.error(f"An error occurred during report generation for timeframe '{timeframe}': {e}", exc_info=True)
                    print(f"❌ An unexpected error occurred. Please check the logs.")

                input("\n✅ Press Enter to continue...")

    async def sync_data(self):
        """
        Orchestrates the Gather, Enrich, and Process pipeline, now intelligently
        skipping unsupported data sources when in Testnet mode.
        """
        self.logger.info("Starting data synchronization pipeline...")
        lookback_config = self.config.get("history_lookback_days", {})

        # Define all possible data sources for production mode
        all_data_sources = {
            "Binance Trade": {"fetcher": self.fetcher.fetch_binance_transactions, "days": lookback_config.get("trades", 90)},
            "Binance Deposit": {"fetcher": self.fetcher.fetch_deposit_history, "days": lookback_config.get("deposits", 90)},
            "Binance Withdrawal": {"fetcher": self.fetcher.fetch_withdrawal_history, "days": lookback_config.get("withdrawals", 90)},
            "Binance P2P Buy": {"fetcher": self.fetcher.fetch_p2p_usdt_buys, "days": lookback_config.get("p2p_buys", 90)},
            "Binance Convert": {"fetcher": self.fetcher.fetch_spot_convert_history, "days": lookback_config.get("spot_convert_history", 90)},
            "Binance Dividend": {"fetcher": self.fetcher.fetch_dividend_history, "days": lookback_config.get("dividend_history", 90)},
            "Binance Simple Earn Reward": {"fetcher": self.fetcher.fetch_simple_earn_rewards, "days": lookback_config.get("simple_earn_rewards", 90)},
            "Binance Simple Earn Subscription": {"fetcher": self.fetcher.fetch_simple_earn_subscriptions, "days": lookback_config.get("simple_earn_subscriptions", 90)},
            "Binance Simple Earn Redemption": {"fetcher": self.fetcher.fetch_simple_earn_redemptions, "days": lookback_config.get("simple_earn_redemptions", 90)},
        }

        data_sources_to_fetch = {}
        fetch_staking = False

        if self.config_manager.is_testnet_mode:
            # In testnet, only fetch spot trade history
            self.logger.warning("TESTNET MODE: Syncing SPOT TRADE history only.")
            data_sources_to_fetch = {
                "Binance Trade": all_data_sources["Binance Trade"]
            }
            fetch_staking = False
        else:
            # In production, fetch everything
            data_sources_to_fetch = all_data_sources
            fetch_staking = True

        fetcher_tasks = []
        for source, config in data_sources_to_fetch.items():
            latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source)
            if latest_known_ts:
                self.logger.info(f"History found for '{source}'. Initiating selective sync from {latest_known_ts.strftime('%Y-%m-%d %H:%M')}.")
            else:
                self.logger.info(f"No history found for '{source}'. Initiating full sync for the last {config['days']} days.")

            fetcher_tasks.append(
                asyncio.to_thread(config["fetcher"], source_name=source, days_back=config['days'], latest_known_ts=latest_known_ts)
            )

        # Only check for staking history if not in testnet mode
        if fetch_staking:
            staking_ts_map = {
                "Binance Staking Subscription": self.db_manager.get_latest_timestamp_for_source("Binance Staking Subscription"),
                "Binance Staking Redemption": self.db_manager.get_latest_timestamp_for_source("Binance Staking Redemption"),
                "Binance Staking Interest": self.db_manager.get_latest_timestamp_for_source("Binance Staking Interest"),
            }
            self.logger.info("Checking for Staking history (Subscriptions, Redemptions, Interest)...")
            fetcher_tasks.append(
                asyncio.to_thread(self.fetcher.fetch_staking_history, source_name="Binance Staking", days_back=lookback_config.get("staking_history", 90), latest_known_ts_map=staking_ts_map)
            )

        if not fetcher_tasks:
            self.logger.info("No data sources to sync in the current mode.")
            return

        # 1. GATHER
        self.logger.info(f"Launching {len(fetcher_tasks)} data fetching tasks concurrently...")
        results = await asyncio.gather(*fetcher_tasks, return_exceptions=True)

        all_raw_transactions = []
        for res in results:
            if isinstance(res, list):
                all_raw_transactions.extend(res)
            elif isinstance(res, Exception):
                self.logger.error(f"An error occurred during a fetching task: {res}", exc_info=False)

        if not all_raw_transactions:
            self.logger.info("No new raw transactions were fetched. Sync complete.")
            return

        # 2. ENRICH
        self.logger.info(f"Passing {len(all_raw_transactions)} raw transactions to the PriceEnricher.")
        enriched_transactions = await self.enricher.enrich_transactions(all_raw_transactions)

        # 3. PROCESS
        if enriched_transactions:
            enriched_transactions.sort(key=lambda x: x['timestamp'])
            self.logger.info(f"Saving {len(enriched_transactions)} enriched transactions to the database...")
            num_inserted = self.db_manager.bulk_insert_transactions(enriched_transactions)
            self.logger.info(f"Database update complete. {num_inserted} new transactions were saved.")
        else:
            self.logger.warning("Enrichment process returned no transactions. Nothing to save.")

        self.update_holdings_from_transactions()
        self.logger.info("Data synchronization pipeline finished successfully.")

    async def run_full_sync(self) -> Dict[str, Any]:
        """Runs the full async data sync and then calculates metrics."""
        await self.sync_data()
        return await self.calculate_portfolio_metrics()

    async def run_rebalancing_backtest(self):
        """
        Handles the user flow for running the rebalancing backtest.
        """
        print("\n--- ⚖️ Rebalancing Strategy Backtesting Mode ---")
        try:
            backtester = RebalancingBacktester(config=self.config)

            initial_capital_str = input("Enter initial capital for simulation (default: 10000): ")
            initial_capital = float(initial_capital_str) if initial_capital_str else 10000.0

            period = ""
            while True:
                period_str = input("Enter backtest period (e.g., 2y, 3y, 5y - default: 3y): ").strip().lower()
                if not period_str:
                    period = "3y"  # Default value
                    break
                # Regex to validate format like "1d", "6m", "5y"
                if re.match(r"^\d+[dmy]$", period_str):
                    period = period_str
                    break
                else:
                    print("❌ Invalid format. Please use a number followed by 'd', 'm', or 'y' (e.g., '90d', '6m', '3y').")

            backtester.run(initial_capital=initial_capital, period=period)

        except Exception as e:
            self.logger.error(f"An error occurred during rebalancing backtest: {e}", exc_info=True)
            print(f"❌ An unexpected error occurred: {e}")

    async def run_manual_trade_session(self):
        """Runs an interactive session for placing a manual trade."""
        print("\n--- TRADE Manual Trading ---")
        is_live = self.config.get("portfolio", {}).get("live_trading_enabled", False)
        if not is_live:
            print("🟡 NOTE: Live Trading is DISABLED. All trades will be simulated (Dry Run).")
        else:
            print("🔴 WARNING: Live Trading is ENABLED. Real orders will be placed.")

        # 1. Get Trade Type
        trade_type = ""
        while trade_type not in ["BUY", "SELL"]:
            trade_type = input("Choose action [BUY / SELL]: ").upper().strip()
            if not trade_type:
                print("Returning to main menu...")
                return

        # 2. Get Asset
        symbol = input("Enter asset symbol (e.g., BTC): ").upper().strip()
        if not symbol:
            print("Returning to main menu...")
            return
        trade_ticker = f"{symbol}USDT"

        # 3. Get Amount
        amount_str_input = input(f"Enter amount to {trade_type} (e.g., '0.1 {symbol}'): ").upper().strip()
        if not amount_str_input:
            print("Returning to main menu...")
            return

        is_quote_qty = "USDT" in amount_str_input
        try:
            # Extract numbers from the string
            numeric_part = re.search(r"[\d\.]+", amount_str_input).group(0)
            amount = float(numeric_part)
        except (AttributeError, ValueError):
            print("❌ Invalid amount format. Could not parse number.")
            return

        # 4. Confirmation
        print("\n" + "="*50)
        print("🚨 PLEASE CONFIRM THE FOLLOWING MARKET ORDER 🚨")
        print(f"   Action: {trade_type}")
        print(f"   Asset:  {symbol}")
        if is_quote_qty:
            print(f"   Amount: {amount:,.2f} USDT")
        else:
            print(f"   Amount: {amount:,.8g} {symbol}")
        print("="*50)

        confirm = input("Type 'EXECUTE' to confirm: ").strip()

        if confirm == "EXECUTE":
            await self._execute_manual_trade(trade_type, symbol, trade_ticker, amount, is_quote_qty, is_live)
        else:
            print("🛑 Trade cancelled by user.")

    async def run_rebalance_and_execute(self):
        """
        Orchestrates the process of getting rebalancing suggestions and allows users
        to execute them all at once or one-by-one.
        """
        self.logger.info("Starting automated rebalancing process...")
        print("\n--- ⚖️ Automated Rebalancing ---")

        # 1. Get Suggestions
        print("🔄 Generating rebalancing suggestions...")
        suggestions_df = await self.get_core_portfolio_rebalance_suggestions_technical()

        if suggestions_df is None or suggestions_df.empty:
            print("\n✅ No rebalancing suggestions available at this time.")
            self.logger.info("No rebalancing suggestions generated. Exiting process.")
            return

        # 2. Calculate TOTAL available USDT (Spot + Earn)
        total_usdt_balance = 0
        try:
            # Fetch Spot Balance
            spot_balance = float(self.binance_client.get_asset_balance(asset="USDT").get("free", 0.0))
            total_usdt_balance += spot_balance

            # Fetch Earn Balance (if not in testnet mode)
            if not self.config_manager.is_testnet_mode:
                earn_positions = self.fetcher.fetch_simple_earn_balances(pd.DataFrame([{"symbol": "USDT"}]))
                earn_balance = earn_positions.get("USDT", 0.0)
                total_usdt_balance += earn_balance

        except Exception as e:
            self.logger.error(f"Could not fetch total USDT balance for display: {e}")

        # 3. Print Suggestions for Review (passing the USDT balance to the printer)
        self.print_rebalance_suggestions(suggestions_df, available_usdt=total_usdt_balance)

        # 4. Check for actionable signals before showing execution prompts
        actionable_trades = suggestions_df[suggestions_df["Signal"].isin(["BUY", "SELL"])]
        if actionable_trades.empty:
            print("\n✅ Your portfolio is balanced. No rebalancing needed at this time.")
            return

        # 4. Add interactive execution choice
        while True:
            action = input(
                "Type EXECUTE ALL or EXECUTE for one-by-one confirmation: "
            ).upper().strip()

            if not action:
                print("Returning to main menu...")
                return

            elif action in ["EXECUTE ALL", "EXECUTE"]:
                self.logger.info("Preparing for rebalance execution...")
                self._sync_binance_client_time(self.binance_client, context="rebalancing")

                # Determine mode and proceed to fetch balances and execute
                interactive_mode = (action == "EXECUTE")

                # Fetch Earn balances ONLY if not in testnet mode
                earn_balances = {}
                if not self.config_manager.is_testnet_mode:
                    print("Verifying balances in Spot and Earn wallets...")
                    spot_balances_df = self.fetch_binance_balances()
                    earn_balances = self.fetcher.fetch_simple_earn_balances(spot_balances_df)
                else:
                    print("🟡 TESTNET MODE: Skipping Earn wallet check.")

                # Pass everything to the Executor with the chosen mode
                await self._execute_rebalancing_trades(suggestions_df, earn_balances, interactive=interactive_mode)
                return # Exit after execution attempt

            else:
                print("Invalid command. Please choose from EXECUTE ALL, EXECUTE.")

    async def run_trading_strategy_backtest(self):
        """
        Handles the user flow for running a directional strategy backtest,
        dynamically discovering and allowing strategies to configure themselves.
        """
        print("\n--- 🧪 Directional Strategy Backtesting Mode ---")

        try:
            analyzer = CryptoTrendAnalyzer(config=self.config, binance_client=self.binance_client)
            backtester = StrategyBacktester(config=self.config, analyzer=analyzer)
        except Exception as e:
            self.logger.error(f"Failed to initialize backtesting components: {e}", exc_info=True)
            return

        # Discover and select a strategy
        available_strategies = {
            name: obj for name, obj in inspect.getmembers(trading_strategies, inspect.isclass)
            if issubclass(obj, trading_strategies.Strategy) and obj is not trading_strategies.Strategy
        }
        if not available_strategies:
            print("❌ No strategy classes found in 'src/trading_strategies.py'."); return

        print("\nAvailable Strategies:")
        strategy_list = list(available_strategies.keys())
        for i, name in enumerate(strategy_list):
            print(f"  {i+1}. {name}")

        strategy_to_run = None
        while strategy_to_run is None:
            try:
                choice_str = input(f"Select the strategy to backtest (1-{len(strategy_list)}): ").strip()
                if not choice_str: return
                choice = int(choice_str) - 1
                if 0 <= choice < len(strategy_list):
                    strategy_name = strategy_list[choice]
                    strategy_class = available_strategies[strategy_name]
                    print(f"\nConfiguring strategy: {strategy_name}")

                    user_params = {}
                    if hasattr(strategy_class, "get_user_params"):
                         user_params = strategy_class.get_user_params()

                    strategy_to_run = strategy_class(analyzer=analyzer, **user_params)
                else:
                    print("❌ Invalid selection.")
            except (ValueError, IndexError):
                print("❌ Invalid input. Please enter a number from the list.")

        if not strategy_to_run: return
        print(f"\nSelected Strategy: {strategy_to_run.name}")

        # Determine the data interval for the strategy
        valid_intervals = strategy_to_run.valid_intervals
        interval = None
        if len(valid_intervals) == 1:
            interval = valid_intervals[0]
            print(f"INFO: This strategy uses the '{interval}' data interval by default.")
        elif len(valid_intervals) > 1:
            print("\nSelect a valid data interval for this strategy:")
            for i, iv in enumerate(valid_intervals): print(f"  {i+1}. {iv}")
            while interval is None:
                try:
                    interval_choice_str = input(f"Select interval (1-{len(valid_intervals)}): ").strip()
                    selected_idx = int(interval_choice_str) - 1
                    if 0 <= selected_idx < len(valid_intervals): interval = valid_intervals[selected_idx]
                    else: print("❌ Invalid selection.")
                except (ValueError, IndexError): print("❌ Invalid input.")

        if interval is None: interval = "1d"
        print(f"Using data interval: {interval}")

        # Select the coin to test
        available_coins = self.config.get("trend_analyzer", {}).get("cryptocurrencies", [])
        if not available_coins:
            self.logger.info("Trend analyzer coin list is empty, automatically using coins from target_allocation.")
            target_coins = self.config.get("target_allocation", {}).keys()
            available_coins = [f"{coin.upper()}-USD" for coin in target_coins]
        if not available_coins:
            print("❌ No cryptocurrencies found in 'target_allocation' or 'trend_analyzer' config section."); return

        available_coins.sort()
        print("\nCoins available for backtesting:")
        num_coins = len(available_coins)
        cols = 3
        for i in range(0, num_coins, cols):
            row = "  ";
            for j in range(cols):
                if i + j < num_coins: row += f"{i+j+1:2d}. {available_coins[i+j]:<15}"
            print(row)

        symbol_to_test = None
        while symbol_to_test is None:
            choice = input("\nEnter the symbol or number of the coin to backtest: ").strip()
            if choice.isdigit():
                try:
                    choice_index = int(choice) - 1
                    if 0 <= choice_index < num_coins: symbol_to_test = available_coins[choice_index]
                    else: print(f"❌ Invalid number. Please enter a number between 1 and {num_coins}.")
                except ValueError: print("❌ Invalid input.")
            else:
                yf_ticker_choice = f"{choice.upper()}-USD"
                if yf_ticker_choice in available_coins: symbol_to_test = yf_ticker_choice
                else: print(f"❌ Symbol '{choice}' not found. Please choose from the list.")

        print(f"\nSelected coin for backtest: {symbol_to_test}")

        try:
            initial_capital_str = input("Enter initial capital (default: 10000): ")
            initial_capital = float(initial_capital_str) if initial_capital_str else 10000.0
            period_str = input("Enter backtest period (e.g., '3y', '60d', default: '3y'): ")
            period = period_str if period_str else "3y"

            if "m" in interval and "y" in period:
                print("⚠️ WARNING: yfinance limits minute data to the last 60 days.")
                print(f"Adjusting backtest period from '{period}' to '60d'.")
                period = "60d"
            elif "h" in interval and "y" in period and int(period.replace("y", "")) > 2:
                 print("⚠️ WARNING: yfinance limits hourly data to the last 730 days (2 years).")
                 print(f"Adjusting backtest period from '{period}' to '729d'.")
                 period = "729d"

        except ValueError:
            print("❌ Invalid capital amount. Using default values."); initial_capital = 10000.0; period = "3y"

        await backtester.run(strategy=strategy_to_run, symbol=symbol_to_test, initial_capital=initial_capital, period=period, interval=interval)
        backtester.generate_report()

    async def run_live_strategy(self):
        """
        Master workflow for running a directional strategy for live signal generation and execution.
        """
        print("\n--- 🤖 Live Trading Strategy Runner ---")

        # 1. Select Account
        accounts = [{"name": "Main Account", "type": "main", **self.config.get("main_api_keys", {})}]
        sub_accounts = self.config.get("sub_accounts", [])
        for sub in sub_accounts:
            # Infer account type from name
            if "swing" in sub["name"].lower():
                sub["type"] = "swing"
            elif "day" in sub["name"].lower():
                sub["type"] = "day"
            else:
                sub["type"] = "main" # Default if not specified
            accounts.append(sub)


        if not any(acc.get("binance_key") for acc in accounts):
            print("❌ No API keys found for any account. Cannot run live strategies.")
            return

        print("\nSelect account to trade on:")
        for i, acc in enumerate(accounts):
            print(f"  {i+1}. {acc["name"]} (Type: {acc.get("type", "N/A")})")

        selected_account = None
        while selected_account is None:
            choice_str = input(f"Select account (1-{len(accounts)}): ").strip()
            if not choice_str:
                print("Returning to main menu...")
                return

            try:
                choice = int(choice_str) - 1
                if 0 <= choice < len(accounts):
                    selected_account = accounts[choice]
                else:
                    print("❌ Invalid selection.")
            except (ValueError, IndexError):
                print("❌ Invalid input.")

        print(f"\nTrading on account: {selected_account["name"]}")
        live_client = self._init_binance_client(
            api_key=selected_account.get("binance_key"),
            api_secret=selected_account.get("binance_secret")
        )
        if not live_client:
            print(f"❌ Failed to initialize Binance client for account: {selected_account["name"]}. Check API keys.")
            return

        # 2. Select and Configure Strategy based on account type
        all_strategies = {name: obj for name, obj in inspect.getmembers(trading_strategies, inspect.isclass) if issubclass(obj, trading_strategies.Strategy) and obj is not trading_strategies.Strategy}

        account_type = selected_account.get("type")
        available_strategies = {}
        if account_type == "main":
            available_strategies = all_strategies
        elif account_type == "swing":
            available_strategies = {k: v for k, v in all_strategies.items() if v.strategy_type in ["swing", "general"]}
        elif account_type == "day":
             available_strategies = {k: v for k, v in all_strategies.items() if v.strategy_type in ["day", "general"]}

        if not available_strategies:
            print(f"❌ No suitable strategies found for an account of type '{account_type}'.")
            return

        print("\nAvailable Strategies:")
        strategy_list = list(available_strategies.keys())
        for i, name in enumerate(strategy_list):
            print(f"  {i+1}. {name}")

        strategy_class = None
        user_params = {}
        strategy_name = ""
        while strategy_class is None:
            choice_str = input(f"Select strategy to run (1-{len(strategy_list)}): ").strip()
            if not choice_str:
                print("Returning to main menu...")
                return

            try:
                choice = int(choice_str) - 1
                if 0 <= choice < len(strategy_list):
                    strategy_name = strategy_list[choice]
                    strategy_class = available_strategies[strategy_name]
                    print(f"\nConfiguring strategy: {strategy_name}")
                    if hasattr(strategy_class, 'get_user_params'):
                        user_params = strategy_class.get_user_params()
                else:
                    print("❌ Invalid selection.")
            except (ValueError, IndexError):
                print("❌ Invalid input.")

        # Create a temporary instance just to get the name for the print message
        temp_strategy_for_name = strategy_class(analyzer=None, **user_params)
        print(f"\n🔄 Running '{temp_strategy_for_name.name}' to generate live signals...")

        # 3. Generate Signals for all portfolio assets
        target_coins = list(self.config.get("target_allocation", {}).keys())
        signals_to_execute = []
        analyzer = CryptoTrendAnalyzer(config=self.config, binance_client=live_client)

        for coin in target_coins:
            yf_ticker = f"{coin}-USD"
            analyzer.set_symbol(yf_ticker)
            state_key = f"{selected_account["name"]}_{strategy_name}_{coin}"
            previous_state = self.strategy_states.get(state_key)

            strategy_instance = strategy_class(
                analyzer=analyzer,
                state=previous_state,
                **user_params
            )

            interval = strategy_instance.valid_intervals[0]
            # Use a shorter period for live trading to be faster
            period = "7d" if "m" in interval or "h" in interval else "1y"
            data = await analyzer.fetch_crypto_data_async(yf_ticker, period=period, interval=interval)

            if data is None or data.empty:
                self.logger.warning(f"Could not fetch data for {yf_ticker}, cannot generate signal.")
                continue

            signal, size, reason = await strategy_instance.generate_signal(data)
            self.strategy_states[state_key] = strategy_instance.get_state()

            if signal in ["BUY", "SELL"]:
                signals_to_execute.append({"Symbol": coin, "Signal": signal, "Size": size, "Reason": reason})

        self._save_strategy_state()

        # 4. Present Trades and Ask for Execution
        if not signals_to_execute:
            print("\n✅ Analysis complete. No new BUY or SELL signals generated by the strategy.")
            return

        is_live = self.config.get("portfolio", {}).get("live_trading_enabled", False)
        is_testnet = self.config.get("apis", {}).get("binance", {}).get("testnet", False)

        print("\n" + "="*80)
        print("🚨 PROPOSED TRADES - PLEASE REVIEW CAREFULLY 🚨")
        print("="*80)

        if is_testnet:
            print("🟡🟡🟡 NOTE: Connected to TESTNET. No real funds will be used. 🟡🟡🟡")

        if is_live:
            print("🔴🔴🔴 WARNING: Live Trading is ENABLED. Real orders will be placed. 🔴🔴🔴")
        else:
            print("🟡🟡🟡 NOTE: Live Trading is DISABLED. This is a DRY RUN. 🟡🟡🟡")

        print("="*80)
        for trade in signals_to_execute:
            print(f"-> {trade["Signal"]} {trade["Symbol"]} (Reason: {trade["Reason"]})")
        print("="*80)

        try:
            confirm = input("Type 'EXECUTE' to proceed with the trades listed above: ")
            if confirm != "EXECUTE":
                print("🛑 Trade execution cancelled by user.")
                return
        except KeyboardInterrupt:
            print("\n🛑 Trade execution cancelled by user.")
            return

        # 5. Execute Trades
        for trade in signals_to_execute:
            self._execute_directional_trade(trade, live_client)

    def run_backup_and_restore_session(self):
        """Orchestrates creating a backup or restoring from one."""
        print("\n--- 🗄️ Database Backup & Restore ---\n")
        print("1. Create a new database backup")
        print("2. Restore from an existing backup")
        print("Press Enter to return to the main menu.")

        choice = input("Select an option: ").strip()

        if choice == '1':
            print("\n💾 Creating database backup...")
            backup_path = self.db_manager.backup_database()
            if backup_path:
                print(f"✅ Backup successful.")
            else:
                print("❌ Backup failed. Please check the logs for details.")
        elif choice == '2':
            self._restore_database_interactive()
        else:
            print("Returning to main menu...")
            return

    def fetch_binance_balances(self) -> pd.DataFrame:
        """Fetch current balances from Binance Spot wallet with retry, explicit normalization, and LD-prefix consolidation."""
        if not self.binance_client:
            self.logger.warning("Binance client not initialized. Cannot fetch Spot balances.")
            return pd.DataFrame(columns=["symbol", "quantity"])

        # --- Close stale connections before making a new request ---
        self.binance_client.session.close()

        retries = 3
        wait_time_seconds = 15
        api_timeout = self.config.get("apis", {}).get("binance", {}).get("timeout", 180)

        while retries > 0:
            try:
                self.logger.debug(f"Attempting to fetch Binance account info (Timeout: {api_timeout}s)...")
                account_info = self.binance_client.get_account()

                balances_raw = account_info.get("balances", [])
                if not balances_raw:
                    self.logger.info("Fetched 0 balances from Binance Spot wallet.")
                    return pd.DataFrame(columns=["symbol", "quantity"])

                processed_balances = []
                for b in balances_raw:
                    free = float(b.get("free", 0.0))
                    locked = float(b.get("locked", 0.0))
                    quantity = free + locked

                    if quantity > 0.00000001:
                        # Get the raw symbol from the API
                        asset_symbol_api = b.get("asset", "")
                        final_symbol = self.symbol_mappings.normalize_symbol(asset_symbol_api)

                        # Only append if the final symbol is valid (not empty)
                        if final_symbol:
                            processed_balances.append({"symbol": final_symbol, "quantity": quantity})


                if not processed_balances:
                    self.logger.info("Found raw balances, but all were zero or negligible after processing.")
                    return pd.DataFrame(columns=["symbol", "quantity"])

                df = pd.DataFrame(processed_balances)
                df = df.groupby("symbol", as_index=False)["quantity"].sum()

                self.logger.debug(f"DEBUG: Balances from SPOT after all processing in fetch_binance_balances: \n{df.to_string() if not df.empty else "EMPTY DF"}")
                self.logger.info(f"Fetched and consolidated {len(df)} non-zero balances from Binance Spot.")
                return df

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                retries -= 1
                self.logger.error(f"Network error fetching Binance Spot balances: {e}. Retries left: {retries}.")
                if retries > 0: self.logger.info(f"Waiting {wait_time_seconds}s before retrying..."); time.sleep(wait_time_seconds)
                else: self.logger.error("Failed to fetch Spot balances after multiple retries due to network errors.")
            except BinanceAPIException as e:
                 self.logger.error(f"Binance API Error fetching Spot balances: {e}. No retries for API errors.")
                 break
            except Exception as e:
                self.logger.error(f"Unexpected error fetching Spot balances: {e}", exc_info=True)
                break
        return pd.DataFrame(columns=["symbol", "quantity"])

    def update_holdings_from_transactions(self):
        """
        Processes all transactions and updates holdings table with FIFO cost basis.
        """
        self.logger.info("Updating holdings from transaction history using FIFO...")
        all_txs = self.db_manager.get_all_transactions()

        if all_txs.empty:
            self.logger.warning("No transactions found in DB. Cannot update holdings.")
            return

        updated_holdings = []
        for symbol, group_df in all_txs.groupby("symbol"):
            self.logger.debug(f"Calculating FIFO for {symbol}...")

            group_df_copy = group_df.copy()
            group_df_copy["price_usd"] = pd.to_numeric(group_df_copy["price_usd"], errors="coerce").fillna(0.0)
            group_df_copy["quantity"] = pd.to_numeric(group_df_copy["quantity"], errors="coerce").fillna(0.0)
            group_df_copy["timestamp"] = pd.to_datetime(group_df_copy["timestamp"], errors="coerce")
            group_df_copy.dropna(subset=["timestamp"], inplace=True)

            # 1. Isolate real trades for cost basis calculation
            cost_basis_tx_df = group_df_copy[
                ~group_df_copy["source"].str.contains("Simple Earn|Asset Transfer|Staking", case=False, na=False)
            ]
            self.logger.info(f"Calculating cost basis for {symbol} using {len(cost_basis_tx_df)} non-transfer transactions.")

            if cost_basis_tx_df.empty:
                self.logger.info(f"No non-transfer transactions for {symbol}. Cannot calculate cost basis.")
                continue  # Skip to the next symbol

            # 2. Calculate cost basis and the remaining quantity FROM THAT BASIS
            cost_basis_qty, avg_cost = calculate_fifo_cost_basis(cost_basis_tx_df)

            # 3. If there"s a valid average cost, save it.
            # The quantity saved here is just a placeholder; the final report uses the live wallet balance.
            if avg_cost > 0:
                self.logger.info(f"Calculated for {symbol}: Qty_from_basis={cost_basis_qty:.8f}, AvgCost={avg_cost:.8f}. Storing avg_cost.")
                updated_holdings.append({
                    "symbol": symbol,
                    "quantity": cost_basis_qty,
                    "average_cost_basis": avg_cost
                })
            else:
                self.logger.info(f"No cost basis calculated for {symbol} (likely no 'BUY' transactions in history).")

        if updated_holdings:
            holdings_df = pd.DataFrame(updated_holdings)
            self.db_manager.update_holdings(holdings_df)
            self.logger.info(f"Successfully updated/inserted {len(holdings_df)} asset holdings in the database with new cost basis.")
        else:
            self.logger.warning("No holdings with valid cost basis to update in the database.")

    def create_portfolio_charts(self, metrics: Dict[str, Any]):
        """Generate portfolio charts."""
        holdings_df = metrics.get("holdings_df"); target_alloc = self.config.get("target_allocation", {})
        if holdings_df is not None: self.visualizer.generate_all_charts(holdings_df, metrics, target_alloc, pd.DataFrame())
        else: self.logger.warning("No holdings data for chart generation.")

    def print_portfolio_summary(self, metrics: Dict[str, Any]):
        """Prints a consolidated summary of the portfolio, including Spot and Earn balances."""
        LINE_WIDTH = 115
        print("\n" + "="*LINE_WIDTH)
        print("📊 CONSOLIDATED PORTFOLIO SUMMARY (Spot + Earn)")
        print("="*LINE_WIDTH)

        if "error" in metrics:
            print(f"❌ Could not generate summary: {metrics["error"]}")
            print("="*LINE_WIDTH)
            return

        timestamp = metrics.get("timestamp", datetime.datetime.now())
        print(f"Timestamp:                   {timestamp.strftime("%Y-%m-%d %H:%M:%S")}")
        db_path = self.config.get("database", {}).get("path", "N/A")
        db_name = Path(db_path).name
        if self.config_manager.is_testnet_mode:
            print(f"Database:                    {db_name} (TESTNET MODE)")
        else:
            print(f"Database:                    {db_name}")

        print("-" * LINE_WIDTH)

        total_invested = metrics.get("total_invested_capital", 0)
        overall_pl_usd = metrics.get("overall_pl_usd", 0)
        overall_pl_pct = metrics.get("overall_pl_percent", 0)
        color_overall = "\033[92m" if overall_pl_usd >= 0 else "\033[91m"
        color_end = "\033[0m"

        print("Performance vs. Invested capital:")
        print(f"Total Invested Capital:      ${total_invested:,.2f}")
        print(f"Overall P/L:                 {color_overall}${overall_pl_usd:,.2f} ({overall_pl_pct:.2f}%){color_end}")
        print("-" * LINE_WIDTH)

        print(f"TOTAL PORTFOLIO VALUE:       ${metrics.get("total_value_usd", 0):,.2f}")
        print("-" * LINE_WIDTH)

        print("Performance vs. Rolling cost basis:")
        print(f"Total Cost Basis (FIFO):     ${metrics.get("total_cost_basis_usd", 0):,.2f}")

        pl_usd = metrics.get("unrealized_pl_usd", 0)
        pl_pct = metrics.get("unrealized_pl_percent", 0)
        color_unrealized = "\033[92m" if pl_usd >= 0 else "\033[91m"

        print(f"Unrealized P/L (FIFO):     {color_unrealized}${pl_usd:,.2f} ({pl_pct:.2f}%){color_end}")
        print("-" * LINE_WIDTH)

        # Print Core Holdings Table
        core_holdings_df = metrics.get("core_holdings_df")
        if core_holdings_df is not None and not core_holdings_df.empty:
            print("\n" + "--- 🎯 Core Portfolio Holdings (Used for Rebalancing) ---".center(LINE_WIDTH))
            # Increased "Asset" width from 8 to 11, decreased "Total Qty" from 18 to 15
            header = f"{"Asset":<11} {"Total Qty":<15} {"Spot Qty":<15} {"Earn Qty":<15} {"Value (USD)":<15} {"Cost Basis":<15} {"P/L (USD)":<15} {"Core Alloc.":<15} {"Total Alloc.":<10}"
            print(header)
            print("-" * len(header))
            for _, row in core_holdings_df.sort_values(by="value_usd", ascending=False).iterrows():
                row_pl_usd = row.get("unrealized_pl_usd", 0)
                row_color_start = "\033[92m" if row_pl_usd >= 0 else "\033[91m"

                core_alloc_str = f"{row.get("core_allocation", 0) * 100:.2f}%"
                total_alloc_str = f"{row.get("allocation", 0) * 100:.2f}%"
                print(
                    f"{row.get("symbol", "N/A"):<11} "
                    f"{row.get("total_quantity", 0):<15,.8g} "
                    f"{row.get("spot_quantity", 0):<15,.8g} "
                    f"{row.get("earn_quantity", 0):<15,.8g} "
                    f"${row.get("value_usd", 0):<14,.2f} "
                    f"${row.get("cost_basis_total", 0):<14,.2f} "
                    f"{row_color_start}${row_pl_usd:<14,.2f}{color_end} "
                    f"{core_alloc_str:<15} "
                    f"{total_alloc_str:<10}"
                )

        # Print Other Holdings Table
        other_holdings_df = metrics.get("other_holdings_df")
        if other_holdings_df is not None and not other_holdings_df.empty:
            print("\n" + "--- 📈 Other Holdings ---".center(LINE_WIDTH))
            # Adjusted header and row formatting for consistency
            header = f"{"Asset":<11} {"Total Qty":<15} {"Spot Qty":<15} {"Earn Qty":<15} {"Value (USD)":<15} {"Cost Basis":<15} {"P/L (USD)":<15} {"Total Alloc.":<10}"
            print(header)
            print("-" * len(header))
            for _, row in other_holdings_df.sort_values(by="value_usd", ascending=False).iterrows():
                row_pl_usd = row.get("unrealized_pl_usd", 0)
                row_color_start = "\033[92m" if row_pl_usd >= 0 else "\033[91m"

                total_alloc_str = f"{row.get("allocation", 0) * 100:.2f}%"
                print(
                    f"{row.get("symbol", "N/A"):<11} "
                    f"{row.get("total_quantity", 0):<15,.8g} "
                    f"{row.get("spot_quantity", 0):<15,.8g} "
                    f"{row.get("earn_quantity", 0):<15,.8g} "
                    f"${row.get("value_usd", 0):<14,.2f} "
                    f"${row.get("cost_basis_total", 0):<14,.2f} "
                    f"{row_color_start}${row_pl_usd:<14,.2f}{color_end} "
                    f"{total_alloc_str:<10}"
                )
        print("="*len(header))

    def print_trend_report(self, report: Dict[str, Any]):
        """Prints a formatted trend analysis report to the console."""
        if not report:
            print("\n--- No Trend Report Data ---")
            return

        print("\n" + "="*80)
        print(f"📈 TREND ANALYSIS REPORT ({report.get("timeframe", "N/A").upper()})")
        print(f"Timestamp: {report.get("timestamp")}")
        print("="*80)

        summary = report.get("market_summary", {})
        print("\n--- 🌍 Market Summary ---")
        print(f"Coins Analyzed: {summary.get("total_coins", "N/A")}")
        print(f"Most Common Condition: {summary.get("most_common_condition", "N/A")}")
        print(f"Bullish Coins: {summary.get("bull_count", 0)} | Bearish Coins: {summary.get("bear_count", 0)}")

        btc = report.get("benchmark_analysis", {})
        if btc:
            print(f"\n--- 🎯 Benchmark Analysis: {btc.get("symbol", "N/A")} ---")
            print(f"  Price: ${btc.get("current_price", 0):,.2f} ({btc.get("price_change_pct", 0):+.2f}%) | RSI: {btc.get("rsi", 0):.2f}")
            print(f"  Support: ${btc.get("support_level", 0):,.2f} | Resistance: ${btc.get("resistance_level", 0):,.2f}")
            print(f"  Active Conditions: {", ".join(btc.get("active_conditions", ["None"]))}")

        print("\n--- 🪙 Coin-by-Coin Analysis ---")
        for symbol, analysis in report.get("coin_analyses", {}).items():
            if symbol == btc.get("symbol"): continue # Skip printing benchmark again

            print(f"\n➡️ {symbol}")
            print(f"  Price: ${analysis.get("current_price", 0):,.2f} ({analysis.get("price_change_pct", 0):+.2f}%) | RSI: {analysis.get("rsi", 0):.2f}")
            print(f"  Support: ${analysis.get("support_level", 0):,.2f} | Resistance: ${analysis.get("resistance_level", 0):,.2f}")
            print(f"  Active Conditions: {", ".join(analysis.get("active_conditions", ["None"]))}")

        print("\n" + "="*80)

    def print_rebalance_suggestions(self, suggestions_df: Optional[pd.DataFrame], available_usdt: Optional[float] = None):
        """Prints the rebalancing suggestions in a formatted way."""
        if suggestions_df is None or suggestions_df.empty:
            print("No rebalancing suggestions available.")
            return

        # Sort the DataFrame before printing for a logical display
        signal_order = {"SELL": 0, "BUY": 1, "HOLD": 2}
        suggestions_df["signal_order"] = suggestions_df["Signal"].map(signal_order)
        suggestions_df = suggestions_df.sort_values(by=["signal_order", "Drift (pts)"], ascending=[True, True])
        suggestions_df = suggestions_df.drop(columns=["signal_order"])

        print("\n" + "="*88)
        print("⚖️  REBALANCING SUGGESTIONS (Multi-Timeframe Analysis)")
        print("="*88)

        # Calculate and display the total value of the core portfolio being rebalanced
        total_core_value = suggestions_df["Current Value (USD)"].sum()
        print("-" * 88)
        print(f"💰 Core Portfolio Value: ${total_core_value:,.2f}")

        # Display the USDT balance inside the header if it was provided
        if available_usdt is not None:
            print(f"💰 Available USDT (Spot + Earn): ${available_usdt:,.2f}")
            print("-" * 88)

        for _, row in suggestions_df.iterrows():
            signal = row["Signal"]
            if signal == "SELL": color_code = "\033[91m"
            elif signal == "BUY": color_code = "\033[92m"
            else: color_code = "\033[93m"
            reset_code = "\033[0m"

            signal_icon = {"SELL": "🔴", "BUY": "🟢", "HOLD": "🟡"}.get(signal, "⚪️")

            print(f"{color_code}{signal_icon} {row["Symbol"]:<7}| Signal: {signal:<4}{reset_code}")
            print(f"   Allocation: {row["Current %"]:.2f}% (Target: {row["Target %"]:.1f}%) | Drift: {row["Drift (pts)"]:.2f} pts | Value: ${row["Current Value (USD)"]:,.2f}")

            price_str = f"${row["TA_Price"]:,.2f}".ljust(12)
            support_str = f"${row.get("Support", 0.0):,.2f}"
            resistance_str = f"${row.get("Resistance", 0.0):,.2f}"

            print(f"   Price: {price_str} | Support: {support_str} | Resistance: {resistance_str}")

            print(f"   Long-Term Trend: {row["TA_Conditions"]}")

            print(f"   Action: {row["Suggested Action Detail"]}")

            if row.name != suggestions_df.index[-1]:
                print("-" * 54)
        print("="*88)

    def print_configuration(self):
        """Prints a security-redacted version of the current configuration."""
        print("\n" + "="*50 + "\n⚙️ Current Configuration\n" + "="*50)

        # Use deepcopy to ensure we don"t accidentally modify the live config object
        safe_config = copy.deepcopy(self.config)

        # Redact main API keys
        if "main_api_keys" in safe_config:
            safe_config["main_api_keys"] = {k: "********" for k in safe_config["main_api_keys"]}

        # --- Redact sub-account keys as well ---
        if "sub_accounts" in safe_config:
            for account in safe_config["sub_accounts"]:
                if "binance_key" in account:
                    account["binance_key"] = "********"
                if "binance_secret" in account:
                    account["binance_secret"] = "********"

        print(json.dumps(safe_config, indent=2))
        print("="*50)

    def export_to_excel(self, metrics: Dict[str, Any]): self.excel_exporter.export(metrics=metrics, holdings_df=metrics.get("holdings_df"))
    def export_to_html(self, metrics: Dict[str, Any]): self.html_exporter.export(metrics=metrics, holdings_df=metrics.get("holdings_df"))
    def export_csv_backup(self): self.csv_exporter.export(transactions_df=self.db_manager.get_all_transactions(), holdings_df=self.db_manager.get_holdings())
    def cleanup_old_data(self): self.db_manager.cleanup_old_data()

    def test_connections(self):
        """Test connections to Binance and CoinGecko."""
        if self.binance_client:
            try: self.binance_client.ping(); print("✅ Binance Connection: SUCCESS")
            except Exception as e: print(f"❌ Binance Connection: FAILED ({e})")
        else: print("⚠️ Binance Connection: SKIPPED (Failed to Initialize Client/No API keys)")

        test_id = "BTC"
        prices = self._get_current_prices([test_id])
        price = prices.get(test_id)

        if price: print(f"✅ CoinGecko Connection: SUCCESS ({test_id.upper()} price: ${price})")
        else: print("❌ CoinGecko Connection: FAILED")
        print("-" * 30)

    def save_snapshot(self, metrics: Dict[str, Any]):
        """Wrapper to save a portfolio snapshot using data from calculated metrics."""
        if "error" in metrics or "total_value_usd" not in metrics:
            self.logger.warning("Skipping snapshot save due to missing data in metrics.")
            return

        timestamp = metrics.get("timestamp", datetime.datetime.now(datetime.timezone.utc))
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
            unrealized_pl_percent=unrealized_pl_percent
        )
