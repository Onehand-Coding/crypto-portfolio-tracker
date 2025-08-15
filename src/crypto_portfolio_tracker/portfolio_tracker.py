"""
Crypto Portfolio Tracker - Main Class
Handles API interactions, data processing, analysis, and orchestration.
"""

import os
import re
import json
import time
import copy
import uuid
import socket
import inspect
import asyncio
import logging
import datetime
import functools
from enum import Enum
from pathlib import Path
from diskcache import Cache
from collections import deque
from urllib3.util.retry import Retry
from datetime import timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple, Union

import requests
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from binance.client import Client
from requests.adapters import HTTPAdapter
from binance.exceptions import BinanceAPIException
from requests.exceptions import ConnectionError

from . import trading_strategies
from .config import ConfigManager
from .database import DatabaseManager
from .visualizations import Visualizer
from .price_enricher import PriceEnricher
from .binance_fetcher import BinanceFetcher
from .utils import calculate_fifo_cost_basis
from .strategy_backtester import StrategyBacktester
from .crypto_trend_analyzer import CryptoTrendAnalyzer
from .rebalancing_backtester import RebalancingBacktester
from .rebalancing_logic import get_live_rebalance_suggestions
from .profit_taking_logic import ProfitTakingAnalyzer, ProfitOpportunity
from .exporters import ExcelExporter, HtmlExporter, CsvExporter
from .exceptions import NetworkOperationError, NetworkUnavailableError

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TradeResult:
    """Result of a trade."""

    success: bool
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


class ExecutionMode(Enum):
    """Execution modes for rebalancing trades."""

    AUTO = "auto"
    BULK = "bulk"
    INTERACTIVE = "interactive"
    CONFIRM = "confirm"


class CryptoPortfolioTracker:
    """Main class for the crypto portfolio tracker."""

    def __init__(self, config_manager: ConfigManager, force_offline: bool = False):
        """Initialize the tracker and its specialist components."""
        self.config_manager = config_manager
        self.config = self.config_manager.config
        self.logger = logging.getLogger(__name__)

        self.offline_mode = force_offline

        # Initialize database with explicit path
        db_path = self.config_manager.get_database_path()
        backup_dir = self.config_manager.get_backup_dir()
        connection_timeout = self.config.get("database", {}).get(
            "connection_timeout", 30
        )
        cleanup_days = self.config.get("database", {}).get("cleanup_days", 90)

        self.db_manager = DatabaseManager(
            db_path=db_path,
            backup_dir=backup_dir,
            connection_timeout=connection_timeout,
            cleanup_days=cleanup_days,
        )

        self.symbol_mappings = self.config_manager.symbol_mapper

        if not self.offline_mode:
            try:
                self.binance_client = self._init_binance_client()
            except NetworkUnavailableError:
                # Let the main entrypoint handle prompting and re-instantiation
                raise
        else:
            self.binance_client = None

        # --- Initialize Caches ---
        self.cache_dir = Path(self.config.get("cache", {}).get("path", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.coingecko_price_cache = Cache(str(self.cache_dir / "coingecko_historical"))
        self.yfinance_disk_cache = Cache(str(self.cache_dir / "yfinance_historical"))
        self.fiat_exchange_rate_cache = Cache(
            str(self.cache_dir / "fiat_exchange_rates")
        )

        if not self.offline_mode:
            # --- Instantiate our new specialist classes ---
            self.fetcher = BinanceFetcher(
                self.binance_client, self.symbol_mappings, self.config
            )
        else:
            self.fetcher = None

        self.enricher = PriceEnricher(
            self.symbol_mappings, self.config, self.coingecko_price_cache
        )
        self.excel_exporter = ExcelExporter(self.config)
        self.html_exporter = HtmlExporter(self.config)
        self.csv_exporter = CsvExporter(self.config)
        self.visualizer = Visualizer(self.config)

        # --- Initialize strategy state ---
        self.strategy_state_path = self.cache_dir.parent / "strategy_state.json"
        self.strategy_states = self._load_strategy_state()

        self.logger.info("Tracker and all components initialized.")

    def _init_binance_client(
        self, api_key: Optional[str] = None, api_secret: Optional[str] = None
    ) -> Optional[Client]:
        """Initialize and return Binance client with robust session, retries, and time sync."""

        if not api_key and not api_secret:
            keys = self.config_manager.get_binance_keys()
            api_key = keys.get("api_key")
            api_secret = keys.get("api_secret")

        if not api_key or not api_secret:
            self.logger.warning(
                "Binance API key/secret not found. Binance client disabled."
            )
            return None

        try:
            # --- Read all Binance settings from config ---
            binance_config = self.config.get("apis", {}).get("binance", {})
            client_timeout = binance_config.get("timeout", 60)
            recv_window = binance_config.get("recv_window", 20000)

            self.logger.info(
                f"Initializing Binance client with timeout: {client_timeout}s."
            )

            client = Client(
                api_key, api_secret, requests_params={"timeout": client_timeout}
            )

            client.RECV_WINDOW = recv_window
            self.logger.debug(
                f"Set Binance client recvWindow to {client.RECV_WINDOW}ms."
            )

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

            client.ping()  # This will raise if network is down
            self.logger.info("Binance client initialized successfully.")
            return client
        except (ConnectionError, socket.gaierror) as e:
            raise NetworkUnavailableError(
                "Network unavailable, entering offline mode."
            ) from e
        except Exception as e:
            # Other errors (bad API key, etc.) should not trigger offline mode
            raise

    def _sync_binance_client_time(self, client: Client, context: str = "general"):
        """
        Reusable utility to synchronize the given Binance client's clock with the server.
        """
        if not client:
            self.logger.warning(
                f"Cannot sync time: Binance client for '{context}' is not available."
            )
            return
        try:
            self.logger.info(f"Synchronizing time for {context}...")
            server_time = client.get_server_time()["serverTime"]
            local_time = int(time.time() * 1000)
            time_offset = server_time - local_time
            client._server_time_offset = time_offset
            self.logger.info(
                f"Time synchronized for {context}. New offset: {time_offset}ms."
            )
        except Exception as e:
            self.logger.error(
                f"Could not sync time for {context}: {e}. Proceeding with old offset."
            )

    def _get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Get current prices for a list of symbols, with a simple retry for rate limiting.
        """
        if not symbols:
            return {}

        prices: Dict[str, Optional[float]] = {symbol: 0.0 for symbol in symbols}
        coingecko_config = self.config.get("apis", {}).get("coingecko", {})

        ids_to_fetch = {
            self.symbol_mappings.get(s.upper())
            for s in symbols
            if self.symbol_mappings.get(s.upper())
        }
        if not ids_to_fetch:
            return prices

        ids_string = ",".join(list(ids_to_fetch))
        url = f"{coingecko_config.get('base_url')}/simple/price"
        params = {"ids": ids_string, "vs_currencies": "usd"}

        last_exception = None
        for attempt in range(3):  # Try up to 3 times
            try:
                self.logger.debug(
                    f"Fetching current prices for {len(ids_to_fetch)} unique assets..."
                )
                response = requests.get(
                    url, params=params, timeout=coingecko_config.get("timeout", 30)
                )
                response.raise_for_status()
                fetched_price_data = response.json()

                for symbol in symbols:
                    coin_id = self.symbol_mappings.get(symbol.upper())
                    if coin_id in fetched_price_data:
                        prices[symbol] = fetched_price_data[coin_id].get("usd")

                return prices  # Return on success

            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response.status_code == 429 and attempt < 2:
                    self.logger.warning(
                        "Rate limited fetching current prices. Waiting 10 seconds before retrying..."
                    )
                    time.sleep(10)
                    continue  # Go to the next attempt
                else:
                    self.logger.error(
                        f"Fatal error fetching batch prices from CoinGecko: {e}"
                    )
                    break
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.error(
                    f"Fatal error fetching batch prices from CoinGecko: {e}"
                )
                break

        # If we reach here, all retries failed
        raise NetworkOperationError(
            f"Failed to fetch current prices from CoinGecko after retries: {last_exception}"
        )

    def _get_coingecko_historical_price(
        self, coin_id: str, date_str: str
    ) -> Optional[float]:
        """Fetch historical price from CoinGecko for a specific date, using disk cache."""
        if not coin_id or not date_str:
            self.logger.error(
                "Coin ID or date string is missing for CoinGecko historical price fetch."
            )
            return None

        cache_key = f"{coin_id}_{date_str}"
        cached_price = self.coingecko_historical_price_disk_cache.get(cache_key)
        if cached_price is not None:
            self.logger.debug(
                f"Disk Cache HIT for CoinGecko: {coin_id} on {date_str} -> ${cached_price:.6f}"
            )
            return cached_price

        self.logger.debug(
            f"Disk Cache MISS for CoinGecko: {coin_id} on {date_str}. Fetching from API."
        )

        base_url = self.coingecko_api.get("base_url")
        timeout = self.coingecko_api.get("timeout", 30)

        try:
            parsed_date = pd.to_datetime(
                date_str, errors="coerce", dayfirst=True
            ).date()
            today = datetime.datetime.now(timezone.utc).date()

            if parsed_date >= today:
                self.logger.warning(
                    f"Requested date {date_str} is today or in the future. Fetching previous day's data instead."
                )
                parsed_date = today - timedelta(days=1)

            api_date_str = parsed_date.strftime("%d-%m-%Y")

        except (ValueError, TypeError):
            self.logger.error(f"Could not parse date '{date_str}' for CoinGecko API.")
            self.coingecko_historical_price_disk_cache.set(cache_key, None, expire=3600)
            return None

        url = (
            f"{base_url}/coins/{coin_id}/history?date={api_date_str}&localization=false"
        )

        max_retries = 3
        initial_wait_seconds = 60
        server_error_codes_to_retry = [500, 502, 503, 504]
        retries_left = max_retries
        last_exception = None

        while retries_left > 0:
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                price = data.get("market_data", {}).get("current_price", {}).get("usd")
                if price is not None:
                    self.coingecko_historical_price_disk_cache.set(cache_key, price)
                    return float(price)
                else:
                    self.coingecko_historical_price_disk_cache.set(
                        cache_key, None, expire=3600 * 24
                    )
                    return None
            except requests.exceptions.HTTPError as e:
                last_exception = e
                should_retry = False
                if e.response.status_code == 429:
                    should_retry = True
                    self.logger.error(
                        f"Rate limited (429) by CoinGecko for {coin_id} on {api_date_str}."
                    )
                elif e.response.status_code in server_error_codes_to_retry:
                    should_retry = True
                    self.logger.error(
                        f"Server error ({e.response.status_code}) from CoinGecko for {coin_id} on {api_date_str}."
                    )

                if should_retry and retries_left > 1:
                    wait_time = initial_wait_seconds * (max_retries - retries_left + 1)
                    self.logger.info(
                        f"Waiting {wait_time}s before retry... ({retries_left - 1} retries left)"
                    )
                    time.sleep(wait_time)
                    retries_left -= 1
                else:
                    if e.response.status_code == 404:
                        self.logger.warning(
                            f"No historical data found on CoinGecko (404) for {coin_id} on {api_date_str}."
                        )
                    else:
                        self.logger.error(
                            f"Failed to fetch CoinGecko historical price for {coin_id} on {api_date_str}: {e}",
                            exc_info=True,
                        )
                    self.coingecko_historical_price_disk_cache.set(
                        cache_key, None, expire=3600 * 24
                    )
                    return None
            except Exception as e:
                last_exception = e
                self.logger.error(
                    f"Unexpected error fetching CoinGecko historical price for {coin_id}: {e}",
                    exc_info=True,
                )
                self.coingecko_historical_price_disk_cache.set(
                    cache_key, None, expire=3600
                )
                break

        # If we reach here, all retries failed
        raise NetworkOperationError(
            f"Failed to fetch historical price for {coin_id} on {date_str} after retries: {last_exception}"
        )

    def _get_historical_fiat_exchange_rate(
        self, date_str_orig: str, from_currency: str, to_currency: str
    ) -> Optional[float]:
        """
        Fetches the historical exchange rate using yfinance.
        date_str_orig should be in "YYYY-MM-DD" or "DD-MM-YYYY".
        Returns how many "to_currency" 1 unit of "from_currency" is worth.
        """
        if not date_str_orig or not from_currency or not to_currency:
            self.logger.error(
                "Date, from_currency, or to_currency missing for fiat exchange rate lookup."
            )
            return None

        if from_currency.upper() == to_currency.upper():
            return 1.0

        try:
            target_dt = pd.to_datetime(date_str_orig)
            api_date_str = target_dt.strftime("%Y-%m-%d")
        except ValueError:
            self.logger.error(
                f"Invalid date format '{date_str_orig}'. Could not parse for yfinance."
            )
            return None

        cache_key = f"yfinance_{api_date_str}_{from_currency}_{to_currency}"
        if cache_key in self.fiat_exchange_rate_cache:
            self.logger.debug(
                f"Cache HIT for yfinance fiat rate: {from_currency}->{to_currency} on {api_date_str}."
            )
            return self.fiat_exchange_rate_cache[cache_key]

        self.logger.debug(
            f"Cache MISS. Fetching yfinance fiat rate: {from_currency}->{to_currency} on {api_date_str}."
        )

        ticker_symbol = f"{from_currency.upper()}{to_currency.upper()}=X"
        rate = None

        start_date_fetch = (target_dt - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        end_date_fetch = (target_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        try:
            self.logger.debug(
                f"yfinance download: Ticker='{ticker_symbol}', Start='{start_date_fetch}', End='{end_date_fetch}'"
            )
            data_window = yf.download(
                ticker_symbol,
                start=start_date_fetch,
                end=end_date_fetch,
                progress=False,
                auto_adjust=True,
            )

            if data_window.empty:
                self.logger.warning(
                    f"No data returned by yfinance for {ticker_symbol} in window {start_date_fetch} to {end_date_fetch}."
                )
            else:
                data_window = data_window.sort_index()
                closest_price_intermediate = data_window["Close"].asof(target_dt)

                self.logger.debug(
                    f"yfinance .asof() result type: {type(closest_price_intermediate)}, value: {closest_price_intermediate}"
                )

                closest_price_scalar = None
                if isinstance(closest_price_intermediate, pd.Series):
                    if not closest_price_intermediate.empty:
                        closest_price_scalar = closest_price_intermediate.iloc[0]
                    else:
                        closest_price_scalar = pd.NA
                else:
                    closest_price_scalar = closest_price_intermediate

                if pd.notna(closest_price_scalar):
                    actual_price_date_idx = data_window.index.get_indexer(
                        [target_dt], method="ffill"
                    )
                    if actual_price_date_idx[0] != -1:
                        actual_price_date = data_window.index[actual_price_date_idx[0]]
                        rate = float(closest_price_scalar)
                        self.logger.debug(
                            f"Fetched yfinance rate for {from_currency}->{to_currency} (for {api_date_str}, using data from {actual_price_date.strftime('%Y-%m-%d')}): {rate}"
                        )
                    else:
                        self.logger.warning(
                            f"Could not determine actual date for 'asof' price for {ticker_symbol} on {api_date_str}, but got a value. Using asof value directly."
                        )
                        rate = float(closest_price_scalar)
                        self.logger.debug(
                            f"Fetched yfinance rate for {from_currency}->{to_currency} (for {api_date_str}, using .asof() value directly): {rate}"
                        )
                else:
                    self.logger.warning(
                        f"No valid data found for {ticker_symbol} on or before {api_date_str} in the fetched window using .asof() (result was NaN or pd.NA)."
                    )

        except Exception as e:
            self.logger.error(
                f"Error during yfinance download or processing for {ticker_symbol} on {api_date_str}: {e}",
                exc_info=True,
            )
            rate = None

        finally:
            self.fiat_exchange_rate_cache[cache_key] = rate
            yf_delay_ms = (
                self.config.get("apis", {})
                .get("yfinance", {})
                .get("request_delay_ms", 200)
            )
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
            self.logger.debug(
                f"Skipping yfinance ticker for stablecoin: {symbol_upper}"
            )
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

    def _redeem_from_earn_if_needed(
        self, asset: str, required_amount: float, is_live: bool
    ) -> Tuple[bool, List[str]]:
        """
        Checks for sufficient spot balance. If needed, it finds the correct productId
        and redeems from Simple Earn only if in live mode.
        Returns (success, messages) tuple for UI-agnostic operation.
        """
        messages = []
        try:
            spot_balance_info = self.binance_client.get_asset_balance(asset=asset)
            free_spot_balance = float(spot_balance_info.get("free", 0.0))

            if free_spot_balance >= required_amount:
                self.logger.info(
                    f"Sufficient {asset} balance in Spot wallet. No action needed."
                )
                return True, messages

            shortfall = required_amount - free_spot_balance
            self.logger.warning(
                f"Shortfall of {shortfall:.8f} {asset}. Checking Simple Earn."
            )

            positions = self.binance_client.get_simple_earn_flexible_product_position(
                asset=asset
            )
            if not positions or not positions.get("rows"):
                messages.append(
                    f"❌ No active Simple Earn Flexible positions found for {asset} to redeem from."
                )
                return False, messages

            product_to_redeem_from = positions["rows"][0]
            product_id = product_to_redeem_from.get("productId")
            available_to_redeem = float(product_to_redeem_from.get("totalAmount", 0.0))

            if shortfall > available_to_redeem:
                messages.append(
                    f"❌ Cannot redeem required amount. Need {shortfall:.8f} {asset}, but only {available_to_redeem:.8f} is available in Simple Earn."
                )
                return False, messages

            if is_live:
                messages.append(
                    f"⚠️ Insufficient Spot balance. Attempting to redeem {shortfall:.8f} {asset} from product {product_id}."
                )
                self.logger.info(
                    f"LIVE MODE: Attempting to redeem {shortfall:.8f} {asset} from productId {product_id}."
                )
                redemption_result = (
                    self.binance_client.redeem_simple_earn_flexible_product(
                        productId=product_id, amount=f"{shortfall:.8f}"
                    )
                )
                if redemption_result and redemption_result.get("success"):
                    self.logger.info(
                        f"Successfully initiated LIVE redemption. Waiting 5 seconds..."
                    )
                    messages.append(
                        f"✅ Redemption initiated for {shortfall:.8f} {asset}. Waiting 5 seconds..."
                    )
                    time.sleep(5)
                    return True, messages  # Assume success and let the trade proceed
                else:
                    self.logger.error(
                        f"Failed to redeem {asset} from {product_id}: {redemption_result}"
                    )
                    messages.append(f"❌ LIVE redemption FAILED for {asset}. See logs.")
                    return False, messages
            else:
                # In Dry Run mode, we only simulate the check.
                messages.append(
                    f"DRY RUN: Would redeem {shortfall:.8f} {asset} from Simple Earn product {product_id}."
                )
                self.logger.info(
                    f"DRY RUN: Would redeem {shortfall:.8f} {asset} from {product_id}."
                )
                return (
                    True,
                    messages,
                )  # Return True to allow the simulated trade to continue

        except BinanceAPIException as e:
            self.logger.error(
                f"API Error during redemption process for {asset}: {e}", exc_info=True
            )
            messages.append(
                f"❌ An API Error occurred when trying to process redemption for {asset}: {e.message}"
            )
            return False, messages
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during redemption check for {asset}: {e}",
                exc_info=True,
            )
            return False, messages

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
                current_price = self._get_current_prices([symbol]).get(symbol, 0)

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
            if self.config_manager.is_testnet_mode
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

                # Check if redemption from Earn is needed
                redemption_success, redemption_messages = (
                    self._redeem_from_earn_if_needed(
                        asset=symbol,
                        required_amount=adjusted_quantity,
                        is_live=is_live,
                    )
                )
                result.messages.extend(redemption_messages)

                if not redemption_success:
                    result.messages.append(
                        f"⚠️ SKIPPING profit-taking for {symbol} due to redemption check failure."
                    )
                    continue

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

                    # Record trade
                    current_price = self._get_current_prices([symbol]).get(symbol, 0)
                    self._record_trade_transaction(
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
                    current_price = self._get_current_prices([symbol]).get(symbol, 0)
                    self._record_trade_transaction(
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

    def _make_tx_hash(
        self,
        source: str,
        batch_id: str,
        symbol: str,
        side: str,
        ts: datetime.datetime,
        order: Optional[dict] = None,
    ) -> str:
        if order and isinstance(order, dict) and order.get("orderId"):
            return f"binance:{order['orderId']}"
        return f"{source}:{batch_id}:{symbol}:{side}:{int(ts.timestamp() * 1000)}"

    def _record_trade_transaction(
        self,
        *,
        symbol: str,
        side: str,  # "BUY" | "SELL"
        quantity: float,
        price_usd: float,
        source: str,  # "REBALANCE" | "DCA"
        mode: str,  # "LIVE" | "TESTNET" | "SIM"
        batch_id: str,
        order: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
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
                # keep only lightweight identifiers
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
                "type": side,  # must be BUY/SELL per schema
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
        except Exception as e:
            self.logger.error(f"Failed to record trade transaction: {e}", exc_info=True)

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
            if self.config_manager.is_testnet_mode
            else ("LIVE" if is_live else "SIM")
        )

        simulated_balances = {}
        all_trade_symbols = set(trades_to_execute["Symbol"].unique()) | {"USDT"}
        for symbol in all_trade_symbols:
            try:
                spot_bal = float(
                    self.binance_client.get_asset_balance(asset=symbol).get("free", 0.0)
                )
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

                    current_price = self._get_current_prices([symbol]).get(symbol, 0)
                    final_notional_value = adjusted_quantity * current_price
                    if final_notional_value < min_trade_usd:
                        result.messages.append(
                            f"⚠️ SKIPPING SELL for {symbol}: Final trade value (~${final_notional_value:,.2f}) is below minimum of ${min_trade_usd:,.2f} after applying exchange rules."
                        )
                        continue

                    # Use the updated _redeem_from_earn_if_needed method
                    redemption_success, redemption_messages = (
                        self._redeem_from_earn_if_needed(
                            asset=symbol,
                            required_amount=adjusted_quantity,
                            is_live=is_live,
                        )
                    )
                    result.messages.extend(redemption_messages)

                    if not redemption_success:
                        result.messages.append(
                            f"⚠️ SKIPPING SELL for {symbol} due to redemption check failure."
                        )
                        continue

                    result.messages.append(
                        f"\nPreparing MARKET SELL for {adjusted_quantity:.8f} {symbol}..."
                    )
                    if is_live:
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
                            # record trade
                            self._record_trade_transaction(
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
                        # record simulated trade
                        self._record_trade_transaction(
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
                    if is_live:
                        try:
                            order = self.binance_client.order_market_buy(
                                symbol=trade_ticker, quoteOrderQty=f"{usd_value:.2f}"
                            )
                            result.messages.append(f"✅ LIVE BUY ORDER PLACED: {order}")
                            self.logger.info(f"LIVE BUY ORDER PLACED: {order}")
                            trades_executed_count += 1
                            simulated_balances["USDT"] -= usd_value
                            # record trade (approx quantity using current price)
                            price_now = (
                                self._get_current_prices([symbol]).get(symbol, 0.0)
                                or 0.0
                            )
                            qty = (usd_value / price_now) if price_now > 0 else 0.0
                            self._record_trade_transaction(
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
                        price_now = (
                            self._get_current_prices([symbol]).get(symbol, 0.0) or 0.0
                        )
                        qty = (usd_value / price_now) if price_now > 0 else 0.0
                        self._record_trade_transaction(
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
                    prices = self._get_current_prices([symbol])
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
                if is_live:
                    order = self.binance_client.order_market_buy(
                        symbol=trade_ticker, quoteOrderQty=f"{usdt_to_spend:.2f}"
                    )
                    result.messages.append(f"LIVE BUY ORDER PLACED: {order}")
                    result.data["order"] = order
                else:
                    result.messages.append(f"(Dry Run) BUY Trade was not placed.")
                result.success = True
            elif trade_type == "SELL":
                coin_quantity_to_sell = amount if not is_quote_qty else 0
                if is_quote_qty:
                    prices = self._get_current_prices([symbol])
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
                if is_live:
                    order = self.binance_client.order_market_sell(
                        symbol=trade_ticker, quantity=f"{adjusted_quantity:.8f}"
                    )
                    result.messages.append(f"LIVE SELL ORDER PLACED: {order}")
                    result.data["order"] = order
                else:
                    result.messages.append(f"(Dry Run) SELL Trade was not placed.")
                result.success = True
            result.messages.append(
                "Recommendation: Run 'Full Sync & Analysis' to update your portfolio with this trade."
            )
        except Exception as e:
            result.errors.append(f"Unexpected Error for {symbol}: {e}")
        return result

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
        live_balances_df = self.fetch_binance_balances()
        if live_balances_df.empty:
            self.logger.error("Could not fetch live balances. Cannot rebalance.")
            return None

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
        if not self.config_manager.is_testnet_mode:
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
        if not self.config_manager.is_testnet_mode:
            futures_balances = self.fetcher.fetch_futures_balance()
        for item in futures_balances:
            if item.get("asset") == "USDT":
                futures_value += float(item.get("balance", 0.0))

        # 3. Calculate Funding Wallet Value
        funding_value = 0
        funding_balances_raw = []
        if not self.config_manager.is_testnet_mode:
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

        total_cost_basis = holdings_df["cost_basis_total"].sum()
        total_pl_usd = spot_earn_value - total_cost_basis
        total_pl_percent = (
            (total_pl_usd / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
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

    async def sync_data(self):
        """
        Orchestrates the Gather, Enrich, and Process pipeline, now intelligently
        skipping unsupported data sources when in Testnet mode.
        """
        self.logger.info("Starting data synchronization pipeline...")
        lookback_config = self.config.get("history_lookback_days", {})

        # Define all possible data sources for production mode
        all_data_sources = {
            "Binance Trade": {
                "fetcher": self.fetcher.fetch_binance_transactions,
                "days": lookback_config.get("trades", 90),
            },
            "Binance Deposit": {
                "fetcher": self.fetcher.fetch_deposit_history,
                "days": lookback_config.get("deposits", 90),
            },
            "Binance Withdrawal": {
                "fetcher": self.fetcher.fetch_withdrawal_history,
                "days": lookback_config.get("withdrawals", 90),
            },
            "Binance P2P Buy": {
                "fetcher": self.fetcher.fetch_p2p_usdt_buys,
                "days": lookback_config.get("p2p_buys", 90),
            },
            "Binance Convert": {
                "fetcher": self.fetcher.fetch_spot_convert_history,
                "days": lookback_config.get("spot_convert_history", 90),
            },
            "Binance Dividend": {
                "fetcher": self.fetcher.fetch_dividend_history,
                "days": lookback_config.get("dividend_history", 90),
            },
            "Binance Simple Earn Reward": {
                "fetcher": self.fetcher.fetch_simple_earn_rewards,
                "days": lookback_config.get("simple_earn_rewards", 90),
            },
            "Binance Simple Earn Subscription": {
                "fetcher": self.fetcher.fetch_simple_earn_subscriptions,
                "days": lookback_config.get("simple_earn_subscriptions", 90),
            },
            "Binance Simple Earn Redemption": {
                "fetcher": self.fetcher.fetch_simple_earn_redemptions,
                "days": lookback_config.get("simple_earn_redemptions", 90),
            },
        }

        data_sources_to_fetch = {}
        fetch_staking = False

        if self.config_manager.is_testnet_mode:
            # In testnet, only fetch spot trade history
            self.logger.warning("TESTNET MODE: Syncing SPOT TRADE history only.")
            data_sources_to_fetch = {"Binance Trade": all_data_sources["Binance Trade"]}
            fetch_staking = False
        else:
            # In production, fetch everything
            data_sources_to_fetch = all_data_sources
            fetch_staking = True

        fetcher_tasks = []
        for source, config in data_sources_to_fetch.items():
            latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source)
            if latest_known_ts:
                self.logger.info(
                    f"History found for '{source}'. Initiating selective sync from {latest_known_ts.strftime('%Y-%m-%d %H:%M')}."
                )
            else:
                self.logger.info(
                    f"No history found for '{source}'. Initiating full sync for the last {config['days']} days."
                )

            fetcher_tasks.append(
                asyncio.to_thread(
                    config["fetcher"],
                    source_name=source,
                    days_back=config["days"],
                    latest_known_ts=latest_known_ts,
                )
            )

        # Only check for staking history if not in testnet mode
        if fetch_staking:
            staking_ts_map = {
                "Binance Staking Subscription": self.db_manager.get_latest_timestamp_for_source(
                    "Binance Staking Subscription"
                ),
                "Binance Staking Redemption": self.db_manager.get_latest_timestamp_for_source(
                    "Binance Staking Redemption"
                ),
                "Binance Staking Interest": self.db_manager.get_latest_timestamp_for_source(
                    "Binance Staking Interest"
                ),
            }
            self.logger.info(
                "Checking for Staking history (Subscriptions, Redemptions, Interest)..."
            )
            fetcher_tasks.append(
                asyncio.to_thread(
                    self.fetcher.fetch_staking_history,
                    source_name="Binance Staking",
                    days_back=lookback_config.get("staking_history", 90),
                    latest_known_ts_map=staking_ts_map,
                )
            )

        if not fetcher_tasks:
            self.logger.info("No data sources to sync in the current mode.")
            return

        # 1. GATHER
        self.logger.info(
            f"Launching {len(fetcher_tasks)} data fetching tasks concurrently..."
        )
        results = await asyncio.gather(*fetcher_tasks, return_exceptions=True)

        all_raw_transactions = []
        for res in results:
            if isinstance(res, list):
                all_raw_transactions.extend(res)
            elif isinstance(res, Exception):
                self.logger.error(
                    f"An error occurred during a fetching task: {res}", exc_info=False
                )

        if not all_raw_transactions:
            self.logger.info("No new raw transactions were fetched. Sync complete.")
            return

        # 2. ENRICH
        self.logger.info(
            f"Passing {len(all_raw_transactions)} raw transactions to the PriceEnricher."
        )
        enriched_transactions = await self.enricher.enrich_transactions(
            all_raw_transactions
        )

        # 3. PROCESS
        if enriched_transactions:
            enriched_transactions.sort(key=lambda x: x["timestamp"])
            self.logger.info(
                f"Saving {len(enriched_transactions)} enriched transactions to the database..."
            )
            num_inserted = self.db_manager.bulk_insert_transactions(
                enriched_transactions
            )
            self.logger.info(
                f"Database update complete. {num_inserted} new transactions were saved."
            )
        else:
            self.logger.warning(
                "Enrichment process returned no transactions. Nothing to save."
            )

        self.update_holdings_from_transactions()
        self.logger.info("Data synchronization pipeline finished successfully.")

    async def run_full_sync(self) -> Dict[str, Any]:
        """Runs the full async data sync and then calculates metrics."""
        await self.sync_data()
        return await self.calculate_portfolio_metrics()

    def fetch_binance_balances(self) -> pd.DataFrame:
        """Fetch current balances from Binance Spot wallet with retry, explicit normalization, and LD-prefix consolidation."""
        if not self.binance_client:
            self.logger.warning(
                "Binance client not initialized. Cannot fetch Spot balances."
            )
            return pd.DataFrame(columns=["symbol", "quantity"])

        # --- Close stale connections before making a new request ---
        self.binance_client.session.close()

        retries = 3
        wait_time_seconds = 15
        api_timeout = self.config.get("apis", {}).get("binance", {}).get("timeout", 180)

        while retries > 0:
            try:
                self.logger.debug(
                    f"Attempting to fetch Binance account info (Timeout: {api_timeout}s)..."
                )
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
                        final_symbol = self.symbol_mappings.normalize_symbol(
                            asset_symbol_api
                        )

                        # Only append if the final symbol is valid (not empty)
                        if final_symbol:
                            processed_balances.append(
                                {"symbol": final_symbol, "quantity": quantity}
                            )

                if not processed_balances:
                    self.logger.info(
                        "Found raw balances, but all were zero or negligible after processing."
                    )
                    return pd.DataFrame(columns=["symbol", "quantity"])

                df = pd.DataFrame(processed_balances)
                df = df.groupby("symbol", as_index=False)["quantity"].sum()

                self.logger.debug(
                    f"DEBUG: Balances from SPOT after all processing in fetch_binance_balances: \n{df.to_string() if not df.empty else 'EMPTY DF'}"
                )
                self.logger.info(
                    f"Fetched and consolidated {len(df)} non-zero balances from Binance Spot."
                )
                return df

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
            except BinanceAPIException as e:
                self.logger.error(
                    f"Binance API Error fetching Spot balances: {e}. No retries for API errors."
                )
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error fetching Spot balances: {e}", exc_info=True
                )
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
            cost_basis_tx_df = group_df_copy[
                ~group_df_copy["source"].str.contains(
                    "Simple Earn|Asset Transfer|Staking", case=False, na=False
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

            # 3. If there"s a valid average cost, save it.
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

    def create_portfolio_charts(self, chart_type: str, metrics: Dict[str, Any]) -> bool:
        """
        Create a specific chart based on the chart type using unified visualizer.

        Args:
            chart_type: Type of chart to create ('allocation_pie', 'allocation_comparison', 'pl_by_asset', 'value_history')
            metrics: Portfolio metrics dictionary

        Returns:
            bool: True if chart was created successfully, False otherwise
        """
        holdings_df = metrics.get("holdings_df")
        snapshots_df = self.db_manager.get_all_snapshots()
        target_alloc = self.config.get("target_allocation", {})

        if holdings_df is None:
            self.logger.warning("No holdings data for chart generation.")
            return False

        try:
            data = {
                "holdings_df": holdings_df,
                "snapshots_df": snapshots_df,
                "target_allocation": target_alloc,
            }

            if chart_type == "allocation_pie":
                self.visualizer.create_portfolio_allocation_pie(
                    holdings_df, metrics, save_to_disk=True
                )
                return True
            elif chart_type == "allocation_comparison":
                self.visualizer.create_allocation_comparison_bar(
                    holdings_df, target_alloc, save_to_disk=True
                )
                return True
            elif chart_type == "pl_by_asset":
                self.visualizer.create_pl_by_asset_bar(holdings_df, save_to_disk=True)
                return True
            elif chart_type == "value_history":
                self.visualizer.create_portfolio_value_history(
                    snapshots_df, save_to_disk=True
                )
                return True
            else:
                self.logger.error(f"Unknown chart type: {chart_type}")
                return False
        except Exception as e:
            self.logger.error(f"Error creating chart {chart_type}: {e}")
            return False

    def create_portfolio_charts_all(self, metrics: Dict[str, Any]):
        """Generate portfolio charts."""
        holdings_df = metrics.get("holdings_df")
        snapshots_df = self.db_manager.get_all_snapshots()
        target_alloc = self.config.get("target_allocation", {})
        if holdings_df is not None:
            self.visualizer.generate_all_charts(
                holdings_df, metrics, target_alloc, snapshots_df
            )
        else:
            self.logger.warning("No holdings data for chart generation.")

    def export_portfolio_summary(self, metrics: Dict[str, Any], format: str):
        """Export portfolio summary to HTML/Excel format."""
        if format.lower() == "html":
            self.html_exporter.export(
                metrics=metrics, holdings_df=metrics.get("holdings_df")
            )
        elif format.lower() == "excel":
            self.excel_exporter.export(
                metrics=metrics, holdings_df=metrics.get("holdings_df")
            )
        elif format.lower() == "csv":
            self.csv_exporter.export(
                metrics=metrics, holdings_df=metrics.get("holdings_df")
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def export_portfolio_summary_all_formats(self, metrics: Dict[str, Any]):
        """Export portfolio summary to all available formats."""
        for format in ["html", "excel"]:
            self.export_portfolio_summary(metrics, format)

    def export_data_backup(self, data_type: str, format: str):
        """Export transactions/holdings to CSV backup file with robust timestamp handling."""
        if data_type == "transactions" and format.lower() == "csv":
            return self.csv_exporter.export(
                transactions_df=self.db_manager.get_all_transactions(),
                data_type="transactions",
            )
        elif data_type == "holdings" and format.lower() == "csv":
            return self.csv_exporter.export(
                holdings_df=self.db_manager.get_holdings(), data_type="holdings"
            )
        elif data_type == "transactions" and format.lower() == "excel":
            return self.excel_exporter.export(
                transactions_df=self.db_manager.get_all_transactions(),
                data_type="transactions",
            )
        elif data_type == "holdings" and format.lower() == "excel":
            return self.excel_exporter.export(
                holdings_df=self.db_manager.get_holdings(), data_type="holdings"
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def export_all_data_backups(self):
        """Export all data backups to all available formats with robust timestamp handling."""
        results = {}
        for data_type in ["transactions", "holdings"]:
            results[data_type] = self.export_data_backup(data_type, "csv")
            results[data_type] = self.export_data_backup(data_type, "excel")
        return results

    def export_trend_report(
        self, report: Dict[str, Any], timeframe: str, export_format: str = "HTML"
    ) -> Optional[Path]:
        """
        Exports a trend analysis report to various formats.

        Args:
            report: The trend analysis report dictionary from CryptoTrendAnalyzer
            timeframe: The timeframe of the analysis (e.g., 'long_term', 'swing', 'day')
            export_format: The export format ('CSV', 'JSON', 'HTML')

        Returns:
            Path to the exported file, or None if export failed
        """
        try:
            # Get export directory from config
            export_dir = Path(
                self.config.get("exports", {}).get("path", "data/exports/")
            )
            export_dir.mkdir(parents=True, exist_ok=True)

            # Create DataFrame from coin analyses for export
            coin_analyses = report.get("coin_analyses", {})
            df_export = pd.DataFrame(
                [
                    {
                        "Symbol": symbol,
                        "Price": analysis.get("current_price", 0),
                        "Change (%)": analysis.get("price_change_pct", 0),
                        "RSI": analysis.get("rsi", 0),
                        "Support": analysis.get("support_level", 0),
                        "Resistance": analysis.get("resistance_level", 0),
                        "Active Conditions": ", ".join(
                            analysis.get("active_conditions", [])
                        ),
                    }
                    for symbol, analysis in coin_analyses.items()
                ]
            )

            # Generate timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_report_{timeframe}_{timestamp}.{export_format.lower()}"
            exported_file = export_dir / filename

            if export_format.upper() == "CSV":
                df_export.to_csv(exported_file, index=False)
                self.logger.info(f"Trend report exported to CSV: {exported_file}")

            elif export_format.upper() == "JSON":
                with open(exported_file, "w") as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Trend report exported to JSON: {exported_file}")

            elif export_format.upper() == "HTML":
                exported_file = self.html_exporter.export_trend_report(
                    report, df_export
                )
                if exported_file:
                    self.logger.info(f"Trend report exported to HTML: {exported_file}")
                else:
                    self.logger.error("Failed to export trend report to HTML")
                    return None
            else:
                self.logger.error(f"Unsupported export format: {export_format}")
                return None

            return exported_file

        except Exception as e:
            self.logger.error(f"Error exporting trend report: {e}", exc_info=True)
            return None

    def export_trend_report_all_formats(
        self, report: Dict[str, Any], timeframe: str
    ) -> Dict[str, Optional[Path]]:
        """
        Exports a trend analysis report to all available formats.

        Args:
            report: The trend analysis report dictionary
            timeframe: The timeframe of the analysis

        Returns:
            Dictionary mapping format to exported file path
        """
        results = {}
        for format_type in ["CSV", "JSON", "HTML"]:
            try:
                file_path = self.export_trend_report(report, timeframe, format_type)
                results[format_type] = file_path
            except Exception as e:
                self.logger.error(f"Failed to export {format_type}: {e}")
                results[format_type] = None

        return results

    def cleanup_old_data(self):
        """Clean up old portfolio data."""
        self.db_manager.cleanup_old_data()

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
            spot_balance = float(
                self.binance_client.get_asset_balance(asset="USDT").get("free", 0.0)
            )
            spot_earn_balance += spot_balance

            # Add Earn balance if not in testnet
            if not self.config_manager.is_testnet_mode:
                earn_positions = self.fetcher.fetch_simple_earn_balances(
                    pd.DataFrame([{"symbol": "USDT"}])
                )
                earn_balance = earn_positions.get("USDT", 0.0)
                spot_earn_balance += earn_balance

            # Get Funding balance (mainnet only)
            if not self.config_manager.is_testnet_mode:
                funding_balances = self.fetcher.fetch_funding_balance()
                for balance in funding_balances:
                    if balance.get("asset") == "USDT":
                        funding_balance = float(balance.get("free", 0.0))
                        break

        except Exception as e:
            self.logger.error(f"Error fetching USDT balances: {e}")

        total_balance = spot_earn_balance + funding_balance

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
        metrics = await self.calculate_portfolio_metrics()

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
            if self.config_manager.is_testnet_mode
            else ("LIVE" if is_live else "SIM")
        )

        if not selected_trades:
            result.messages.append("No trades selected for execution")
            result.success = False  # Set to False if no trades
            return result

        self.logger.info(
            f"Executing {len(selected_trades)} DCA trades using {method} method"
        )

        for trade in selected_trades:
            asset = trade["asset"]
            amount = trade["amount"]

            # Determine trade type based on amount
            trade_type = "BUY" if amount > 0 else "SELL"
            trade_amount = abs(amount)

            self.logger.info(
                f"Executing DCA {trade_type} for {asset}: ${trade_amount:,.2f}"
            )

            # Use existing trade execution method
            trade_result = await self.execute_manual_trade_core(
                trade_type=trade_type,
                symbol=asset,
                trade_ticker=f"{asset}USDT",
                amount=trade_amount,
                is_quote_qty=True,
                is_live=is_live,
            )

            # Aggregate results
            result.success &= trade_result.success
            result.messages.extend(trade_result.messages)
            result.errors.extend(trade_result.errors)
            # Record trade using actual order data
            order_data = trade_result.data.get("order", {})
            if order_data and "price" in order_data:
                price_usd = float(order_data.get("price", 0))
                qty = float(order_data.get("executedQty", trade_amount))
            else:
                price_usd = 0.0
                qty = trade_amount
            self._record_trade_transaction(
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

        result.messages.append("\n" + "=" * 80)
        result.messages.append(
            f"✅ {len(selected_trades)} DCA trade(s) executed successfully!"
        )
        result.data["batch_id"] = batch_id
        result.success = len(selected_trades) > 0
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

    def save_snapshot(self, metrics: Dict[str, Any]):
        """Wrapper to save a portfolio snapshot using data from calculated metrics."""
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

    async def transfer_funding_to_spot(
        self, asset: str, amount: float, is_live: bool = False
    ) -> TradeResult:
        """
        Transfer funds from funding wallet to spot wallet.

        Args:
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)

        Returns:
            TradeResult with success status, messages, and any errors
        """
        result = TradeResult(success=False)

        try:
            # Validate inputs
            if not asset or amount <= 0:
                result.errors.append("Invalid asset or amount")
                return result

            # Check if we have a Binance client
            if not self.binance_client:
                result.errors.append("Binance client not initialized")
                return result

            # Sync time with Binance server
            self._sync_binance_client_time(self.binance_client, context="transfer")

            # Execute the transfer
            transfer_result = self.fetcher.transfer_funding_to_spot(
                asset=asset, amount=amount, is_live=is_live
            )

            # Process the result
            if transfer_result["success"]:
                result.success = True
                result.messages.extend(transfer_result["messages"])
                if transfer_result.get("transfer_id"):
                    result.data["transfer_id"] = transfer_result["transfer_id"]
            else:
                result.errors.extend(transfer_result["errors"])

        except Exception as e:
            error_msg = f"Unexpected error during transfer: {e}"
            self.logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)

        return result
