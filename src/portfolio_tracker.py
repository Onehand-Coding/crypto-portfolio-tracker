"""
Crypto Portfolio Tracker - Main Class
Handles API interactions, data processing, analysis, and orchestration.
"""
import re
import os
import json
import time
import asyncio
import logging
import datetime
import functools
import diskcache
from pathlib import Path
from diskcache import Cache
from collections import deque
from urllib3.util.retry import Retry
from typing import Dict, Any, Optional, List
from datetime import timezone, timedelta

import requests
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from binance.client import Client
from requests.adapters import HTTPAdapter
from binance.exceptions import BinanceAPIException, BinanceRequestException

from config import ConfigManager
from backtester import Backtester
from database import DatabaseManager
from visualizations import Visualizer
from crypto_trend_analyzer import CryptoTrendAnalyzer
from rebalancing_backtester import RebalancingBacktester
from exporters import ExcelExporter, HtmlExporter, CsvExporter

logger = logging.getLogger(__name__)


def calculate_fifo_cost_basis(transactions_df: pd.DataFrame):
    """
    Calculates current quantity and average cost basis using FIFO.
    Assumes transactions_df is for a single asset, sorted by timestamp.
    Incorporates fee_usd into the cost of BUY/DEPOSIT transactions.
    """
    buy_lots = deque()  # Queue to store {'qty': quantity, 'price': effective_price_usd_per_unit}
    current_quantity = 0.0
    total_cost_basis = 0.0 # This will track the cost basis of *remaining* lots

    if transactions_df.empty:
        symbol = transactions_df['symbol'].iloc[0] if not transactions_df.empty and 'symbol' in transactions_df.columns else 'Unknown'
        logger.debug(f"No transactions provided to calculate_fifo_cost_basis for symbol {symbol}. Returning 0, 0.")
        return 0.0, 0.0

    # Ensure correct dtypes and sorting
    transactions_df = transactions_df.copy()
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'], errors='coerce')
    transactions_df = transactions_df.sort_values(by='timestamp').reset_index(drop=True)

    if transactions_df['timestamp'].isna().any():
        logger.warning(f"Found NaT timestamps for symbol {transactions_df['symbol'].iloc[0] if not transactions_df.empty else 'Unknown'} after coercion. Dropping these rows for FIFO.")
        transactions_df.dropna(subset=['timestamp'], inplace=True)
        if transactions_df.empty:
            logger.warning(f"All transactions dropped for symbol due to NaT timestamps. Returning 0, 0.")
            return 0.0,0.0

    for _, row in transactions_df.iterrows():
        tx_type = row.get('type')

        try:
            quantity = float(row.get('quantity', 0.0))
        except (ValueError, TypeError):
            logger.warning(f"Invalid quantity '{row.get('quantity')}' for tx {row.get('transaction_hash')}. Using 0.0.")
            quantity = 0.0

        if quantity == 0.0 and tx_type in ['BUY', 'SELL', 'DEPOSIT', 'WITHDRAWAL']:
            logger.debug(f"Skipping zero quantity {tx_type} transaction: {row.get('transaction_hash')}")
            continue

        price = row.get('price_usd')

        fee_usd = row.get('fee_usd', 0.0)
        if pd.isna(fee_usd) or not isinstance(fee_usd, (int, float)):
            logger.debug(f"Invalid or missing fee_usd ('{row.get('fee_usd')}') for tx {row.get('transaction_hash')}. Using 0.0 for fee_usd.")
            fee_usd = 0.0
        else:
            try:
                fee_usd = float(fee_usd)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert fee_usd '{row.get('fee_usd')}' to float for tx {row.get('transaction_hash')}. Using 0.0 for fee_usd.")
                fee_usd = 0.0

        if tx_type == 'BUY' or tx_type == 'DEPOSIT':
            actual_tx_price_usd = 0.0
            if price is not None and isinstance(price, (int, float)) and price > 0:
                actual_tx_price_usd = float(price)
            elif price == 0.0 and tx_type == 'DEPOSIT':
                actual_tx_price_usd = 0.0
            elif price is None or not isinstance(price, (int, float)) or price <=0 :
                logger.warning(f"{tx_type} tx for {row.get('symbol', 'N/A')} (ID: {row.get('transaction_hash', 'N/A')}) "
                               f"has invalid/missing price ({price}). Using $0 price component for this lot.")
                actual_tx_price_usd = 0.0

            # <<< FEE INCORPORATION FOR BUYS/DEPOSITS >>>
            cost_for_this_transaction_lot = (quantity * actual_tx_price_usd) + fee_usd
            effective_price_per_unit = cost_for_this_transaction_lot / quantity if quantity > 0 else 0.0 # Price per unit *including* fee

            buy_lots.append({'qty': quantity, 'price': effective_price_per_unit})
            current_quantity += quantity
            total_cost_basis += cost_for_this_transaction_lot # Add full cost (value + fee)

        elif tx_type == 'SELL' or tx_type == 'WITHDRAWAL':
            sell_qty = quantity
            current_quantity -= sell_qty

            while sell_qty > 0 and buy_lots:
                oldest_buy = buy_lots[0]
                # oldest_buy['price'] is the effective_price_per_unit from its acquisition (already includes its buy fee)

                if oldest_buy['qty'] <= sell_qty:
                    qty_removed_from_lot = oldest_buy['qty']
                    cost_basis_removed = qty_removed_from_lot * oldest_buy['price']
                    total_cost_basis -= cost_basis_removed
                    sell_qty -= qty_removed_from_lot
                    buy_lots.popleft()
                else:
                    cost_basis_removed = sell_qty * oldest_buy['price']
                    total_cost_basis -= cost_basis_removed
                    oldest_buy['qty'] -= sell_qty
                    sell_qty = 0

            if sell_qty > 0:
                logger.warning(f"{tx_type} of {sell_qty} {row.get('symbol', 'N/A')} more than available from history. Cost basis may be affected.")

        total_cost_basis = max(0, total_cost_basis)

    average_cost_basis = total_cost_basis / current_quantity if current_quantity > 0 else 0.0
    current_quantity = max(0, current_quantity)

    logger.debug(f"FIFO Result for {transactions_df['symbol'].iloc[0] if not transactions_df.empty else 'Unknown'}: Qty={current_quantity:.8f}, AvgCost={average_cost_basis:.8f}, TotalCostBasisForAsset={total_cost_basis:.8f}")
    return current_quantity, average_cost_basis


class CryptoPortfolioTracker:
    """Main class for the crypto portfolio tracker."""

    def __init__(self, config_data: Dict[str, Any]):
        """Initialize the tracker with a pre-loaded configuration dictionary."""
        self.config = config_data
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(self.config)
        self.excel_exporter = ExcelExporter(self.config)
        self.html_exporter = HtmlExporter(self.config)
        self.csv_exporter = CsvExporter(self.config)
        self.visualizer = Visualizer(self.config)
        self.binance_client = self._init_binance_client()
        self.coingecko_api = self.config.get("apis", {}).get("coingecko", {})
        self.binance_api_config = self.config.get("apis", {}).get("binance", {})
        self.frankfurter_api_config = self.config.get("apis", {}).get("frankfurter_api", {})
        self.symbol_mappings = self.config.get("symbol_mappings", {}).get("coingecko_ids", {})
        self.norm_map = self.config.get("symbol_normalization_map", {})
        self.stablecoin_symbols = [s.upper() for s in self.config.get("portfolio", {}).get("stablecoin_symbols", ["USDT", "USDC", "BUSD", "DAI"])]
        self.fiat_exchange_rate_cache: Dict[str, Optional[float]] = {}
        self.cache_dir = Path(self.config.get("cache", {}).get("path", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.coingecko_historical_price_disk_cache = Cache(str(self.cache_dir / "coingecko_historical"))
        logger.info(f"Disk cache for CoinGecko historical prices initialized at: {self.cache_dir / 'coingecko_historical'}")
        self.yfinance_disk_cache = Cache(str(self.cache_dir / "yfinance_ohlcv"))
        logger.info(f"Disk cache for yfinance historical data initialized at: {self.cache_dir / 'yfinance_ohlcv'}")
        self.yfinance_config = self.config.get("apis", {}).get("yfinance", {})
        target_coins_from_config = list(self.config.get("target_allocation", {}).keys())
        self.target_assets_for_sync = set(s.upper() for s in target_coins_from_config)
        self.target_assets_for_sync.add("USDT")
        logger.info(f"Tracker initialized. Will focus sync on target assets: {list(self.target_assets_for_sync)}")

    def _init_binance_client(self) -> Optional[Client]:
        """Initialize and return Binance client with robust session and retries."""
        api_keys = self.config.get("api_keys", {})
        api_key = api_keys.get("binance_key")
        api_secret = api_keys.get("binance_secret")

        if not api_key or not api_secret:
            self.logger.warning("Binance API key/secret not found. Binance disabled.")
            return None

        try:
            client_timeout = self.config.get("apis", {}).get("binance", {}).get("timeout", 180)
            self.logger.info(f"Initializing Binance client with timeout: {client_timeout}s and retries.")

            # Initialize the client, which creates its own session with default headers.
            client = Client(api_key, api_secret, requests_params={'timeout': client_timeout})

            # Create the retry strategy.
            retry_strategy = Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            # Create the adapter with the retry strategy.
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount the adapter to the client's EXISTING session.
            # This adds our retry logic without overwriting the client's default headers.
            client.session.mount("https://", adapter)
            client.session.mount("http://", adapter)

            if self.config.get("apis", {}).get("binance", {}).get("testnet", False):
                client.API_URL = 'https://testnet.binance.vision/api'

            # Ping to confirm connection works with the modified session.
            client.ping()
            self.logger.info("Binance client initialized with robust session.")
            return client
        except Exception as e:
            self.logger.error(f"Failed to init Binance client: {e}", exc_info=True)
            return None

    def _get_coingecko_price(self, coin_id: str) -> Optional[float]:
        """Fetch current price from CoinGecko."""
        base_url = self.coingecko_api.get("base_url"); timeout = self.coingecko_api.get("timeout", 30)
        url = f"{base_url}/simple/price?ids={coin_id}&vs_currencies=usd"
        try:
            response = requests.get(url, timeout=timeout); response.raise_for_status(); data = response.json()
            return data.get(coin_id, {}).get("usd")
        except requests.exceptions.RequestException as e: logger.error(f"CoinGecko price fetch error for {coin_id}: {e}"); return None

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
            parsed_date = pd.to_datetime(date_str, errors='coerce', dayfirst=True).date()
            today = datetime.datetime.now(timezone.utc).date()

            # --- FIX: Prevent requests for today or future dates ---
            if parsed_date >= today:
                self.logger.warning(f"Requested date {date_str} is today or in the future. Fetching previous day's data instead.")
                parsed_date = today - timedelta(days=1)

            api_date_str = parsed_date.strftime('%d-%m-%Y')

        except (ValueError, TypeError):
            self.logger.error(f"Could not parse date '{date_str}' for CoinGecko API.")
            self.coingecko_historical_price_disk_cache.set(cache_key, None, expire=3600)
            return None

        url = f"{base_url}/coins/{coin_id}/history?date={api_date_str}&localization=false"

        # Retry logic remains the same
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
        date_str_orig should be in 'YYYY-MM-DD' or 'DD-MM-YYYY'.
        Returns how many 'to_currency' 1 unit of 'from_currency' is worth.
        """
        if not date_str_orig or not from_currency or not to_currency:
            logger.error("Date, from_currency, or to_currency missing for fiat exchange rate lookup.")
            return None

        if from_currency.upper() == to_currency.upper():
            return 1.0

        try:
            target_dt = pd.to_datetime(date_str_orig)
            api_date_str = target_dt.strftime('%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format '{date_str_orig}'. Could not parse for yfinance.")
            return None

        cache_key = f"yfinance_{api_date_str}_{from_currency}_{to_currency}"
        if cache_key in self.fiat_exchange_rate_cache:
            logger.debug(f"Cache HIT for yfinance fiat rate: {from_currency}->{to_currency} on {api_date_str}.")
            return self.fiat_exchange_rate_cache[cache_key]

        logger.debug(f"Cache MISS. Fetching yfinance fiat rate: {from_currency}->{to_currency} on {api_date_str}.")

        ticker_symbol = f"{from_currency.upper()}{to_currency.upper()}=X"
        rate = None

        start_date_fetch = (target_dt - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        end_date_fetch = (target_dt + pd.Timedelta(days=2)).strftime('%Y-%m-%d')

        try:
            logger.debug(f"yfinance download: Ticker='{ticker_symbol}', Start='{start_date_fetch}', End='{end_date_fetch}'")
            data_window = yf.download(ticker_symbol, start=start_date_fetch, end=end_date_fetch, progress=False, auto_adjust=True)

            if data_window.empty:
                logger.warning(f"No data returned by yfinance for {ticker_symbol} in window {start_date_fetch} to {end_date_fetch}.")
            else:
                data_window = data_window.sort_index()
                closest_price_intermediate = data_window['Close'].asof(target_dt)

                logger.debug(f"yfinance .asof() result type: {type(closest_price_intermediate)}, value: {closest_price_intermediate}")

                closest_price_scalar = None
                if isinstance(closest_price_intermediate, pd.Series):
                    if not closest_price_intermediate.empty:
                        closest_price_scalar = closest_price_intermediate.iloc[0]
                    else:
                        closest_price_scalar = pd.NA
                else:
                    closest_price_scalar = closest_price_intermediate

                if pd.notna(closest_price_scalar):
                    actual_price_date_idx = data_window.index.get_indexer([target_dt], method='ffill')
                    if actual_price_date_idx[0] != -1 :
                        actual_price_date = data_window.index[actual_price_date_idx[0]]
                        rate = float(closest_price_scalar)
                        logger.debug(f"Fetched yfinance rate for {from_currency}->{to_currency} (for {api_date_str}, using data from {actual_price_date.strftime('%Y-%m-%d')}): {rate}")
                    else:
                        logger.warning(f"Could not determine actual date for 'asof' price for {ticker_symbol} on {api_date_str}, but got a value. Using asof value directly.")
                        rate = float(closest_price_scalar)
                        logger.debug(f"Fetched yfinance rate for {from_currency}->{to_currency} (for {api_date_str}, using .asof() value directly): {rate}")
                else:
                    logger.warning(f"No valid data found for {ticker_symbol} on or before {api_date_str} in the fetched window using .asof() (result was NaN or pd.NA).")

        except Exception as e:
            logger.error(f"Error during yfinance download or processing for {ticker_symbol} on {api_date_str}: {e}", exc_info=True)
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
            logger.debug(f"Skipping yfinance ticker for stablecoin: {symbol_upper}")
            return None

        # Add more specific mappings if needed, e.g., RENDER might be RNDR-USD
        # This is a common source of issues, ensure tickers are correct for yfinance.
        # Example: self.config.get("yfinance_ticker_map", {}).get(symbol_upper, f"{symbol_upper}-USD")
        return f"{symbol_upper}-USD"

    def _calculate_technical_indicators(self, symbol: str, df_weekly: Optional[pd.DataFrame], df_daily: Optional[pd.DataFrame]) -> Dict[str, Any]:
        indicators = {
            "current_price": None,
            "rsi_weekly": None,
            "ma_200w": None,
            "price_vs_200w_ma_percent": None
        }

        # Get RSI period from config
        rsi_period = self.config.get("rebalance_technical", {}).get("rsi_period_weekly", 14)
        logger.debug(f"Using RSI period {rsi_period} for {symbol}")

        # Process Daily Data (Current Price)
        if df_daily is not None and not df_daily.empty and 'Close' in df_daily.columns:
            daily_close_series = df_daily['Close']
            if not daily_close_series.empty:
                last_daily_close_raw = daily_close_series.iloc[-1]
                if isinstance(last_daily_close_raw, (float, int)) and pd.notna(last_daily_close_raw):
                    indicators["current_price"] = float(last_daily_close_raw)
                else:
                    logger.warning(f"Last daily close for {symbol} was not a scalar: {last_daily_close_raw}.")

        # Process Weekly Data (RSI and 200-week MA)
        if df_weekly is not None and not df_weekly.empty and 'Close' in df_weekly.columns:
            latest_weekly_close_raw = df_weekly['Close'].iloc[-1]
            latest_weekly_close: Optional[float] = None
            if isinstance(latest_weekly_close_raw, (float, int)) and pd.notna(latest_weekly_close_raw):
                latest_weekly_close = float(latest_weekly_close_raw)

            if indicators["current_price"] is None and latest_weekly_close is not None:
                indicators["current_price"] = latest_weekly_close

            # Calculate RSI
            if len(df_weekly) > rsi_period:
                rsi_series = df_weekly.ta.rsi(length=rsi_period)
                if rsi_series is not None and not rsi_series.empty:
                    last_rsi_raw = rsi_series.iloc[-1]
                    if isinstance(last_rsi_raw, (float, int)) and pd.notna(last_rsi_raw):
                        indicators["rsi_weekly"] = float(last_rsi_raw)
                    else:
                        logger.info(f"RSI for {symbol} was NaN or not scalar: {last_rsi_raw}.")
                else:
                    logger.info(f"RSI series calculation for {symbol} returned None or empty.")

            # Calculate 200-week MA
            if len(df_weekly) >= 200:
                ma_200w_series = df_weekly.ta.sma(length=200)
                if ma_200w_series is not None and not ma_200w_series.empty:
                    ma_200w_raw = ma_200w_series.iloc[-1]
                    if isinstance(ma_200w_raw, (float, int)) and pd.notna(ma_200w_raw):
                        indicators["ma_200w"] = float(ma_200w_raw)
                    else:
                        logger.info(f"200w MA for {symbol} was NaN or not scalar: {ma_200w_raw}.")
                    if indicators["ma_200w"] is not None and indicators["ma_200w"] > 0 and latest_weekly_close is not None:
                        indicators["price_vs_200w_ma_percent"] = ((latest_weekly_close - indicators["ma_200w"]) / indicators["ma_200w"]) * 100

        logger.debug(f"Calculated TA indicators for {symbol}: RSI={indicators['rsi_weekly']}, PriceForTA={indicators['current_price']}, MA200w={indicators['ma_200w']}")
        return indicators

    async def _fetch_historical_data_async(self, symbol: str, period_str: str, interval_str: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data, trying yfinance first and falling back to Binance API.
        This increases reliability for assets on Binance that may have poor yfinance coverage.
        """
        yf_ticker = self._get_yfinance_ticker(symbol)
        # Use a more descriptive cache key that includes the source
        cache_key = f"historical_data_{yf_ticker}_{period_str}_{interval_str}"

        # --- Caching logic ---
        cached_data = self.yfinance_disk_cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Disk Cache HIT for historical data: {cache_key}")
            return cached_data

        logger.debug(f"Disk Cache MISS for historical data: {cache_key}. Fetching from APIs.")

        # --- Fallback Strategy ---
        # 1. Try yfinance first
        data = await self._fetch_historical_data_yfinance_async(yf_ticker, period_str, interval_str)

        # 2. If yfinance fails, try Binance API as a fallback
        if data is None or data.empty:
            logger.warning(f"yfinance failed for {yf_ticker}. Falling back to Binance API.")
            try:
                # Define mapping from our interval string to the Binance client's constants
                interval_map = {
                    "1d": Client.KLINE_INTERVAL_1DAY,
                    "1wk": Client.KLINE_INTERVAL_1WEEK,
                    "1mo": Client.KLINE_INTERVAL_1MONTH,
                }
                binance_interval = interval_map.get(interval_str)

                if not binance_interval:
                    logger.error(f"Unsupported interval '{interval_str}' for Binance API fallback.")
                    return None

                # The python-binance client can interpret human-readable start strings
                start_str = f"{period_str.replace('y', ' years').replace('mo', ' months')} ago UTC"

                # Fetch klines from Binance
                binance_pair = f"{symbol.upper()}USDT"
                logger.info(f"Fetching from Binance API: Pair='{binance_pair}', Interval='{binance_interval}', Start='{start_str}'")
                klines = self.binance_client.get_historical_klines(binance_pair, binance_interval, start_str)

                if not klines:
                    logger.warning(f"Binance API returned no kline data for {binance_pair}.")
                    self.yfinance_disk_cache.set(cache_key, None, expire=3600 * 24) # Cache failure
                    return None

                # Convert to a clean DataFrame, matching the yfinance format
                cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time',
                        'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
                data = pd.DataFrame(klines, columns=cols)
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
                data.set_index('timestamp', inplace=True)

                # Keep only the necessary columns and ensure they are numeric
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')
                logger.info(f"Successfully fetched {len(data)} data points for {symbol} from Binance API fallback.")

            except Exception as e:
                logger.error(f"Binance API fallback also failed for {symbol}: {e}", exc_info=True)
                self.yfinance_disk_cache.set(cache_key, None, expire=3600 * 24) # Cache failure
                return None

        # Cache the successful result (from either yfinance or Binance) and return it
        self.yfinance_disk_cache.set(cache_key, data)
        return data

    async def _fetch_historical_data_yfinance_async(self, yf_ticker: str, period_str: str, interval_str: str) -> Optional[pd.DataFrame]:
        """
        Asynchronously fetches historical data from yfinance with improved error handling.
        This function is now designed to be called by the main _fetch_historical_data_async wrapper.
        """
        if not yf_ticker:
            logger.warning("yf_ticker was empty in _fetch_historical_data_yfinance_async.")
            return None

        try:
            loop = asyncio.get_event_loop()
            # yfinance can be slow, run it in a thread pool executor
            partial_yf_download = functools.partial(
                yf.download,
                tickers=yf_ticker,
                period=period_str, # yfinance handles periods like "4y", "60d" well
                interval=interval_str,
                progress=False,
                auto_adjust=True
            )

            logger.debug(f"Executing yfinance download: Ticker='{yf_ticker}', Period='{period_str}', Interval='{interval_str}'")
            data = await loop.run_in_executor(None, partial_yf_download)

            if data.empty:
                logger.warning(f"No historical data from yfinance for {yf_ticker} (Period:{period_str}, Interval:{interval_str}).")
                return None

            # --- Flatten MultiIndex logic ---
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            data.columns = [str(col).capitalize() for col in data.columns]
            data.index = pd.to_datetime(data.index, utc=True)
            logger.debug(f"Fetched {len(data)} yfinance rows for {yf_ticker}.")
            return data

        except Exception as e:
            # This is broad, but yfinance can raise various errors.
            # The YFInvalidPeriodError for TAO is one example.
            logger.error(f"yfinance download failed for {yf_ticker} with period '{period_str}': {e}", exc_info=True)
            return None

    async def get_core_portfolio_rebalance_suggestions_technical(self) -> Optional[pd.DataFrame]:
        logger.info("Calculating Core Portfolio rebalance suggestions with technical indicators...")

        live_balances_df = self.fetch_binance_balances()
        if live_balances_df.empty:
            logger.error("Could not fetch live balances from Binance. Cannot rebalance.")
            return None

        rebalance_config = self.config.get("rebalance_technical", {})
        rsi_period = rebalance_config.get("rsi_period_weekly", 14)
        drift_threshold = rebalance_config.get("allocation_drift_threshold", 0.1)
        rsi_overbought = rebalance_config.get("rsi_overbought", 70)
        rsi_oversold = rebalance_config.get("rsi_oversold", 30)
        price_vs_ma_above = rebalance_config.get("price_vs_ma_above", 25)
        price_vs_ma_near_below = rebalance_config.get("price_vs_ma_near_below", 0)
        sell_multiplier = rebalance_config.get("sell_percentage_multiplier", 0.15)
        buy_multiplier = rebalance_config.get("buy_amount_multiplier", 0.75)
        buy_portfolio_cap = rebalance_config.get("buy_portfolio_cap", 0.1)

        core_portfolio_symbols_config = list(self.config.get("target_allocation", {}).keys())
        core_portfolio_symbols = [self.norm_map.get(k.upper(), k.upper()) for k in core_portfolio_symbols_config]
        core_balances_df = live_balances_df[live_balances_df['symbol'].isin(core_portfolio_symbols)].copy()
        all_symbols_to_price = list(set(live_balances_df['symbol'].tolist() + core_portfolio_symbols))
        current_prices_dict = self.get_current_prices(all_symbols_to_price)
        core_balances_df['current_price'] = core_balances_df['symbol'].map(current_prices_dict).fillna(0.0)
        core_balances_df['value_usd'] = core_balances_df['quantity'] * core_balances_df['current_price']
        total_portfolio_value = core_balances_df['value_usd'].sum()

        if total_portfolio_value == 0:
            return pd.DataFrame()

        target_allocation_normalized = { self.norm_map.get(k.upper(), k.upper()): v for k, v in self.config.get("target_allocation", {}).items() }
        suggestions_data = []

        def format_coin_amount(amount):
            return f"{amount:,.8g}"

        for symbol_upper_case in target_allocation_normalized.keys():
            logger.info(f"--- Analyzing: {symbol_upper_case} ---")
            current_row = live_balances_df[live_balances_df['symbol'] == symbol_upper_case]
            current_qty = current_row['quantity'].iloc[0] if not current_row.empty else 0.0
            live_current_price = current_prices_dict.get(symbol_upper_case, 0.0)
            current_value = current_qty * live_current_price
            target_pct = target_allocation_normalized.get(symbol_upper_case, 0.0)
            target_value_for_symbol = total_portfolio_value * target_pct
            current_pct_of_portfolio = (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0

            # --- THIS IS THE KEY CALCULATION ---
            relative_drift = (current_value - target_value_for_symbol) / target_value_for_symbol if target_value_for_symbol > 0 else float('inf') if current_value > 0 else 0.0

            df_weekly_hist = await self._fetch_historical_data_async(symbol_upper_case, period_str="4y", interval_str="1wk")
            df_daily_hist = await self._fetch_historical_data_async(symbol_upper_case, period_str="60d", interval_str="1d")
            ta_indicators = self._calculate_technical_indicators(symbol_upper_case, df_weekly_hist, df_daily_hist)
            rsi = ta_indicators.get("rsi_weekly")
            price_vs_200w_ma = ta_indicators.get("price_vs_200w_ma_percent")

            signal = "HOLD"
            action_text = "Hold: Allocation is within tolerance."

            # Use the 'relative_drift' for the logic check
            is_significantly_overweight = relative_drift > drift_threshold
            is_significantly_underweight = relative_drift < -drift_threshold

            rsi_very_overbought = rsi is not None and rsi > rsi_overbought
            rsi_very_oversold = rsi is not None and rsi < rsi_oversold
            price_well_above_ma = price_vs_200w_ma is not None and price_vs_200w_ma > price_vs_ma_above
            price_near_or_below_ma = price_vs_200w_ma is not None and price_vs_200w_ma <= price_vs_ma_near_below

            if is_significantly_overweight or (rsi_very_overbought and price_well_above_ma):
                signal = "SELL"
                sell_percentage_of_position = min(0.1, relative_drift * sell_multiplier)
                action_value_usd = current_value * sell_percentage_of_position
                coin_amount_to_sell = action_value_usd / live_current_price if live_current_price > 0 else 0
                action_text = (f"Sell ~{sell_percentage_of_position * 100:.1f}% of position (~${action_value_usd:,.2f}), which is {format_coin_amount(coin_amount_to_sell)} {symbol_upper_case}")

            elif is_significantly_underweight or (rsi_very_oversold and price_near_or_below_ma):
                signal = "BUY"
                underweight_amount_usd = target_value_for_symbol - current_value
                action_value_usd = min(underweight_amount_usd * buy_multiplier, abs(relative_drift) * total_portfolio_value * buy_portfolio_cap)
                coin_amount_to_buy = action_value_usd / live_current_price if live_current_price > 0 else 0
                action_text = (f"Buy ~${action_value_usd:,.2f} worth ({format_coin_amount(coin_amount_to_buy)} {symbol_upper_case})")

            suggestions_data.append({
                "Symbol": symbol_upper_case,
                "Target %": target_pct * 100,
                "Current %": current_pct_of_portfolio,
                "Current Value (USD)": current_value,
                "Relative Drift %": relative_drift * 100, # Store as percentage
                "RSI (14W)": rsi,
                "Price vs 200w MA (%)": price_vs_200w_ma,
                "Signal": signal,
                "Suggested Action Detail": action_text
            })
            await asyncio.sleep(self.yfinance_config.get("request_delay_ms", 200) / 1000.0)

        if not suggestions_data:
            return pd.DataFrame()

        df_suggestions = pd.DataFrame(suggestions_data)
        return df_suggestions

    async def sync_data(self):
        """Asynchronously synchronize data from all sources, and update holdings."""
        logger.info("Starting data synchronization...")

        # This part remains synchronous as it's just config loading
        target_coins_from_config = list(self.config.get("target_allocation", {}).keys())
        self.target_assets_for_sync = set(s.upper() for s in target_coins_from_config)
        self.target_assets_for_sync.add("USDT")
        logger.info(f"SYNC FOCUS: Will be focusing on transactions for target assets: {list(self.target_assets_for_sync)}")

        lookback_config = self.config.get("history_lookback_days", {})

        # Create tasks for each data fetching method using asyncio.to_thread
        # This allows the synchronous, network-blocking calls inside each method to run concurrently without blocking the main program
        tasks = [
            asyncio.to_thread(self.fetch_binance_transactions),
            asyncio.to_thread(self.fetch_deposit_history, days_back=lookback_config.get("deposits", 90)),
            asyncio.to_thread(self.fetch_withdrawal_history, days_back=lookback_config.get("withdrawals", 90)),
            asyncio.to_thread(self.fetch_p2p_usdt_buys, days_back=lookback_config.get("p2p_buys", 90)),
            asyncio.to_thread(self.fetch_internal_transfers, days_back=lookback_config.get("internal_transfers", 90)),
            asyncio.to_thread(self.fetch_spot_futures_transfers, asset="USDT", days_back=lookback_config.get("spot_futures_transfers", 90)),
            asyncio.to_thread(self.fetch_spot_convert_history, days_back=lookback_config.get("spot_convert_history", 90)),
            asyncio.to_thread(self.fetch_simple_earn_flexible_rewards, days_back=lookback_config.get("simple_earn_rewards", 90)),
            asyncio.to_thread(self.fetch_simple_earn_flexible_subscriptions, days_back=lookback_config.get("simple_earn_subscriptions", 90)),
            asyncio.to_thread(self.fetch_simple_earn_flexible_redemptions, days_back=lookback_config.get("simple_earn_redemptions", 90)),
            asyncio.to_thread(self.fetch_dividend_history, days_back=lookback_config.get("dividend_history", 90)),
            asyncio.to_thread(self.fetch_staking_history, days_back=lookback_config.get("staking_history", 90)),
        ]

        logger.info(f"Launching {len(tasks)} data fetching tasks concurrently...")

        # asyncio.gather runs all tasks concurrently and waits for them all to complete
        results_from_all_tasks = await asyncio.gather(*tasks)

        # Combine results from all tasks into a single list
        all_new_transactions = []
        for result_list in results_from_all_tasks:
            if result_list:
                all_new_transactions.extend(result_list)

        if all_new_transactions:
            logger.info(f"Processing a total of {len(all_new_transactions)} fetched/parsed transaction items.")
            all_new_transactions.sort(key=lambda x: x['timestamp']) # Sort all txs chronologically before DB insertion

            num_inserted_updated = self.db_manager.bulk_insert_transactions(all_new_transactions)
            logger.info(f"Attempted to process {len(all_new_transactions)} transaction records. Database reported {num_inserted_updated if num_inserted_updated is not None else 'unknown'} changes/insertions.")
        else:
            logger.info("No new transactions fetched or processed from any source.")

        self.update_holdings_from_transactions()
        logger.info("Data synchronization finished.")

    async def run_full_sync(self) -> Dict[str, Any]:
        """Runs the full async data sync and then calculates metrics."""
        await self.sync_data()
        return self.calculate_portfolio_metrics()

    async def view_trends(self):
        """
        Provides an interactive menu to view cryptocurrency trend analysis reports.
        Initializes the trend analyzer with the central config and manages user flow.
        """
        try:
            # Pass the tracker's config object to the analyzer
            analyzer = CryptoTrendAnalyzer(config=self.config)
            print("✅ Trend Analyzer initialized with centralized config.")
        except Exception as e:
            logger.error(f"Failed to initialize CryptoTrendAnalyzer: {e}", exc_info=True)
            print("❌ Error: Could not initialize the Trend Analyzer. Please check the logs.")
            return

        while True:
            print("\n--- 📈 Crypto Trend Analysis ---")
            print("Select the timeframe for the analysis:")
            print("1. Long-term (1 Year)")
            print("2. Swing (3 Months)")
            print("3. Day (1 Month)")
            print("4. Back to Main Menu")

            try:
                choice_str = input("Select option (1-4): ")
                if not choice_str.isdigit():
                    print("❌ Invalid input. Please enter a number.")
                    time.sleep(2)
                    continue
                choice = int(choice_str)
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
                time.sleep(2)
                continue

            timeframe = None
            if choice == 1:
                timeframe = "long_term"
            elif choice == 2:
                timeframe = "swing"
            elif choice == 3:
                timeframe = "day"
            elif choice == 4:
                break
            else:
                print("❌ Invalid option. Please try again.")
                time.sleep(2)
                continue

            if timeframe:
                print(f"\n🔄 Generating {timeframe.replace('_', ' ')} trend report...")
                try:
                    # Run the synchronous generate_report in a separate thread
                    report = await asyncio.to_thread(analyzer.generate_report, timeframe)

                    if report:
                        self.print_trend_report(report)
                        # Ask the user if they want to export the report
                        export_choice = input("\nDo you want to export this report? (y/n): ").lower()
                        if export_choice == 'y':
                            self.export_trend_report(report, timeframe)
                    else:
                        print(f"❌ Could not generate the {timeframe} trend report. See logs for details.")

                except Exception as e:
                    logger.error(f"An error occurred during report generation for timeframe '{timeframe}': {e}", exc_info=True)
                    print(f"❌ An unexpected error occurred. Please check the logs.")

                input("\n✅ Press Enter to continue...")

    async def run_rebalancing_backtest(self):
        """
        Handles the user flow for running the rebalancing backtest.
        """
        print("\n--- ⚖️ Rebalancing Strategy Backtesting Mode ---")

        try:
            # The RebalancingBacktester needs the tracker instance itself to access its logic
            backtester = RebalancingBacktester(config=self.config, tracker=self)

            initial_capital_str = input("Enter initial capital for simulation (default: 10000): ")
            initial_capital = float(initial_capital_str) if initial_capital_str else 10000.0

            backtest_period_str = input("Enter backtest period (e.g., 2y, 3y, 5y - default: 3y): ")
            backtest_period = backtest_period_str if backtest_period_str else "3y"

            # Run the backtest and generate the report
            # This is an async method now because the suggestion method is async
            await backtester.run(initial_capital=initial_capital, backtest_period=backtest_period)
            backtester.generate_report()

        except Exception as e:
            logger.error(f"An error occurred during rebalancing backtest: {e}", exc_info=True)
            print(f"❌ An unexpected error occurred: {e}")

    async def run_rebalance_and_execute(self):
        """
        Orchestrates the process of getting rebalancing suggestions and executing them.
        This version now fetches Earn balances to pass to the executor.
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

        # 2. Print Suggestions for Review
        self.print_rebalance_suggestions(suggestions_df)

        # 3. Fetch Earn balances to make the executor smarter
        print("Verifying balances in Spot and Earn wallets...")
        earn_balances = self.fetch_simple_earn_balances()

        # 4. Pass everything to the Executor for Confirmation and Trading
        self.execute_rebalancing_trades(suggestions_df, earn_balances)

    def fetch_binance_balances(self) -> pd.DataFrame:
        """Fetch current balances from Binance Spot wallet with retry, explicit normalization, and LD-prefix consolidation."""
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Spot balances.")
            return pd.DataFrame(columns=['symbol', 'quantity'])

        # --- Close stale connections before making a new request ---
        self.binance_client.session.close()

        retries = 3
        wait_time_seconds = 15
        api_timeout = self.config.get("apis", {}).get("binance", {}).get("timeout", 180)

        while retries > 0:
            try:
                logger.debug(f"Attempting to fetch Binance account info (Timeout: {api_timeout}s)...")
                account_info = self.binance_client.get_account()

                balances_raw = account_info.get('balances', [])
                if not balances_raw:
                    logger.info("Fetched 0 balances from Binance Spot wallet.")
                    return pd.DataFrame(columns=['symbol', 'quantity'])

                processed_balances = []
                for b in balances_raw:
                    free = float(b.get('free', 0.0))
                    locked = float(b.get('locked', 0.0))
                    quantity = free + locked
                    if quantity > 0.00000001:
                        asset_symbol_api = b.get('asset').upper()
                        normalized_symbol_s1 = self.norm_map.get(asset_symbol_api, asset_symbol_api)
                        final_symbol = normalized_symbol_s1
                        if normalized_symbol_s1.startswith('LD') and len(normalized_symbol_s1) > 2:
                            base_equivalent = normalized_symbol_s1[2:]
                            if normalized_symbol_s1 in self.symbol_mappings and base_equivalent in self.symbol_mappings:
                                final_symbol = base_equivalent
                                logger.debug(f"Consolidated API symbol '{asset_symbol_api}' (norm1: '{normalized_symbol_s1}') to base '{final_symbol}' based on coingecko_ids structure.")
                        processed_balances.append({'symbol': final_symbol, 'quantity': quantity})

                if not processed_balances:
                    logger.info("Found raw balances, but all were zero or negligible after processing.")
                    return pd.DataFrame(columns=['symbol', 'quantity'])

                df = pd.DataFrame(processed_balances)
                df = df.groupby('symbol', as_index=False)['quantity'].sum()

                logger.debug(f"DEBUG: Balances from SPOT after all processing in fetch_binance_balances: \n{df.to_string() if not df.empty else 'EMPTY DF'}")
                logger.info(f"Fetched and consolidated {len(df)} non-zero balances from Binance Spot.")
                return df

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                retries -= 1
                logger.error(f"Network error fetching Binance Spot balances: {e}. Retries left: {retries}.")
                if retries > 0: logger.info(f"Waiting {wait_time_seconds}s before retrying..."); time.sleep(wait_time_seconds)
                else: logger.error("Failed to fetch Spot balances after multiple retries due to network errors.")
            except BinanceAPIException as e:
                 logger.error(f"Binance API Error fetching Spot balances: {e}. No retries for API errors.")
                 break
            except Exception as e:
                logger.error(f"Unexpected error fetching Spot balances: {e}", exc_info=True)
                break
        return pd.DataFrame(columns=['symbol', 'quantity'])

    def fetch_binance_transactions(self) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch transactions.")
            return []

        # # --- TEMPORARY MODIFICATION FOR FULL HISTORY SYNC ---
        # days_back = 1095  # Override to 3 years for one-time full sync
        # logger.warning(f"TEMPORARY OVERRIDE: Fetching full trade history for the last {days_back} days.")
        # # --- END TEMPORARY MODIFICATION ---

        transactions = []
        norm_map = self.config.get("symbol_normalization_map", {})

        stablecoin_quotes = self.config.get("portfolio",{}).get("stablecoin_symbols", ["USDT"])
        crypto_quotes_from_targets = [s for s in self.config.get("portfolio", {}).get("crypto_quotes", ["BTC", "ETH"]) if s in self.target_assets_for_sync]
        all_quotes_to_check = sorted(list(set(stablecoin_quotes + crypto_quotes_from_targets)))
        potential_base_assets = [s for s in self.target_assets_for_sync if s not in stablecoin_quotes]

        logger.info(f"Attempting to fetch trades for target base symbols: {potential_base_assets} against quotes: {all_quotes_to_check}")

        source_trade = "Binance Trade"
        source_synthetic = "Binance Synthetic"
        latest_ts_trade = self.db_manager.get_latest_timestamp_for_source(source_trade)
        latest_ts_synthetic = self.db_manager.get_latest_timestamp_for_source(source_synthetic)

        overall_latest_known_ts_for_trades = None
        if latest_ts_trade and latest_ts_synthetic:
            overall_latest_known_ts_for_trades = max(latest_ts_trade, latest_ts_synthetic)
        elif latest_ts_trade:
            overall_latest_known_ts_for_trades = latest_ts_trade
        elif latest_ts_synthetic:
            overall_latest_known_ts_for_trades = latest_ts_synthetic

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        lookback_days_trades = self.config.get("history_lookback_days", {}).get("trades", 90)

        overall_sync_start_time_dt: datetime.datetime
        if overall_latest_known_ts_for_trades:
            # Fetch from a bit before to ensure no gaps, up to the general lookback days
            effective_start_from_db = overall_latest_known_ts_for_trades - datetime.timedelta(minutes=60) # 1 hour buffer
            fallback_start_from_days = now_utc - datetime.timedelta(days=lookback_days_trades)
            overall_sync_start_time_dt = max(effective_start_from_db, fallback_start_from_days)
            logger.info(f"Selective sync for Binance Trades/Synthetic: Effective overall start date {overall_sync_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            overall_sync_start_time_dt = now_utc - datetime.timedelta(days=lookback_days_trades)
            logger.info(f"Full sync for Binance Trades/Synthetic: Fetching last {lookback_days_trades} days from {overall_sync_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if overall_sync_start_time_dt >= now_utc:
            logger.info("Trade history is very recent. No new trades to fetch based on overall selective sync time.")
            return []

        processed_pairs = set()
        max_retries = self.config.get("apis", {}).get("binance", {}).get("max_retries_per_batch", 3)
        wait_time_seconds = self.config.get("apis", {}).get("binance", {}).get("retry_delay_sec", 15)
        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms_csv", 1500))
        binance_api_delay_ms = self.config.get("apis",{}).get("binance",{}).get("request_delay_ms", 250)

        # Define a safe batch duration, e.g., 23 hours
        batch_duration_for_trades = datetime.timedelta(hours=23)

        for base_asset in potential_base_assets:
            for quote_asset in all_quotes_to_check:
                if base_asset == quote_asset:
                    continue
                pair = f"{base_asset}{quote_asset}"
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                logger.debug(f"Processing pair: {pair}")

                # Iterate through the lookback period in 23-hour chunks for this pair
                current_chunk_start_dt = overall_sync_start_time_dt

                while current_chunk_start_dt < now_utc:
                    current_chunk_end_dt = min(current_chunk_start_dt + batch_duration_for_trades, now_utc)

                    api_start_time_ms_chunk = int(current_chunk_start_dt.timestamp() * 1000)
                    api_end_time_ms_chunk = int(current_chunk_end_dt.timestamp() * 1000)

                    if api_start_time_ms_chunk >= api_end_time_ms_chunk: # Should not happen if loop is correct
                        break

                    trades_for_pair_chunk = []
                    retries_left = max_retries
                    while retries_left > 0:
                        try:
                            logger.debug(f"Fetching trades for {pair} (Chunk: {current_chunk_start_dt.strftime('%Y-%m-%d %H:%M')} to {current_chunk_end_dt.strftime('%Y-%m-%d %H:%M')}, Attempt {max_retries - retries_left + 1})")
                            trades_for_pair_chunk = self.binance_client.get_my_trades(
                                symbol=pair,
                                startTime=api_start_time_ms_chunk,
                                endTime=api_end_time_ms_chunk,
                                limit=1000 # Max limit per call
                            )
                            if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                            break
                        except BinanceAPIException as e_api:
                            if e_api.status_code == 400 and (e_api.code == -1121 or "Invalid symbol" in str(e_api.message)):
                                logger.debug(f"Invalid or non-existent pair: {pair}. Skipping this pair.")
                                trades_for_pair_chunk = None # Signal to break outer pair loop
                                break # Break retry loop
                            elif e_api.code == -1127: # Should be avoided by batching, but good to log if it still happens
                                logger.error(f"API Error -1127 (More than 24h) for {pair} even with batching: {e_api}. Chunk: {current_chunk_start_dt} to {current_chunk_end_dt}. Skipping this chunk.")
                                break # Break retry loop, move to next chunk
                            elif e_api.code == -1021 :
                                logger.warning(f"Timestamp error for {pair}, may need to sync system clock or adjust recvWindow. {e_api}")
                                # Potentially break or retry based on strategy
                            else:
                                logger.error(f"API Error fetching trades for {pair} (Chunk): {e_api}")
                            # Handle retry logic for other API errors
                            retries_left -=1
                            if retries_left == 0: logger.error(f"Max retries for {pair} (Chunk)."); break
                            time.sleep(wait_time_seconds)
                        except Exception as e_net:
                            retries_left -=1
                            logger.error(f"Network/Other error for {pair} (Chunk): {e_net}. Retries left: {retries_left}")
                            if retries_left == 0: logger.error(f"Max retries for {pair} (Chunk)."); break
                            time.sleep(wait_time_seconds)

                    if trades_for_pair_chunk is None: # Break from outer pair loop if symbol is invalid
                        break

                    if not trades_for_pair_chunk:
                        # logger.debug(f"No trades found for {pair} in chunk: {current_chunk_start_dt} to {current_chunk_end_dt}")
                        # Move to the next chunk for this pair
                        current_chunk_start_dt = current_chunk_end_dt # Next chunk starts where this one ended
                        if current_chunk_start_dt >= now_utc: break # Reached current time
                        continue

                    logger.info(f"Fetched {len(trades_for_pair_chunk)} trades for {pair} in chunk: {current_chunk_start_dt.date()} to {current_chunk_end_dt.date()}")
                    for trade in trades_for_pair_chunk:
                        timestamp_obj = pd.to_datetime(trade['time'], unit='ms', utc=True).to_pydatetime()

                        # Selective sync: skip if older than determined overall_sync_start_time_dt
                        if timestamp_obj < overall_sync_start_time_dt:
                            continue
                        # Additional check: if overall_latest_known_ts_for_trades exists, skip if not newer
                        if overall_latest_known_ts_for_trades and timestamp_obj <= overall_latest_known_ts_for_trades:
                            continue

                        normalized_base_asset = base_asset
                        normalized_quote_asset = quote_asset
                        quantity_base = float(trade['qty'])
                        price_in_quote_terms = float(trade['price'])
                        is_buy_of_base = trade['isBuyer']
                        fee_quantity_raw = float(trade['commission'])
                        fee_currency_raw = trade['commissionAsset'].upper()
                        normalized_fee_currency = norm_map.get(fee_currency_raw, fee_currency_raw)
                        trade_date_str = timestamp_obj.strftime('%d-%m-%Y')
                        price_base_in_usd = 0.0
                        price_quote_in_usd_for_trade = None
                        coingecko_called_for_trade_pricing = False

                        # ... (Rest of your existing pricing logic for base, quote, and fees)
                        if normalized_quote_asset in self.stablecoin_symbols:
                            price_base_in_usd = price_in_quote_terms
                            price_quote_in_usd_for_trade = 1.0
                        elif normalized_base_asset in self.stablecoin_symbols:
                            price_base_in_usd = 1.0
                            price_quote_in_usd_for_trade = 1.0 / price_in_quote_terms if price_in_quote_terms > 0 else 0.0
                        else:
                            quote_coin_id = self.symbol_mappings.get(normalized_quote_asset)
                            if quote_coin_id:
                                price_quote_in_usd_for_trade = self._get_coingecko_historical_price(quote_coin_id, trade_date_str)
                                coingecko_called_for_trade_pricing = True
                                if price_quote_in_usd_for_trade is not None:
                                    price_base_in_usd = price_in_quote_terms * price_quote_in_usd_for_trade
                                else: logger.warning(f"Trade {pair}: Could not get hist price for quote {normalized_quote_asset}. Base USD price will be $0.")
                            else: logger.warning(f"Trade {pair}: No CoinGecko ID for quote {normalized_quote_asset}. Base USD price will be $0.")

                        fee_usd = 0.0
                        if normalized_fee_currency in self.stablecoin_symbols:
                            fee_usd = fee_quantity_raw
                        elif normalized_fee_currency == normalized_base_asset and price_base_in_usd > 0.0:
                            fee_usd = fee_quantity_raw * price_base_in_usd
                        elif normalized_fee_currency == normalized_quote_asset :
                            if normalized_quote_asset in self.stablecoin_symbols: fee_usd = fee_quantity_raw
                            elif price_quote_in_usd_for_trade is not None : fee_usd = fee_quantity_raw * price_quote_in_usd_for_trade
                            else: # Attempt to price fee currency if it was the quote and quote wasn't priced yet
                                fee_quote_coin_id = self.symbol_mappings.get(normalized_quote_asset)
                                if fee_quote_coin_id:
                                    temp_price = self._get_coingecko_historical_price(fee_quote_coin_id, trade_date_str)
                                    if temp_price is not None: fee_usd = fee_quantity_raw * temp_price
                                    if temp_price is not None and not coingecko_called_for_trade_pricing: time.sleep(cg_delay_ms / 1000.0) # Delay if fresh call
                        else: # Fee in a third currency
                            fee_asset_coin_id = self.symbol_mappings.get(normalized_fee_currency)
                            if fee_asset_coin_id:
                                price_fee_asset_in_usd = self._get_coingecko_historical_price(fee_asset_coin_id, trade_date_str)
                                if price_fee_asset_in_usd is not None:
                                    fee_usd = fee_quantity_raw * price_fee_asset_in_usd
                                    if not coingecko_called_for_trade_pricing: time.sleep(cg_delay_ms / 1000.0)
                        # --- End of Fee Pricing ---

                        transactions.append({
                            "symbol": normalized_base_asset, "timestamp": timestamp_obj,
                            "type": 'BUY' if is_buy_of_base else 'SELL',
                            "quantity": quantity_base, "price_usd": price_base_in_usd,
                            "fee_quantity": fee_quantity_raw, "fee_currency": normalized_fee_currency, "fee_usd": fee_usd,
                            "source": source_trade,
                            "transaction_hash": f"binance_trade_{trade['id']}_{normalized_base_asset}",
                            "notes": f"Pair: {pair}, Price: {price_in_quote_terms} {quote_asset}, Fee: {fee_quantity_raw} {fee_currency_raw}"
                        })

                        if normalized_quote_asset in self.target_assets_for_sync:
                            qty_quote_exchanged = quantity_base * price_in_quote_terms
                            price_quote_for_synth = 0.0
                            if normalized_quote_asset in self.stablecoin_symbols: price_quote_for_synth = 1.0
                            elif price_quote_in_usd_for_trade is not None : price_quote_for_synth = price_quote_in_usd_for_trade

                            transactions.append({
                                "symbol": normalized_quote_asset, "timestamp": timestamp_obj,
                                "type": 'SELL' if is_buy_of_base else 'BUY',
                                "quantity": qty_quote_exchanged, "price_usd": price_quote_for_synth,
                                "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                                "source": source_synthetic,
                                "transaction_hash": f"binance_trade_{trade['id']}_synth_{normalized_quote_asset}",
                                "notes": f"Synthetic for {pair} trade. Original fee: {fee_quantity_raw} {fee_currency_raw}"
                            })
                        if coingecko_called_for_trade_pricing and (price_base_in_usd > 0.0 or (price_quote_in_usd_for_trade is not None and price_quote_in_usd_for_trade > 0.0)):
                            if cg_delay_ms > 0: time.sleep(cg_delay_ms / 1000.0)
                    # End of for trade in trades_for_pair_chunk

                    # Move to next chunk for this pair
                    current_chunk_start_dt = current_chunk_end_dt
                    if current_chunk_start_dt >= now_utc: break # Reached current time for this pair
                # End of while current_chunk_start_dt < now_utc (chunking loop for a pair)
                if trades_for_pair_chunk is None: # If pair was invalid, break from quote asset loop
                    break
            # End of for quote_asset loop
            if potential_base_assets and binance_api_delay_ms > 0 and base_asset != potential_base_assets[-1]: # Delay between different base assets
                 time.sleep(binance_api_delay_ms / 1000.0)
        # End of for base_asset loop

        logger.info(f"Fetched a total of {len(transactions)} new spot trade transactions (including synthetic, filtered for target assets) using selective sync and daily batching.")
        return transactions

    def fetch_deposit_history(self, days_back: int = 90) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch deposit history.")
            return []

        deposits_data_unfiltered = []
        norm_map = self.config.get("symbol_normalization_map", {})
        pepe_gift_config = self.config.get("pepe_gift_details", {})
        pepe_gift_symbol = pepe_gift_config.get("symbol", "PEPE").upper()
        try:
            your_pepe_gift_amount = float(pepe_gift_config.get("amount", "0"))
        except ValueError:
            your_pepe_gift_amount = 0.0
            logger.warning(f"PEPE gift amount '{pepe_gift_config.get('amount')}' is invalid for deposits.")

        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms_deposit", 1500))

        source_name = "Binance Deposit"
        latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        endTime = int(now_utc.timestamp() * 1000)
        startTime = 0

        specific_lookback_days = self.config.get("history_lookback_days", {}).get("deposits", days_back)

        if latest_known_ts:
            start_datetime_obj = latest_known_ts - datetime.timedelta(minutes=5)
            startTime = int(start_datetime_obj.timestamp() * 1000)
            logger.info(f"Selective sync for '{source_name}': Fetching since ~{start_datetime_obj} (last known: {latest_known_ts})")
        else:
            start_datetime_obj = now_utc - datetime.timedelta(days=specific_lookback_days)
            startTime = int(start_datetime_obj.timestamp() * 1000)
            logger.info(f"Full sync for '{source_name}': Fetching last {specific_lookback_days} days since {start_datetime_obj}")

        if startTime >= endTime:
            logger.info(f"'{source_name}' history is very recent. Setting startTime to endTime-1min to fetch minimal window.")
            startTime = endTime - (60 * 1000)

        try:
            logger.info(f"Fetching deposit history (startTime: {pd.to_datetime(startTime, unit='ms', utc=True)}, endTime: {pd.to_datetime(endTime, unit='ms', utc=True)})...")
            all_deposits = self.binance_client.get_deposit_history(startTime=startTime, endTime=endTime, status=1)
            logger.info(f"Fetched {len(all_deposits)} successful deposit records (startTime: {startTime}, endTime: {endTime}, pre-filter).")

            for i, deposit in enumerate(all_deposits):
                logger.debug(f"Processing raw deposit record {i+1}/{len(all_deposits)}: {deposit}")
                symbol_original = deposit.get('coin')
                if not symbol_original:
                    logger.warning(f"Deposit record {i+1} missing 'coin' field (TxID: {deposit.get('txId')}). Skipping.")
                    continue

                normalized_symbol = norm_map.get(symbol_original.upper(), symbol_original.upper())

                if normalized_symbol not in self.target_assets_for_sync:
                    # logger.debug(f"Skipping deposit for non-target asset: {normalized_symbol}") # Optional: can be verbose
                    continue

                insert_time_raw = deposit.get('insertTime')
                if insert_time_raw is None:
                    logger.error(f"Deposit for {normalized_symbol} (TxID: {deposit.get('txId')}) has missing 'insertTime'. Skipping.")
                    continue

                deposit_timestamp_obj_pandas = pd.to_datetime(insert_time_raw, unit='ms', utc=True)
                if pd.isna(deposit_timestamp_obj_pandas):
                    logger.error(f"Parsed timestamp is NaT for {normalized_symbol} (TxID: {deposit.get('txId')}) from raw value '{insert_time_raw}'. Skipping.")
                    continue
                deposit_timestamp_py = deposit_timestamp_obj_pandas.to_pydatetime()

                price_usd_at_deposit = 0.0
                is_pepe_gift = (normalized_symbol == pepe_gift_symbol and
                                abs(float(deposit.get('amount', 0)) - your_pepe_gift_amount) < 1e-9 and
                                your_pepe_gift_amount > 0)

                coingecko_called = False
                if is_pepe_gift:
                    logger.info(f"Identified PEPE gift deposit ({deposit.get('amount')} {normalized_symbol}). Assigning $0 cost.")
                elif normalized_symbol in self.stablecoin_symbols:
                    price_usd_at_deposit = 1.0
                else:
                    coin_id = self.symbol_mappings.get(normalized_symbol)
                    if coin_id:
                        date_str = deposit_timestamp_obj_pandas.strftime('%d-%m-%Y')
                        historical_price = self._get_coingecko_historical_price(coin_id, date_str)
                        coingecko_called = True
                        if historical_price is not None:
                            price_usd_at_deposit = historical_price
                            logger.info(f"Fetched historical price for {normalized_symbol} (deposit) on {date_str}: ${price_usd_at_deposit:.6f}")
                        else:
                            logger.warning(f"Could not fetch historical price for {normalized_symbol} (deposit) on {date_str}. Cost for deposit will be $0.")
                    else:
                        logger.warning(f"No CoinGecko ID for {normalized_symbol} (deposit). Cost for deposit will be $0.")

                tx_hash = deposit.get('txId', f"binance_deposit_{deposit.get('id', i)}_{insert_time_raw}")

                # Ensure this deposit is newer than the latest known to avoid re-processing if overlap is used
                if latest_known_ts and deposit_timestamp_py <= latest_known_ts:
                    # This check might be redundant if ON CONFLICT handles it, but can prevent re-pricing
                    # logger.debug(f"Skipping deposit {tx_hash} as its timestamp ({deposit_timestamp_py}) is not newer than last known ({latest_known_ts}) for source {source_name}")
                    continue

                deposits_data_unfiltered.append({
                    "symbol": normalized_symbol,
                    "timestamp": deposit_timestamp_py,
                    "type": "DEPOSIT",
                    "quantity": float(deposit.get('amount', 0)),
                    "price_usd": price_usd_at_deposit,
                    "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                    "source": source_name,
                    "transaction_hash": tx_hash,
                    "notes": f"Status: {deposit.get('status', 'N/A')}, Network: {deposit.get('network')}"
                })
                if coingecko_called and historical_price is not None and historical_price > 0.0:
                    time.sleep(cg_delay_ms / 1000.0)

        except BinanceAPIException as e:
            logger.error(f"API Error fetching deposit history: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching deposit history: {e}", exc_info=True)

        logger.info(f"Fetched {len(deposits_data_unfiltered)} targeted deposit transactions for source '{source_name}'.")
        return deposits_data_unfiltered

    def fetch_withdrawal_history(self, days_back: int = 90) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch withdrawal history.")
            return []

        withdrawals_data_unfiltered = []
        norm_map = self.config.get("symbol_normalization_map", {})
        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms_deposit", 1500))

        source_name = "Binance Withdrawal"
        latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        endTime = int(now_utc.timestamp() * 1000)
        startTime = 0

        specific_lookback_days = self.config.get("history_lookback_days", {}).get("withdrawals", days_back)

        if latest_known_ts:
            start_datetime_obj = latest_known_ts - datetime.timedelta(minutes=5)
            startTime = int(start_datetime_obj.timestamp() * 1000)
            logger.info(f"Selective sync for '{source_name}': Fetching since ~{start_datetime_obj}")
        else:
            start_datetime_obj = now_utc - datetime.timedelta(days=specific_lookback_days)
            startTime = int(start_datetime_obj.timestamp() * 1000)
            logger.info(f"Full sync for '{source_name}': Fetching last {specific_lookback_days} days.")

        if startTime >= endTime: # Ensure startTime is not after endTime
            logger.info(f"'{source_name}' history is very recent. Setting startTime to endTime-1min.")
            startTime = endTime - (60 * 1000)

        try:
            logger.info(f"Fetching withdrawal history (startTime: {pd.to_datetime(startTime, unit='ms', utc=True)}, endTime: {pd.to_datetime(endTime, unit='ms', utc=True)})...")
            all_withdrawals = self.binance_client.get_withdraw_history(startTime=startTime, endTime=endTime, status=6)
            logger.info(f"Fetched {len(all_withdrawals)} completed withdrawal records (startTime: {startTime}, endTime: {endTime}, pre-filter).")

            for i, withdrawal in enumerate(all_withdrawals):
                logger.debug(f"Processing raw withdrawal record {i+1}/{len(all_withdrawals)}: {withdrawal}")
                symbol_original = withdrawal.get('coin')
                if not symbol_original:
                    logger.warning(f"Withdrawal record {i+1} missing 'coin' field. Skipping. Data: {withdrawal}")
                    continue

                normalized_symbol = norm_map.get(symbol_original.upper(), symbol_original.upper())

                if normalized_symbol not in self.target_assets_for_sync:
                    # logger.debug(f"Skipping withdrawal for non-target asset: {normalized_symbol}")
                    continue

                apply_time_raw = withdrawal.get('applyTime') # Using applyTime as it reflects initiation
                if not apply_time_raw:
                    logger.warning(f"Withdrawal for {normalized_symbol} (TxID: {withdrawal.get('txId')}) has missing 'applyTime'. Skipping.")
                    continue

                try:
                    # applyTime can be a string like "2021-09-15 10:00:00" or sometimes a timestamp
                    withdrawal_timestamp_obj_pandas = pd.to_datetime(apply_time_raw, utc=True)
                except (ValueError, TypeError):
                    try:
                        withdrawal_timestamp_obj_pandas = pd.to_datetime(int(apply_time_raw), unit='ms', utc=True)
                    except (ValueError, TypeError):
                        logger.error(f"Could not parse applyTime '{apply_time_raw}' for {normalized_symbol} (TxID: {withdrawal.get('txId')}). Skipping.")
                        continue

                if pd.isna(withdrawal_timestamp_obj_pandas):
                    logger.error(f"Parsed withdrawal timestamp is NaT for {normalized_symbol} (TxID: {withdrawal.get('txId')}). Skipping.")
                    continue
                withdrawal_timestamp_py = withdrawal_timestamp_obj_pandas.to_pydatetime()

                tx_hash = withdrawal.get('txId', f"binance_withdraw_{withdrawal.get('id', i)}_{apply_time_raw}")
                if latest_known_ts and withdrawal_timestamp_py <= latest_known_ts:
                    # logger.debug(f"Skipping withdrawal {tx_hash} as its timestamp ({withdrawal_timestamp_py}) is not newer than last known ({latest_known_ts})")
                    continue

                fee_quantity = float(withdrawal.get('transactionFee', 0.0))
                fee_currency_api = withdrawal.get('network') # For withdrawals, fee is often implied by network, or is the asset itself
                                                         # The 'commissionAsset' field is not typical for withdrawal history.
                                                         # Assuming fee is paid in the withdrawn asset.
                fee_currency = normalized_symbol


                price_usd_at_withdrawal = 0.0
                fee_usd_at_withdrawal = 0.0
                coingecko_called = False

                if normalized_symbol in self.stablecoin_symbols:
                    price_usd_at_withdrawal = 1.0
                    fee_usd_at_withdrawal = fee_quantity * 1.0
                else:
                    coin_id = self.symbol_mappings.get(normalized_symbol)
                    if coin_id:
                        date_str = withdrawal_timestamp_obj_pandas.strftime('%d-%m-%Y')
                        historical_price = self._get_coingecko_historical_price(coin_id, date_str)
                        coingecko_called = True
                        if historical_price is not None:
                            price_usd_at_withdrawal = historical_price
                            fee_usd_at_withdrawal = fee_quantity * historical_price
                            logger.info(f"Fetched historical price for {normalized_symbol} (withdrawal) on {date_str}: ${price_usd_at_withdrawal:.6f}")
                        else:
                            logger.warning(f"Could not fetch historical price for {normalized_symbol} (withdrawal) on {date_str}. Price/Fee USD for withdrawal will be $0.")
                    else:
                        logger.warning(f"No CoinGecko ID for {normalized_symbol} (withdrawal). Price/Fee USD for withdrawal will be $0.")

                withdrawals_data_unfiltered.append({
                    "symbol": normalized_symbol,
                    "timestamp": withdrawal_timestamp_py,
                    "type": "WITHDRAWAL",
                    "quantity": float(withdrawal.get('amount', 0.0)),
                    "price_usd": price_usd_at_withdrawal,
                    "fee_quantity": fee_quantity,
                    "fee_currency": fee_currency,
                    "fee_usd": fee_usd_at_withdrawal,
                    "source": source_name,
                    "transaction_hash": tx_hash,
                    "notes": f"Network: {withdrawal.get('network', 'N/A')}, Address: {withdrawal.get('address', 'N/A')}, Fee: {fee_quantity} {fee_currency}"
                })
                if coingecko_called and price_usd_at_withdrawal > 0.0:
                    time.sleep(cg_delay_ms / 1000.0)

        except BinanceAPIException as e:
            logger.error(f"API Error fetching withdrawal history: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching withdrawal history: {e}", exc_info=True)

        logger.info(f"Fetched {len(withdrawals_data_unfiltered)} targeted withdrawal transactions for source '{source_name}'.")
        return withdrawals_data_unfiltered

    def fetch_p2p_usdt_buys(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch P2P history.")
            return []

        all_p2p_transactions = []
        p2p_fiat_currency = self.config.get("portfolio", {}).get("p2p_fiat_currency", "PHP").upper()
        target_usd_currency = "USD"
        source_name = "Binance P2P Buy"

        specific_lookback_days = self.config.get("history_lookback_days", {}).get("p2p_buys", days_back if days_back is not None else 90)

        latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name)
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        overall_start_dt_for_lookback: datetime.datetime
        if latest_known_ts:
            effective_start_from_db = latest_known_ts - datetime.timedelta(minutes=60)
            fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
            overall_start_dt_for_lookback = max(effective_start_from_db, fallback_start_from_days)
            logger.info(f"Selective sync for '{source_name}': Effective start date {overall_start_dt_for_lookback.strftime('%Y-%m-%d %H:%M:%S %Z')} (last known: {latest_known_ts.strftime('%Y-%m-%d %H:%M:%S %Z') if latest_known_ts else 'None'})")
        else:
            overall_start_dt_for_lookback = now_utc - datetime.timedelta(days=specific_lookback_days)
            logger.info(f"Full sync for '{source_name}': Fetching last {specific_lookback_days} days from {overall_start_dt_for_lookback.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if overall_start_dt_for_lookback >= now_utc:
            logger.info(f"'{source_name}' history is very recent. No new data to fetch.")
            return []

        current_batch_end_dt = now_utc # Start batching from now, going backwards
        rows_per_page = 50
        binance_api_delay_ms = self.binance_api_config.get("request_delay_ms", 700)
        max_retries = self.config.get("apis",{}).get("binance",{}).get("max_retries_per_batch", 3)
        retry_delay_seconds = self.config.get("apis",{}).get("binance",{}).get("retry_delay_sec", 15)


        logger.info(f"Fetching P2P USDT buy history against fiat: {p2p_fiat_currency} from {overall_start_dt_for_lookback.date()} to {current_batch_end_dt.date()} (batched).")

        while current_batch_end_dt > overall_start_dt_for_lookback:
            # Determine the start of the current batch (max 30 days prior or overall_start_dt_for_lookback)
            current_batch_start_dt = current_batch_end_dt - datetime.timedelta(days=30)
            if current_batch_start_dt < overall_start_dt_for_lookback:
                current_batch_start_dt = overall_start_dt_for_lookback

            start_ms = int(current_batch_start_dt.timestamp() * 1000)
            end_ms = int(current_batch_end_dt.timestamp() * 1000)

            if start_ms >= end_ms:
                logger.debug(f"P2P fetch: Batch start_ms {start_ms} is not before end_ms {end_ms}. Ending batch processing for this period.")
                break

            current_page = 1
            max_pages_to_try_per_batch = 20

            while current_page <= max_pages_to_try_per_batch:
                logger.debug(f"Fetching P2P BUY for {p2p_fiat_currency}/USDT. Period: {current_batch_start_dt.date()} to {current_batch_end_dt.date()}, Page: {current_page}")
                try:
                    history = self.binance_client.get_c2c_trade_history(
                        tradeType='BUY', page=current_page, rows=rows_per_page,
                        startTimestamp=start_ms, endTimestamp=end_ms
                    )
                    if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)

                    trades_in_page = history.get('data', [])
                    if not history or not trades_in_page:
                        logger.debug(f"No P2P BUY data on page {current_page} for this batch, or end of data for this period.")
                        break # Exit this page loop, move to next older batch

                    new_transactions_in_page = 0
                    for trade in trades_in_page:
                        create_time_ms = trade.get('createTime')
                        if not create_time_ms: continue
                        timestamp_dt = pd.to_datetime(create_time_ms, unit='ms', utc=True).to_pydatetime()

                        if timestamp_dt < overall_start_dt_for_lookback:
                            continue

                        if latest_known_ts and timestamp_dt <= latest_known_ts:
                            continue

                        if trade.get('asset', '').upper() == 'USDT' and trade.get('fiat', '').upper() == p2p_fiat_currency:
                            order_number = trade.get('orderNumber')
                            try:
                                usdt_quantity = float(trade.get('amount', 0))
                                fiat_amount_paid = float(trade.get('totalPrice', 0))
                            except (ValueError, TypeError) as e_parse:
                                logger.error(f"Could not parse P2P amount/totalPrice for order {order_number}: {e_parse}. Trade: {trade}")
                                continue
                            if usdt_quantity <= 0 or fiat_amount_paid <= 0: continue

                            trade_date_str_yyyy_mm_dd = timestamp_dt.strftime('%Y-%m-%d')
                            fiat_to_usd_rate = self._get_historical_fiat_exchange_rate(
                                trade_date_str_yyyy_mm_dd, p2p_fiat_currency, target_usd_currency
                            )
                            actual_usdt_price_in_usd = 1.0
                            notes_detail = ""
                            if fiat_to_usd_rate is not None and fiat_to_usd_rate > 0:
                                usd_equivalent_of_fiat_paid = fiat_amount_paid * fiat_to_usd_rate
                                if usdt_quantity > 0:
                                    actual_usdt_price_in_usd = usd_equivalent_of_fiat_paid / usdt_quantity
                                notes_detail = (f"Fiat paid: {fiat_amount_paid:.2f} {p2p_fiat_currency}. "
                                                f"Rate: 1 {p2p_fiat_currency} = {fiat_to_usd_rate:.6f} {target_usd_currency}. "
                                                f"USD Cost: ${usd_equivalent_of_fiat_paid:.2f} for {usdt_quantity:.2f} USDT. "
                                                f"Effective USDT/USD Price: ${actual_usdt_price_in_usd:.6f}.")
                                logger.info(f"P2P Trade {order_number}: {notes_detail}")
                            else:
                                logger.warning(f"P2P Trade {order_number}: Could not get {p2p_fiat_currency}/{target_usd_currency} rate for {trade_date_str_yyyy_mm_dd}. Using placeholder $1.00/USDT cost.")
                                notes_detail = f"Fiat paid: {fiat_amount_paid:.2f} {p2p_fiat_currency}. USD rate lookup failed. Using placeholder cost."
                            notes = f"P2P Buy. OrderNo: {order_number}. {notes_detail}"

                            all_p2p_transactions.append({
                                "symbol": "USDT", "timestamp": timestamp_dt, "type": "BUY",
                                "quantity": usdt_quantity, "price_usd": actual_usdt_price_in_usd,
                                "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                                "source": source_name, "transaction_hash": str(order_number), "notes": notes,
                            })
                            new_transactions_in_page +=1

                    logger.debug(f"Added {new_transactions_in_page} new P2P transactions from page {current_page}.")
                    current_page += 1
                    if history.get('total') is not None and ((current_page -1) * rows_per_page >= int(history.get('total', 0))):
                        logger.debug(f"Fetched all P2P records ({history.get('total')}) for this batch based on API total count.")
                        break

                except BinanceAPIException as e_binance:
                    logger.error(f"Binance API Error fetching P2P history (Batch {current_batch_start_dt.date()}-{current_batch_end_dt.date()}, Page {current_page}): {e_binance}")
                    break
                except Exception as e_generic:
                    logger.error(f"Unexpected error fetching P2P history (Batch {current_batch_start_time_dt.date()}-{current_batch_end_dt.date()}, Page {current_page}): {e_generic}", exc_info=True)
                    break
            # End of 'while current_page <= max_pages_to_try_per_batch:' loop

            # *** CORRECTED LINE FOR NEXT BATCH ITERATION ***
            current_batch_end_dt = current_batch_start_dt

            if current_batch_end_dt <= overall_start_dt_for_lookback:
                 logger.info(f"P2P fetching has processed batches up to or before the overall start date limit. Stopping.")
                 break
        # End of 'while current_batch_end_dt > overall_start_dt_for_lookback:' loop

        logger.info(f"Fetched a total of {len(all_p2p_transactions)} new P2P USDT buy transactions against {p2p_fiat_currency}.")
        return all_p2p_transactions

    def fetch_internal_transfers(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch internal transfers.")
            return []

        all_processed_transfers = []
        config_portfolio = self.config.get("portfolio", {})
        config_apis_binance = self.config.get("apis", {}).get("binance", {})
        config_apis_coingecko = self.config.get("apis", {}).get("coingecko", {})

        default_assets_to_check = ['USDT']
        assets_to_check_list = config_portfolio.get("assets_for_internal_transfer_check", default_assets_to_check)
        # Filter by target_assets_for_sync, but always include USDT if it was in the original check list
        assets_to_check_for_transfers = [
            asset.upper() for asset in assets_to_check_list
            if asset.upper() in self.target_assets_for_sync or asset.upper() == "USDT"
        ]

        if not assets_to_check_for_transfers:
            logger.info("No target assets configured for internal transfer check that overlap with sync targets. Skipping.")
            return []
        # Ensure USDT is first if present, for consistent logging if nothing else
        if "USDT" in assets_to_check_for_transfers and assets_to_check_for_transfers[0] != "USDT":
            assets_to_check_for_transfers.insert(0, assets_to_check_for_transfers.pop(assets_to_check_for_transfers.index("USDT")))
        assets_to_check_for_transfers = sorted(list(set(assets_to_check_for_transfers))) # Unique and sorted


        batch_days = min(config_apis_binance.get("transfer_history_batch_days", 7), 7) # Max 7 for this API, ensure positive
        limit_per_page = 100
        max_retries = config_apis_binance.get("max_retries_per_batch", 3)
        retry_delay_seconds = config_apis_binance.get("retry_delay_sec", 30)
        cg_delay_ms = config_apis_coingecko.get("request_delay_ms_internal_transfer", config_apis_coingecko.get("request_delay_ms",10000))
        binance_api_delay_ms = config_apis_binance.get("request_delay_ms_transfer_history", config_apis_binance.get("request_delay_ms", 500))
        recv_window_ms = config_apis_binance.get("recv_window", 60000)

        specific_lookback_days = self.config.get("history_lookback_days", {}).get("internal_transfers", days_back if days_back is not None else 90)

        processed_tran_ids = set()
        endpoint_path = 'asset/transfer' # SAPI v1

        transfer_configs = [
            {"api_transferType_param": "FUNDING_MAIN", "flow_type": "DEPOSIT", "log_label": "Funding to Spot", "source_name": "Binance API FUNDING_MAIN (Asset Transfer)"},
            {"api_transferType_param": "MAIN_FUNDING", "flow_type": "WITHDRAWAL", "log_label": "Spot to Funding", "source_name": "Binance API MAIN_FUNDING (Asset Transfer)"}
        ]

        now_utc = datetime.datetime.now(datetime.timezone.utc)

        for config_entry in transfer_configs:
            api_transfer_type = config_entry["api_transferType_param"]
            transaction_flow_type = config_entry["flow_type"]
            log_label = config_entry["log_label"]
            source_name_current_type = config_entry["source_name"]

            latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name_current_type)
            overall_start_time_dt_for_type: datetime.datetime
            if latest_known_ts:
                effective_start_from_db = latest_known_ts - datetime.timedelta(minutes=10) # Small buffer
                fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
                overall_start_time_dt_for_type = max(effective_start_from_db, fallback_start_from_days)
                logger.info(f"Selective sync for Internal Transfers '{source_name_current_type}': Effective start date {overall_start_time_dt_for_type.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                overall_start_time_dt_for_type = now_utc - datetime.timedelta(days=specific_lookback_days)
                logger.info(f"Full sync for Internal Transfers '{source_name_current_type}': Fetching last {specific_lookback_days} days from {overall_start_time_dt_for_type.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            if overall_start_time_dt_for_type >= now_utc:
                logger.info(f"'{source_name_current_type}' history is very recent. Skipping fetch for this type.")
                continue

            logger.info(f"Fetching {log_label} ({api_transfer_type}) for assets: {assets_to_check_for_transfers} from {overall_start_time_dt_for_type.date()}.")

            for asset_symbol_for_api in assets_to_check_for_transfers:
                logger.info(f"Processing asset for {api_transfer_type} transfers: {asset_symbol_for_api}")
                asset_specific_transfers_for_current_asset_type = []
                current_batch_end_time_dt = now_utc

                while current_batch_end_time_dt > overall_start_time_dt_for_type:
                    current_batch_start_time_dt = current_batch_end_time_dt - datetime.timedelta(days=batch_days)
                    if current_batch_start_time_dt < overall_start_time_dt_for_type:
                        current_batch_start_time_dt = overall_start_time_dt_for_type

                    start_ms = int(current_batch_start_time_dt.timestamp() * 1000)
                    end_ms = int(current_batch_end_time_dt.timestamp() * 1000)

                    if start_ms >= end_ms: break

                    logger.debug(f"Fetching {api_transfer_type} for {asset_symbol_for_api}: Batch Period {current_batch_start_time_dt.strftime('%Y-%m-%d %H:%M')} to {current_batch_end_time_dt.strftime('%Y-%m-%d %H:%M')}")
                    current_page_api = 1; fetched_all_for_batch = False
                    while not fetched_all_for_batch:
                        fetched_rows_in_page = None; api_response_total_count = 0
                        for attempt in range(max_retries):
                            try:
                                params = {
                                    "type": api_transfer_type, "asset": asset_symbol_for_api,
                                    "startTime": start_ms, "endTime": end_ms,
                                    "current": current_page_api, "size": limit_per_page,
                                    "recvWindow": recv_window_ms
                                }
                                logger.debug(f"API Call ({api_transfer_type} page {current_page_api}) attempt {attempt+1} for {asset_symbol_for_api}. EP: '{endpoint_path}', Params: {params}")
                                transfer_history = self.binance_client._request_margin_api('get', endpoint_path, True, data=params) # SAPI is fine
                                fetched_rows_in_page = transfer_history.get('rows', [])
                                api_response_total_count = transfer_history.get('total', 0)
                                logger.debug(f"Raw API response for {api_transfer_type} page {current_page_api} {asset_symbol_for_api} (attempt {attempt+1}): Fetched {len(fetched_rows_in_page) if fetched_rows_in_page is not None else 'None'} rows. API 'total': {api_response_total_count}. Sample: {str(transfer_history)[:300]}...")
                                if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                                break
                            except Exception as e_api_internal:
                                logger.error(f"API/Net Error attempt {attempt+1} for {api_transfer_type} {asset_symbol_for_api} page {current_page_api}: {e_api_internal}. Code: {getattr(e_api_internal, 'code', 'N/A')}")
                                if attempt < max_retries - 1: time.sleep(retry_delay_seconds)
                                else: logger.error(f"Max retries for {api_transfer_type} {asset_symbol_for_api} page {current_page_api}."); fetched_all_for_batch = True

                        if fetched_rows_in_page:
                            for item in fetched_rows_in_page:
                                timestamp_ms = item.get('timestamp')
                                if not timestamp_ms: continue
                                timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True).to_pydatetime()

                                if latest_known_ts and timestamp <= latest_known_ts:
                                    continue

                                tran_id_val = item.get('tranId')
                                if tran_id_val is None: # tranId can be integer or string, ensure it's a string for set
                                    tran_id = f"{api_transfer_type}_pg{current_page_api}_{item.get('asset')}_{timestamp_ms}" # Fallback unique ID
                                else:
                                    tran_id = str(tran_id_val)

                                if tran_id in processed_tran_ids: continue
                                processed_tran_ids.add(tran_id)

                                normalized_symbol = self.norm_map.get(item['asset'].upper(), item['asset'].upper())
                                quantity = float(item['amount'])
                                price_at_transfer = 1.0 # Default for USDT or if price fetch fails
                                if normalized_symbol not in self.stablecoin_symbols:
                                    coin_id_to_fetch = self.symbol_mappings.get(normalized_symbol)
                                    if coin_id_to_fetch:
                                        fetched_price = self._get_coingecko_historical_price(coin_id_to_fetch, timestamp.strftime('%d-%m-%Y'))
                                        if fetched_price is not None: price_at_transfer = fetched_price
                                        else: price_at_transfer = 0.0 # Mark as 0 if fetch failed
                                        if cg_delay_ms > 0 and fetched_price is not None: time.sleep(cg_delay_ms / 1000.0)
                                    else: price_at_transfer = 0.0 # No mapping

                                asset_specific_transfers_for_current_asset_type.append({
                                    "symbol": normalized_symbol, "timestamp": timestamp, "type": transaction_flow_type,
                                    "quantity": quantity, "price_usd": price_at_transfer,
                                    "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                                    "source": source_name_current_type, "transaction_hash": tran_id,
                                    "notes": f"{log_label} ({item['asset']}): {quantity:.8f} {normalized_symbol}"
                                })
                            if not fetched_rows_in_page or (api_response_total_count > 0 and current_page_api * limit_per_page >= api_response_total_count) or len(fetched_rows_in_page) < limit_per_page :
                                fetched_all_for_batch = True
                            else: current_page_api += 1
                        else: fetched_all_for_batch = True # No rows or error after retries

                        if fetched_all_for_batch:
                            logger.debug(f"Completed batch for {asset_symbol_for_api} ({api_transfer_type}).")
                            break

                    current_batch_end_time_dt = current_batch_start_time_dt - datetime.timedelta(milliseconds=1)
                    # Removed general sleep, it's now after successful API call.
                    if current_batch_end_time_dt <= overall_start_time_dt_for_type: break

                if asset_specific_transfers_for_current_asset_type:
                    all_processed_transfers.extend(asset_specific_transfers_for_current_asset_type)
                    logger.info(f"Added {len(asset_specific_transfers_for_current_asset_type)} new transfers for {asset_symbol_for_api} ({api_transfer_type}).")

                if asset_symbol_for_api != assets_to_check_for_transfers[-1] and binance_api_delay_ms > 0:
                    time.sleep(binance_api_delay_ms / 1000.0) # Delay between assets if checking multiple

            if config_entry != transfer_configs[-1] and binance_api_delay_ms > 0:
                time.sleep(binance_api_delay_ms * 2 / 1000.0) # Longer delay between FUNDING_MAIN and MAIN_FUNDING runs

        logger.info(f"Fetched a total of {len(all_processed_transfers)} new internal transfers after selective sync.")
        return all_processed_transfers

    def fetch_spot_futures_transfers(self, asset: str = "USDT", days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        asset_upper = asset.upper()

        specific_lookback_days = self.config.get("history_lookback_days", {}).get(
            "spot_futures_transfers", days_back if days_back is not None else 90
        )

        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Spot-Futures transfer history.")
            return []

        all_spot_futures_transfers = []
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        transfer_configs = [
            {"api_call_type_param_value": "MAIN_UMFUTURE", "spot_equivalent_action": "SELL", "log_label": "Spot to USD-M Futures", "source_name": "Binance API MAIN_UMFUTURE"},
            {"api_call_type_param_value": "UMFUTURE_MAIN", "spot_equivalent_action": "BUY", "log_label": "USD-M Futures to Spot", "source_name": "Binance API UMFUTURE_MAIN"}
        ]

        config_apis_binance = self.config.get("apis", {}).get("binance", {})
        batch_days = min(config_apis_binance.get("transfer_history_batch_days", 7), 7) # Max 7 for this API usually, ensure positive
        limit_per_page = 100
        max_retries = config_apis_binance.get("max_retries_per_batch", 3)
        retry_delay_seconds = config_apis_binance.get("retry_delay_sec", 30)
        binance_api_delay_ms = config_apis_binance.get("request_delay_ms_transfer_history", config_apis_binance.get("request_delay_ms", 500))
        recv_window_ms = config_apis_binance.get("recv_window", 60000)
        endpoint_path = 'asset/transfer'

        logger.info(f"Fetching Spot <-> USDⓈ-M Futures transfers for {asset_upper} (lookback: {specific_lookback_days} days), Batch size: {batch_days} days.")
        processed_tran_ids_spot_futures = set()

        for config_entry in transfer_configs:
            api_call_transfer_type_value = config_entry["api_call_type_param_value"]
            spot_action = config_entry["spot_equivalent_action"]
            log_label_detail = config_entry["log_label"]
            source_name_current_type = config_entry["source_name"]

            latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name_current_type)
            overall_start_time_dt_for_type: datetime.datetime
            if latest_known_ts:
                effective_start_from_db = latest_known_ts - datetime.timedelta(minutes=10)
                fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
                overall_start_time_dt_for_type = max(effective_start_from_db, fallback_start_from_days)
                logger.info(f"Selective sync for '{source_name_current_type}' ({asset_upper}): Start date {overall_start_time_dt_for_type.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                overall_start_time_dt_for_type = now_utc - datetime.timedelta(days=specific_lookback_days)
                logger.info(f"Full sync for '{source_name_current_type}' ({asset_upper}): Fetching last {specific_lookback_days} days from {overall_start_time_dt_for_type.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            if overall_start_time_dt_for_type >= now_utc:
                logger.info(f"'{source_name_current_type}' ({asset_upper}) history is very recent. Skipping.")
                continue

            logger.info(f"Fetching {log_label_detail} ({api_call_transfer_type_value}) for {asset_upper} (Spot action: {spot_action}) from {overall_start_time_dt_for_type.date()}.")
            current_batch_end_time_dt = now_utc

            while current_batch_end_time_dt > overall_start_time_dt_for_type:
                current_batch_start_time_dt = current_batch_end_time_dt - datetime.timedelta(days=batch_days)
                if current_batch_start_time_dt < overall_start_time_dt_for_type:
                    current_batch_start_time_dt = overall_start_time_dt_for_type
                start_ms = int(current_batch_start_time_dt.timestamp() * 1000)
                end_ms = int(current_batch_end_time_dt.timestamp() * 1000)
                if start_ms >= end_ms: break

                logger.debug(f"Fetching {api_call_transfer_type_value} for {asset_upper}: Batch {current_batch_start_time_dt.strftime('%Y-%m-%d %H:%M')} to {current_batch_end_time_dt.strftime('%Y-%m-%d %H:%M')}")
                current_page_api = 1; fetched_all_for_this_batch_asset_type = False
                while not fetched_all_for_this_batch_asset_type:
                    fetched_rows_in_page = None; api_response_total_count = 0
                    for attempt in range(max_retries):
                        try:
                            params = {
                                "type": api_call_transfer_type_value, "asset": asset_upper,
                                "startTime": start_ms, "endTime": end_ms,
                                "current": current_page_api, "size": limit_per_page,
                                "recvWindow": recv_window_ms,
                            }
                            logger.debug(f"API Call ({api_call_transfer_type_value} page {current_page_api}) attempt {attempt+1} for {asset_upper}. EP: '{endpoint_path}', Params: {params}")
                            history_page = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                            fetched_rows_in_page = history_page.get('rows', [])
                            api_response_total_count = history_page.get('total', 0)
                            logger.debug(f"Raw API response for {api_call_transfer_type_value} page {current_page_api} {asset_upper} (attempt {attempt+1}): Fetched {len(fetched_rows_in_page) if fetched_rows_in_page is not None else 'None'} rows. API 'total': {api_response_total_count}. Sample: {str(history_page)[:300]}...")
                            if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                            break
                        except Exception as e_api_sf:
                            logger.error(f"API/Net Error attempt {attempt+1} for {api_call_transfer_type_value} {asset_upper} page {current_page_api}: {e_api_sf}. Code: {getattr(e_api_sf, 'code', 'N/A')}")
                            if attempt < max_retries - 1: time.sleep(retry_delay_seconds)
                            else: logger.error(f"Max retries for {api_call_transfer_type_value} {asset_upper} page {current_page_api}."); fetched_all_for_this_batch_asset_type = True

                    if fetched_rows_in_page:
                        for t in fetched_rows_in_page:
                            timestamp_ms = t.get('timestamp')
                            if not timestamp_ms: continue
                            tx_timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True).to_pydatetime()

                            if latest_known_ts and tx_timestamp <= latest_known_ts:
                                continue

                            tran_id_val = t.get('tranId')
                            if tran_id_val is None: tran_id = f"{api_call_transfer_type_value.lower()}_{asset_upper.lower()}_{timestamp_ms}_{current_page_api}"
                            else: tran_id = str(tran_id_val)

                            if tran_id in processed_tran_ids_spot_futures: continue
                            processed_tran_ids_spot_futures.add(tran_id)

                            quantity = float(t['amount'])
                            price_usd = 1.0 # Default for USDT
                            if asset_upper != "USDT":
                                logger.warning(f"Spot-Futures transfer of non-USDT asset {asset_upper} - price lookup might be needed and is not implemented here yet for this transfer type.")
                                # If this asset is a target non-stablecoin, you might want to price it here too.
                                # For simplicity, assuming only USDT is transferred or other assets are valued at $0 for these transfers.

                            all_spot_futures_transfers.append({
                                "symbol": asset_upper, "timestamp": tx_timestamp, "type": spot_action,
                                "quantity": quantity, "price_usd": price_usd,
                                "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                                "source": source_name_current_type, "transaction_hash": tran_id,
                                "notes": f"{log_label_detail} of {quantity:.8f} {asset_upper}. TxID: {t.get('tranId','N/A')}"
                            })
                        if not fetched_rows_in_page or (api_response_total_count > 0 and current_page_api * limit_per_page >= api_response_total_count) or len(fetched_rows_in_page) < limit_per_page:
                            fetched_all_for_this_batch_asset_type = True
                        else: current_page_api += 1
                    else: fetched_all_for_this_batch_asset_type = True

                    if fetched_all_for_this_batch_asset_type:
                        logger.debug(f"Completed fetching pages for batch {current_batch_start_time_dt.strftime('%Y-%m-%d')} to {current_batch_end_time_dt.strftime('%Y-%m-%d')} for {asset_upper} ({api_call_transfer_type_value}).")
                        break
                current_batch_end_time_dt = current_batch_start_time_dt - datetime.timedelta(milliseconds=1)
                # Delay moved into the API call loop attempt
                if current_batch_end_time_dt <= overall_start_time_dt_for_type: break

            if config_entry != transfer_configs[-1] and binance_api_delay_ms > 0:
                 time.sleep(binance_api_delay_ms * 2 / 1000.0) # Delay between MAIN_UMFUTURE and UMFUTURE_MAIN runs

        logger.info(f"Fetched a total of {len(all_spot_futures_transfers)} new Spot <-> USDⓈ-M Futures transfer records processed for {asset_upper}.")
        return all_spot_futures_transfers

    def fetch_spot_convert_history(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Spot Convert History.")
            return []

        source_name = "Binance API Spot Convert" # Define source name
        latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        specific_lookback_days = self.config.get("history_lookback_days", {}).get("spot_convert_history", days_back if days_back is not None else 90)

        overall_start_time_dt = None
        if latest_known_ts:
            effective_start_from_db = latest_known_ts - datetime.timedelta(minutes=60) # 1 hour buffer
            fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
            overall_start_time_dt = max(effective_start_from_db, fallback_start_from_days)
            logger.info(f"Selective sync for '{source_name}': Effective start date {overall_start_time_dt}")
        else:
            overall_start_time_dt = now_utc - datetime.timedelta(days=specific_lookback_days)
            logger.info(f"Full sync for '{source_name}': Fetching last {specific_lookback_days} days from {overall_start_time_dt}")

        if overall_start_time_dt >= now_utc:
            logger.info(f"'{source_name}' history is very recent. No new data to fetch.")
            return []

        logger.info(f"Fetching Spot Convert History from {overall_start_time_dt.date()} to {now_utc.date()}.")
        all_convert_transactions_filtered = []

        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms", 1500))
        binance_api_delay_ms = self.config.get("apis",{}).get("binance",{}).get("request_delay_ms", 250)

        batch_days = 29 # Convert API can take up to 30 days range, use 29 for safety
        limit_per_batch = 1000 # Max limit for this endpoint
        max_retries = self.config.get("apis", {}).get("binance", {}).get("max_retries_per_batch", 3)
        retry_delay_seconds = self.config.get("apis", {}).get("binance", {}).get("retry_delay_sec", 15)

        current_iteration_end_time_dt = now_utc
        processed_quote_ids = set() # To avoid double processing if a quote_id appears across fetches (unlikely with ON CONFLICT)

        while current_iteration_end_time_dt > overall_start_time_dt:
            current_iteration_start_time_dt = current_iteration_end_time_dt - datetime.timedelta(days=batch_days)
            if current_iteration_start_time_dt < overall_start_time_dt:
                current_iteration_start_time_dt = overall_start_time_dt

            start_ms = int(current_iteration_start_time_dt.timestamp() * 1000)
            end_ms = int(current_iteration_end_time_dt.timestamp() * 1000)

            if start_ms >= end_ms: break

            logger.debug(f"Fetching Spot Convert History batch: {current_iteration_start_time_dt.strftime('%Y-%m-%d %H:%M')} to {current_iteration_end_time_dt.strftime('%Y-%m-%d %H:%M')}")
            current_batch_data = None
            for attempt in range(max_retries):
                try:
                    history_page = self.binance_client.get_convert_trade_history(
                        startTime=start_ms, endTime=end_ms, limit=limit_per_batch
                    )
                    current_batch_data = history_page.get('list', [])
                    logger.debug(f"Successfully fetched Spot Convert History batch (attempt {attempt+1}). Count: {len(current_batch_data) if current_batch_data else 0}")
                    if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                    break
                except Exception as e:
                    logger.error(f"Error fetching Spot Convert History batch (attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1: time.sleep(retry_delay_seconds)
                    else: logger.error("Max retries for Spot Convert History batch."); break

            if current_batch_data:
                for trade in current_batch_data:
                    timestamp = pd.to_datetime(trade['createTime'], unit='ms', utc=True).to_pydatetime()
                    if latest_known_ts and timestamp <= latest_known_ts:
                        # logger.debug(f"Skipping already processed convert trade {trade.get('quoteId')} based on timestamp")
                        continue

                    quote_id = trade.get('quoteId', trade.get('orderId', f"convert_{trade['fromAsset']}_{trade['toAsset']}_{trade['createTime']}"))
                    if quote_id in processed_quote_ids: continue # Skip if already processed in this run
                    processed_quote_ids.add(quote_id)

                    if trade.get('orderStatus') == "SUCCESS":
                        # ... (rest of your existing logic for processing successful convert trades)
                        # Ensure "source": source_name is used
                        from_asset_orig = trade['fromAsset'].upper()
                        from_asset = self.norm_map.get(from_asset_orig, from_asset_orig)
                        from_amount = float(trade['fromAmount'])
                        to_asset_orig = trade['toAsset'].upper()
                        to_asset = self.norm_map.get(to_asset_orig, to_asset_orig)
                        to_amount = float(trade['toAmount'])

                        if not (from_asset in self.target_assets_for_sync or to_asset in self.target_assets_for_sync):
                            # logger.debug(f"Skipping Spot Convert trade {quote_id} as neither {from_asset} nor {to_asset} are target assets.")
                            continue

                        price_ratio = float(trade.get('ratio', "0"))
                        sell_price_usd = 0.0; buy_price_usd = 0.0
                        cg_call_made_for_this_convert = False # Reset for each trade

                        if from_asset in self.stablecoin_symbols: sell_price_usd = 1.0
                        elif from_asset in self.target_assets_for_sync:
                            coin_id_from = self.symbol_mappings.get(from_asset)
                            if coin_id_from:
                                date_str = timestamp.strftime('%d-%m-%Y')
                                fetched_price = self._get_coingecko_historical_price(coin_id_from, date_str)
                                cg_call_made_for_this_convert = True
                                if fetched_price is not None: sell_price_usd = fetched_price
                                else: logger.warning(f"Spot Convert (Sell {from_asset}): Could not get hist price. Using $0.")
                            else: logger.warning(f"Spot Convert (Sell {from_asset}): No CoinGecko ID. Using $0.")

                        if to_asset in self.stablecoin_symbols: buy_price_usd = 1.0
                        elif to_asset in self.target_assets_for_sync:
                            coin_id_to = self.symbol_mappings.get(to_asset)
                            if coin_id_to:
                                date_str = timestamp.strftime('%d-%m-%Y')
                                fetched_price = self._get_coingecko_historical_price(coin_id_to, date_str)
                                if not cg_call_made_for_this_convert : cg_call_made_for_this_convert = True
                                elif cg_call_made_for_this_convert == True: cg_call_made_for_this_convert = "Done"

                                if fetched_price is not None: buy_price_usd = fetched_price
                                else: logger.warning(f"Spot Convert (Buy {to_asset}): Could not get hist price. Using $0.")
                            else: logger.warning(f"Spot Convert (Buy {to_asset}): No CoinGecko ID. Using $0.")

                        if from_asset in self.target_assets_for_sync:
                            all_convert_transactions_filtered.append({
                                "symbol": from_asset, "timestamp": timestamp, "type": "SELL",
                                "quantity": from_amount, "price_usd": sell_price_usd,
                                "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                                "source": source_name, "transaction_hash": f"convert_sell_{quote_id}",
                                "notes": f"Convert: {from_amount:.8f} {from_asset} to {to_amount:.8f} {to_asset} @ {price_ratio}"
                            })
                        if to_asset in self.target_assets_for_sync:
                            all_convert_transactions_filtered.append({
                                "symbol": to_asset, "timestamp": timestamp, "type": "BUY",
                                "quantity": to_amount, "price_usd": buy_price_usd,
                                "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                                "source": source_name, "transaction_hash": f"convert_buy_{quote_id}",
                                "notes": f"Convert: {to_amount:.8f} {to_asset} from {from_amount:.8f} {from_asset} @ {price_ratio}"
                            })
                        if cg_call_made_for_this_convert == True and (sell_price_usd > 0.0 or buy_price_usd > 0.0):
                            if cg_delay_ms > 0: time.sleep(cg_delay_ms / 1000.0)
                    else:
                        logger.info(f"Skipping non-SUCCESS Spot Convert trade: {quote_id} - Status: {trade.get('orderStatus')}")

            current_iteration_end_time_dt = current_iteration_start_time_dt - datetime.timedelta(milliseconds=1)
            # Removed the general Coingecko delay here; it's now per-priced item.
            if current_iteration_end_time_dt <= overall_start_time_dt: break

        logger.info(f"Processed {len(all_convert_transactions_filtered)} new targeted individual transactions from Spot Convert History API.")
        return all_convert_transactions_filtered

    def fetch_simple_earn_balances(self) -> Dict[str, float]:
        """Fetch current balances from Binance Simple Earn Flexible products, ensuring normalized base symbols."""
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Simple Earn balances.")
            return {}

        earn_balances_aggregated: Dict[str, float] = {}
        assets_in_config = list(self.symbol_mappings.keys())
        potential_earn_assets = set()
        for symbol_variant_config_key in assets_in_config:
            normalized_symbol = self.norm_map.get(symbol_variant_config_key.upper(), symbol_variant_config_key.upper())
            if not normalized_symbol.startswith('LD'):
                 potential_earn_assets.add(normalized_symbol)

        common_bases = ['BTC', 'ETH', 'USDT', 'SOL', 'PEPE', 'HMSTR', 'TAO', 'RENDER'] # Make sure USDT is here
        for base in common_bases:
            potential_earn_assets.add(base.upper())

        assets_to_check_api = sorted(list(potential_earn_assets))
        logger.info(f"Fetching Simple Earn Flexible balances for up to {len(assets_to_check_api)} potential base assets...")


        for asset_api_name in assets_to_check_api:
            try:
                logger.debug(f"Fetching Simple Earn position for {asset_api_name}...")
                # Ensure asset_api_name is not empty
                if not asset_api_name:
                    logger.debug(f"Skipping empty asset_api_name for Simple Earn fetch.")
                    continue
                positions = self.binance_client.get_simple_earn_flexible_product_position(asset=asset_api_name)

                if positions and isinstance(positions.get('rows'), list):
                    total_amount_for_asset = 0.0
                    for pos in positions['rows']:
                        try: total_amount_for_asset += float(pos.get('totalAmount', 0.0))
                        except (ValueError, TypeError): logger.warning(f"Could not parse 'totalAmount' for {asset_api_name}: {pos.get('totalAmount')}")

                    if total_amount_for_asset > 0:
                        current_bal = earn_balances_aggregated.get(asset_api_name, 0.0)
                        earn_balances_aggregated[asset_api_name] = current_bal + total_amount_for_asset
                        logger.info(f"Found {total_amount_for_asset:.8f} {asset_api_name} in Simple Earn.")
                time.sleep(0.3) # Be kind to the API
            except BinanceAPIException as e:
                if e.code == -6001 or "product does not exist" in str(e).lower() or "not supported" in str(e).lower() or "invalid asset" in str(e).lower(): # Added "invalid asset"
                     logger.debug(f"No Simple Earn for {asset_api_name} or asset not supported (API Info: {e}).")
                else: logger.error(f"API Error Simple Earn for {asset_api_name}: {e}")
            except Exception as e: logger.error(f"Unexpected error Simple Earn for {asset_api_name}: {e}", exc_info=True)
        logger.info(f"DEBUG: Balances from EARN after all processing in fetch_simple_earn_balances: {earn_balances_aggregated if earn_balances_aggregated else 'EMPTY DICT'}")
        logger.info(f"Fetched {len(earn_balances_aggregated)} asset balances from Simple Earn (after normalization).")
        return earn_balances_aggregated

    def fetch_simple_earn_flexible_rewards(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Simple Earn Flexible rewards.")
            return []

        source_name_prefix = "Binance API Simple Earn Reward"
        # We'll query for the latest timestamp using a wildcard to cover all reward types initially
        latest_known_ts = self.db_manager.get_latest_timestamp_for_source(f"{source_name_prefix} (%)")

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        specific_lookback_days = self.config.get("history_lookback_days", {}).get("simple_earn_rewards", days_back if days_back is not None else 90)

        overall_start_time_dt: datetime.datetime
        if latest_known_ts:
            effective_start_from_db = latest_known_ts - datetime.timedelta(hours=1) # 1 hour buffer for rewards
            fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
            overall_start_time_dt = max(effective_start_from_db, fallback_start_from_days)
            logger.info(f"Selective sync for '{source_name_prefix}': Effective start {overall_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} (last known relevant reward: {latest_known_ts.strftime('%Y-%m-%d %H:%M:%S %Z') if latest_known_ts else 'None'})")
        else:
            overall_start_time_dt = now_utc - datetime.timedelta(days=specific_lookback_days)
            logger.info(f"Full sync for '{source_name_prefix}': Fetching last {specific_lookback_days} days from {overall_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if overall_start_time_dt >= now_utc:
            logger.info(f"'{source_name_prefix}' history is very recent. No new data to fetch.")
            return []

        logger.info(f"Fetching Simple Earn Flexible rewards from {overall_start_time_dt.date()} (all types, filtered for target assets).")
        all_rewards_transactions = []

        config_apis_binance = self.config.get("apis", {}).get("binance", {})
        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms_csv", 1500))
        batch_days = config_apis_binance.get("transfer_history_batch_days", 7) # Consistent batching
        limit_per_page = 100
        max_retries = config_apis_binance.get("max_retries_per_batch", 2)
        retry_delay_seconds = config_apis_binance.get("retry_delay_sec", 30)
        binance_api_delay_ms = config_apis_binance.get("request_delay_ms", 250)
        recv_window_ms = config_apis_binance.get("recv_window", 10000)
        endpoint_path = 'simple-earn/flexible/history/rewardRecord'
        processed_reward_ids = set()
        current_batch_end_time_dt = now_utc

        while current_batch_end_time_dt > overall_start_time_dt:
            current_batch_start_time_dt = current_batch_end_time_dt - datetime.timedelta(days=batch_days)
            if current_batch_start_time_dt < overall_start_time_dt:
                current_batch_start_time_dt = overall_start_time_dt
            start_ms = int(current_batch_start_time_dt.timestamp() * 1000)
            end_ms = int(current_batch_end_time_dt.timestamp() * 1000)
            if start_ms >= end_ms: break

            logger.debug(f"Fetching SE Rewards: Batch Period {current_batch_start_time_dt.strftime('%Y-%m-%d %H:%M')} to {current_batch_end_time_dt.strftime('%Y-%m-%d %H:%M')}")
            current_page_api = 1; fetched_all_for_batch = False
            while not fetched_all_for_batch:
                response_data = None; fetched_rows = None; api_total = 0
                for attempt in range(max_retries):
                    try:
                        params = {"startTime": start_ms, "endTime": end_ms, "current": current_page_api, "size": limit_per_page, "recvWindow": recv_window_ms}
                        logger.debug(f"API Call (SE Rewards Pg: {current_page_api}) Att: {attempt + 1}. EP: '{endpoint_path}', Params: {params}")
                        response_data = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                        fetched_rows = response_data.get('rows', [])
                        api_total = response_data.get('total', 0)
                        logger.debug(f"Raw API (SE Rewards Pg: {current_page_api}) Att: {attempt + 1}: Fetched {len(fetched_rows) if fetched_rows else '0'} rows. API Total: {api_total}. Sample: {str(response_data)[:200]}...")
                        if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                        break
                    except Exception as e:
                        logger.error(f"Error (SE Rewards Pg: {current_page_api}) Att: {attempt + 1}: {e}. Code: {getattr(e, 'code', 'N/A')}")
                        if attempt < max_retries - 1: time.sleep(retry_delay_seconds)
                        else: fetched_all_for_batch = True

                if fetched_rows:
                    for item in fetched_rows:
                        timestamp_ms = item.get('time')
                        if not timestamp_ms: continue
                        timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True).to_pydatetime()

                        if latest_known_ts and timestamp <= latest_known_ts:
                            continue

                        asset_rewarded = item.get('asset','').upper()
                        normalized_symbol = self.norm_map.get(asset_rewarded, asset_rewarded)
                        if normalized_symbol not in self.target_assets_for_sync:
                            continue

                        reward_type_from_item = item.get('type', 'UNKNOWN_REWARD_TYPE')
                        reward_id_str = str(item.get('tranId', f"ser_{reward_type_from_item}_{asset_rewarded}_{timestamp_ms}_{item.get('rewards')}"))
                        if reward_id_str in processed_reward_ids: continue
                        processed_reward_ids.add(reward_id_str)

                        quantity = float(item.get('rewards', 0.0))
                        if quantity == 0: continue

                        current_item_source_name = f"{source_name_prefix} ({reward_type_from_item})"

                        price_usd_at_reward = 0.0
                        coingecko_called_for_item = False
                        if normalized_symbol in self.stablecoin_symbols:
                            price_usd_at_reward = 1.0
                        else:
                            coin_id = self.symbol_mappings.get(normalized_symbol)
                            if coin_id:
                                date_str_for_price = timestamp.strftime('%d-%m-%Y')
                                fetched_price = self._get_coingecko_historical_price(coin_id, date_str_for_price)
                                coingecko_called_for_item = True
                                if fetched_price is not None:
                                    price_usd_at_reward = fetched_price
                                    logger.debug(f"SE Reward: Fetched hist_price ${price_usd_at_reward:.6f} for {normalized_symbol} on {date_str_for_price}.")
                                else:
                                    logger.warning(f"SE Reward: Could not get hist price for {normalized_symbol} on {date_str_for_price}. Using $0.00.")
                            else:
                                logger.warning(f"SE Reward: No CoinGecko ID for {normalized_symbol}. Using $0.00.")

                        all_rewards_transactions.append({
                            "symbol": normalized_symbol, "timestamp": timestamp, "type": "DEPOSIT",
                            "quantity": quantity, "price_usd": price_usd_at_reward,
                            "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                            "source": current_item_source_name,
                            "transaction_hash": reward_id_str,
                            "notes": f"Simple Earn Reward: {reward_type_from_item} {quantity:.8f} {normalized_symbol}"
                        })
                        if coingecko_called_for_item and price_usd_at_reward > 0.0 and cg_delay_ms > 0:
                            time.sleep(cg_delay_ms / 1000.0)

                    if not fetched_rows or (api_total > 0 and current_page_api * limit_per_page >= api_total) or len(fetched_rows) < limit_per_page:
                        fetched_all_for_batch = True
                    else:
                        current_page_api += 1
                else: fetched_all_for_batch = True

                if fetched_all_for_batch:
                    logger.debug(f"Completed SE Rewards batch for {current_batch_start_time_dt.strftime('%Y-%m-%d %H:%M')} to {current_batch_end_time_dt.strftime('%Y-%m-%d %H:%M')}.")
                    break

            current_batch_end_time_dt = current_batch_start_time_dt - datetime.timedelta(milliseconds=1)
            if current_batch_end_time_dt <= overall_start_time_dt: break

        logger.info(f"Fetched and processed {len(all_rewards_transactions)} new Simple Earn Flexible reward transactions for target assets.")
        return all_rewards_transactions

    def fetch_simple_earn_flexible_subscriptions(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Simple Earn Flexible subscriptions.")
            return []

        source_name = "Binance API Simple Earn Subscription"
        latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        specific_lookback_days = self.config.get("history_lookback_days", {}).get("simple_earn_subscriptions", days_back if days_back is not None else 90)

        overall_start_time_dt: datetime.datetime
        if latest_known_ts:
            effective_start_from_db = latest_known_ts - datetime.timedelta(minutes=10)
            fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
            overall_start_time_dt = max(effective_start_from_db, fallback_start_from_days)
            logger.info(f"Selective sync for '{source_name}': Effective start date {overall_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            overall_start_time_dt = now_utc - datetime.timedelta(days=specific_lookback_days)
            logger.info(f"Full sync for '{source_name}': Fetching last {specific_lookback_days} days from {overall_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if overall_start_time_dt >= now_utc:
            logger.info(f"'{source_name}' history is very recent. No new data to fetch.")
            return []

        logger.info(f"Fetching Simple Earn Flexible subscriptions from {overall_start_time_dt.date()}.")
        all_subscription_transactions = []

        config_apis_binance = self.config.get("apis", {}).get("binance", {})
        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms_csv", 1500))
        batch_days = config_apis_binance.get("transfer_history_batch_days", 7)
        limit_per_page = 100
        max_retries = config_apis_binance.get("max_retries_per_batch", 2)
        retry_delay_seconds = config_apis_binance.get("retry_delay_sec", 30)
        binance_api_delay_ms = config_apis_binance.get("request_delay_ms", 250)
        recv_window_ms = config_apis_binance.get("recv_window", 10000)
        endpoint_path = 'simple-earn/flexible/history/subscriptionRecord'
        processed_ids = set()
        current_batch_end_time_dt = now_utc

        while current_batch_end_time_dt > overall_start_time_dt:
            current_batch_start_time_dt = current_batch_end_time_dt - datetime.timedelta(days=batch_days)
            if current_batch_start_time_dt < overall_start_time_dt:
                current_batch_start_time_dt = overall_start_time_dt
            start_ms = int(current_batch_start_time_dt.timestamp() * 1000)
            end_ms = int(current_batch_end_time_dt.timestamp() * 1000)
            if start_ms >= end_ms: break

            logger.debug(f"Fetching SE Subscriptions: Batch {current_batch_start_time_dt.strftime('%Y-%m-%d %H:%M')} to {current_batch_end_time_dt.strftime('%Y-%m-%d %H:%M')}")
            current_page_api = 1; fetched_all_for_batch = False
            while not fetched_all_for_batch:
                history_page = None; fetched_rows_in_page = None; api_total_for_this_query = 0
                for attempt in range(max_retries):
                    try:
                        params = {"startTime": start_ms, "endTime": end_ms, "current": current_page_api, "size": limit_per_page, "recvWindow": recv_window_ms}
                        logger.debug(f"API Call (SE Subs Pg: {current_page_api}) Att: {attempt + 1}. EP: '{endpoint_path}', Params: {params}")
                        history_page = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                        fetched_rows_in_page = history_page.get('rows', [])
                        api_total_for_this_query = history_page.get('total', 0)
                        logger.debug(f"Raw API (SE Subs Pg: {current_page_api}) Att: {attempt + 1}: Fetched {len(fetched_rows_in_page) if fetched_rows_in_page else '0'} rows. API total: {api_total_for_this_query}. Sample: {str(history_page)[:200]}...")
                        if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                        break
                    except Exception as e:
                        logger.error(f"API/Net Error attempt {attempt + 1} for SE Subs page {current_page_api}: {e}. Code: {getattr(e, 'code', 'N/A')}")
                        if attempt < max_retries - 1: time.sleep(retry_delay_seconds)
                        else: logger.error(f"Max retries for SE Subs page {current_page_api}."); fetched_all_for_batch = True

                if fetched_rows_in_page:
                    for item in fetched_rows_in_page:
                        timestamp_ms = item.get('time')
                        if not timestamp_ms: continue
                        timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True).to_pydatetime()

                        if latest_known_ts and timestamp <= latest_known_ts:
                            continue

                        purchase_id = str(item.get('purchaseId', f"ses_{item.get('asset')}_{timestamp_ms}_{item.get('amount')}"))
                        if purchase_id in processed_ids: continue
                        processed_ids.add(purchase_id)

                        asset_subscribed = item.get('asset','').upper()
                        normalized_symbol = self.norm_map.get(asset_subscribed, asset_subscribed)
                        if normalized_symbol not in self.target_assets_for_sync:
                            continue
                        quantity = float(item.get('amount', 0.0))
                        if quantity == 0: continue

                        price_usd = 0.0
                        coingecko_called_for_item = False
                        if normalized_symbol in self.stablecoin_symbols: price_usd = 1.0
                        else:
                            coin_id = self.symbol_mappings.get(normalized_symbol)
                            if coin_id:
                                date_str_for_price = timestamp.strftime('%d-%m-%Y')
                                fetched_price = self._get_coingecko_historical_price(coin_id, date_str_for_price)
                                coingecko_called_for_item = True
                                if fetched_price is not None:
                                    price_usd = fetched_price
                                    logger.debug(f"SE Sub: Fetched hist_price ${price_usd:.6f} for {normalized_symbol} on {date_str_for_price}.")
                                else: logger.warning(f"SE Sub: Could not get hist price for {normalized_symbol} on {date_str_for_price}. Using $0.00.")
                            else: logger.warning(f"SE Sub: No CoinGecko ID for {normalized_symbol}. Using $0.00.")

                        all_subscription_transactions.append({
                            "symbol": normalized_symbol, "timestamp": timestamp, "type": "SELL",
                            "quantity": quantity, "price_usd": price_usd,
                            "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                            "source": source_name, "transaction_hash": purchase_id,
                            "notes": f"Simple Earn Subscription: {quantity:.8f} {normalized_symbol}"
                        })
                        if coingecko_called_for_item and price_usd > 0.0 and cg_delay_ms > 0:
                             time.sleep(cg_delay_ms / 1000.0)

                    if not fetched_rows_in_page or (api_total_for_this_query > 0 and current_page_api * limit_per_page >= api_total_for_this_query) or len(fetched_rows_in_page) < limit_per_page:
                        fetched_all_for_batch = True
                    else: current_page_api += 1
                else: fetched_all_for_batch = True

                if fetched_all_for_batch:
                    logger.debug(f"Completed SE Subscriptions batch {current_batch_start_time_dt.strftime('%Y-%m-%d')} to {current_batch_end_time_dt.strftime('%Y-%m-%d')}.")
                    break

            current_batch_end_time_dt = current_batch_start_time_dt - datetime.timedelta(milliseconds=1)
            if current_batch_end_time_dt <= overall_start_time_dt: break

        logger.info(f"Fetched and processed {len(all_subscription_transactions)} new Simple Earn Flexible subscription transactions for target assets.")
        return all_subscription_transactions

    def fetch_simple_earn_flexible_redemptions(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Simple Earn Flexible redemptions.")
            return []

        source_name = "Binance API Simple Earn Redemption"
        latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        specific_lookback_days = self.config.get("history_lookback_days", {}).get("simple_earn_redemptions", days_back if days_back is not None else 90)

        overall_start_time_dt: datetime.datetime
        if latest_known_ts:
            effective_start_from_db = latest_known_ts - datetime.timedelta(minutes=10)
            fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
            overall_start_time_dt = max(effective_start_from_db, fallback_start_from_days)
            logger.info(f"Selective sync for '{source_name}': Effective start date {overall_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            overall_start_time_dt = now_utc - datetime.timedelta(days=specific_lookback_days)
            logger.info(f"Full sync for '{source_name}': Fetching last {specific_lookback_days} days from {overall_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if overall_start_time_dt >= now_utc:
            logger.info(f"'{source_name}' history is very recent. No new data to fetch.")
            return []

        logger.info(f"Fetching Simple Earn Flexible redemptions from {overall_start_time_dt.date()}.")
        all_redemption_transactions = []

        config_apis_binance = self.config.get("apis", {}).get("binance", {})
        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms_csv", 1500))
        batch_days = config_apis_binance.get("transfer_history_batch_days", 7)
        limit_per_page = 100
        max_retries = config_apis_binance.get("max_retries_per_batch", 2)
        retry_delay_seconds = config_apis_binance.get("retry_delay_sec", 30)
        binance_api_delay_ms = config_apis_binance.get("request_delay_ms", 250)
        recv_window_ms = config_apis_binance.get("recv_window", 10000)
        endpoint_path = 'simple-earn/flexible/history/redemptionRecord'
        processed_ids = set()
        current_batch_end_time_dt = now_utc

        while current_batch_end_time_dt > overall_start_time_dt:
            current_batch_start_time_dt = current_batch_end_time_dt - datetime.timedelta(days=batch_days)
            if current_batch_start_time_dt < overall_start_time_dt:
                current_batch_start_time_dt = overall_start_time_dt
            start_ms = int(current_batch_start_time_dt.timestamp() * 1000)
            end_ms = int(current_batch_end_time_dt.timestamp() * 1000)
            if start_ms >= end_ms: break

            logger.debug(f"Fetching SE Redemptions: Batch {current_batch_start_time_dt.strftime('%Y-%m-%d %H:%M')} to {current_batch_end_time_dt.strftime('%Y-%m-%d %H:%M')}")
            current_page_api = 1; fetched_all_for_batch = False
            while not fetched_all_for_batch:
                history_page = None; fetched_rows_in_page = None; api_total_for_this_query = 0
                for attempt in range(max_retries):
                    try:
                        params = {"startTime": start_ms, "endTime": end_ms, "current": current_page_api, "size": limit_per_page, "recvWindow": recv_window_ms}
                        logger.debug(f"API Call (SE Redemptions Pg: {current_page_api}) Att: {attempt + 1}. EP: '{endpoint_path}', Params: {params}")
                        history_page = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                        fetched_rows_in_page = history_page.get('rows', [])
                        api_total_for_this_query = history_page.get('total', 0)
                        logger.debug(f"Raw API (SE Redemptions Pg: {current_page_api}) Att: {attempt + 1}: Fetched {len(fetched_rows_in_page) if fetched_rows_in_page else '0'} rows. API total: {api_total_for_this_query}. Sample: {str(history_page)[:200]}...")
                        if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                        break
                    except Exception as e:
                        logger.error(f"API/Net Error attempt {attempt + 1} for SE Redemptions page {current_page_api}: {e}. Code: {getattr(e, 'code', 'N/A')}")
                        if attempt < max_retries - 1: time.sleep(retry_delay_seconds)
                        else: logger.error(f"Max retries for SE Redemptions page {current_page_api}."); fetched_all_for_batch = True

                if fetched_rows_in_page:
                    for item in fetched_rows_in_page:
                        timestamp_ms = item.get('time')
                        if not timestamp_ms: continue
                        timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True).to_pydatetime()

                        if latest_known_ts and timestamp <= latest_known_ts:
                            continue

                        redeem_id = str(item.get('redeemId', f"serd_{item.get('asset')}_{timestamp_ms}_{item.get('amount')}"))
                        if redeem_id in processed_ids: continue
                        processed_ids.add(redeem_id)

                        asset_redeemed = item.get('asset','').upper()
                        normalized_symbol = self.norm_map.get(asset_redeemed, asset_redeemed)
                        if normalized_symbol not in self.target_assets_for_sync:
                            continue
                        quantity = float(item.get('amount', 0.0))
                        if quantity == 0: continue

                        price_usd = 0.0
                        coingecko_called_for_item = False
                        if normalized_symbol in self.stablecoin_symbols: price_usd = 1.0
                        else:
                            coin_id = self.symbol_mappings.get(normalized_symbol)
                            if coin_id:
                                date_str_for_price = timestamp.strftime('%d-%m-%Y')
                                fetched_price = self._get_coingecko_historical_price(coin_id, date_str_for_price)
                                coingecko_called_for_item = True
                                if fetched_price is not None:
                                    price_usd = fetched_price
                                    logger.debug(f"SE Redemp: Fetched hist_price ${price_usd:.6f} for {normalized_symbol} on {date_str_for_price}.")
                                else: logger.warning(f"SE Redemp: Could not get hist price for {normalized_symbol} on {date_str_for_price}. Using $0.00.")
                            else: logger.warning(f"SE Redemp: No CoinGecko ID for {normalized_symbol}. Using $0.00.")

                        all_redemption_transactions.append({
                            "symbol": normalized_symbol, "timestamp": timestamp, "type": "BUY",
                            "quantity": quantity, "price_usd": price_usd,
                            "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                            "source": source_name, "transaction_hash": redeem_id,
                            "notes": f"Simple Earn Redemption: {quantity:.8f} {normalized_symbol}"
                        })
                        if coingecko_called_for_item and price_usd > 0.0 and cg_delay_ms > 0:
                             time.sleep(cg_delay_ms / 1000.0)

                    if not fetched_rows_in_page or (api_total_for_this_query > 0 and current_page_api * limit_per_page >= api_total_for_this_query) or len(fetched_rows_in_page) < limit_per_page:
                        fetched_all_for_batch = True
                    else: current_page_api += 1
                else: fetched_all_for_batch = True

                if fetched_all_for_batch:
                    logger.debug(f"Completed SE Redemptions batch {current_batch_start_time_dt.strftime('%Y-%m-%d')} to {current_batch_end_time_dt.strftime('%Y-%m-%d')}.")
                    break

            current_batch_end_time_dt = current_batch_start_time_dt - datetime.timedelta(milliseconds=1)
            if current_batch_end_time_dt <= overall_start_time_dt: break

        logger.info(f"Fetched and processed {len(all_redemption_transactions)} new Simple Earn Flexible redemption transactions for target assets.")
        return all_redemption_transactions

    def fetch_dividend_history(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch dividend history.")
            return []

        source_name = "Binance API Dividend"
        latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        specific_lookback_days = self.config.get("history_lookback_days", {}).get("dividend_history", days_back if days_back is not None else 90)

        overall_start_time_dt: datetime.datetime
        if latest_known_ts:
            effective_start_from_db = latest_known_ts - datetime.timedelta(hours=1) # 1 hour buffer
            fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
            overall_start_time_dt = max(effective_start_from_db, fallback_start_from_days)
            logger.info(f"Selective sync for '{source_name}': Effective start date {overall_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            overall_start_time_dt = now_utc - datetime.timedelta(days=specific_lookback_days)
            logger.info(f"Full sync for '{source_name}': Fetching last {specific_lookback_days} days from {overall_start_time_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if overall_start_time_dt >= now_utc:
            logger.info(f"'{source_name}' history is very recent. No new data to fetch.")
            return []

        logger.info(f"Fetching asset dividend history from {overall_start_time_dt.date()}.")
        all_dividend_transactions = []

        config_apis_binance = self.config.get("apis", {}).get("binance", {})
        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms_csv", 1500))
        batch_max_days = config_apis_binance.get("transfer_history_batch_days", 7) # Consistent batching
        limit_per_api_call = 500 # Max for this endpoint
        max_retries = config_apis_binance.get("max_retries_per_batch", 2)
        retry_delay_seconds = config_apis_binance.get("retry_delay_sec", 30)
        binance_api_delay_ms = config_apis_binance.get("request_delay_ms", 250)
        processed_tran_ids = set()
        current_batch_end_time_dt = now_utc

        while current_batch_end_time_dt > overall_start_time_dt:
            current_batch_start_time_dt = current_batch_end_time_dt - datetime.timedelta(days=batch_max_days)
            if current_batch_start_time_dt < overall_start_time_dt:
                current_batch_start_time_dt = overall_start_time_dt
            start_ms = int(current_batch_start_time_dt.timestamp() * 1000)
            end_ms = int(current_batch_end_time_dt.timestamp() * 1000)
            if start_ms >= end_ms: break

            logger.debug(f"Fetching dividend history: Batch {current_batch_start_time_dt.strftime('%Y-%m-%d %H:%M')} to {current_batch_end_time_dt.strftime('%Y-%m-%d %H:%M')}")
            fetched_dividends_for_batch = None
            for attempt in range(max_retries):
                try:
                    fetched_dividends_for_batch = self.binance_client.get_asset_dividend_history(
                        startTime=start_ms, endTime=end_ms, limit=limit_per_api_call
                    )
                    rows_fetched_count = len(fetched_dividends_for_batch.get('rows', [])) if fetched_dividends_for_batch and 'rows' in fetched_dividends_for_batch else 0
                    logger.debug(f"Raw API (Dividends) Att {attempt + 1}: Fetched {rows_fetched_count} records. Sample: {str(fetched_dividends_for_batch)[:200]}...")
                    if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                    break
                except Exception as e_api_div:
                    logger.error(f"API/Net Error attempt {attempt + 1} for dividend history: {e_api_div}. Code: {getattr(e_api_div, 'code', 'N/A')}")
                    if attempt < max_retries - 1: time.sleep(retry_delay_seconds)
                    else: logger.error(f"Max retries for dividend history batch."); # No fetched_all_for_batch here, will just process what we have if any

            if fetched_dividends_for_batch and fetched_dividends_for_batch.get('rows'):
                for item in fetched_dividends_for_batch['rows']:
                    timestamp_ms = item.get('divTime')
                    if not timestamp_ms: continue
                    timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True).to_pydatetime()

                    if latest_known_ts and timestamp <= latest_known_ts:
                        continue

                    tran_id = str(item.get('tranId'))
                    if not tran_id or tran_id == 'None' or tran_id in processed_tran_ids :
                        logger.debug(f"Skipping dividend item with missing or duplicate tranId: {item}")
                        continue
                    processed_tran_ids.add(tran_id)

                    asset_received = item.get('asset','').upper()
                    if not asset_received: logger.warning(f"Skipping dividend item with missing asset: {item}"); continue
                    normalized_symbol = self.norm_map.get(asset_received, asset_received)
                    if normalized_symbol not in self.target_assets_for_sync:
                        continue
                    quantity = float(item.get('amount', 0.0))
                    if quantity == 0: continue

                    price_usd = 0.0
                    coingecko_called_for_item = False
                    if normalized_symbol in self.stablecoin_symbols: price_usd = 1.0
                    else:
                        coin_id = self.symbol_mappings.get(normalized_symbol)
                        if coin_id:
                            date_str_for_price = timestamp.strftime('%d-%m-%Y')
                            fetched_price = self._get_coingecko_historical_price(coin_id, date_str_for_price)
                            coingecko_called_for_item = True
                            if fetched_price is not None:
                                price_usd = fetched_price
                                logger.debug(f"Dividend: Fetched hist_price ${price_usd:.6f} for {normalized_symbol} on {date_str_for_price}.")
                            else: logger.warning(f"Dividend: Could not get hist price for {normalized_symbol} on {date_str_for_price}. Using $0.00.")
                        else: logger.warning(f"Dividend: No CoinGecko ID for {normalized_symbol}. Using $0.00.")

                    all_dividend_transactions.append({
                        "symbol": normalized_symbol, "timestamp": timestamp, "type": "DEPOSIT",
                        "quantity": quantity, "price_usd": price_usd,
                        "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                        "source": source_name, "transaction_hash": tran_id,
                        "notes": f"Dividend: {item.get('enInfo', '')} - {quantity:.8f} {normalized_symbol}"
                    })
                    if coingecko_called_for_item and price_usd > 0.0 and cg_delay_ms > 0 :
                         time.sleep(cg_delay_ms / 1000.0)

            current_batch_end_time_dt = current_batch_start_time_dt - datetime.timedelta(milliseconds=1)
            if current_batch_end_time_dt <= overall_start_time_dt: break

        logger.info(f"Fetched and processed {len(all_dividend_transactions)} new dividend transactions for target assets.")
        return all_dividend_transactions

    def fetch_staking_history(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch staking history.")
            return []

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        specific_lookback_days = self.config.get("history_lookback_days", {}).get("staking_history", days_back if days_back is not None else 90)

        logger.info(f"Fetching staking history (Product: STAKING) with lookback: {specific_lookback_days} days.")
        all_transactions = []

        config_apis_binance = self.config.get("apis", {}).get("binance", {})
        cg_config = self.config.get("apis", {}).get("coingecko", {})
        cg_delay_ms = cg_config.get("request_delay_ms_generic_historical", cg_config.get("request_delay_ms_csv", 1500))
        batch_days = 89
        limit_per_page = 100
        max_retries = config_apis_binance.get("max_retries_per_batch", 2)
        retry_delay_seconds = config_apis_binance.get("retry_delay_sec", 30)
        binance_api_delay_ms = config_apis_binance.get("request_delay_ms", 250)
        recv_window_ms = config_apis_binance.get("recv_window", 10000)
        endpoint_path = 'staking/history'
        processed_txn_ids_this_run = set() # Avoid reprocessing within the same sync_data call if tranId appears multiple times
        staking_product_type = "STAKING"
        transaction_types_to_fetch = ["SUBSCRIPTION", "REDEMPTION", "INTEREST"]

        for txn_type_filter in transaction_types_to_fetch:
            source_name_current_filter = f"Binance API Staking ({staking_product_type} {txn_type_filter})"
            latest_known_ts = self.db_manager.get_latest_timestamp_for_source(source_name_current_filter)

            overall_start_time_dt_for_filter: datetime.datetime
            if latest_known_ts:
                effective_start_from_db = latest_known_ts - datetime.timedelta(minutes=10)
                fallback_start_from_days = now_utc - datetime.timedelta(days=specific_lookback_days)
                overall_start_time_dt_for_filter = max(effective_start_from_db, fallback_start_from_days)
                logger.info(f"Selective sync for '{source_name_current_filter}': Effective start date {overall_start_time_dt_for_filter.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                overall_start_time_dt_for_filter = now_utc - datetime.timedelta(days=specific_lookback_days)
                logger.info(f"Full sync for '{source_name_current_filter}': Fetching last {specific_lookback_days} days from {overall_start_time_dt_for_filter.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            if overall_start_time_dt_for_filter >= now_utc:
                logger.info(f"'{source_name_current_filter}' history is very recent. Skipping.")
                continue

            logger.info(f"Fetching staking history for Product: {staking_product_type}, TxnType: {txn_type_filter} from {overall_start_time_dt_for_filter.date()}.")
            current_batch_end_time_dt = now_utc

            while current_batch_end_time_dt > overall_start_time_dt_for_filter:
                current_batch_start_time_dt = current_batch_end_time_dt - datetime.timedelta(days=batch_days)
                if current_batch_start_time_dt < overall_start_time_dt_for_filter:
                    current_batch_start_time_dt = overall_start_time_dt_for_filter
                start_ms = int(current_batch_start_time_dt.timestamp() * 1000)
                end_ms = int(current_batch_end_time_dt.timestamp() * 1000)
                if start_ms >= end_ms: break

                logger.debug(f"Fetching staking ({staking_product_type}/{txn_type_filter}): Batch {current_batch_start_time_dt.strftime('%Y-%m-%d')} to {current_batch_end_time_dt.strftime('%Y-%m-%d')}")
                current_page_api = 1; fetched_all_for_batch_prod_type = False
                while not fetched_all_for_batch_prod_type:
                    response_data = None; fetched_rows_in_page = None
                    for attempt in range(max_retries):
                        try:
                            params = {
                                "product": staking_product_type, "txnType": txn_type_filter,
                                "startTime": start_ms, "endTime": end_ms,
                                "current": current_page_api, "size": limit_per_page,
                                "recvWindow": recv_window_ms
                            }
                            logger.debug(f"API Call (Staking {staking_product_type}/{txn_type_filter}, Pg: {current_page_api}) Att {attempt + 1}. EP: '{endpoint_path}', Params: {params}")
                            response_data = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                            fetched_rows_in_page = response_data if isinstance(response_data, list) else []
                            logger.debug(f"Raw API (Staking {staking_product_type}/{txn_type_filter}, pg {current_page_api}, att {attempt+1}): Fetched {len(fetched_rows_in_page)} records. Sample: {str(response_data)[:200]}...")
                            if binance_api_delay_ms > 0: time.sleep(binance_api_delay_ms / 1000.0)
                            break
                        except Exception as e_api_stake:
                            logger.error(f"API/Net Error staking ({staking_product_type}/{txn_type_filter}, pg {current_page_api}, att {attempt+1}): {e_api_stake}. Code: {getattr(e_api_stake, 'code', 'N/A')}")
                            if attempt < max_retries - 1: time.sleep(retry_delay_seconds)
                            else: logger.error(f"Max retries for staking history."); fetched_all_for_batch_prod_type = True

                    if fetched_rows_in_page:
                        for item in fetched_rows_in_page:
                            timestamp_ms = item.get('time')
                            if not timestamp_ms: continue
                            timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True).to_pydatetime()

                            if latest_known_ts and timestamp <= latest_known_ts:
                                continue

                            txn_id_val = item.get('txnId')
                            if txn_id_val is None: txn_id = f"stk_{item.get('asset')}_{timestamp_ms}_{item.get('amount')}_{txn_type_filter}"
                            else: txn_id = str(txn_id_val)

                            if txn_id in processed_txn_ids_this_run: continue
                            processed_txn_ids_this_run.add(txn_id)

                            asset = item.get('asset','').upper()
                            normalized_symbol = self.norm_map.get(asset, asset)
                            if normalized_symbol not in self.target_assets_for_sync:
                                continue
                            quantity = float(item.get('amount', 0.0))
                            if quantity == 0: continue

                            tx_type_mapped = ""; notes_prefix = ""
                            if txn_type_filter == "SUBSCRIPTION": tx_type_mapped = "SELL"; notes_prefix = "Staking Subscription"
                            elif txn_type_filter == "REDEMPTION": tx_type_mapped = "BUY"; notes_prefix = "Staking Redemption"
                            elif txn_type_filter == "INTEREST": tx_type_mapped = "DEPOSIT"; notes_prefix = "Staking Interest"
                            else: logger.warning(f"Unknown staking txnType: {txn_type_filter} for item {item}"); continue

                            price_usd = 0.0
                            coingecko_called_for_item = False
                            if normalized_symbol in self.stablecoin_symbols: price_usd = 1.0
                            else:
                                coin_id = self.symbol_mappings.get(normalized_symbol)
                                if coin_id:
                                    date_str_for_price = timestamp.strftime('%d-%m-%Y')
                                    fetched_price = self._get_coingecko_historical_price(coin_id, date_str_for_price)
                                    coingecko_called_for_item = True
                                    if fetched_price is not None:
                                        price_usd = fetched_price
                                        logger.debug(f"Staking ({notes_prefix}): Fetched hist_price ${price_usd:.6f} for {normalized_symbol} on {date_str_for_price}.")
                                    else: logger.warning(f"Staking ({notes_prefix}): Could not get hist price for {normalized_symbol} on {date_str_for_price}. Using $0.00.")
                                else: logger.warning(f"Staking ({notes_prefix}): No CoinGecko ID for {normalized_symbol}. Using $0.00.")

                            all_transactions.append({
                                "symbol": normalized_symbol, "timestamp": timestamp, "type": tx_type_mapped,
                                "quantity": quantity, "price_usd": price_usd,
                                "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                                "source": source_name_current_filter,
                                "transaction_hash": txn_id,
                                "notes": f"{notes_prefix}: {quantity:.8f} {normalized_symbol}"
                            })
                            if coingecko_called_for_item and price_usd > 0.0 and cg_delay_ms > 0:
                                 time.sleep(cg_delay_ms / 1000.0)

                        if len(fetched_rows_in_page) < limit_per_page: fetched_all_for_batch_prod_type = True
                        else: current_page_api += 1
                    else: fetched_all_for_batch_prod_type = True

                    if fetched_all_for_batch_prod_type: break

                current_batch_end_time_dt = current_batch_start_time_dt - datetime.timedelta(milliseconds=1)
                if current_batch_end_time_dt <= overall_start_time_dt_for_filter: break

            if binance_api_delay_ms > 0 and txn_type_filter != transaction_types_to_fetch[-1] :
                time.sleep(binance_api_delay_ms * 2 / 1000.0)

        logger.info(f"Fetched and processed {len(all_transactions)} new staking transactions (Product: {staking_product_type}) for target assets.")
        return all_transactions

    def update_holdings_from_transactions(self):
        """
        Processes all transactions and updates holdings table with FIFO cost basis.
        """
        logger.info("Updating holdings from transaction history using FIFO...")
        all_txs = self.db_manager.get_all_transactions()

        if all_txs.empty:
            logger.warning("No transactions found in DB. Cannot update holdings.")
            return

        updated_holdings = []
        for symbol, group_df in all_txs.groupby('symbol'):
            logger.debug(f"Calculating FIFO for {symbol}...")

            group_df_copy = group_df.copy()
            group_df_copy['price_usd'] = pd.to_numeric(group_df_copy['price_usd'], errors='coerce').fillna(0.0)
            group_df_copy['quantity'] = pd.to_numeric(group_df_copy['quantity'], errors='coerce').fillna(0.0)
            group_df_copy['timestamp'] = pd.to_datetime(group_df_copy['timestamp'], errors='coerce')
            group_df_copy.dropna(subset=['timestamp'], inplace=True)

            # 1. Isolate real trades for cost basis calculation
            cost_basis_tx_df = group_df_copy[
                ~group_df_copy['source'].str.contains("Simple Earn|Asset Transfer|Staking", case=False, na=False)
            ]
            logger.info(f"Calculating cost basis for {symbol} using {len(cost_basis_tx_df)} non-transfer transactions.")

            if cost_basis_tx_df.empty:
                logger.info(f"No non-transfer transactions for {symbol}. Cannot calculate cost basis.")
                continue  # Skip to the next symbol

            # 2. Calculate cost basis and the remaining quantity FROM THAT BASIS
            cost_basis_qty, avg_cost = calculate_fifo_cost_basis(cost_basis_tx_df)

            # 3. If there's a valid average cost, save it.
            # The quantity saved here is just a placeholder; the final report uses the live wallet balance.
            if avg_cost > 0:
                logger.info(f"Calculated for {symbol}: Qty_from_basis={cost_basis_qty:.8f}, AvgCost={avg_cost:.8f}. Storing avg_cost.")
                updated_holdings.append({
                    "symbol": symbol,
                    "quantity": cost_basis_qty,
                    "average_cost_basis": avg_cost
                })
            else:
                logger.info(f"No cost basis calculated for {symbol} (likely no 'BUY' transactions in history).")

        if updated_holdings:
            holdings_df = pd.DataFrame(updated_holdings)
            self.db_manager.update_holdings(holdings_df)
            logger.info(f"Successfully updated/inserted {len(holdings_df)} asset holdings in the database with new cost basis.")
        else:
            logger.warning("No holdings with valid cost basis to update in the database.")

    def get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for a list of symbols using CoinGecko (Batched with Retry).
        Correctly handles multiple symbols mapping to the same CoinGecko ID."""
        prices: Dict[str, Optional[float]] = {symbol: 0.0 for symbol in symbols}
        symbols_to_fetch_cg_ids = {}
        unique_coingecko_ids_to_fetch = set()

        logger.info(f"Mapping {len(symbols)} symbols to CoinGecko IDs for price fetching...")
        for symbol_upper in set(s.upper() for s in symbols):
            original_case_symbol = next(s for s in symbols if s.upper() == symbol_upper)
            coin_id = self.symbol_mappings.get(symbol_upper)
            if not coin_id:
                logger.warning(f"No CoinGecko ID mapping found for {original_case_symbol}. Price will be $0.")
                prices[original_case_symbol] = 0.0
            else:
                symbols_to_fetch_cg_ids[original_case_symbol] = coin_id
                unique_coingecko_ids_to_fetch.add(coin_id)

        if not unique_coingecko_ids_to_fetch:
            logger.warning("No valid CoinGecko IDs to fetch prices for.")
            return prices

        ids_string = ",".join(list(unique_coingecko_ids_to_fetch))
        base_url = self.coingecko_api.get("base_url")
        timeout = self.coingecko_api.get("timeout", 30)
        url = f"{base_url}/simple/price?ids={ids_string}&vs_currencies=usd"
        logger.info(f"Fetching {len(unique_coingecko_ids_to_fetch)} unique prices from CoinGecko...")

        retries = 1
        fetched_price_data = None
        while retries >= 0:
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                fetched_price_data = response.json()
                break
            except requests.exceptions.HTTPError as e:
                 if e.response.status_code == 429 and retries > 0:
                     logger.error(f"Rate limited (429) fetching batch prices. Waiting 60s before retry...")
                     time.sleep(60)
                     retries -= 1
                 else:
                     logger.error(f"HTTP error fetching batch prices: {e}")
                     fetched_price_data = {}
                     break
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching batch prices from CoinGecko: {e}")
                fetched_price_data = {}
                break

        if fetched_price_data:
            for original_symbol, coin_id in symbols_to_fetch_cg_ids.items():
                if coin_id in fetched_price_data and "usd" in fetched_price_data[coin_id]:
                    prices[original_symbol] = fetched_price_data[coin_id]["usd"]
                else:
                    logger.warning(f"USD price not found for {original_symbol} (ID: {coin_id}) in CoinGecko response. Setting to $0.")
                    prices[original_symbol] = 0.0
        else:
            logger.error("Failed to fetch any price data from CoinGecko. All prices will be $0.")
        logger.info("Price fetching complete.")
        return prices

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculates key portfolio metrics using a consolidated view of holdings,
        correctly reflecting that the Earn balance is the authoritative total for assets in Earn.
        """
        logger.info("Calculating consolidated portfolio metrics (Spot + Earn)...")

        # 1. Fetch balances from both wallets
        cost_basis_df = self.db_manager.get_holdings()
        # This is the total balance from the main account endpoint
        total_api_df = self.fetch_binance_balances().rename(columns={'quantity': 'total_quantity_api'})
        earn_dict = self.fetch_simple_earn_balances()
        earn_df = pd.DataFrame(list(earn_dict.items()), columns=['symbol', 'earn_quantity'])

        # 2. Merge the dataframes
        holdings_df = pd.merge(total_api_df, earn_df, on='symbol', how='outer').fillna(0)

        # 3. --- THIS IS THE KEY LOGIC CHANGE ---
        # For each asset, the true total quantity is the larger of the two values reported by the APIs.
        # This handles the discrepancy and correctly reflects that the Earn balance is the true total if an asset is staked.
        holdings_df['total_quantity'] = holdings_df[['total_quantity_api', 'earn_quantity']].max(axis=1)

        # 4. Now that we have a definitive total, the breakdown is simple and robust.
        holdings_df['spot_quantity'] = (holdings_df['total_quantity'] - holdings_df['earn_quantity']).clip(lower=0)

        # --- The rest of the function remains the same ---
        holdings_df = holdings_df[holdings_df['total_quantity'] > 0.00000001].reset_index(drop=True)

        if holdings_df.empty:
            logger.warning("No non-zero holdings found. Portfolio value is $0.")
            return {"total_value_usd": 0, "holdings_df": pd.DataFrame()}

        # Merge cost basis and price data
        if not cost_basis_df.empty:
            holdings_df = pd.merge(holdings_df, cost_basis_df[['symbol', 'average_cost_basis']], on='symbol', how='left')
        holdings_df['average_cost_basis'] = holdings_df['average_cost_basis'].fillna(0.0)

        prices = self.get_current_prices(holdings_df['symbol'].tolist())
        holdings_df['current_price'] = holdings_df['symbol'].map(prices).fillna(0.0)

        # Perform all financial calculations
        holdings_df['value_usd'] = holdings_df['total_quantity'] * holdings_df['current_price']
        holdings_df['cost_basis_total'] = holdings_df['total_quantity'] * holdings_df['average_cost_basis']
        holdings_df['unrealized_pl_usd'] = holdings_df['value_usd'] - holdings_df['cost_basis_total']
        holdings_df['unrealized_pl_percent'] = 0.0
        mask = holdings_df['cost_basis_total'] > 0
        holdings_df.loc[mask, 'unrealized_pl_percent'] = \
            (holdings_df.loc[mask, 'unrealized_pl_usd'] / holdings_df.loc[mask, 'cost_basis_total']) * 100

        total_value = holdings_df['value_usd'].sum()
        total_portfolio_cost_basis = holdings_df['cost_basis_total'].sum()
        holdings_df['allocation'] = holdings_df['value_usd'] / total_value if total_value > 0 else 0
        total_pl_usd = total_value - total_portfolio_cost_basis
        total_pl_percent = (total_pl_usd / total_portfolio_cost_basis * 100) if total_portfolio_cost_basis > 0 else 0.0

        metrics = {
            "total_value_usd": total_value,
            "total_cost_basis_usd": total_portfolio_cost_basis,
            "unrealized_pl_usd": total_pl_usd,
            "unrealized_pl_percent": total_pl_percent,
            "holdings_df": holdings_df,
            "timestamp": datetime.datetime.now()
        }
        logger.info(f"Consolidated Metrics Calculated: Total Value = ${total_value:,.2f}")
        return metrics

    def print_portfolio_summary(self, metrics: Dict[str, Any]):
        """Prints a consolidated summary of the portfolio, including Spot and Earn balances."""
        print("\n" + "="*100)
        print("📊 CONSOLIDATED PORTFOLIO SUMMARY (Spot + Earn)")
        print("="*100)
        if "error" in metrics:
            print(f"❌ Could not generate summary: {metrics['error']}")
            print("="*100)
            return

        timestamp = metrics.get('timestamp')
        if isinstance(timestamp, datetime.datetime):
            print(f"Timestamp:             {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"Timestamp:             N/A")

        print(f"Total Portfolio Value: ${metrics.get('total_value_usd', 0):,.2f}")
        print(f"Total Cost Basis:      ${metrics.get('total_cost_basis_usd', 0):,.2f}")

        pl_usd = metrics.get('unrealized_pl_usd', 0)
        pl_pct = metrics.get('unrealized_pl_percent', 0)
        color_start = "\033[92m" if pl_usd >= 0 else "\033[91m"
        color_end = "\033[0m"
        print(f"Unrealized P/L:        {color_start}${pl_usd:,.2f} ({pl_pct:.2f}%){color_end}")
        print("-" * 100)

        holdings_df = metrics.get('holdings_df')
        if holdings_df is not None and not holdings_df.empty:
            # --- UPDATED HEADER TO MATCH NEW COLUMNS ---
            header = f"{'Asset':<8} {'Total Qty':<18} {'Spot Qty':<15} {'Earn Qty':<15} {'Value (USD)':<15} {'P/L (USD)':<15} {'Allocation':<10}"
            print(header)
            print("-" * 100)

            for _, row in holdings_df.sort_values(by='value_usd', ascending=False).iterrows():
                row_pl_usd = row.get('unrealized_pl_usd', 0)
                row_color_start = "\033[92m" if row_pl_usd >= 0 else "\033[91m"
                row_color_end = "\033[0m"

                # --- UPDATED ROW TO USE NEW COLUMN NAMES ---
                print(
                    f"{row.get('symbol', 'N/A'):<8} "
                    f"{row.get('total_quantity', 0):<18,.8g} "
                    f"{row.get('spot_quantity', 0):<15,.8g} "
                    f"{row.get('earn_quantity', 0):<15,.8g} "
                    f"${row.get('value_usd', 0):<14,.2f} "
                    f"{row_color_start}${row_pl_usd:<14,.2f}{row_color_end} "
                    f"{row.get('allocation', 0) * 100:<9.2f}%"
                )
            print("="*100)
        else:
            print("No holdings data to display.")
            print("="*100)

    def print_rebalance_suggestions(self, suggestions_df: Optional[pd.DataFrame]):
        """Prints rebalancing suggestions with relative drift in a clean, readable format."""
        if suggestions_df is None or suggestions_df.empty:
            print("\nCould not generate rebalancing suggestions.")
            return

        # Sort by the absolute value of relative drift to show the most deviated assets first
        if 'Relative Drift %' in suggestions_df.columns:
            suggestions_df = suggestions_df.sort_values(by='Relative Drift %', key=abs, ascending=False)

        print("\n" + "="*80)
        print("⚖️ REBALANCING SUGGESTIONS (Core Portfolio - Technical Analysis)")
        print("="*80)

        for _, row in suggestions_df.iterrows():
            symbol = row.get('Symbol', 'N/A')
            signal = row.get('Signal', 'N/A')
            current_pct = row.get('Current %', 0)
            target_pct = row.get('Target %', 0)
            relative_drift = row.get('Relative Drift %', 0) # Get the new value
            current_val = row.get('Current Value (USD)', 0)
            action = row.get('Suggested Action Detail', 'N/A')
            rsi = row.get('RSI (14W)')
            ma_dist = row.get('Price vs 200w MA (%)')

            if signal == "BUY": color_start, color_end, icon = "\033[92m", "\033[0m", "🟢"
            elif signal == "SELL": color_start, color_end, icon = "\033[91m", "\033[0m", "🔴"
            else: color_start, color_end, icon = "\033[93m", "\033[0m", "🟡"

            print(f"{color_start}{icon} {symbol.ljust(7)} | Signal: {signal.ljust(5)}{color_end}")

            # --- UPDATED LINE TO SHOW RELATIVE DRIFT ---
            print(f"   Allocation: {current_pct:.2f}% (Target: {target_pct:.1f}%) | Relative Drift: {relative_drift:+.1f}% | Value: ${current_val:,.2f}")

            ta_info = []
            if pd.notna(rsi): ta_info.append(f"RSI (14W): {rsi:.1f}")
            if pd.notna(ma_dist): ta_info.append(f"Price vs 200w MA: {ma_dist:+.1f}%")
            else:
                if symbol not in self.stablecoin_symbols:
                    ta_info.append("Price vs 200w MA: (No Data)")
            if ta_info:
                print(f"   TA: {', '.join(ta_info)}")

            print(f"   Action: {action}")
            print("-" * 60)

        print("\n" + "="*80)

    def export_to_excel(self, metrics: Dict[str, Any]): self.excel_exporter.export(metrics=metrics, holdings_df=metrics.get('holdings_df'))
    def export_to_html(self, metrics: Dict[str, Any]): self.html_exporter.export(metrics=metrics, holdings_df=metrics.get('holdings_df'))
    def export_csv_backup(self): self.csv_exporter.export(transactions_df=self.db_manager.get_all_transactions(), holdings_df=self.db_manager.get_holdings())
    def cleanup_old_data(self): self.db_manager.cleanup_old_data()

    def create_portfolio_charts(self, metrics: Dict[str, Any]):
        """Generate portfolio charts."""
        holdings_df = metrics.get('holdings_df'); target_alloc = self.config.get("target_allocation", {})
        if holdings_df is not None: self.visualizer.generate_all_charts(holdings_df, metrics, target_alloc, pd.DataFrame())
        else: logger.warning("No holdings data for chart generation.")

    def print_configuration(self):
        """Print the current configuration (excluding sensitive data)."""
        print("\n" + "="*50 + "\n⚙️ Current Configuration\n" + "="*50)
        safe_config = self.config.copy()
        if "api_keys" in safe_config: safe_config["api_keys"] = {k: '********' for k in safe_config["api_keys"]}
        print(json.dumps(safe_config, indent=2) + "\n" + "="*50)

    def test_connections(self):
        """Test connections to Binance and CoinGecko."""
        if self.binance_client:
            try: self.binance_client.ping(); print("✅ Binance Connection: SUCCESS")
            except Exception as e: print(f"❌ Binance Connection: FAILED ({e})")
        else: print("⚠️ Binance Connection: SKIPPED (Failed to Initialize Client/No API keys)")
        test_id = list(self.symbol_mappings.values())[0] if self.symbol_mappings else 'bitcoin'
        price = self._get_coingecko_price(test_id)
        if price: print(f"✅ CoinGecko Connection: SUCCESS ({test_id.capitalize()} price: ${price})")
        else: print("❌ CoinGecko Connection: FAILED")
        print("-" * 30)

    def save_snapshot(self, metrics: Dict[str, Any]):
        """Wrapper to save a portfolio snapshot using data from calculated metrics."""
        if "error" in metrics or "total_value_usd" not in metrics:
            logger.warning("Skipping snapshot save due to missing data in metrics.")
            return

        # --- This is the corrected section ---
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

    def print_trend_report(self, report: Dict[str, Any]):
        """Prints a formatted trend analysis report to the console."""
        if not report:
            print("\n--- No Trend Report Data ---")
            return

        print("\n" + "="*80)
        print(f"📈 TREND ANALYSIS REPORT ({report.get('timeframe', 'N/A').upper()})")
        print(f"Timestamp: {report.get('timestamp')}")
        print("="*80)

        summary = report.get('market_summary', {})
        print("\n--- 🌍 Market Summary ---")
        print(f"Overall Sentiment: {summary.get('overall_sentiment', 'N/A')}")
        print(f"Coins Analyzed: {summary.get('total_coins', 0)}")
        print(f"Coins Outperforming Benchmark: {summary.get('outperforming_count', 0)}")

        btc = report.get('benchmark_analysis', {})
        print(f"\n--- 🎯 Benchmark Analysis: {btc.get('symbol', 'N/A')} ---")
        print(f"  Trend: {btc.get('trend', 'N/A')} (Confidence: {btc.get('confidence', 0):.0%})")
        print(f"  Price: ${btc.get('current_price', 0):,.2f} ({btc.get('price_change_pct', 0):+.2f}%)")
        print(f"  RSI: {btc.get('rsi', 0):.2f}")
        print(f"  Position vs SMAs: {btc.get('sma_position')}")

        print("\n--- 🪙 Coin-by-Coin Analysis ---")
        for symbol, analysis in report.get('coin_analyses', {}).items():
            rel_perf = report.get('relative_performances', {}).get(symbol, {})
            outperforming_str = "🔥 Outperforming" if rel_perf.get('outperforming') else "🧊 Underperforming"
            vs_btc_str = f"({rel_perf.get('vs_btc_return', 0):+.2f}% vs BTC)"

            print(f"\n➡️ {symbol} - {outperforming_str} {vs_btc_str}")
            print(f"  Trend: {analysis.get('trend', 'N/A')} (Confidence: {analysis.get('confidence', 0):.0%})")
            print(f"  Price: ${analysis.get('current_price', 0):,.2f} ({analysis.get('price_change_pct', 0):+.2f}%)")
            print(f"  RSI: {analysis.get('rsi', 0):.2f} | SMAs: {analysis.get('sma_position')}")
            print(f"  Support: ${analysis.get('support_level', 0):,.2f} | Resistance: ${analysis.get('resistance_level', 0):,.2f}")

        print("\n" + "="*80)

    def export_trend_report(self, report: Dict[str, Any], timeframe: str):
        """Exports the trend report to JSON."""
        if not report:
            print("❌ No report data to export.")
            return

        report_dir = self.config.get('trend_analyzer', {}).get('report_directory', 'data/trend_reports')
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(report_dir, f"trend_report_{timeframe}_{timestamp}.json")

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            print(f"\n✅ Trend report successfully saved to {filename}")
            logger.info(f"Trend report saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save trend report: {e}")
            logger.error(f"Could not save trend report to {filename}: {e}", exc_info=True)

    def run_backtest(self):
        """
        Handles the user flow for running a backtest with an improved UI.
        """
        print("\n--- 🧪 Backtesting Mode ---")

        # Initialize the analyzer and backtester
        try:
            analyzer = CryptoTrendAnalyzer(config=self.config)
            backtester = Backtester(config=self.config, analyzer=analyzer)
        except Exception as e:
            self.logger.error(f"Failed to initialize backtesting components: {e}", exc_info=True)
            print("❌ Error: Could not initialize the backtester. Please check logs.")
            return

        available_coins = self.config.get('trend_analyzer', {}).get('cryptocurrencies', [])
        if not available_coins:
            print("❌ No cryptocurrencies found in your configuration.")
            return

        # --- UI IMPROVEMENT ---
        available_coins.sort() # Sort coins alphabetically for a consistent order

        print("\nCoins available for backtesting:")

        # Print the list in 3 neat columns for better readability
        num_coins = len(available_coins)
        cols = 3
        for i in range(0, num_coins, cols):
            row = "  "
            for j in range(cols):
                if i + j < num_coins:
                    # Format each item with a number and fixed width
                    row += f"{i+j+1:2d}. {available_coins[i+j]:<15}"
            print(row)

        symbol_to_test = None
        while symbol_to_test is None:
            choice = input("\nEnter the symbol or number of the coin to backtest: ").strip()

            if choice.isdigit():
                # Handle numeric input
                try:
                    choice_index = int(choice) - 1
                    if 0 <= choice_index < num_coins:
                        symbol_to_test = available_coins[choice_index]
                    else:
                        print(f"❌ Invalid number. Please enter a number between 1 and {num_coins}.")
                except ValueError:
                     print(f"❌ Invalid input. Please enter a valid symbol or number.")
            else:
                # Handle string input (the symbol itself)
                if choice.upper() in available_coins:
                    symbol_to_test = choice.upper()
                else:
                    print(f"❌ Symbol '{choice}' not found. Please choose from the list above.")
        # --- END OF UI IMPROVEMENT ---

        print(f"\nSelected coin for backtest: {symbol_to_test}")

        try:
            initial_capital_str = input("Enter initial capital (default: 10000): ")
            initial_capital = float(initial_capital_str) if initial_capital_str else 10000.0
        except ValueError:
            print("❌ Invalid capital amount. Using default $10,000.")
            initial_capital = 10000.0

        # Run the backtest and generate the report
        backtester.run(symbol=symbol_to_test, initial_capital=initial_capital)
        backtester.generate_report()

    def _redeem_from_earn_if_needed(self, asset: str, required_amount: float, is_live: bool) -> bool:
        """
        Checks for sufficient spot balance. If needed, it finds the correct productId
        and redeems from Simple Earn only if in live mode.
        Returns True if funds are sufficient, False otherwise.
        """
        try:
            spot_balance_info = self.binance_client.get_asset_balance(asset=asset)
            free_spot_balance = float(spot_balance_info.get('free', 0.0))

            if free_spot_balance >= required_amount:
                logger.info(f"Sufficient {asset} balance in Spot wallet. No action needed.")
                return True

            shortfall = required_amount - free_spot_balance
            logger.warning(f"Shortfall of {shortfall:.8f} {asset}. Checking Simple Earn.")

            positions = self.binance_client.get_simple_earn_flexible_product_position(asset=asset)
            if not positions or not positions.get('rows'):
                print(f"❌ No active Simple Earn Flexible positions found for {asset} to redeem from.")
                return False

            product_to_redeem_from = positions['rows'][0]
            product_id = product_to_redeem_from.get('productId')
            available_to_redeem = float(product_to_redeem_from.get('totalAmount', 0.0))

            if shortfall > available_to_redeem:
                print(f"❌ Cannot redeem required amount. Need {shortfall:.8f} {asset}, but only {available_to_redeem:.8f} is available in Simple Earn.")
                return False

            # --- This is the new, critical check ---
            if is_live:
                print(f"⚠️ Insufficient Spot balance. Attempting to redeem {shortfall:.8f} {asset} from product {product_id}.")
                logger.info(f"LIVE MODE: Attempting to redeem {shortfall:.8f} {asset} from productId {product_id}.")
                redemption_result = self.binance_client.redeem_simple_earn_flexible_product(
                    productId=product_id,
                    amount=f"{shortfall:.8f}"
                )
                if redemption_result and redemption_result.get('success'):
                    logger.info(f"Successfully initiated LIVE redemption. Waiting 5 seconds...")
                    print(f"✅ Redemption initiated for {shortfall:.8f} {asset}. Waiting 5 seconds...")
                    time.sleep(5)
                    return True # Assume success and let the trade proceed
                else:
                    logger.error(f"Failed to redeem {asset} from {product_id}: {redemption_result}")
                    print(f"❌ LIVE redemption FAILED for {asset}. See logs.")
                    return False
            else:
                # In Dry Run mode, we only simulate the check.
                print(f"DRY RUN: Would redeem {shortfall:.8f} {asset} from Simple Earn product {product_id}.")
                logger.info(f"DRY RUN: Would redeem {shortfall:.8f} {asset} from {product_id}.")
                return True # Return True to allow the simulated trade to continue

        except BinanceAPIException as e:
            logger.error(f"API Error during redemption process for {asset}: {e}", exc_info=True)
            print(f"❌ An API Error occurred when trying to process redemption for {asset}: {e.message}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during redemption check for {asset}: {e}", exc_info=True)
            return False

    def execute_rebalancing_trades(self, suggestions_df: pd.DataFrame, earn_balances: Dict[str, float]):
        """
        Executes rebalancing trades, enforcing minimum trade size during execution.
        """
        if suggestions_df.empty or 'Signal' not in suggestions_df.columns:
            print("No rebalancing suggestions to execute.")
            return

        trades_to_execute = suggestions_df[suggestions_df['Signal'].isin(['BUY', 'SELL'])]
        if trades_to_execute.empty:
            print("\n✅ No BUY or SELL actions suggested. Nothing to execute.")
            return

        portfolio_config = self.config.get("portfolio", {})
        min_trade_usd = portfolio_config.get("minimum_trade_usd", 10.0)
        is_live = self.config.get("rebalance_technical", {}).get("live_trading_enabled", False)

        simulated_balances = {}
        all_trade_symbols = set(trades_to_execute['Symbol'].unique()) | {'USDT'}
        for symbol in all_trade_symbols:
            try:
                spot_bal = float(self.binance_client.get_asset_balance(asset=symbol).get('free', 0.0))
                earn_bal = earn_balances.get(symbol, 0.0)
                simulated_balances[symbol] = spot_bal + earn_bal
                logger.info(f"Initialized simulated balance for {symbol}: {simulated_balances[symbol]:.8f}")
            except Exception as e:
                logger.error(f"Could not fetch balance for {symbol}: {e}")
                simulated_balances[symbol] = 0.0

        print("\n" + "="*80)
        if is_live: print("🔴🔴🔴 WARNING: Live Trading is ENABLED. 🔴🔴🔴")
        else: print("🟡🟡🟡 NOTE: Live Trading is DISABLED. 🟡🟡🟡")
        print("="*80)
        print("🚨 PROPOSED TRADES - PLEASE REVIEW CAREFULLY 🚨")
        print("="*80)
        print(trades_to_execute[['Symbol', 'Signal', 'Suggested Action Detail']].to_string(index=False))
        print("="*80)

        try:
            confirm = input("Type 'EXECUTE' to proceed with the trades listed above: ")
            if confirm != 'EXECUTE':
                print("🛑 Trade execution cancelled by user."); return
        except KeyboardInterrupt:
            print("\n🛑 Trade execution cancelled by user."); return

        self.logger.info("User confirmed trade execution. Proceeding...")

        for _, row in trades_to_execute.iterrows():
            symbol = row['Symbol']
            signal = row['Signal']
            details = row['Suggested Action Detail']
            trade_ticker = f"{symbol}USDT"

            try:
                if signal == 'SELL':
                    qty_match = re.search(r"which is ([\d\.e\-\+]+) " + symbol, details)
                    usd_match = re.search(r"~\$([\d,\.]+)\)", details)
                    if not (qty_match and usd_match):
                        print(f"\n⚠️ SKIPPING SELL for {symbol}: Could not parse trade details from action string.")
                        logger.warning(f"Could not parse sell details: {details}")
                        continue

                    qty_to_sell = float(qty_match.group(1))
                    sell_value_usd = float(usd_match.group(1).replace(",", ""))

                    if sell_value_usd < min_trade_usd:
                        print(f"\n⚠️ SKIPPING SELL for {symbol}: Suggested trade value (~${sell_value_usd:,.2f}) is below the minimum of ${min_trade_usd:,.2f}.")
                        continue

                    if qty_to_sell > simulated_balances.get(symbol, 0.0):
                        print(f"⚠️ SKIPPING SELL for {symbol}: Required quantity ({qty_to_sell:.8f}) exceeds simulated available balance ({simulated_balances.get(symbol, 0.0):.8f}).")
                        continue

                    if not self._redeem_from_earn_if_needed(asset=symbol, required_amount=qty_to_sell, is_live=is_live):
                        print(f"⚠️ SKIPPING SELL for {symbol} due to redemption check failure.")
                        continue

                    simulated_balances[symbol] -= qty_to_sell
                    print(f"\nPreparing MARKET SELL for {qty_to_sell:.8g} {symbol}...")

                    if is_live:
                        try:
                            print("🚀 PLACING LIVE ORDER...")
                            order = self.binance_client.order_market_sell(symbol=trade_ticker, quantity=f"{qty_to_sell:.8g}")
                            print(f"✅ LIVE SELL ORDER PLACED: {order}")
                            logger.info(f"LIVE SELL ORDER PLACED: {order}")
                        except BinanceAPIException as e:
                            print(f"❌ LIVE SELL FAILED for {symbol}: {e}")
                            logger.error(f"LIVE SELL FAILED for {symbol}: {e}")
                    else:
                        print("✅ (Dry Run) Trade was not placed.")

                elif signal == 'BUY':
                    usd_match = re.search(r"\$([\d,\.]+) worth", details)
                    if not usd_match: continue
                    buy_value_usd = float(usd_match.group(1).replace(",", ""))

                    if buy_value_usd < min_trade_usd:
                        print(f"\n⚠️ SKIPPING BUY for {symbol}: Suggested trade value (~${buy_value_usd:,.2f}) is below the minimum of ${min_trade_usd:,.2f}.")
                        continue

                    if buy_value_usd > simulated_balances.get('USDT', 0.0):
                        print(f"\n⚠️ SKIPPING BUY for {symbol}: Required USDT (${buy_value_usd:,.2f}) exceeds simulated available USDT balance (${simulated_balances.get('USDT', 0.0):,.2f}).")
                        continue

                    if not self._redeem_from_earn_if_needed(asset='USDT', required_amount=buy_value_usd, is_live=is_live):
                        print(f"⚠️ SKIPPING BUY for {symbol} due to insufficient USDT after redemption check.")
                        continue

                    simulated_balances['USDT'] -= buy_value_usd
                    print(f"\nPreparing MARKET BUY for ${buy_value_usd:,.2f} of {symbol}...")

                    if is_live:
                        try:
                            print("🚀 PLACING LIVE ORDER...")
                            order = self.binance_client.order_market_buy(symbol=trade_ticker, quoteOrderQty=f"{buy_value_usd:.2f}")
                            print(f"✅ LIVE BUY ORDER PLACED: {order}")
                            logger.info(f"LIVE BUY ORDER PLACED: {order}")
                        except BinanceAPIException as e:
                            print(f"❌ LIVE BUY FAILED for {symbol}: {e}")
                            logger.error(f"LIVE BUY FAILED for {symbol}: {e}")
                    else:
                        print("✅ (Dry Run) Trade was not placed.")

            except Exception as e:
                self.logger.error(f"An unexpected error occurred executing trade for {symbol}: {e}", exc_info=True)
                print(f"❌ Unexpected Error for {symbol}: {e}")
