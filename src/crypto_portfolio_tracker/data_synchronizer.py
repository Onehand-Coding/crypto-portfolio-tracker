"""
Data Synchronizer - Handles all external data source interactions
Moved from CryptoPortfolioTracker to separate concerns.
"""

import asyncio
import datetime
import logging
import socket
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yfinance as yf
from binance.client import Client
from binance.exceptions import BinanceAPIException
from diskcache import Cache
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
from urllib3.util.retry import Retry

from .binance_fetcher import BinanceFetcher
from .config import ConfigManager
from .database import DatabaseManager
from .exceptions import NetworkOperationError, NetworkUnavailableError
from .models import TradeResult


class DataSynchronizer:
    """
    Handles all data synchronization operations including:
    - Binance API client initialization and management
    - Current price fetching from CoinGecko
    - Historical price data from CoinGecko
    - Fiat exchange rates
    - yFinance ticker data
    - Full portfolio data synchronization
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        db_manager: DatabaseManager,
        binance_client: Optional[Client] = None,
        fetcher: Optional[BinanceFetcher] = None,
        cache_dir: Path = None,
        coingecko_price_cache: Cache = None,
        yfinance_disk_cache: Cache = None,
        fiat_exchange_rate_cache: Cache = None,
        symbol_mappings: Dict = None,
        offline_mode: bool = False,
    ):
        """
        Initialize DataSynchronizer with necessary dependencies.

        Args:
            config_manager: Configuration manager instance
            db_manager: Database manager instance
            binance_client: Optional Binance client instance
            fetcher: Optional BinanceFetcher instance
            cache_dir: Cache directory path
            coingecko_price_cache: CoinGecko price cache
            yfinance_disk_cache: yFinance disk cache
            fiat_exchange_rate_cache: Fiat exchange rate cache
            symbol_mappings: Symbol mappings dictionary
            offline_mode: Whether operating in offline mode
        """
        self.config_manager = config_manager
        self.config = self.config_manager.config
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.binance_client = binance_client
        self.fetcher = fetcher
        self.symbol_mappings = symbol_mappings or {}
        self.offline_mode = offline_mode

        # Cache setup
        self.cache_dir = cache_dir or Path(
            self.config.get("cache", {}).get("path", "data/cache")
        )
        self.coingecko_price_cache = coingecko_price_cache
        self.yfinance_disk_cache = yfinance_disk_cache
        self.fiat_exchange_rate_cache = fiat_exchange_rate_cache

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
        except Exception:
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
        self,
        symbol: str,
        target_date: datetime.date,
        coin_id: Optional[str] = None,
        fallback_to_nearest: bool = True,
    ) -> Optional[float]:
        """
        Get historical price for a symbol from CoinGecko with disk caching.
        """
        symbol_upper = symbol.upper()

        if not coin_id:
            coin_id = self.symbol_mappings.get(symbol_upper)

        if not coin_id:
            self.logger.warning(
                f"No CoinGecko mapping found for symbol {symbol_upper}."
            )
            return None

        cache_key = f"{coin_id}_{target_date.strftime('%Y%m%d')}"

        # Check cache first
        if self.coingecko_price_cache and cache_key in self.coingecko_price_cache:
            cached_price = self.coingecko_price_cache[cache_key]
            if cached_price is not None:
                return cached_price

        coingecko_config = self.config.get("apis", {}).get("coingecko", {})
        base_url = coingecko_config.get("base_url")
        timeout = coingecko_config.get("timeout", 30)

        target_date_str = target_date.strftime("%d-%m-%Y")
        url = f"{base_url}/coins/{coin_id}/history"
        params = {"date": target_date_str}

        try:
            self.logger.debug(
                f"Fetching historical price for {symbol_upper} ({coin_id}) on {target_date_str}..."
            )
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            # Extract USD price
            market_data = data.get("market_data", {})
            current_price = market_data.get("current_price", {})
            price = current_price.get("usd")

            if price is not None:
                self.logger.debug(
                    f"Successfully fetched historical price for {symbol_upper}: ${price:.6f}"
                )
                # Cache the result
                if self.coingecko_price_cache:
                    self.coingecko_price_cache[cache_key] = price
                return price
            else:
                self.logger.warning(
                    f"No USD price found in CoinGecko response for {symbol_upper} on {target_date_str}"
                )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                self.logger.warning(
                    f"Rate limited fetching historical price for {symbol_upper}. Consider implementing delays."
                )
            else:
                self.logger.error(
                    f"HTTP error fetching historical price for {symbol_upper}: {e}"
                )
        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Request error fetching historical price for {symbol_upper}: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching historical price for {symbol_upper}: {e}"
            )

        # If we reach here, the exact date failed
        if not fallback_to_nearest:
            if self.coingecko_price_cache:
                self.coingecko_price_cache[cache_key] = None
            return None

        # Try fallback to nearest date
        self.logger.info(
            f"Attempting fallback to nearest date for {symbol_upper} around {target_date_str}..."
        )

        for days_offset in range(1, 8):  # Try up to 7 days before and after
            for direction in [-1, 1]:  # Try both before and after
                fallback_date = target_date + timedelta(days=days_offset * direction)
                fallback_date_str = fallback_date.strftime("%d-%m-%Y")
                fallback_cache_key = f"{coin_id}_{fallback_date.strftime('%Y%m%d')}"

                # Check cache for fallback date
                if (
                    self.coingecko_price_cache
                    and fallback_cache_key in self.coingecko_price_cache
                ):
                    cached_fallback = self.coingecko_price_cache[fallback_cache_key]
                    if cached_fallback is not None:
                        self.logger.info(
                            f"Using cached fallback price for {symbol_upper} on {fallback_date_str}: ${cached_fallback:.6f}"
                        )
                        # Cache this result for the original date too
                        self.coingecko_price_cache[cache_key] = cached_fallback
                        return cached_fallback

                # Fetch fallback date
                fallback_params = {"date": fallback_date_str}
                try:
                    response = requests.get(
                        url, params=fallback_params, timeout=timeout
                    )
                    response.raise_for_status()
                    data = response.json()

                    market_data = data.get("market_data", {})
                    current_price = market_data.get("current_price", {})
                    price = current_price.get("usd")

                    if price is not None:
                        self.logger.info(
                            f"Fallback successful for {symbol_upper} on {fallback_date_str}: ${price:.6f}"
                        )
                        # Cache both the fallback date and original date
                        if self.coingecko_price_cache:
                            self.coingecko_price_cache[fallback_cache_key] = price
                            self.coingecko_price_cache[cache_key] = price
                        return price

                except requests.exceptions.RequestException:
                    continue  # Try next fallback date

        self.logger.error(
            f"Failed to get historical price for {symbol_upper} around {target_date_str} after trying fallback dates"
        )
        if self.coingecko_price_cache:
            self.coingecko_price_cache[cache_key] = None
        return None

    def _get_historical_fiat_exchange_rate(
        self,
        target_date: datetime.date,
        from_currency: str = "USD",
        to_currency: str = "CAD",
    ) -> Optional[float]:
        """
        Get historical fiat exchange rate using Exchange Rates API with disk caching.
        """
        if from_currency == to_currency:
            return 1.0

        cache_key = f"{from_currency}_{to_currency}_{target_date.strftime('%Y%m%d')}"

        # Check cache first
        if self.fiat_exchange_rate_cache and cache_key in self.fiat_exchange_rate_cache:
            cached_rate = self.fiat_exchange_rate_cache[cache_key]
            if cached_rate is not None:
                return cached_rate

        # Get exchange rates API config
        exchange_config = self.config.get("apis", {}).get("exchange_rates", {})
        base_url = exchange_config.get(
            "base_url", "https://api.exchangerate-api.com/v4"
        )
        timeout = exchange_config.get("timeout", 10)

        target_date_str = target_date.strftime("%Y-%m-%d")
        url = f"{base_url}/historical/{target_date_str}"

        try:
            self.logger.debug(
                f"Fetching historical exchange rate {from_currency} to {to_currency} for {target_date_str}..."
            )
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            # Get the base currency (should be USD for this API)
            base_currency = data.get("base")
            rates = data.get("rates", {})

            if base_currency != from_currency:
                self.logger.warning(
                    f"Expected base currency {from_currency}, got {base_currency}"
                )

            rate = rates.get(to_currency)
            if rate is not None:
                self.logger.debug(
                    f"Successfully fetched exchange rate {from_currency}/{to_currency} for {target_date_str}: {rate:.6f}"
                )
                # Cache the result
                if self.fiat_exchange_rate_cache:
                    self.fiat_exchange_rate_cache[cache_key] = rate
                return rate
            else:
                self.logger.warning(
                    f"No {to_currency} rate found for {target_date_str}"
                )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                self.logger.warning(
                    f"Rate limited fetching exchange rate for {target_date_str}"
                )
            else:
                self.logger.error(
                    f"HTTP error fetching exchange rate for {target_date_str}: {e}"
                )
        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Request error fetching exchange rate for {target_date_str}: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching exchange rate for {target_date_str}: {e}"
            )

        # Cache the failure
        if self.fiat_exchange_rate_cache:
            self.fiat_exchange_rate_cache[cache_key] = None
        return None

    def _get_yfinance_ticker(self, symbol: str) -> Optional[yf.Ticker]:
        """
        Get yFinance ticker object with disk caching for performance.
        """
        cache_key = f"ticker_{symbol}"

        # Check if ticker is in cache
        if self.yfinance_disk_cache and cache_key in self.yfinance_disk_cache:
            try:
                cached_ticker = self.yfinance_disk_cache[cache_key]
                if cached_ticker:
                    return cached_ticker
            except Exception as e:
                self.logger.warning(f"Failed to load cached ticker for {symbol}: {e}")

        try:
            # Create new ticker
            ticker = yf.Ticker(symbol)

            # Cache the ticker object
            if self.yfinance_disk_cache:
                self.yfinance_disk_cache[cache_key] = ticker

            return ticker
        except Exception as e:
            self.logger.error(f"Failed to create yFinance ticker for {symbol}: {e}")
            return None

    async def sync_data(self, enricher=None):
        """
        Orchestrates the Gather, Enrich, and Process pipeline, now intelligently
        skipping unsupported data sources when in Testnet mode.
        """
        if self.offline_mode:
            self.logger.info("Skipping data sync - offline mode active")
            return True

        self.logger.info("Starting data synchronization pipeline...")
        lookback_config = self.config.get("history_lookback_days", {})

        # Define all possible data sources for production mode
        all_data_sources = {
            "Binance Trade": {
                "fetcher": self.fetcher.fetch_binance_transactions,
                "days": lookback_config.get("trades", 90),
            },
            "Binance Transfer": {
                "fetcher": self.fetcher.fetch_transfer_history,
                "days": lookback_config.get("transfers", 90),
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
            "Binance P2P Sell": {
                "fetcher": self.fetcher.fetch_p2p_usdt_sells,
                "days": lookback_config.get("p2p_sells", 90),
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
                asyncio.ensure_future(
                    asyncio.to_thread(
                        config["fetcher"],
                        source_name=source,
                        days_back=config["days"],
                        latest_known_ts=latest_known_ts,
                    )
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
                asyncio.ensure_future(
                    asyncio.to_thread(
                        self.fetcher.fetch_staking_history,
                        source_name="Binance Staking",
                        days_back=lookback_config.get("staking_history", 90),
                        latest_known_ts_map=staking_ts_map,
                    )
                )
            )

        if not fetcher_tasks:
            self.logger.info("No data sources to sync in the current mode.")
            return True

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
            return True

        # 2. ENRICH
        self.logger.info(
            f"Passing {len(all_raw_transactions)} raw transactions to the PriceEnricher."
        )
        if enricher:
            enriched_transactions = await enricher.enrich_transactions(
                all_raw_transactions
            )
        else:
            self.logger.warning("No enricher provided, cannot enrich transactions")
            return False

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

        self.logger.info("Data synchronization pipeline finished successfully.")
        return True

    async def run_full_sync(self, enricher=None) -> bool:
        """Run a complete data synchronization."""
        return await self.sync_data(enricher)

    async def transfer_funding_to_spot(
        self, asset: str, amount: float, is_live: bool = False
    ) -> "TradeResult":
        """
        Transfer funds from funding wallet to spot wallet using Binance API.

        Args:
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)

        Returns:
            TradeResult with success status, messages, and any errors
        """
        return await self._execute_wallet_transfer(
            "FUNDING_MAIN", asset, amount, is_live, "funding", "spot"
        )

    async def transfer_spot_to_funding(
        self, asset: str, amount: float, is_live: bool = False
    ) -> "TradeResult":
        """
        Transfer funds from spot wallet to funding wallet using Binance API.

        Args:
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)

        Returns:
            TradeResult with success status, messages, and any errors
        """
        return await self._execute_wallet_transfer(
            "MAIN_FUNDING", asset, amount, is_live, "spot", "funding"
        )

    async def transfer_spot_to_futures(
        self, asset: str, amount: float, is_live: bool = False
    ) -> "TradeResult":
        """
        Transfer funds from spot wallet to futures wallet using Binance API.

        Args:
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)

        Returns:
            TradeResult with success status, messages, and any errors
        """
        return await self._execute_wallet_transfer(
            "MAIN_UMFUTURE", asset, amount, is_live, "spot", "futures"
        )

    async def transfer_futures_to_spot(
        self, asset: str, amount: float, is_live: bool = False
    ) -> "TradeResult":
        """
        Transfer funds from futures wallet to spot wallet using Binance API.

        Args:
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)

        Returns:
            TradeResult with success status, messages, and any errors
        """
        return await self._execute_wallet_transfer(
            "UMFUTURE_MAIN", asset, amount, is_live, "futures", "spot"
        )

    async def transfer_funding_to_futures(
        self, asset: str, amount: float, is_live: bool = False
    ) -> "TradeResult":
        """
        Transfer funds from funding wallet to futures wallet using Binance API.

        Args:
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)

        Returns:
            TradeResult with success status, messages, and any errors
        """
        return await self._execute_wallet_transfer(
            "FUNDING_UMFUTURE", asset, amount, is_live, "funding", "futures"
        )

    async def transfer_futures_to_funding(
        self, asset: str, amount: float, is_live: bool = False
    ) -> "TradeResult":
        """
        Transfer funds from futures wallet to funding wallet using Binance API.

        Args:
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)

        Returns:
            TradeResult with success status, messages, and any errors
        """
        return await self._execute_wallet_transfer(
            "UMFUTURE_FUNDING", asset, amount, is_live, "futures", "funding"
        )

    async def _execute_wallet_transfer(
        self,
        transfer_type: str,
        asset: str,
        amount: float,
        is_live: bool,
        source_wallet: str,
        dest_wallet: str,
    ) -> "TradeResult":
        """
        Internal method to execute wallet transfers using Binance API.

        Args:
            transfer_type: Binance transfer type (e.g., "FUNDING_MAIN")
            asset: Asset to transfer (e.g., 'USDT')
            amount: Amount to transfer
            is_live: Whether to execute the transfer (False for dry run)
            source_wallet: Source wallet name for display
            dest_wallet: Destination wallet name for display

        Returns:
            TradeResult with success status, messages, and any errors
        """
        from .models import TradeResult

        result = TradeResult(success=False)

        try:
            # Validate inputs
            if not asset or amount <= 0:
                result.errors.append("Invalid asset or amount")
                return result

            # Check source wallet balance first
            source_balance = 0.0
            if source_wallet == "funding":
                balances = self.fetcher.fetch_funding_balance()
                for balance in balances:
                    if balance.get("asset") == asset:
                        source_balance = float(balance.get("free", 0.0))
                        break
            elif source_wallet == "spot":
                balances = self.fetcher.fetch_binance_balances()
                for _, row in balances.iterrows():
                    if row["symbol"] == asset:
                        source_balance = float(row["quantity"])
                        break
            elif source_wallet == "futures":
                balances = self.fetcher.fetch_futures_balance()
                for balance in balances:
                    if balance.get("asset") == asset:
                        source_balance = float(balance.get("balance", 0.0))
                        break

            if source_balance < amount:
                result.errors.append(
                    f"Insufficient {asset} in {source_wallet} wallet. Available: {source_balance:.8f}, Required: {amount:.8f}"
                )
                return result

            result.messages.append(
                f"✅ {source_wallet.capitalize()} wallet has sufficient {asset} balance: {source_balance:.8f}"
            )

            if not is_live:
                result.messages.append(
                    f"(Dry Run) Would transfer {amount:.8f} {asset} from {source_wallet} to {dest_wallet} wallet"
                )
                result.success = True
                return result

            # Execute the transfer using Binance Asset Transfer API
            transfer_params = {
                "type": transfer_type,
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
                result.data["transfer_id"] = transfer_result["tranId"]
                result.messages.append(
                    f"✅ Successfully transferred {amount:.8f} {asset} from {source_wallet} to {dest_wallet} wallet"
                )
                result.messages.append(f"Transfer ID: {transfer_result['tranId']}")
                result.success = True
            else:
                result.errors.append(
                    f"Transfer failed - no transfer ID returned. Response: {transfer_result}"
                )

        except BinanceAPIException as e:
            error_msg = f"Binance API error during transfer: {e}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during transfer: {e}"
            self.logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)

        return result
