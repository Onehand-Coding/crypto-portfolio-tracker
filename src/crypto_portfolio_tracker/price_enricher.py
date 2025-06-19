import asyncio
import logging
import datetime
from typing import List, Dict, Any, Optional, Set, Tuple

import pytz
import httpx
import pandas as pd
import yfinance as yf
from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, wait_exponential

from .symbol_mapper import SymbolMapper

logger = logging.getLogger(__name__)


class PriceEnricher:
    """Takes raw transactions and enriches them with historical USD prices."""

    def __init__(self, symbol_mapper: SymbolMapper, config: Dict[str, Any], disk_cache: Cache):
        self.symbol_mappings = symbol_mapper
        self.config = config
        self.coingecko_config = config.get("apis", {}).get("coingecko", {})
        self.stablecoin_symbols = config.get("portfolio", {}).get("stablecoin_symbols", [])
        self.disk_cache = disk_cache
        self.price_cache: Dict[str, Optional[float]] = {}
        self.fiat_rate_cache: Dict[str, Optional[float]] = {}
        self.target_assets_for_sync = set(self.config.get("target_allocation", {}).keys())
        self.target_assets_for_sync.add("USDT")
        self.coingecko_semaphore = asyncio.Semaphore(5)

        # --- Log whether we are using an API key ---
        if self.coingecko_config.get("api_key"):
            logger.info("PriceEnricher initialized with a CoinGecko API Key.")
        else:
            logger.info("PriceEnricher initialized without a CoinGecko API Key (using public access).")
        logger.info(f"Concurrency limit set to {self.coingecko_semaphore._value} for price fetching.")

    def _get_price_from_cache(self, symbol: str, date: datetime.datetime) -> float:
        if not symbol: return 0.0
        if symbol in self.stablecoin_symbols: return 1.0

        coin_id = self.symbol_mappings.get(symbol)
        if not coin_id: return 0.0

        date_str = date.strftime('%d-%m-%Y')
        cache_key = f"{coin_id}_{date_str}"

        # 1. Check in-memory cache first for performance
        price = self.price_cache.get(cache_key)
        if price is not None:
            return price

        # 2. If not in memory, check the disk cache
        price = self.disk_cache.get(cache_key)
        if price is not None:
            self.price_cache[cache_key] = price # Add to memory cache for next time
            return price

        # 3. If not in any cache, return 0.0
        return 0.0

    def _get_historical_fiat_exchange_rate(self, date: datetime.datetime, from_currency: str, to_currency: str) -> float:
        """
        Fetches and caches historical fiat exchange rates using yfinance.
        This version ensures timezone consistency and handles ambiguous Series truth values.
        """
        if from_currency == to_currency: return 1.0

        date_str = date.strftime('%Y-%m-%d')
        cache_key = f"yfinance_{date_str}_{from_currency}_{to_currency}"
        if cache_key in self.fiat_rate_cache:
            return self.fiat_rate_cache.get(cache_key) or 0.0

        ticker = f"{from_currency.upper()}{to_currency.upper()}=X"
        rate = None
        try:
            start_date_fetch = (date - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
            end_date_fetch = (date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')

            data = yf.download(ticker, start=start_date_fetch, end=end_date_fetch, progress=False, auto_adjust=True)

            if not data.empty:
                if data.index.tz is None:
                    data = data.tz_localize('UTC')

                # The .asof() method can sometimes return a Series. We handle this explicitly.
                price_or_series = data['Close'].asof(date)

                closest_price = None
                if isinstance(price_or_series, pd.Series):
                    if not price_or_series.empty:
                        closest_price = price_or_series.iloc[0] # Take the first element if it's a Series
                else:
                    closest_price = price_or_series # It was a single value as expected

                if pd.notna(closest_price):
                    rate = float(closest_price)

        except Exception as e:
            logger.warning(f"Could not fetch yfinance rate for {ticker} on {date_str}: {e}")

        self.fiat_rate_cache[cache_key] = rate
        return rate or 0.0

    async def _fetch_price_for_date(self, client: httpx.AsyncClient, coin_id: str, date_str: str):
        """Asynchronously fetches a single historical price, optionally using an API key."""
        cache_key = f"{coin_id}_{date_str}"
        if self.price_cache.get(cache_key) is not None or self.disk_cache.get(cache_key) is not None:
            if self.price_cache.get(cache_key) is None:
                self.price_cache[cache_key] = self.disk_cache.get(cache_key)
            return

        async with self.coingecko_semaphore:
            await asyncio.sleep(self.coingecko_config.get("request_delay_ms", 1000) / 1000.0) # Reduced delay
            base_url = self.coingecko_config.get("base_url", "https://api.coingecko.com/api/v3")
            url = f"{base_url}/coins/{coin_id}/history"

            # --- FIX: Add API key to params if it exists ---
            params = {'date': date_str, 'localization': 'false'}
            api_key = self.coingecko_config.get("api_key")
            if api_key:
                params['x_cg_demo_api_key'] = api_key

            try:
                response = await client.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                price = data.get("market_data", {}).get("current_price", {}).get("usd")
                if price is not None:
                    logger.info(f"API SUCCESS: Fetched price for {coin_id} on {date_str}: ${price}")
                    self.disk_cache.set(cache_key, price)
                    self.price_cache[cache_key] = float(price)
                else:
                    self.price_cache[cache_key] = None
            except httpx.HTTPStatusError as e:
                logger.warning(f"API FAILED for {coin_id} on {date_str}: {e}")
                self.price_cache[cache_key] = None
            except Exception as e:
                logger.error(f"An unexpected error in _fetch_price_for_date for {coin_id}: {e}")
                self.price_cache[cache_key] = None

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def _batch_fetch_prices(self, price_requests: Set[Tuple[str, str]]):
        """
        Manages batch fetching with retries. This version checks for success *after* the
        batch and raises an error to trigger a retry if any requests failed.
        """
        if not price_requests:
            return

        logger.info(f"Starting controlled batch fetch for {len(price_requests)} unique prices...")

        # Only fetch what's missing.
        missing_requests = {(coin_id, date_str) for coin_id, date_str in price_requests
                            if self.price_cache.get(f"{coin_id}_{date_str}") is None and self.disk_cache.get(f"{coin_id}_{date_str}") is None}

        if not missing_requests:
            logger.info("All requested prices were already in the cache. Nothing to fetch.")
            return

        logger.info(f"Fetching {len(missing_requests)} missing prices...")

        async with httpx.AsyncClient() as client:
            tasks = [self._fetch_price_for_date(client, coin_id, date_str) for coin_id, date_str in missing_requests]
            await asyncio.gather(*tasks)

        # After the batch, check if all requests were successful.
        all_successful = True
        for coin_id, date_str in missing_requests:
            cache_key = f"{coin_id}_{date_str}"
            if self.price_cache.get(cache_key) is None and self.disk_cache.get(cache_key) is None:
                all_successful = False
                logger.warning(f"Verification failed: Price for {coin_id} on {date_str} is still missing after batch.")
                break

        if not all_successful:
            logger.error("One or more price fetches failed in the batch. Raising error to trigger retry.")
            raise httpx.HTTPStatusError(
                "Batch fetch incomplete, triggering retry.",
                request=None,
                response=httpx.Response(status_code=429)
            )

        logger.info("Batch price fetch complete and all prices verified.")

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(10)
    )
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Asynchronously fetches current prices for a list of symbols, with retries and optional API key."""
        if not symbols:
            return {}

        prices: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        ids_to_fetch = {self.symbol_mappings.get(s.upper()) for s in symbols if self.symbol_mappings.get(s.upper())}
        if not ids_to_fetch:
            return prices

        ids_string = ",".join(list(ids_to_fetch))
        url = f"{self.coingecko_config.get('base_url')}/simple/price"

        # --- Add API key to params if it exists ---
        params = {'ids': ids_string, 'vs_currencies': 'usd'}
        api_key = self.coingecko_config.get("api_key")
        if api_key:
            params['x_cg_demo_api_key'] = api_key

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, timeout=self.coingecko_config.get("timeout", 30))
                response.raise_for_status()
                fetched_price_data = response.json()

                for symbol in symbols:
                    coin_id = self.symbol_mappings.get(symbol.upper())
                    if coin_id in fetched_price_data:
                        prices[symbol] = fetched_price_data[coin_id].get("usd", 0.0)
                return prices

            except httpx.HTTPStatusError as e:
                logger.warning(f"Rate limited fetching current prices: {e}. Retrying...")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred while fetching current prices: {e}")
                return prices
        return prices

    async def enrich_transactions(self, raw_txs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Orchestrates the enrichment process, correctly handling transaction-specific logic.
        """
        if not raw_txs: return []

        # 1. Gather all unique price requests
        price_requests: Set[Tuple[str, str]] = set()
        for tx in raw_txs:
            if not tx.get('timestamp') or not isinstance(tx.get('timestamp'), datetime.datetime): continue
            date_str = tx['timestamp'].strftime('%d-%m-%Y')
            raw = tx['raw_data']
            symbols_to_price = []

            tx_type = tx['tx_type']
            if tx_type == 'TRADE':
                symbols_to_price.extend([raw.get('quote_asset'), raw.get('fee_currency')])
            elif tx_type == 'WITHDRAWAL':
                symbols_to_price.extend([raw.get('symbol'), raw.get('fee_currency')])
            elif tx_type == 'CONVERT':
                symbols_to_price.extend([raw.get('from_asset'), raw.get('to_asset')])
            elif tx_type != 'P2P_BUY':
                symbols_to_price.append(raw.get('symbol'))

            for symbol in symbols_to_price:
                if symbol and symbol not in self.stablecoin_symbols:
                    coin_id = self.symbol_mappings.get(symbol)
                    if coin_id: price_requests.add((coin_id, date_str))

        # 2. Batch-fetch all prices
        await self._batch_fetch_prices(price_requests)

        # 3. Apply prices and transform raw data into the final format
        enriched_txs = []
        for tx in raw_txs:
            raw = tx['raw_data']
            ts = tx['timestamp']

            if tx['tx_type'] == 'CONVERT':
                from_asset, to_asset = raw['from_asset'], raw['to_asset']
                # Only create the SELL leg if the asset being sold is a target asset
                if from_asset in self.target_assets_for_sync:
                    price_usd = self._get_price_from_cache(from_asset, ts)
                    enriched_txs.append({'symbol': from_asset, 'timestamp': ts, 'type': 'SELL', 'quantity': raw['from_quantity'], 'price_usd': price_usd, 'fee_quantity': 0, 'fee_currency': None, 'fee_usd': 0, 'source': tx['source'], 'transaction_hash': f"convert_sell_{tx['transaction_hash']}", 'notes': f"Converted to {raw['to_quantity']} {to_asset}"})
                # Only create the BUY leg if the asset being bought is a target asset
                if to_asset in self.target_assets_for_sync:
                    price_usd = self._get_price_from_cache(to_asset, ts)
                    enriched_txs.append({'symbol': to_asset, 'timestamp': ts, 'type': 'BUY', 'quantity': raw['to_quantity'], 'price_usd': price_usd, 'fee_quantity': 0, 'fee_currency': None, 'fee_usd': 0, 'source': tx['source'], 'transaction_hash': f"convert_buy_{tx['transaction_hash']}", 'notes': f"Converted from {raw['from_quantity']} {from_asset}"})

            elif tx['tx_type'] == 'TRADE':
                quote_price_usd = self._get_price_from_cache(raw['quote_asset'], ts)
                price_usd = raw['price'] * quote_price_usd
                fee_price_usd = self._get_price_from_cache(raw['fee_currency'], ts)
                fee_usd = raw['fee_quantity'] * fee_price_usd
                enriched_txs.append({'symbol': raw['base_asset'], 'timestamp': ts, 'type': 'BUY' if raw['is_buyer'] else 'SELL', 'quantity': raw['quantity'], 'price_usd': price_usd, 'fee_quantity': raw['fee_quantity'], 'fee_currency': raw['fee_currency'], 'fee_usd': fee_usd, 'source': tx['source'], 'transaction_hash': tx['transaction_hash'], 'notes': f"Pair: {raw['base_asset']}/{raw['quote_asset']}"})
                if raw['quote_asset'] in self.target_assets_for_sync:
                    enriched_txs.append({'symbol': raw['quote_asset'], 'timestamp': ts, 'type': 'SELL' if raw['is_buyer'] else 'BUY', 'quantity': raw['quantity'] * raw['price'], 'price_usd': quote_price_usd, 'fee_quantity': 0, 'fee_currency': None, 'fee_usd': 0, 'source': 'Binance Synthetic', 'transaction_hash': f"synth_{tx['transaction_hash']}", 'notes': f"Synthetic for {raw['base_asset']}/{raw['quote_asset']} trade"})

            elif tx['tx_type'] in ['DEPOSIT', 'EARN_REWARD', 'DIVIDEND', 'STAKING_INTEREST', 'EARN_REDEMPTION', 'STAKING_REDEMPTION']:
                price_usd = self._get_price_from_cache(raw['symbol'], ts)
                enriched_txs.append({'symbol': raw['symbol'], 'timestamp': ts, 'type': 'DEPOSIT', 'quantity': raw['quantity'], 'price_usd': price_usd, 'fee_quantity': 0, 'fee_currency': None, 'fee_usd': 0, 'source': tx['source'], 'transaction_hash': tx['transaction_hash'], 'notes': raw.get('notes', '')})

            elif tx['tx_type'] in ['WITHDRAWAL', 'EARN_SUBSCRIPTION', 'STAKING_SUBSCRIBE']:
                price_usd = self._get_price_from_cache(raw['symbol'], ts)
                fee_price_usd = self._get_price_from_cache(raw.get('fee_currency', raw['symbol']), ts)
                fee_usd = raw.get('fee_quantity', 0.0) * fee_price_usd
                tx_type_final = 'WITHDRAWAL' if tx['tx_type'] == 'WITHDRAWAL' else 'SELL'
                enriched_txs.append({'symbol': raw['symbol'], 'timestamp': ts, 'type': tx_type_final, 'quantity': raw['quantity'], 'price_usd': price_usd, 'fee_quantity': raw.get('fee_quantity', 0.0), 'fee_currency': raw.get('fee_currency'), 'fee_usd': fee_usd, 'source': tx['source'], 'transaction_hash': tx['transaction_hash'], 'notes': raw.get('notes', '')})

            elif tx['tx_type'] == 'P2P_BUY':
                fiat_rate = self._get_historical_fiat_exchange_rate(ts, raw['fiat_currency'], 'USD')
                price_usd = (raw['fiat_amount'] * fiat_rate) / raw['quantity'] if raw['quantity'] > 0 else 0
                enriched_txs.append({'symbol': raw['asset'], 'timestamp': ts, 'type': 'BUY', 'quantity': raw['quantity'], 'price_usd': price_usd, 'fee_quantity': 0, 'fee_currency': None, 'fee_usd': 0, 'source': tx['source'], 'transaction_hash': tx['transaction_hash'], 'notes': f"P2P Buy: {raw['fiat_amount']} {raw['fiat_currency']}"})

        logger.info(f"Enrichment complete. Returning {len(enriched_txs)} final transactions.")
        return enriched_txs
