import asyncio
import logging
import datetime
from typing import List, Dict, Any, Optional, Set, Tuple

import httpx
import yfinance as yf
from diskcache import Cache

from src.symbol_mapper import SymbolMapper

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
        logger.info("PriceEnricher initialized with a concurrency limit of 5 for price fetching.")

    def _get_price_from_cache(self, symbol: str, date: datetime.datetime) -> float:
        if not symbol: return 0.0
        if symbol in self.stablecoin_symbols: return 1.0
        coin_id = self.symbol_mappings.get(symbol)
        if not coin_id: return 0.0
        date_str = date.strftime('%d-%m-%Y')
        price = self.price_cache.get(f"{coin_id}_{date_str}")
        return price if price is not None else 0.0

    def _get_historical_fiat_exchange_rate(self, date: datetime.datetime, from_currency: str, to_currency: str) -> float:
        """Fetches and caches historical fiat exchange rates using yfinance, now robust against Series objects."""
        if from_currency == to_currency: return 1.0

        date_str = date.strftime('%Y-%m-%d')
        cache_key = f"yfinance_{date_str}_{from_currency}_{to_currency}"
        if cache_key in self.fiat_rate_cache:
            return self.fiat_rate_cache.get(cache_key) or 0.0

        ticker = f"{from_currency.upper()}{to_currency.upper()}=X"
        rate = None
        try:
            # Fetch a small window around the target date for reliability
            start_date_fetch = (date - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
            end_date_fetch = (date + datetime.timedelta(days=2)).strftime('%Y-%m-%d')

            data = yf.download(ticker, start=start_date_fetch, end=end_date_fetch, progress=False, auto_adjust=True)

            if not data.empty:
                # Use .asof to find the last known value on or before the requested date
                closest_price = data['Close'].asof(date)
                if pd.notna(closest_price):
                    rate = float(closest_price)
        except Exception as e:
            logger.warning(f"Could not fetch yfinance rate for {ticker} on {date_str}: {e}")

        self.fiat_rate_cache[cache_key] = rate
        return rate or 0.0

    async def _fetch_price_for_date(self, client: httpx.AsyncClient, coin_id: str, date_str: str):
        """Asynchronously fetches a single historical price, respecting the semaphore and all caches."""
        cache_key = f"{coin_id}_{date_str}"

        #  Check the in-memory cache first
        if cache_key in self.price_cache:
            return

        # Then check the disk cache
        if self.disk_cache.get(cache_key) is not None:
            self.price_cache[cache_key] = self.disk_cache.get(cache_key)
            return

        # If not in any cache, proceed to fetch from API
        async with self.coingecko_semaphore:
            await asyncio.sleep(self.coingecko_config.get("request_delay_ms", 1500) / 1000.0)
            base_url = self.coingecko_config.get("base_url", "https://api.coingecko.com/api/v3")
            url = f"{base_url}/coins/{coin_id}/history?date={date_str}&localization=false"

            try:
                response = await client.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                price = data.get("market_data", {}).get("current_price", {}).get("usd")

                if price is not None:
                    logger.info(f"API SUCCESS: Fetched price for {coin_id} on {date_str}: ${price}")
                    self.disk_cache.set(cache_key, price)
                    self.price_cache[cache_key] = float(price)
                else:
                    self.price_cache[cache_key] = None
            except Exception as e:
                logger.warning(f"API FAILED for {coin_id} on {date_str}: {e}")
                self.price_cache[cache_key] = None

    async def _batch_fetch_prices(self, price_requests: Set[Tuple[str, str]]):
        if not price_requests: return
        logger.info(f"Starting controlled batch fetch for {len(price_requests)} unique prices...")
        async with httpx.AsyncClient() as client:
            tasks = [self._fetch_price_for_date(client, coin_id, date_str) for coin_id, date_str in price_requests]
            await asyncio.gather(*tasks)
        logger.info("Batch price fetch complete.")

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Asynchronously fetches current prices for a list of symbols, with retries."""
        if not symbols:
            return {}

        prices: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        ids_to_fetch = {self.symbol_mappings.get(s.upper()) for s in symbols if self.symbol_mappings.get(s.upper())}
        if not ids_to_fetch:
            return prices

        ids_string = ",".join(list(ids_to_fetch))
        url = f"{self.coingecko_config.get('base_url')}/simple/price"
        params = {'ids': ids_string, 'vs_currencies': 'usd'}

        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                try:
                    response = await client.get(url, params=params, timeout=self.coingecko_config.get("timeout", 30))
                    response.raise_for_status()
                    fetched_price_data = response.json()

                    # Map prices back
                    for symbol in symbols:
                        coin_id = self.symbol_mappings.get(symbol.upper())
                        if coin_id in fetched_price_data:
                            prices[symbol] = fetched_price_data[coin_id].get("usd", 0.0)
                    return prices # Return on success

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < 2:
                        logger.warning("Rate limited fetching current prices. Waiting 10 seconds...")
                        await asyncio.sleep(10)
                    else:
                        logger.error(f"Failed to fetch current prices: {e}")
                        return prices # Return default prices on final failure
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
