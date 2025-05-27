"""
Crypto Portfolio Tracker - Main Class
Handles API interactions, data processing, analysis, and orchestration.
"""
import os
import json
import time
import logging
import datetime
from collections import deque
from typing import Dict, Any, Optional, List

import requests
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from config import ConfigManager
from database import DatabaseManager
from exporters import ExcelExporter, HtmlExporter, CsvExporter
from visualizations import Visualizer

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
    logger = logging.getLogger(__name__)

    if transactions_df.empty:
        logger.debug(f"No transactions provided to calculate_fifo_cost_basis for symbol {transactions_df['symbol'].iloc[0] if not transactions_df.empty else 'Unknown'}. Returning 0, 0.")
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

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the tracker."""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.db_manager = DatabaseManager(self.config)
        self.excel_exporter = ExcelExporter(self.config)
        self.html_exporter = HtmlExporter(self.config)
        self.csv_exporter = CsvExporter(self.config)
        self.visualizer = Visualizer(self.config)
        self.binance_client = self._init_binance_client()
        self.coingecko_api = self.config.get("apis", {}).get("coingecko", {})
        self.binance_api_config = self.config.get("apis", {}).get("binance", {})
        self.symbol_mappings = self.config.get("symbol_mappings", {}).get("coingecko_ids", {})
        self.norm_map = self.config.get("symbol_normalization_map", {})
        self.historical_price_cache: Dict[str, Optional[float]] = {}

    def _init_binance_client(self) -> Optional[Client]:
        """Initialize and return Binance client."""
        api_keys = self.config.get("api_keys", {}); api_key = api_keys.get("binance_key"); api_secret = api_keys.get("binance_secret")
        if not api_key or not api_secret: logger.warning("Binance API key/secret not found. Binance disabled."); return None
        try:
            client = Client(api_key, api_secret, requests_params={'timeout': 120})
            if self.config.get("apis", {}).get("binance", {}).get("testnet", False): client.API_URL = 'https://testnet.binance.vision/api'
            client.ping(); logger.info("Binance client initialized."); return client
        except Exception as e: logger.error(f"Failed to init Binance client: {e}"); return None

    def _get_coingecko_price(self, coin_id: str) -> Optional[float]:
        """Fetch current price from CoinGecko."""
        base_url = self.coingecko_api.get("base_url"); timeout = self.coingecko_api.get("timeout", 30)
        url = f"{base_url}/simple/price?ids={coin_id}&vs_currencies=usd"
        try:
            response = requests.get(url, timeout=timeout); response.raise_for_status(); data = response.json()
            return data.get(coin_id, {}).get("usd")
        except requests.exceptions.RequestException as e: logger.error(f"CoinGecko price fetch error for {coin_id}: {e}"); return None

    def _get_coingecko_historical_price(self, coin_id: str, date_str: str) -> Optional[float]:
        """
        Fetches the historical price of a coin from CoinGecko for a specific date.
        Uses a cache and includes robust retry logic with exponential backoff.
        Date string should be in 'dd-mm-yyyy' format.
        """
        if not coin_id or not date_str:
            return None

        cache_key = f"{coin_id}_{date_str}"
        if cache_key in self.historical_price_cache:
            logger.debug(f"Cache HIT for {coin_id} on {date_str}.")
            return self.historical_price_cache[cache_key]

        logger.debug(f"Cache MISS. Fetching historical price for {coin_id} on {date_str}.")

        base_url = self.coingecko_api.get("base_url")
        timeout = self.coingecko_api.get("timeout", 30)
        url = f"{base_url}/coins/{coin_id}/history?date={date_str}&localization=false"

        # <<< INCREASED RETRIES & BACKOFF >>>
        max_retries = 3 # Increase from 1 (total 4 attempts)
        initial_wait_seconds = 60
        server_error_codes_to_retry = [500, 502, 503, 504]

        retries_left = max_retries
        while retries_left >= 0:
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status() # Will raise HTTPError for 4xx/5xx
                data = response.json()
                price = None
                if "market_data" in data and "current_price" in data["market_data"] and "usd" in data["market_data"]["current_price"]:
                    price = float(data["market_data"]["current_price"]["usd"])
                else:
                    logger.warning(f"Could not find USD price in historical data for {coin_id} on {date_str}. Response: {data}")

                self.historical_price_cache[cache_key] = price
                return price

            except requests.exceptions.HTTPError as e:
                should_retry = False
                if e.response.status_code == 429: # Too Many Requests
                    should_retry = True
                    logger.error(f"Rate limited (429) by CoinGecko for {coin_id} on {date_str}.")
                elif e.response.status_code in server_error_codes_to_retry:
                    should_retry = True
                    logger.error(f"Server error ({e.response.status_code}) from CoinGecko for {coin_id} on {date_str}.")

                if should_retry and retries_left > 0:
                    # Exponential backoff (60s, 120s, 180s, etc.)
                    wait_time = initial_wait_seconds * (max_retries - retries_left + 1)
                    logger.info(f"Waiting {wait_time}s before retry... ({retries_left} retries left)")
                    time.sleep(wait_time)
                    retries_left -= 1
                else: # If not a retryable error, or no retries left
                    if e.response.status_code == 404:
                         logger.warning(f"No historical data found on CoinGecko for {coin_id} on {date_str}.")
                    elif not (e.response.status_code == 404):
                         logger.error(f"HTTP error fetching historical price for {coin_id} on {date_str} after all retries (if any): {e}")
                    self.historical_price_cache[cache_key] = None
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching historical price for {coin_id} on {date_str}: {e}", exc_info=True)
                self.historical_price_cache[cache_key] = None
                return None # Break on other unexpected errors

        logger.error(f"Failed to fetch historical price for {coin_id} on {date_str} due to exhausted retries.")
        self.historical_price_cache[cache_key] = None
        return None

    def fetch_binance_balances(self) -> pd.DataFrame:
        """Fetch current balances from Binance Spot wallet with retry logic and normalization."""
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Spot balances.")
            return pd.DataFrame(columns=['symbol', 'quantity'])

        retries = 3
        wait_time_seconds = 15
        api_timeout = self.config.get("apis", {}).get("binance", {}).get("timeout", 120)

        while retries > 0:
            try:
                logger.debug(f"Attempting to fetch Binance account info (Timeout: {api_timeout}s)...")
                account_info = self.binance_client.get_account() # Uses timeout from client init

                balances_raw = account_info.get('balances', [])
                if not balances_raw:
                    logger.info("Fetched 0 balances from Binance Spot wallet.")
                    return pd.DataFrame(columns=['symbol', 'quantity'])

                processed_balances = []
                for b in balances_raw:
                    free = float(b.get('free', 0.0))
                    locked = float(b.get('locked', 0.0))
                    quantity = free + locked
                    if quantity > 0.00000001: # Added a small threshold
                        asset_symbol_api = b.get('asset').upper()

                        # Apply explicit normalization from config (e.g., RNDR -> RENDER, LDTAO -> TAO)
                        normalized_symbol = self.norm_map.get(asset_symbol_api, asset_symbol_api)

                        processed_balances.append({'symbol': normalized_symbol, 'quantity': quantity})

                if not processed_balances:
                    logger.info("Found raw balances, but all were zero or negligible after processing.")
                    return pd.DataFrame(columns=['symbol', 'quantity'])

                df = pd.DataFrame(processed_balances)
                # Group by normalized symbol and sum quantities (in case API returns multiple small lots for same asset)
                df = df.groupby('symbol', as_index=False)['quantity'].sum()

                # <<< THIS IS THE DEBUG LOG WE NEED TO SEE >>>
                logger.info(f"DEBUG: Balances from SPOT after explicit normalization & group by: \n{df.to_string() if not df.empty else 'EMPTY DF'}")

                logger.info(f"Fetched and normalized {len(df)} non-zero balances from Binance Spot.")
                return df

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                retries -= 1
                logger.error(f"Network error fetching Binance Spot balances: {e}. Retries left: {retries}.")
                if retries > 0:
                    logger.info(f"Waiting {wait_time_seconds}s before retrying...")
                    time.sleep(wait_time_seconds)
                else:
                    logger.error("Failed to fetch Spot balances after multiple retries due to network errors.")
            except BinanceAPIException as e:
                 logger.error(f"Binance API Error fetching Spot balances: {e}. No retries for API errors.")
                 break
            except Exception as e:
                logger.error(f"Unexpected error fetching Binance Spot balances: {e}", exc_info=True)
                break
        return pd.DataFrame(columns=['symbol', 'quantity'])

    def fetch_binance_balances(self) -> pd.DataFrame:
        """Fetch current balances from Binance Spot wallet with retry, explicit normalization, and LD-prefix consolidation."""
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch Spot balances.")
            return pd.DataFrame(columns=['symbol', 'quantity'])

        retries = 3
        wait_time_seconds = 15
        api_timeout = self.config.get("apis", {}).get("binance", {}).get("timeout", 120)

        while retries > 0:
            try:
                logger.debug(f"Attempting to fetch Binance account info (Timeout: {api_timeout}s)...")
                account_info = self.binance_client.get_account() # Uses timeout from client init

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

                        # Step 1: Apply explicit normalization from config's symbol_normalization_map (e.g., RNDR -> RENDER)
                        normalized_symbol_s1 = self.norm_map.get(asset_symbol_api, asset_symbol_api)

                        # Step 2: Consolidate "LD" prefixed symbols to their base if both are in symbol_mappings.coingecko_ids
                        final_symbol = normalized_symbol_s1
                        if normalized_symbol_s1.startswith('LD') and len(normalized_symbol_s1) > 2:
                            base_equivalent = normalized_symbol_s1[2:] # e.g., BTC from LDBTC
                            # Check if both the LD-variant AND its base equivalent are configured in coingecko_ids
                            if normalized_symbol_s1 in self.symbol_mappings and base_equivalent in self.symbol_mappings:
                                final_symbol = base_equivalent
                                logger.debug(f"Consolidated API symbol '{asset_symbol_api}' (norm1: '{normalized_symbol_s1}') to base '{final_symbol}' based on coingecko_ids structure.")
                            # else: it's an "LD" prefixed symbol that doesn't have a direct base counterpart in mappings, treat as is.

                        processed_balances.append({'symbol': final_symbol, 'quantity': quantity})

                if not processed_balances:
                    logger.info("Found raw balances, but all were zero or negligible after processing.")
                    return pd.DataFrame(columns=['symbol', 'quantity'])

                df = pd.DataFrame(processed_balances)
                df = df.groupby('symbol', as_index=False)['quantity'].sum()

                logger.info(f"DEBUG: Balances from SPOT after all processing in fetch_binance_balances: \n{df.to_string() if not df.empty else 'EMPTY DF'}")
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
        """
        Fetch trade history from Binance for known symbols with retry logic.
        Creates synthetic SELL transactions for the quote currency on BUY trades,
        and synthetic BUY transactions for the quote currency on SELL trades.
        """
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch transactions.")
            return []

        transactions = []
        norm_map = self.config.get("symbol_normalization_map", {})
        tracked_symbols_for_cg_ids = list(self.config.get("symbol_mappings", {}).get("coingecko_ids", {}).keys())
        stablecoin_quotes = ['USDT', 'BUSD', 'USDC', 'FDUSD', 'TUSD']
        crypto_quotes = ['BTC', 'ETH']
        all_quotes_to_check = stablecoin_quotes + crypto_quotes
        logger.info(f"Attempting to fetch trades for {len(tracked_symbols_for_cg_ids)} base symbols...")
        processed_pairs = set()

        # <<< Retry Parameters >>>
        max_retries = 3
        wait_time_seconds = 15

        for symbol_to_find_orig_case in tracked_symbols_for_cg_ids:
            symbol_to_find = symbol_to_find_orig_case.upper()
            if symbol_to_find in all_quotes_to_check and symbol_to_find not in crypto_quotes:
                 if symbol_to_find in stablecoin_quotes : continue
            if symbol_to_find.startswith('LD'):
                continue

            for quote_candidate in all_quotes_to_check:
                quote_c = quote_candidate.upper()
                pair = f"{symbol_to_find}{quote_c}"

                if symbol_to_find == quote_c or pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                trades = [] # Initialize trades as empty list
                retries_left = max_retries

                # <<< ADD RETRY LOOP >>>
                while retries_left > 0:
                    try:
                        logger.debug(f"Fetching trades for {pair} (Attempt {max_retries - retries_left + 1})...")
                        trades = self.binance_client.get_my_trades(symbol=pair)
                        # If successful, break the retry loop
                        break
                    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                        retries_left -= 1
                        logger.error(f"Network error fetching trades for {pair}: {e}. Retries left: {retries_left}.")
                        if retries_left > 0:
                            logger.info(f"Waiting {wait_time_seconds}s before retrying...")
                            time.sleep(wait_time_seconds)
                        else:
                            logger.error(f"Failed to fetch {pair} after multiple retries due to network errors.")
                            trades = [] # Ensure trades is empty on failure
                    except BinanceAPIException as e:
                        if e.status_code not in [400, -1121]: # -1121 is "Invalid symbol"
                             logger.error(f"API Error fetching trades for {pair}: {e}")
                        else:
                             logger.debug(f"API Info: No trades or invalid pair for {pair} ({e.code}).")
                        trades = [] # Ensure trades is empty
                        break # No retry for API errors
                    except Exception as e:
                        logger.error(f"Unexpected error fetching trades for {pair}: {e}")
                        trades = [] # Ensure trades is empty
                        break # No retry for unexpected errors
                # <<< END RETRY LOOP >>>

                if not trades:
                    continue # Move to the next pair if no trades fetched/found

                logger.info(f"Fetched {len(trades)} trades for {pair}.")

                # ... (Keep the rest of the trade processing logic exactly as it was) ...
                for trade in trades:
                        base_asset_traded = symbol_to_find
                        quote_asset_traded = quote_c

                        normalized_base_asset = norm_map.get(base_asset_traded, base_asset_traded)
                        normalized_quote_asset = norm_map.get(quote_asset_traded, quote_asset_traded)

                        is_buy_of_base = trade['isBuyer']
                        quantity_base = float(trade['qty'])
                        price_in_quote = float(trade['price'])
                        fee_quantity_raw = float(trade['commission'])
                        fee_currency_raw = trade['commissionAsset'].upper()
                        normalized_fee_currency = norm_map.get(fee_currency_raw, fee_currency_raw)
                        price_base_in_usd = price_in_quote if quote_asset_traded in stablecoin_quotes else None

                        if price_base_in_usd is None:
                             logger.warning(f"Trade {trade['id']} for {pair} (Base: {normalized_base_asset}): Quote '{quote_asset_traded}' is not a tracked stablecoin. "
                                            f"price_usd for base asset requires historical price (Not yet implemented).")

                        fee_usd_for_primary_trade = None
                        if normalized_fee_currency in stablecoin_quotes:
                            fee_usd_for_primary_trade = fee_quantity_raw
                        elif normalized_fee_currency == normalized_base_asset and price_base_in_usd is not None:
                            fee_usd_for_primary_trade = fee_quantity_raw * price_base_in_usd
                        elif normalized_fee_currency == normalized_quote_asset and quote_asset_traded in stablecoin_quotes:
                            fee_usd_for_primary_trade = fee_quantity_raw

                        transactions.append({
                            "symbol": normalized_base_asset,
                            "timestamp": pd.to_datetime(trade['time'], unit='ms').to_pydatetime(),
                            "type": 'BUY' if is_buy_of_base else 'SELL',
                            "quantity": quantity_base,
                            "price_usd": price_base_in_usd,
                            "fee_quantity": fee_quantity_raw,
                            "fee_currency": normalized_fee_currency,
                            "fee_usd": fee_usd_for_primary_trade,
                            "source": "Binance Trade",
                            "transaction_hash": f"binance_{trade['id']}",
                            "notes": f"Pair: {pair}, Price: {price_in_quote} {quote_asset_traded}, Fee: {fee_quantity_raw} {fee_currency_raw}"
                        })

                        if normalized_quote_asset in self.symbol_mappings:
                            qty_quote_exchanged = quantity_base * price_in_quote
                            if normalized_fee_currency == normalized_quote_asset:
                                if is_buy_of_base:
                                    qty_quote_exchanged += fee_quantity_raw
                                else:
                                    qty_quote_exchanged -= fee_quantity_raw

                            price_quote_in_usd = 1.0 if normalized_quote_asset in stablecoin_quotes else None
                            if price_quote_in_usd is None:
                                quote_coin_id = self.symbol_mappings.get(normalized_quote_asset)
                                if quote_coin_id:
                                    trade_date_str = pd.to_datetime(trade['time'], unit='ms').strftime('%d-%m-%Y')
                                    price_quote_in_usd = self._get_coingecko_historical_price(quote_coin_id, trade_date_str)
                                    if price_quote_in_usd:
                                         logger.info(f"Fetched historical price for quote {normalized_quote_asset} on {trade_date_str}: ${price_quote_in_usd:.6f}")
                                         time.sleep(0.5) # Add back sleep specifically after historical price calls
                                    else:
                                         logger.warning(f"Could not get historical price for quote {normalized_quote_asset} for synthetic tx.")
                                else:
                                    logger.warning(f"No CoinGecko ID for quote {normalized_quote_asset} for synthetic tx.")

                            transactions.append({
                                "symbol": normalized_quote_asset,
                                "timestamp": pd.to_datetime(trade['time'], unit='ms').to_pydatetime(),
                                "type": 'SELL' if is_buy_of_base else 'BUY',
                                "quantity": qty_quote_exchanged,
                                "price_usd": price_quote_in_usd,
                                "fee_quantity": 0.0,
                                "fee_currency": None,
                                "fee_usd": 0.0,
                                "source": "Binance Synthetic",
                                "transaction_hash": f"binance_{trade['id']}_synth_{normalized_quote_asset}",
                                "notes": f"Synthetic for {pair} trade. Original fee: {fee_quantity_raw} {fee_currency_raw}"
                            })

                time.sleep(0.5) # Keep the sleep between *different pairs*

        logger.info(f"Fetched a total of {len(transactions)} transactions (including synthetic).")
        return transactions

    def fetch_internal_transfers(self, days_back: int = 365, batch_days: int = 6) -> List[Dict[str, Any]]:
        """
        Fetch internal asset transfer history (type FUNDING_MAIN) in batches.
        Iterates through specified assets and fetches their history in 'batch_days' chunks
        up to 'days_back' days ago.
        """
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch internal transfer history.")
            return []

        all_fetched_transfers_data = []
        norm_map = self.config.get("symbol_normalization_map", {})

        # Determine which symbols to query for internal transfers (existing logic from your diff)
        stablecoin_symbols_config = self.config.get("portfolio", {}).get("stablecoin_symbols", ['USDT', 'USDC', 'BUSD', 'DAI', 'FDUSD', 'TUSD'])
        target_assets_config = list(self.config.get("target_allocation", {}).keys())
        assets_to_query_initially = ["USDT"] + \
                                    [s.upper() for s in self.config.get("symbol_mappings", {}).get("coingecko_ids", {}).keys()
                                     if s.upper() in stablecoin_symbols_config]
        for ta in target_assets_config:
            if ta.upper() not in assets_to_query_initially:
                assets_to_query_initially.append(ta.upper())
        assets_to_check_for_transfers_normalized = set()
        for s_orig in assets_to_query_initially:
            assets_to_check_for_transfers_normalized.add(norm_map.get(s_orig.upper(), s_orig.upper()))
        assets_to_check_for_transfers = sorted(list(assets_to_check_for_transfers_normalized))

        pepe_gift_config = self.config.get("pepe_gift_details", {})
        pepe_gift_symbol_config = pepe_gift_config.get("symbol", "PEPE").upper()
        try:
            pepe_gift_amount_config = float(pepe_gift_config.get("amount", "0"))
        except ValueError:
            pepe_gift_amount_config = 0.0

        logger.info(f"Proceeding with REGULAR internal transfer history fetch (Funding to Spot, {days_back} days lookback in {batch_days}-day batches) for assets: {', '.join(assets_to_check_for_transfers)}")
        overall_start_time_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back) # Ensure UTC

        for asset_symbol_for_api in assets_to_check_for_transfers:
            logger.debug(f"Processing asset for internal transfers: {asset_symbol_for_api}")
            current_asset_transfers_for_loop = []

            num_periods = (days_back + batch_days - 1) // batch_days

            for i in range(num_periods):
                period_end_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=i * batch_days) # Ensure UTC
                period_start_dt = period_end_dt - datetime.timedelta(days=batch_days)

                if period_start_dt < overall_start_time_dt:
                    period_start_dt = overall_start_time_dt

                if period_start_dt >= period_end_dt:
                    if i == 0 : logger.debug(f"Initial period_start_dt {period_start_dt} is already >= period_end_dt {period_end_dt} for asset {asset_symbol_for_api}. Skipping asset or reducing days_back further might be needed if this is unexpected for all assets.")
                    break # Break from inner loop (batches) for this asset if condition met.

                start_ms = int(period_start_dt.timestamp() * 1000)
                end_ms = int(period_end_dt.timestamp() * 1000)

                retries = 2
                current_batch_fetched_rows = []
                while retries > 0:
                    try:
                        logger.debug(f"Fetching FUNDING_MAIN transfers for {asset_symbol_for_api} from "
                                     f"{period_start_dt.strftime('%Y-%m-%d %H:%M')} to "
                                     f"{period_end_dt.strftime('%Y-%m-%d %H:%M')}")
                        transfer_history = self.binance_client.get_universal_transfer_history(
                            type="TRANSFER",
                            asset="USDT",
                            startTime=start_ms,
                            endTime=end_ms,
                            size=100 # Max results per page (default is 10, max 100 for this endpoint)
                        )
                        current_batch_fetched_rows = transfer_history.get('rows', [])
                        if current_batch_fetched_rows:
                            logger.info(f"Fetched {len(current_batch_fetched_rows)} 'FUNDING_MAIN' transfers for {asset_symbol_for_api} in this batch (approx. {period_start_dt.date()} to {period_end_dt.date()}).")
                        # else:
                            # logger.debug(f"No FUNDING_MAIN transfers found for {asset_symbol_for_api} in this specific batch.")
                        break # Success
                    except BinanceAPIException as e:
                        retries -=1
                        logger.error(f"API Error fetching internal transfers for {asset_symbol_for_api} (batch ending {period_end_dt.strftime('%Y-%m-%d')}): {e}")
                        if e.code == -9000: # Specific error for "Only X days of history can be returned"
                             logger.warning(f"  ⤷ Received API error {e.code} for {asset_symbol_for_api}. This asset or timeframe may be restricted or have less history available via this endpoint.")
                             # If -9000, maybe break from retries for this batch as it won't succeed.
                             break
                        if retries == 0:
                            logger.error(f"  ⤷ Failed to fetch batch for {asset_symbol_for_api} after multiple retries.")
                        else:
                            logger.info(f"  ⤷ Retrying batch for {asset_symbol_for_api}, {retries} attempts left...")
                            time.sleep(2) # Increased sleep on retry
                    except Exception as e:
                        retries -= 1
                        logger.error(f"Unexpected error fetching internal transfers for {asset_symbol_for_api} (batch ending {period_end_dt.strftime('%Y-%m-%d')}): {e}", exc_info=True)
                        if retries == 0:
                             logger.error(f"  ⤷ Failed to fetch batch for {asset_symbol_for_api} after multiple retries due to unexpected error.")
                        else:
                            logger.info(f"  ⤷ Retrying batch for {asset_symbol_for_api} (unexpected error), {retries} attempts left...")
                            time.sleep(2)

                current_asset_transfers_for_loop.extend(current_batch_fetched_rows)
                if not current_batch_fetched_rows and period_start_dt == overall_start_time_dt:
                    logger.debug(f"No transfers found for {asset_symbol_for_api} in the oldest batch reaching overall_start_time_dt. Stopping for this asset.")
                    break # Stop for this asset if the oldest period has no data.

                time.sleep(0.7) # Increased sleep between batches for the same asset to be kinder to API limits

            # Process all successfully fetched transfers for this asset
            for i_tx, transfer_item in enumerate(current_asset_transfers_for_loop):
                asset_from_api = transfer_item.get('asset')
                if not asset_from_api or asset_from_api.upper() != asset_symbol_for_api.upper() : # Ensure we only process the asset we queried for
                    logger.debug(f"Skipping transfer item, asset mismatch: expected {asset_symbol_for_api}, got {asset_from_api}. Item: {transfer_item}")
                    continue

                normalized_symbol_for_storage = asset_symbol_for_api # Already normalized

                quantity = float(transfer_item.get('amount', 0.0))
                if quantity == 0: continue

                transfer_timestamp_ms = int(transfer_item.get('timestamp', 0))
                if transfer_timestamp_ms == 0: continue

                transfer_datetime_obj = pd.to_datetime(transfer_timestamp_ms, unit='ms')
                # Convert to Python datetime and ensure UTC
                python_timestamp = transfer_datetime_obj.to_pydatetime()
                if python_timestamp.tzinfo is None:
                    python_timestamp = python_timestamp.replace(tzinfo=datetime.timezone.utc)
                else:
                    python_timestamp = python_timestamp.astimezone(datetime.timezone.utc)

                price_usd_at_transfer = 0.0
                is_pepe_gift_transfer = (normalized_symbol_for_storage == pepe_gift_symbol_config and
                                         abs(quantity - pepe_gift_amount_config) < 1e-9 and
                                         pepe_gift_amount_config > 0)

                coingecko_sleep_needed = False
                if is_pepe_gift_transfer:
                    logger.info(f"Identified PEPE gift via internal transfer ({quantity} {normalized_symbol_for_storage}). Assigning $0 cost.")
                else:
                    coin_id = self.symbol_mappings.get(normalized_symbol_for_storage)
                    if coin_id:
                        date_str = transfer_datetime_obj.strftime('%d-%m-%Y')
                        historical_price = self._get_coingecko_historical_price(coin_id, date_str)
                        if historical_price is not None:
                            price_usd_at_transfer = historical_price
                            logger.info(f"Fetched historical price for {normalized_symbol_for_storage} (internal transfer) on {date_str}: ${price_usd_at_transfer:.6f}")
                            coingecko_sleep_needed = True
                        else:
                            logger.warning(f"Could not fetch historical for {normalized_symbol_for_storage} (internal transfer) on {date_str}. Cost for this transfer will be $0.")
                    else:
                        logger.warning(f"No CoinGecko ID for {normalized_symbol_for_storage} (internal transfer). Cost for this transfer will be $0.")

                tran_id_from_api = transfer_item.get('tranId')
                if tran_id_from_api is None: # tranId can be None for some older universal transfers
                     tran_id_from_api = f"internal_manual_id_{transfer_timestamp_ms}_{normalized_symbol_for_storage}_{quantity}"
                     logger.debug(f"tranId was None for internal transfer. Generated manual ID: {tran_id_from_api}")


                all_fetched_transfers_data.append({
                    "symbol": normalized_symbol_for_storage,
                    "timestamp": python_timestamp,
                    "type": "DEPOSIT", "quantity": quantity, "price_usd": price_usd_at_transfer,
                    "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                    "source": "Binance Internal Transfer", "transaction_hash": str(tran_id_from_api), # Ensure tranId is string
                    "notes": f"Funding to Spot: {quantity} {normalized_symbol_for_storage}"
                })
                if coingecko_sleep_needed: # Sleep only if CoinGecko was successfully called
                    # Access coingecko specific delay, or a general one
                    cg_delay = self.config.get("apis",{}).get("coingecko",{}).get("request_delay_ms_internal_transfer", 1200) / 1000.0
                    time.sleep(cg_delay)

            # Delay before processing next asset in assets_to_check_for_transfers
            if len(assets_to_check_for_transfers) > 1 : # if there are more assets to process
                 time.sleep(1.5) # Increased sleep between processing different assets

        logger.info(f"Fetched a total of {len(all_fetched_transfers_data)} internal transfer records to be processed as deposits after iterating all configured assets.")
        return all_fetched_transfers_data

    def fetch_p2p_usdt_buys(self, days_back: int = 365) -> List[Dict[str, Any]]:
        """
        Fetch P2P USDT buy history against a configured fiat currency,
        using monthly batches and start/end timestamps.
        """
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch P2P history.")
            return []

        all_p2p_transactions = []
        p2p_fiat_currency = self.config.get("portfolio", {}).get("p2p_fiat_currency", "PHP").upper()
        logger.info(f"Fetching P2P USDT buy history against fiat: {p2p_fiat_currency} for last {days_back} days (batched by month).")

        num_months_to_check = (days_back + 29) // 30
        current_loop_end_dt = datetime.datetime.now(datetime.timezone.utc)
        overall_start_dt_for_lookback = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)
        rows_per_page = 50 # Define rows_per_page once, outside the loop

        for m_idx in range(num_months_to_check):
            current_loop_start_dt = current_loop_end_dt - datetime.timedelta(days=30)

            if current_loop_start_dt < overall_start_dt_for_lookback:
                current_loop_start_dt = overall_start_dt_for_lookback

            # Ensure the 30-day window isn't exceeded for the API call
            if (current_loop_end_dt - current_loop_start_dt).days > 30:
                 actual_start_for_api = current_loop_end_dt - datetime.timedelta(days=30)
            else:
                 actual_start_for_api = current_loop_start_dt

            if actual_start_for_api >= current_loop_end_dt:
                logger.debug(f"P2P fetch: Batch start {actual_start_for_api.date()} is not before batch end {current_loop_end_dt.date()}. Stopping monthly batches.")
                break

            start_ms = int(actual_start_for_api.timestamp() * 1000)
            end_ms = int(current_loop_end_dt.timestamp() * 1000)

            current_page = 1
            max_pages_to_try_per_batch = 20 # Safety break
            processed_trades_older_than_overall_lookback_in_batch = False

            while current_page <= max_pages_to_try_per_batch:
                logger.debug(f"Fetching P2P BUY trades for {p2p_fiat_currency}/USDT. "
                             f"Period: {actual_start_for_api.date()} to {current_loop_end_dt.date()}, "
                             f"Page: {current_page}")
                try:
                    # <<< MODIFIED: Added startTimestamp and endTimestamp >>>
                    history = self.binance_client.get_c2c_trade_history(
                        tradeType='BUY',
                        page=current_page,
                        rows=rows_per_page,
                        startTimestamp=start_ms,
                        endTimestamp=end_ms
                    )
                    time.sleep(self.config.get("apis", {}).get("binance", {}).get("request_delay_ms", 700) / 1000.0)

                    trades_in_page = history.get('data', [])
                    if not history or not trades_in_page:
                        logger.debug(f"No P2P BUY data on page {current_page} for this batch, or end of data.")
                        break

                    for trade in trades_in_page:
                        create_time_ms = trade.get('createTime')
                        if not create_time_ms: continue

                        timestamp_dt = pd.to_datetime(create_time_ms, unit='ms', utc=True).to_pydatetime()

                        if timestamp_dt < overall_start_dt_for_lookback:
                            processed_trades_older_than_overall_lookback_in_batch = True
                            continue

                        # Keep this check, just in case API doesn't filter perfectly
                        if not (actual_start_for_api <= timestamp_dt < current_loop_end_dt):
                             logger.warning(f"P2P trade {trade.get('orderNumber')} timestamp {timestamp_dt} "
                                            f"is outside the expected batch window "
                                            f"{actual_start_for_api} - {current_loop_end_dt}. Skipping.")
                             continue

                        if trade.get('asset', '').upper() == 'USDT' and \
                           trade.get('fiat', '').upper() == p2p_fiat_currency:
                            order_number = trade.get('orderNumber')
                            try:
                                usdt_quantity = float(trade.get('amount', 0))
                                fiat_amount_paid = float(trade.get('totalPrice', 0))
                            except (ValueError, TypeError) as e:
                                logger.error(f"Could not parse P2P amount/totalPrice for order {order_number}: {e}. Trade: {trade}")
                                continue
                            if usdt_quantity <= 0 or fiat_amount_paid <= 0: continue

                            price_usd_placeholder = 1.0
                            fiat_currency_from_trade = trade.get('fiat', '').upper()
                            notes = f"P2P Buy: {usdt_quantity:.8f} USDT for {fiat_amount_paid:.2f} {fiat_currency_from_trade}. " \
                                    f"OrderNo: {order_number}. USD price is placeholder."

                            all_p2p_transactions.append({
                                "symbol": "USDT", "timestamp": timestamp_dt, "type": "BUY",
                                "quantity": usdt_quantity, "price_usd": price_usd_placeholder,
                                "fee_quantity": 0.0, "fee_currency": None, "fee_usd": 0.0,
                                "source": "Binance P2P Buy", "transaction_hash": str(order_number),
                                "notes": notes, "p2p_fiat_currency": fiat_currency_from_trade,
                                "p2p_fiat_amount_paid": fiat_amount_paid
                            })

                    current_page += 1
                    if history.get('total') is not None and ((current_page -1) * rows_per_page >= int(history.get('total', 0))):
                        logger.debug(f"Fetched all P2P records for this batch based on total count ({history.get('total')}).")
                        break

                except BinanceAPIException as e:
                    logger.error(f"Binance API Error fetching P2P history (Batch {actual_start_for_api.date()}-{current_loop_end_dt.date()}, Page {current_page}): {e}")
                    # If error occurs (e.g., invalid timestamp params), stop trying for this batch.
                    break
                except Exception as e:
                    logger.error(f"Unexpected error fetching P2P history (Batch {actual_start_for_api.date()}-{current_loop_end_dt.date()}, Page {current_page}): {e}", exc_info=True)
                    break

            # Update end_of_current_period_utc for the next iteration
            current_loop_end_dt = actual_start_for_api

            if processed_trades_older_than_overall_lookback_in_batch and current_loop_start_dt <= overall_start_dt_for_lookback:
                 logger.info(f"P2P fetching reached overall start date limit ({overall_start_dt_for_lookback.date()}). Stopping P2P history fetch.")
                 break

        logger.info(f"Fetched a total of {len(all_p2p_transactions)} P2P USDT buy transactions against {p2p_fiat_currency}.")
        return all_p2p_transactions

    def fetch_deposit_history(self, days_back: int = 90) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch deposit history.")
            return []

        deposits_data = []
        norm_map = self.config.get("symbol_normalization_map", {})
        pepe_gift_config = self.config.get("pepe_gift_details", {})
        pepe_gift_symbol = pepe_gift_config.get("symbol", "PEPE").upper()
        try:
            your_pepe_gift_amount = float(pepe_gift_config.get("amount", "0"))
        except ValueError:
            your_pepe_gift_amount = 0.0
            logger.warning(f"PEPE gift amount '{pepe_gift_config.get('amount')}' is invalid for deposits.")

        try:
            logger.info(f"Fetching deposit history (last {days_back} days)...") # Updated log
            # Ensure current time is timezone-aware (UTC)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            endTime = int(now_utc.timestamp() * 1000)
            startTime = int((now_utc - datetime.timedelta(days=days_back)).timestamp() * 1000) # Use days_back

            # Status 1 means completed deposits
            all_deposits = self.binance_client.get_deposit_history(startTime=startTime, endTime=endTime, status=1)
            logger.info(f"Fetched {len(all_deposits)} successful deposit records within the last {days_back} days.")

            for i, deposit in enumerate(all_deposits):
                logger.debug(f"Processing raw deposit record {i+1}/{len(all_deposits)}: {deposit}")
                symbol_original = deposit.get('coin')
                if not symbol_original:
                    logger.warning(f"Deposit record {i+1} missing 'coin' field (TxID: {deposit.get('txId')}). Skipping.")
                    continue

                normalized_symbol = norm_map.get(symbol_original.upper(), symbol_original.upper())
                insert_time_raw = deposit.get('insertTime')
                if insert_time_raw is None:
                    logger.error(f"Deposit for {normalized_symbol} (TxID: {deposit.get('txId')}) has missing 'insertTime'. Skipping.")
                    continue

                deposit_timestamp_obj_pandas = pd.to_datetime(insert_time_raw, unit='ms', utc=True) # Ensure UTC
                if pd.isna(deposit_timestamp_obj_pandas):
                    logger.error(f"Parsed timestamp is NaT for {normalized_symbol} (TxID: {deposit.get('txId')}) from raw value '{insert_time_raw}'. Skipping.")
                    continue

                deposit_timestamp_py = deposit_timestamp_obj_pandas.to_pydatetime() # Already UTC-aware

                price_usd_at_deposit = 0.0
                is_pepe_gift = (normalized_symbol == pepe_gift_symbol and
                                abs(float(deposit.get('amount', 0)) - your_pepe_gift_amount) < 1e-9 and
                                your_pepe_gift_amount > 0)

                coingecko_called = False
                if is_pepe_gift:
                    logger.info(f"Identified PEPE gift deposit ({deposit.get('amount')} {normalized_symbol}). Assigning $0 cost.")
                else:
                    coin_id = self.symbol_mappings.get(normalized_symbol) # self.symbol_mappings should be loaded in __init__
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

                deposits_data.append({
                    "symbol": normalized_symbol,
                    "timestamp": deposit_timestamp_py, # UTC-aware datetime
                    "type": "DEPOSIT",
                    "quantity": float(deposit.get('amount', 0)),
                    "price_usd": price_usd_at_deposit,
                    "fee_quantity": 0.0, # Assuming deposit fees are not relevant here or handled by net amount
                    "fee_currency": None,
                    "fee_usd": 0.0,
                    "source": "Binance Deposit",
                    "transaction_hash": deposit.get('txId', f"binance_deposit_{deposit.get('id', i)}_{insert_time_raw}"),
                    "notes": f"Status: {deposit.get('status', 'N/A')}, Network: {deposit.get('network')}"
                })
                if coingecko_called and historical_price is not None: # Sleep if coingecko was successfully called
                     time.sleep(self.config.get("apis", {}).get("coingecko", {}).get("request_delay_ms_deposit", 1200) / 1000.0)

        except BinanceAPIException as e:
            logger.error(f"API Error fetching deposit history: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching deposit history: {e}", exc_info=True)
        return deposits_data

    def fetch_withdrawal_history(self, days_back: int = 90) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch withdrawal history.")
            return []

        withdrawals_data = []
        norm_map = self.config.get("symbol_normalization_map", {})
        try:
            logger.info(f"Fetching withdrawal history (last {days_back} days)...") # Updated log
            now_utc = datetime.datetime.now(datetime.timezone.utc) # Ensure UTC
            endTime = int(now_utc.timestamp() * 1000)
            startTime = int((now_utc - datetime.timedelta(days=days_back)).timestamp() * 1000) # Use days_back

            # Status 6 means completed withdrawals
            all_withdrawals = self.binance_client.get_withdraw_history(startTime=startTime, endTime=endTime, status=6)
            logger.info(f"Fetched {len(all_withdrawals)} completed withdrawal records within the last {days_back} days.")

            for i, withdrawal in enumerate(all_withdrawals):
                logger.debug(f"Processing raw withdrawal record {i+1}/{len(all_withdrawals)}: {withdrawal}")
                symbol_original = withdrawal.get('coin')
                if not symbol_original:
                     logger.warning(f"Withdrawal record {i+1} missing 'coin' field. Skipping. Data: {withdrawal}")
                     continue

                normalized_symbol = norm_map.get(symbol_original.upper(), symbol_original.upper())
                apply_time_raw = withdrawal.get('applyTime') # applyTime is often more relevant than insertTime for withdrawals
                if not apply_time_raw: # applyTime might be a string like "2021-07-14 10:00:00" or epoch ms
                    logger.warning(f"Withdrawal for {normalized_symbol} (TxID: {withdrawal.get('txId')}) has missing 'applyTime'. Skipping.")
                    continue

                # Try parsing as string first, then as ms if it fails (Binance API can be inconsistent)
                try:
                    # If applyTime is already in a parseable string format including timezone or assumed UTC by pandas
                    withdrawal_timestamp_obj_pandas = pd.to_datetime(apply_time_raw, utc=True)
                except (ValueError, TypeError):
                    try: # Try as milliseconds if string parsing fails
                        withdrawal_timestamp_obj_pandas = pd.to_datetime(int(apply_time_raw), unit='ms', utc=True)
                    except (ValueError, TypeError):
                        logger.error(f"Could not parse applyTime '{apply_time_raw}' for {normalized_symbol} (TxID: {withdrawal.get('txId')}). Skipping.")
                        continue

                if pd.isna(withdrawal_timestamp_obj_pandas):
                    logger.error(f"Parsed withdrawal timestamp is NaT for {normalized_symbol} (TxID: {withdrawal.get('txId')}). Skipping.")
                    continue

                withdrawal_timestamp_py = withdrawal_timestamp_obj_pandas.to_pydatetime() # Already UTC-aware

                # For withdrawals, price_usd at withdrawal is for realized P/L, not directly for FIFO cost reduction of remaining assets
                # Fee is important as it's an additional reduction of the asset.
                fee_quantity = float(withdrawal.get('transactionFee', 0.0))
                # Fee currency is usually the same as the withdrawn coin.
                fee_currency = normalized_symbol
                # TODO: Calculate fee_usd if needed, using historical price of fee_currency at withdrawal_timestamp_py
                # For now, fee_usd is not calculated for simplicity in this part.

                withdrawals_data.append({
                    "symbol": normalized_symbol,
                    "timestamp": withdrawal_timestamp_py, # UTC-aware datetime
                    "type": "WITHDRAWAL",
                    "quantity": float(withdrawal.get('amount', 0.0)),
                    "price_usd": None, # Market value at withdrawal; not directly used by current FIFO for cost of remaining
                    "fee_quantity": fee_quantity,
                    "fee_currency": fee_currency,
                    "fee_usd": None, # Placeholder - needs calculation
                    "source": "Binance Withdrawal",
                    "transaction_hash": withdrawal.get('txId', f"binance_withdraw_{withdrawal.get('id', i)}_{apply_time_raw}"),
                    "notes": f"Network: {withdrawal.get('network', 'N/A')}, Address: {withdrawal.get('address', 'N/A')}, Fee: {fee_quantity} {fee_currency}"
                })
        except BinanceAPIException as e:
            logger.error(f"API Error fetching withdrawal history: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching withdrawal history: {e}", exc_info=True)
        return withdrawals_data

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
            if not normalized_symbol.startswith('LD'): # Simple Earn API uses base asset names
                 potential_earn_assets.add(normalized_symbol)

        common_bases = ['BTC', 'ETH', 'USDT', 'SOL', 'PEPE', 'HMSTR', 'TAO', 'RENDER']
        for base in common_bases:
            potential_earn_assets.add(base.upper()) # Ensure common bases are checked

        assets_to_check_api = sorted(list(potential_earn_assets))

        logger.info(f"Fetching Simple Earn Flexible balances for up to {len(assets_to_check_api)} potential base assets: {assets_to_check_api}")

        for asset_api_name in assets_to_check_api:
            try:
                logger.debug(f"Fetching Simple Earn position for {asset_api_name}...")
                positions = self.binance_client.get_simple_earn_flexible_product_position(asset=asset_api_name)

                if positions and isinstance(positions.get('rows'), list):
                    total_amount_for_asset = 0.0
                    for pos in positions['rows']:
                        try: total_amount_for_asset += float(pos.get('totalAmount', 0.0))
                        except (ValueError, TypeError): logger.warning(f"Could not parse 'totalAmount' for {asset_api_name}: {pos.get('totalAmount')}")

                    if total_amount_for_asset > 0:
                        # asset_api_name is already a normalized base symbol here
                        current_bal = earn_balances_aggregated.get(asset_api_name, 0.0)
                        earn_balances_aggregated[asset_api_name] = current_bal + total_amount_for_asset
                        logger.info(f"Found {total_amount_for_asset:.8f} {asset_api_name} in Simple Earn.")
                time.sleep(0.3)
            except BinanceAPIException as e:
                if e.code == -6001 or "product does not exist" in str(e).lower() or "not supported" in str(e).lower():
                     logger.debug(f"No Simple Earn for {asset_api_name} (API Info: {e}).")
                else: logger.error(f"API Error Simple Earn for {asset_api_name}: {e}")
            except Exception as e: logger.error(f"Unexpected error Simple Earn for {asset_api_name}: {e}", exc_info=True)

        # <<< THIS IS THE DEBUG LOG WE NEED TO SEE >>>
        logger.info(f"DEBUG: Balances from EARN after all processing in fetch_simple_earn_balances: {earn_balances_aggregated if earn_balances_aggregated else 'EMPTY DICT'}")

        logger.info(f"Fetched {len(earn_balances_aggregated)} asset balances from Simple Earn (after normalization).") # Message adjusted
        return earn_balances_aggregated

    def update_holdings_from_transactions(self):
        """
        Processes all transactions and updates holdings table with FIFO cost basis.
        """
        logger.info("Updating holdings from transaction history using FIFO...")
        all_txs = self.db_manager.get_all_transactions()

        if all_txs.empty:
            logger.warning("No transactions found in DB. Cannot update holdings.")
            return

        # <<< START NEW DEBUG BLOCK >>>
        logger.debug(f"Data types of all_txs DataFrame immediately after DB read:\n{all_txs.dtypes}")
        deposit_txs_debug = all_txs[all_txs['type'] == 'DEPOSIT'].copy() # Use .copy() to avoid SettingWithCopyWarning on next lines

        if not deposit_txs_debug.empty:
            logger.debug(f"Content of DEPOSIT transactions from DB (first 5 rows if many) BEFORE any processing in this function:\n{deposit_txs_debug.head().to_string()}")
            # Specifically log the problematic columns for these deposits
            logger.debug(f"Problematic 'timestamp' for DEPOSITs from DB: \n{deposit_txs_debug['timestamp'].head().to_string()}")
            logger.debug(f"Problematic 'price_usd' for DEPOSITs from DB: \n{deposit_txs_debug['price_usd'].head().to_string()}")
        else:
            logger.debug("No DEPOSIT transactions found in all_txs DataFrame immediately after DB read.")
        # <<< END NEW DEBUG BLOCK >>>

        updated_holdings = []
        for symbol, group_df in all_txs.groupby('symbol'):
            logger.debug(f"Calculating FIFO for {symbol}...")
            # Ensure necessary columns are numeric and handle N/A
            group_df['price_usd'] = pd.to_numeric(group_df['price_usd'], errors='coerce').fillna(0)
            group_df['quantity'] = pd.to_numeric(group_df['quantity'], errors='coerce').fillna(0)

            if symbol.upper() in ['HMSTR', 'USDT']: # Check for upper case to be safe
                logger.debug(f"Transactions for {symbol} being passed to FIFO:\n{group_df[['timestamp', 'type', 'quantity', 'price_usd']]}")

            final_qty, avg_cost = calculate_fifo_cost_basis(group_df)

            if final_qty > 0.000001: # Use a small threshold to avoid dust
                updated_holdings.append({
                    "symbol": symbol,
                    "quantity": final_qty,
                    "average_cost_basis": avg_cost
                })
            else:
                logger.info(f"Final quantity for {symbol} is ~zero. Not adding to holdings.")

        if updated_holdings:
            holdings_df = pd.DataFrame(updated_holdings)
            self.db_manager.update_holdings(holdings_df)
            logger.info(f"Successfully updated {len(holdings_df)} holdings in the database.")
        else:
            logger.warning("No holdings to update after FIFO calculation.")

    def sync_data(self):
        """Synchronize data from APIs, P2P, CSV, and update holdings."""
        logger.info("Starting data synchronization...")

        logger.info("Fetching new spot trades...")
        binance_trades = self.fetch_binance_transactions()

        logger.info("Fetching new external deposits...")
        deposit_lookback_days = self.config.get("history_lookback_days", {}).get("deposits", 90)
        binance_deposits = self.fetch_deposit_history(days_back=deposit_lookback_days)

        logger.info("Fetching new withdrawals...")
        withdrawal_lookback_days = self.config.get("history_lookback_days", {}).get("withdrawals", 90)
        binance_withdrawals = self.fetch_withdrawal_history(days_back=withdrawal_lookback_days)

        logger.info("Fetching internal transfers (API - might be limited)...") # Updated log
        internal_transfer_lookback_days = self.config.get("history_lookback_days", {}).get("internal_transfers", 365)
        binance_internal_transfers = self.fetch_internal_transfers(days_back=internal_transfer_lookback_days)

        logger.info("Fetching P2P USDT buy history...")
        p2p_lookback_days = self.config.get("history_lookback_days", {}).get("p2p_buys", 365)
        p2p_buys = self.fetch_p2p_usdt_buys(days_back=p2p_lookback_days)

        logger.info("Parsing transactions from Binance CSV...")
        csv_transactions = self._parse_binance_csv()

        # Combine all fetched transactions
        all_new_transactions = (binance_trades +
                                binance_deposits +
                                binance_withdrawals +
                                binance_internal_transfers +
                                p2p_buys +
                                csv_transactions)

        if all_new_transactions:
            logger.info(f"Processing a total of {len(all_new_transactions)} fetched/parsed transaction items.") # Updated log

            # +++ Standardize all timestamps to UTC-aware before sorting +++
            processed_transactions_for_sorting = []
            for tx_idx, tx in enumerate(all_new_transactions):
                try:
                    if 'timestamp' not in tx or tx['timestamp'] is None:
                        logger.warning(f"Transaction at index {tx_idx} is missing a timestamp or is None. Skipping. Data: {tx}")
                        continue

                    current_timestamp = tx['timestamp']
                    if isinstance(current_timestamp, datetime.datetime):
                        if current_timestamp.tzinfo is None:
                            tx['timestamp'] = current_timestamp.replace(tzinfo=datetime.timezone.utc)
                        else:
                            tx['timestamp'] = current_timestamp.astimezone(datetime.timezone.utc)
                        processed_transactions_for_sorting.append(tx)
                    elif isinstance(current_timestamp, (int, float)):
                        tx['timestamp'] = pd.to_datetime(current_timestamp, unit='ms', utc=True).to_pydatetime()
                        processed_transactions_for_sorting.append(tx)
                    else:
                        logger.error(f"Transaction {tx.get('transaction_hash', 'N/A_in_sync_data')} has an unexpected timestamp type ({type(current_timestamp)}): {current_timestamp}. Skipping.")
                        continue
                except Exception as e_ts:
                    logger.error(f"Error processing timestamp for transaction at index {tx_idx} (Data: {tx}): {e_ts}", exc_info=True)
                    continue

            all_new_transactions = processed_transactions_for_sorting
            # +++ End timestamp standardization +++

            if all_new_transactions:
                all_new_transactions.sort(key=lambda x: x['timestamp'])
                # <<< MODIFY THIS LOG LINE >>>
                num_inserted_or_ignored = self.db_manager.bulk_insert_transactions(all_new_transactions)
                logger.info(f"Attempted to process {len(all_new_transactions)} valid transaction records. Database reported {num_inserted_or_ignored if num_inserted_or_ignored is not None else 'unknown'} changes/insertions.")

            else:
                logger.info("No valid transactions remained after timestamp processing.")

        else:
            logger.info("No new transactions (API, P2P, or CSV) fetched or processed.") # Updated log

        self.update_holdings_from_transactions()
        logger.info("Data synchronization finished.")

    def run_full_sync(self) -> Dict[str, Any]: self.sync_data(); return self.calculate_portfolio_metrics()

    def get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for a list of symbols using CoinGecko (Batched with Retry).
        Correctly handles multiple symbols mapping to the same CoinGecko ID."""
        prices: Dict[str, Optional[float]] = {symbol: 0.0 for symbol in symbols} # Initialize all to 0.0

        symbols_to_fetch_cg_ids = {} # Map original symbol to its coingecko_id
        unique_coingecko_ids_to_fetch = set()

        logger.info(f"Mapping {len(symbols)} symbols to CoinGecko IDs for price fetching...")
        for symbol_upper in set(s.upper() for s in symbols): # Use upper for mapping
            # Find original case symbol (first occurrence) for accurate keying in `prices`
            original_case_symbol = next(s for s in symbols if s.upper() == symbol_upper)

            coin_id = self.symbol_mappings.get(symbol_upper)
            if not coin_id:
                logger.warning(f"No CoinGecko ID mapping found for {original_case_symbol}. Price will be $0.")
                prices[original_case_symbol] = 0.0 # Ensure it's explicitly 0.0
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
                fetched_price_data = response.json() # Price data from CoinGecko, keyed by coin_id
                break # Success
            except requests.exceptions.HTTPError as e:
                 if e.response.status_code == 429 and retries > 0:
                     logger.error(f"Rate limited (429) fetching batch prices. Waiting 60s before retry...")
                     time.sleep(60)
                     retries -= 1
                 else:
                     logger.error(f"HTTP error fetching batch prices: {e}")
                     fetched_price_data = {} # Ensure it's an empty dict on error
                     break
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching batch prices from CoinGecko: {e}")
                fetched_price_data = {} # Ensure it's an empty dict on error
                break

        if fetched_price_data:
            # Now map the fetched prices (keyed by coin_id) back to ALL original symbols
            for original_symbol, coin_id in symbols_to_fetch_cg_ids.items():
                if coin_id in fetched_price_data and "usd" in fetched_price_data[coin_id]:
                    prices[original_symbol] = fetched_price_data[coin_id]["usd"]
                else:
                    logger.warning(f"USD price not found for {original_symbol} (ID: {coin_id}) in CoinGecko response. Setting to $0.")
                    prices[original_symbol] = 0.0
        else:
            logger.error("Failed to fetch any price data from CoinGecko. All prices will be $0.")
            # All prices are already 0.0 from initialization

        logger.info("Price fetching complete.")
        return prices

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate key portfolio metrics. Assumes Spot balances from get_account() are comprehensive."""
        logger.info("Calculating portfolio metrics...")

        cost_basis_df = self.db_manager.get_holdings()
        if cost_basis_df.empty:
            logger.warning("No cost basis info found in DB. Cost basis will be $0 for all assets.")
            cost_basis_df_to_merge = pd.DataFrame(columns=['symbol', 'average_cost_basis'])
        else:
            cost_basis_df_to_merge = cost_basis_df[['symbol', 'average_cost_basis']].copy()

        # >>> Use Spot balances (get_account) as the primary source of TRUTH for current quantities <<<
        # fetch_binance_balances already applies normalization using self.norm_map
        holdings_df = self.fetch_binance_balances()

        # Log Simple Earn balances for information, but DO NOT add them to holdings_df if get_account() is comprehensive
        # This is based on the theory that get_account() already reflects assets in Simple Earn Flexible for auto-subscribe.
        earn_balances_dict_info_only = self.fetch_simple_earn_balances()
        logger.info(f"Informational: Simple Earn Balances fetched: {earn_balances_dict_info_only}")


        if holdings_df.empty:
            logger.error("Could not fetch ANY current balances from Binance Spot (get_account). Cannot calculate metrics accurately.")
            return {"error": "No holdings data could be fetched from Spot.", "total_value_usd": 0, "holdings_df": pd.DataFrame()}

        holdings_df = holdings_df[holdings_df['quantity'] > 0.00000001].reset_index(drop=True)

        if holdings_df.empty:
            logger.warning("No non-zero holdings found after fetching Spot balances. Portfolio value is $0.")
            return {"total_value_usd": 0, "holdings_df": pd.DataFrame()}

        holdings_df = pd.merge(holdings_df, cost_basis_df_to_merge, on='symbol', how='left')
        holdings_df['average_cost_basis'] = holdings_df['average_cost_basis'].fillna(0.0)

        prices = self.get_current_prices(holdings_df['symbol'].tolist())
        holdings_df['current_price'] = holdings_df['symbol'].map(prices).fillna(0.0)

        holdings_df['value_usd'] = holdings_df['quantity'] * holdings_df['current_price']
        holdings_df['cost_basis_total'] = holdings_df['quantity'] * holdings_df['average_cost_basis']
        holdings_df['unrealized_pl_usd'] = holdings_df['value_usd'] - holdings_df['cost_basis_total']

        holdings_df['unrealized_pl_percent'] = 0.0
        mask = holdings_df['cost_basis_total'] != 0
        holdings_df.loc[mask, 'unrealized_pl_percent'] = \
            (holdings_df.loc[mask, 'unrealized_pl_usd'] / holdings_df.loc[mask, 'cost_basis_total']) * 100

        total_value = holdings_df['value_usd'].sum()
        total_portfolio_cost_basis = holdings_df['cost_basis_total'].sum()

        holdings_df['allocation'] = (holdings_df['value_usd'] / total_value) if total_value > 0 else 0.0

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
        logger.info(f"Metrics Calculated: Total Value = ${total_value:,.2f}, Total Cost Basis = ${total_portfolio_cost_basis:,.2f}")
        return metrics

    def get_rebalance_suggestions_by_cost(self) -> Optional[pd.DataFrame]:
        """Calculate rebalance suggestions based on target cost basis, using LIVE quantities."""
        logger.info("Calculating rebalance suggestions by cost basis...")

        # 1. Get cost basis info from DB
        db_holdings = self.db_manager.get_holdings()
        if db_holdings.empty or 'average_cost_basis' not in db_holdings.columns:
            logger.warning("Need holdings with average_cost_basis from DB. Run sync first.")
            return None
        cost_basis_df_to_merge = db_holdings[['symbol', 'average_cost_basis']].copy()

        # 2. Get LIVE quantities from Binance API
        live_balances_df = self.fetch_binance_balances()
        if live_balances_df.empty:
            logger.error("Could not fetch LIVE balances. Cannot rebalance accurately.")
            return None

        # 3. Merge live quantities with DB costs (This is the key change!)
        holdings_df = pd.merge(live_balances_df, cost_basis_df_to_merge, on='symbol', how='left')
        holdings_df['average_cost_basis'] = holdings_df['average_cost_basis'].fillna(0.0)

        # 4. Calculate current cost basis using LIVE quantity
        holdings_df['current_cost_basis'] = holdings_df['quantity'] * holdings_df['average_cost_basis']

        # --- Continue with existing rebalancing logic using this new 'holdings_df' ---

        strategy_config = self.config.get("rebalancing_strategy", {})
        target_allocation_config = self.config.get("target_allocation", {})
        norm_map = self.config.get("symbol_normalization_map", {})
        allow_selling = strategy_config.get("allow_selling", True)
        never_sell = [s.upper() for s in strategy_config.get("never_sell_symbols", [])]

        if not target_allocation_config:
            logger.warning("Need target allocation for suggestions.")
            return None

        target_allocation_normalized = {norm_map.get(k.upper(), k.upper()): v for k, v in target_allocation_config.items()}
        target_asset_symbols = [s for s, p in target_allocation_normalized.items() if p > 0]

        # Filter *this new holdings_df* for relevant assets
        relevant_holdings_df = holdings_df[holdings_df['symbol'].isin(target_asset_symbols)].copy()

        # Recalculate total cost basis using LIVE quantities for *target assets only*
        total_relevant_portfolio_cost_basis = relevant_holdings_df['current_cost_basis'].sum()

        if total_relevant_portfolio_cost_basis == 0 and any(p > 0 for p in target_allocation_normalized.values()):
            logger.warning(
                "Total cost basis for assets in target_allocation is $0.00. "
                "Rebalance suggestions to 'buy' will be $0. "
                "This might be due to $0 cost basis (e.g., price fetch issues) or assets not held."
            )
        logger.info(f"Total Portfolio Cost Basis (for target assets only, using LIVE Qty): ${total_relevant_portfolio_cost_basis:,.2f}")

        suggestions = []
        all_symbols_to_consider = sorted(list(set(list(target_allocation_normalized.keys()) + holdings_df['symbol'].tolist())))
        current_prices = self.get_current_prices(all_symbols_to_consider)

        for symbol in all_symbols_to_consider:
            target_pct = target_allocation_normalized.get(symbol, 0.0)
            target_cost_basis_for_symbol = total_relevant_portfolio_cost_basis * target_pct

            # Use the *new* holdings_df with live quantities and its calculated cost basis
            current_row = holdings_df[holdings_df['symbol'] == symbol]
            current_actual_cost_for_symbol = current_row['current_cost_basis'].iloc[0] if not current_row.empty else 0.0

            rebalance_amount_usd = target_cost_basis_for_symbol - current_actual_cost_for_symbol
            amount_to_buy_usd = 0.0; amount_to_sell_usd = 0.0
            buy_qty_coin = 0.0; sell_qty_coin = 0.0
            current_price = current_prices.get(symbol, 0.0)

            if rebalance_amount_usd > 0:
                amount_to_buy_usd = rebalance_amount_usd
                if current_price and current_price > 0:
                    buy_qty_coin = amount_to_buy_usd / current_price
            elif rebalance_amount_usd < 0:
                if allow_selling and symbol.upper() not in never_sell:
                    amount_to_sell_usd = abs(rebalance_amount_usd)
                    if current_price and current_price > 0:
                        sell_qty_coin = amount_to_sell_usd / current_price

            # Add to suggestions if it's a target asset OR if it's currently held
            if target_pct > 0 or current_actual_cost_for_symbol > 0:
                 # Check if the asset is currently held (even if cost is 0, quantity might be > 0)
                 is_held = not current_row.empty and current_row['quantity'].iloc[0] > 0
                 if target_pct > 0 or is_held:
                    suggestions.append({
                        "Symbol": symbol,
                        "Target %": f"{target_pct * 100:.2f}%",
                        "Cost (USD)": f"${current_actual_cost_for_symbol:,.2f}", # This will now use live_qty * avg_cost
                        "Target Cost (USD)": f"${target_cost_basis_for_symbol:,.2f}",
                        "Buy (USD)": f"${amount_to_buy_usd:,.2f}",
                        "Buy (Qty)": f"{buy_qty_coin:,.6f}".rstrip('0').rstrip('.'),
                        "Sell (USD)": f"${amount_to_sell_usd:,.2f}",
                        "Sell (Qty)": f"{sell_qty_coin:,.6f}".rstrip('0').rstrip('.')
                    })


        if not suggestions: return pd.DataFrame()

        df = pd.DataFrame(suggestions)
        cols = ["Symbol", "Target %", "Cost (USD)", "Target Cost (USD)",
                "Buy (USD)", "Buy (Qty)", "Sell (USD)", "Sell (Qty)"]
        for col in cols:
            if col not in df.columns:
                df[col] = "$0.00" if "USD" in col else "0"
        return df[cols]

    def print_portfolio_summary(self, metrics: Dict[str, Any]):
        """Print a summary of the portfolio to the console."""
        print("\n" + "="*80)
        print("📊 CRYPTO PORTFOLIO SUMMARY")
        print("="*80)

        # Check for errors first
        if "error" in metrics:
            print(f"❌ Could not generate summary: {metrics['error']}")
            print("="*80)
            return

        timestamp = metrics.get('timestamp')
        # Check if timestamp is a datetime object before formatting
        if isinstance(timestamp, datetime.datetime):
            print(f"Timestamp:             {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"Timestamp:             N/A")

        print(f"Total Portfolio Value: ${metrics.get('total_value_usd', 0):,.2f}")
        print(f"Total Cost Basis:      ${metrics.get('total_cost_basis_usd', 0):,.2f}")
        print(f"Unrealized P/L:        ${metrics.get('unrealized_pl_usd', 0):,.2f} ({metrics.get('unrealized_pl_percent', 0):.2f}%)")
        print("-" * 80)

        holdings_df = metrics.get('holdings_df')
        if holdings_df is not None and not holdings_df.empty:
            print(f"{'Asset':<10} {'Quantity':<15} {'Price (USD)':<15} {'Value (USD)':<18} {'P/L (USD)':<15} {'Allocation':<10}")
            print("-" * 80)
            for _, row in holdings_df.iterrows():
                print(
                    f"{row['symbol']:<10} "
                    f"{row['quantity']:<15,.4f} "
                    f"${row['current_price']:<14,.2f} "
                    f"${row['value_usd']:<17,.2f} "
                    f"${row['unrealized_pl_usd']:<14,.2f} "
                    f"{row['allocation'] * 100:<9.2f}%"
                )
            print("="*80)
        else:
            print("No holdings data to display.")
            print("="*80)

    def export_to_excel(self, metrics: Dict[str, Any]): self.excel_exporter.export(metrics=metrics, holdings_df=metrics.get('holdings_df'))
    def export_to_html(self, metrics: Dict[str, Any]): self.html_exporter.export(metrics=metrics, holdings_df=metrics.get('holdings_df'))
    def export_csv_backup(self): self.csv_exporter.export(transactions_df=self.db_manager.get_all_transactions(), holdings_df=self.db_manager.get_holdings())

    def cleanup_old_data(self): self.db_manager.cleanup_old_data()

    def create_portfolio_charts(self, metrics: Dict[str, Any]):
        """Generate portfolio charts."""
        holdings_df = metrics.get('holdings_df'); target_alloc = self.config.get("target_allocation", {})
        if holdings_df is not None: self.visualizer.generate_all_charts(holdings_df, metrics, target_alloc, pd.DataFrame()) # Pass empty snapshots for now
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

    def _parse_binance_csv(self) -> List[Dict[str, Any]]:
        """
        Parses the Binance transaction history CSV to extract transactions
        not reliably fetched via API (Internal Transfers, Simple Earn).
        """
        csv_path = self.config.get("portfolio", {}).get("binance_csv_path")
        if not csv_path or not os.path.exists(csv_path):
            logger.warning(f"Binance CSV path '{csv_path}' not found or not configured. Skipping CSV parsing.")
            return []

        logger.info(f"Parsing Binance transaction history from CSV: {csv_path}")
        transactions = []
        cg_delay_ms = self.config.get("apis",{}).get("coingecko",{}).get("request_delay_ms_csv", 1200)

        try:
            df = pd.read_csv(csv_path, dtype={'Change': str})
            df = df.sort_values(by='UTC_Time').reset_index(drop=True)
            logger.info(f"Read {len(df)} rows from CSV.")

            for index, row in df.iterrows():
                account = row.get('Account')
                operation = row.get('Operation')
                coin = row.get('Coin')
                change_str = row.get('Change')
                utc_time_str = row.get('UTC_Time')

                if not all([account, operation, coin, change_str, utc_time_str]):
                    logger.debug(f"Skipping CSV row {index+2} due to missing data: {row.to_dict()}")
                    continue

                if account != 'Spot':
                    logger.debug(f"Skipping CSV row {index+2} (Account: {account})")
                    continue

                normalized_symbol = self.norm_map.get(coin.upper(), coin.upper())

                try:
                    quantity = float(change_str)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse 'Change' value '{change_str}' in CSV row {index+2}. Skipping.")
                    continue

                try:
                    timestamp = pd.to_datetime(utc_time_str, utc=True).to_pydatetime()
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse 'UTC_Time' value '{utc_time_str}' in CSV row {index+2}. Skipping.")
                    continue

                tx_type = None
                price_usd = 0.0
                notes = f"CSV: {operation}"
                source = "Binance CSV"
                get_hist_price = False

                if operation == 'Transfer Between Main and Funding Wallet' and quantity > 0:
                    tx_type = 'DEPOSIT'
                    notes = "CSV: Funding to Spot Transfer"
                    get_hist_price = True
                elif operation.startswith('Simple Earn Flexible'):
                    logger.debug(f"Skipping Simple Earn operation '{operation}' in CSV row {index+2} to avoid Qty/Cost issues. Relying on API balances & trades.")
                    continue
                elif operation in ['Transaction Buy', 'Transaction Fee', 'Transaction Spend', 'Deposit', 'Withdrawal', 'P2P Trading']:
                    logger.debug(f"Skipping known API/P2P operation '{operation}' in CSV row {index+2}.")
                    continue
                else:
                    logger.warning(f"Skipping unrecognized CSV operation '{operation}' in row {index+2}.")
                    continue

                if tx_type:
                    if get_hist_price:
                        coin_id = self.symbol_mappings.get(normalized_symbol)
                        if normalized_symbol == 'USDT':
                            price_usd = 1.0
                        elif coin_id:
                            date_str = timestamp.strftime('%d-%m-%Y')
                            hist_price = self._get_coingecko_historical_price(coin_id, date_str)
                            if hist_price is not None:
                                price_usd = hist_price
                                logger.debug(f"Sleeping {cg_delay_ms / 1000.0}s after CoinGecko call/cache for CSV...")
                                time.sleep(cg_delay_ms / 1000.0) # Use the delay (e.g., 1.2 seconds)
                            else:
                                logger.warning(f"Could not get hist price for {normalized_symbol} on {date_str} for CSV tx. Using $0.")
                                price_usd = 0.0
                        else:
                            logger.warning(f"No CoinGecko ID for {normalized_symbol} for CSV tx. Using $0.")
                            price_usd = 0.0

                    if operation in ['Simple Earn Flexible Interest', 'Simple Earn Flexible Airdrop'] and price_usd == 0.0:
                         logger.info(f"Assigning $0 cost basis for CSV '{operation}' of {normalized_symbol}.")
                         price_usd = 0.0

                    tx_hash = f"csv_{index}_{int(timestamp.timestamp() * 1000)}_{normalized_symbol}_{quantity:.8f}"

                    transactions.append({
                        "symbol": normalized_symbol,
                        "timestamp": timestamp,
                        "type": tx_type,
                        "quantity": quantity,
                        "price_usd": price_usd,
                        "fee_quantity": 0.0,
                        "fee_currency": None,
                        "fee_usd": 0.0,
                        "source": source,
                        "transaction_hash": tx_hash,
                        "notes": notes
                    })
                # Add a tiny sleep *between each row* to be generally kinder,
                # even if caching hits.
                time.sleep(0.05)

            logger.info(f"Parsed {len(transactions)} relevant transactions from CSV.")
            return transactions
        except Exception as e:
            logger.error(f"Error parsing Binance CSV file: {e}", exc_info=True)
            return []
