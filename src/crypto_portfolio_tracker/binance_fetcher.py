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
    """A class dedicated to fetching all types of raw data from the Binance API."""

    def __init__(self, client: Optional[Client], symbol_mapper: SymbolMapper, config: Dict[str, Any]):
        if not client:
            raise ValueError("BinanceFetcher requires an initialized Binance client.")
        self.logger = logging.getLogger(__name__)
        self.binance_client = client
        self.symbol_mappings = symbol_mapper
        self.config = config
        self.target_assets_for_sync = set(self.config.get("target_allocation", {}).keys())
        self.target_assets_for_sync.add("USDT")
        self.logger.debug("BinanceFetcher initialized.")

    def _get_start_end_timestamps(self, source_name: str, days_back: int, latest_known_ts: Optional[datetime.datetime]) -> Optional[tuple[int, int]]:
        """Helper to calculate the start and end timestamps for an API call."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        end_ts = int(now_utc.timestamp() * 1000)

        start_dt: datetime.datetime
        if latest_known_ts:
            effective_start_from_db = latest_known_ts - datetime.timedelta(minutes=60) # 1 hour buffer
            fallback_start_from_days = now_utc - datetime.timedelta(days=days_back)
            start_dt = max(effective_start_from_db, fallback_start_from_days)
        else:
            start_dt = now_utc - datetime.timedelta(days=days_back)

        if start_dt >= now_utc:
            self.logger.debug(f"[{source_name}] History is up-to-date. Skipping fetch.")
            return None

        start_ts = int(start_dt.timestamp() * 1000)
        self.logger.debug(f"Fetching [{source_name}] data from {start_dt.strftime('%Y-%m-%d %H:%M')} to {now_utc.strftime('%Y-%m-%d %H:%M')}")
        return start_ts, end_ts

    def _fetch_paginated_history(self, endpoint_path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generic helper to handle paginated endpoints that use 'current' and 'size'."""
        all_rows = []
        params['current'] = 1
        params['size'] = 100 # Max for many of these endpoints

        while True:
            try:
                history = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                rows = history.get('rows', []) if isinstance(history, dict) else history if isinstance(history, list) else []
                if not rows:
                    break
                all_rows.extend(rows)
                if len(rows) < params['size']:
                    break # Reached the last page
                params['current'] += 1
            except Exception as e:
                self.logger.error(f"Error fetching paginated data from {endpoint_path}: {e}")
                break
        return all_rows

    def fetch_binance_balances(self) -> pd.DataFrame:
        """Fetch current balances from Binance Spot wallet."""
        if not self.binance_client:
            self.logger.warning("Binance client not initialized. Cannot fetch Spot balances.")
            return pd.DataFrame(columns=['symbol', 'quantity'])

        try:
            self.logger.info("Re-synchronizing time with Binance server for fresh data...")
            server_time = self.binance_client.get_server_time()['serverTime']
            local_time = int(time.time() * 1000)
            time_offset = server_time - local_time
            self.binance_client._server_time_offset = time_offset
            self.logger.info(f"Time re-synchronized for balance fetch. Offset: {time_offset}ms.")
        except Exception as e:
            self.logger.error(f"Could not re-sync time for balance fetch: {e}. Proceeding with old offset.")

        self.logger.info("Fetching current spot wallet balances...")
        try:
            account_info = self.binance_client.get_account()
            balances_raw = account_info.get('balances', [])

            processed_balances = []
            for b in balances_raw:
                quantity = float(b.get('free', 0.0)) + float(b.get('locked', 0.0))
                if quantity > 1e-8:
                    final_symbol = self.symbol_mappings.normalize_symbol(b.get('asset', ''))

                    # Skip known fiat currencies to prevent unnecessary discovery attempts
                    if final_symbol in ['EUR', 'JPY', 'BRL', 'ARS', 'CZK', 'MXN', 'UAH', 'ZAR', 'TRY']:
                        self.logger.debug(f"Skipping known fiat currency: {final_symbol}")
                        continue

                    if final_symbol:
                        processed_balances.append({'symbol': final_symbol, 'quantity': quantity})

            if not processed_balances:
                return pd.DataFrame(columns=['symbol', 'quantity'])

            df = pd.DataFrame(processed_balances)
            # Consolidate any duplicate symbols (e.g. from LD-prefixed assets)
            df = df.groupby('symbol', as_index=False)['quantity'].sum()

            self.logger.debug(f"Fetched and consolidated {len(df)} non-zero balances.")
            return df
        except Exception as e:
            self.logger.error(f"Unexpected error fetching Spot balances: {e}", exc_info=True)
            raise NetworkOperationError(f"Failed to fetch Binance Spot balances: {e}")

    def fetch_binance_transactions(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ts, _ = time_window
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_chunk_start_dt = pd.to_datetime(start_ts, unit='ms', utc=True).to_pydatetime()
        raw_transactions = []
        stablecoin_quotes = self.config.get("portfolio", {}).get("stablecoin_symbols", ["USDT"])
        crypto_quotes = [s for s in self.config.get("portfolio", {}).get("crypto_quotes", ["BTC"]) if s in self.target_assets_for_sync]
        all_quotes_to_check = sorted(list(set(stablecoin_quotes + crypto_quotes)))
        potential_base_assets = [s for s in self.target_assets_for_sync if s not in stablecoin_quotes]
        processed_pairs = set()

        try:
            for base_asset in potential_base_assets:
                for quote_asset in all_quotes_to_check:
                    if base_asset == quote_asset: continue
                    pair = f"{base_asset}{quote_asset}"
                    if pair in processed_pairs: continue
                    processed_pairs.add(pair)

                    chunk_start = current_chunk_start_dt
                    while chunk_start < now_utc:
                        chunk_end = min(chunk_start + datetime.timedelta(hours=23), now_utc)
                        try:
                            trades = self.binance_client.get_my_trades(symbol=pair, startTime=int(chunk_start.timestamp()*1000), endTime=int(chunk_end.timestamp()*1000), limit=1000)
                            for trade in trades:
                                timestamp = pd.to_datetime(trade.get('time'), unit='ms', utc=True)
                                if pd.isna(timestamp):
                                    self.logger.warning(f"Skipping trade with invalid timestamp: {trade}")
                                    continue
                                raw_transactions.append({'tx_type': 'TRADE', 'timestamp': timestamp, 'source': 'Binance Trade', 'transaction_hash': f"binance_trade_{trade['id']}", 'raw_data': {'base_asset': self.symbol_mappings.normalize_symbol(base_asset), 'quote_asset': self.symbol_mappings.normalize_symbol(quote_asset), 'is_buyer': trade['isBuyer'], 'quantity': float(trade['qty']), 'price': float(trade['price']), 'fee_quantity': float(trade['commission']), 'fee_currency': self.symbol_mappings.normalize_symbol(trade['commissionAsset'])}})
                        except BinanceAPIException as e:
                            if e.code == -1121: self.logger.debug(f"Invalid pair: {pair}. Skipping.")
                            else: self.logger.error(f"[{source_name}] API Error fetching trades for {pair}: {e}")
                            break
                        except Exception as e:
                            self.logger.error(f"[{source_name}] Error fetching trades for {pair}: {e}")
                        chunk_start = chunk_end
            return raw_transactions
        except Exception as e:
            self.logger.error(f"Unexpected error fetching Binance transactions: {e}", exc_info=True)
            raise NetworkOperationError(f"Failed to fetch Binance transactions: {e}")

    def fetch_deposit_history(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ms, end_ms = time_window
        raw_transactions = []
        try:
            for deposit in self.binance_client.get_deposit_history(startTime=start_ms, endTime=end_ms, status=1):
                timestamp = pd.to_datetime(deposit.get('insertTime'), unit='ms', utc=True)
                if pd.isna(timestamp): continue
                normalized_symbol = self.symbol_mappings.normalize_symbol(deposit.get('coin').upper())
                if normalized_symbol not in self.target_assets_for_sync: continue
                raw_transactions.append({'tx_type': 'DEPOSIT', 'timestamp': timestamp, 'source': 'Binance Deposit', 'transaction_hash': deposit.get('txId'), 'raw_data': {'symbol': normalized_symbol, 'quantity': float(deposit.get('amount', 0)), 'notes': f"Network: {deposit.get('network')}"}})
            return raw_transactions
        except BinanceAPIException as e:
            if getattr(e, 'code', None) == -1021:
                self.logger.warning("BinanceAPIException -1021: Timestamp out of recvWindow. Attempting to re-sync time and retry once.")
                self._resync_time()
                try:
                    for deposit in self.binance_client.get_deposit_history(startTime=start_ms, endTime=end_ms, status=1):
                        timestamp = pd.to_datetime(deposit.get('insertTime'), unit='ms', utc=True)
                        if pd.isna(timestamp): continue
                        normalized_symbol = self.symbol_mappings.normalize_symbol(deposit.get('coin').upper())
                        if normalized_symbol not in self.target_assets_for_sync: continue
                        raw_transactions.append({'tx_type': 'DEPOSIT', 'timestamp': timestamp, 'source': 'Binance Deposit', 'transaction_hash': deposit.get('txId'), 'raw_data': {'symbol': normalized_symbol, 'quantity': float(deposit.get('amount', 0)), 'notes': f"Network: {deposit.get('network')}"}})
                    return raw_transactions
                except BinanceAPIException as e2:
                    if getattr(e2, 'code', None) == -1021:
                        self.logger.error("Failed after time re-sync: Timestamp still out of recvWindow. Your system clock may be out of sync with Binance servers. Please check your time settings.")
                        print("❌ Binance API error: Your system clock may be out of sync with Binance servers. Please check your time settings.")
                    else:
                        self.logger.error(f"Error fetching deposit history after retry: {e2}", exc_info=True)
                        print(f"❌ Binance API error: {e2}")
            elif getattr(e, 'code', None) == -2015:
                self.logger.error("BinanceAPIException -2015: Invalid API-key, IP, or permissions for action.")
                print("❌ Binance API error: Invalid API-key, IP, or permissions for action. Please check your API credentials and permissions.")
            else:
                self.logger.error(f"Error fetching deposit history: {e}", exc_info=True)
                print(f"❌ Binance API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching deposit history: {e}", exc_info=True)
            print(f"❌ Unexpected error: {e}")

    def fetch_withdrawal_history(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ms, end_ms = time_window
        raw_transactions = []
        try:
            for withdrawal in self.binance_client.get_withdraw_history(startTime=start_ms, endTime=end_ms, status=6):
                timestamp = pd.to_datetime(withdrawal.get('applyTime'), utc=True)
                if pd.isna(timestamp): continue
                normalized_symbol = self.symbol_mappings.normalize_symbol(withdrawal.get('coin').upper())
                if normalized_symbol not in self.target_assets_for_sync: continue
                raw_transactions.append({'tx_type': 'WITHDRAWAL', 'timestamp': timestamp, 'source': 'Binance Withdrawal', 'transaction_hash': withdrawal.get('txId') or f"binance_withdraw_{withdrawal.get('id')}", 'raw_data': {'symbol': normalized_symbol, 'quantity': float(withdrawal.get('amount', 0)), 'fee_quantity': float(withdrawal.get('transactionFee', 0.0)), 'fee_currency': normalized_symbol}})
            return raw_transactions
        except BinanceAPIException as e:
            if getattr(e, 'code', None) == -1021:
                self.logger.warning("BinanceAPIException -1021: Timestamp out of recvWindow. Attempting to re-sync time and retry once.")
                self._resync_time()
                try:
                    for withdrawal in self.binance_client.get_withdraw_history(startTime=start_ms, endTime=end_ms, status=6):
                        timestamp = pd.to_datetime(withdrawal.get('applyTime'), utc=True)
                        if pd.isna(timestamp): continue
                        normalized_symbol = self.symbol_mappings.normalize_symbol(withdrawal.get('coin').upper())
                        if normalized_symbol not in self.target_assets_for_sync: continue
                        raw_transactions.append({'tx_type': 'WITHDRAWAL', 'timestamp': timestamp, 'source': 'Binance Withdrawal', 'transaction_hash': withdrawal.get('txId') or f"binance_withdraw_{withdrawal.get('id')}", 'raw_data': {'symbol': normalized_symbol, 'quantity': float(withdrawal.get('amount', 0)), 'fee_quantity': float(withdrawal.get('transactionFee', 0.0)), 'fee_currency': normalized_symbol}})
                    return raw_transactions
                except BinanceAPIException as e2:
                    if getattr(e2, 'code', None) == -1021:
                        self.logger.error("Failed after time re-sync: Timestamp still out of recvWindow. Your system clock may be out of sync with Binance servers. Please check your time settings.")
                        print("❌ Binance API error: Your system clock may be out of sync with Binance servers. Please check your time settings.")
                    else:
                        self.logger.error(f"Error fetching withdrawal history after retry: {e2}", exc_info=True)
                        print(f"❌ Binance API error: {e2}")
            else:
                self.logger.error(f"Error fetching withdrawal history: {e}", exc_info=True)
                print(f"❌ Binance API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching withdrawal history: {e}", exc_info=True)
            print(f"❌ Unexpected error: {e}")

    def fetch_p2p_usdt_buys(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetches P2P buy history, filtering for COMPLETED trades first and then
        de-duplicating by orderNumber to ensure absolute accuracy.
        """
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ms, end_ms = time_window

        p2p_fiat_currency = self.config.get("portfolio", {}).get("p2p_fiat_currency", "PHP").upper()

        all_completed_trades = []
        current_page = 1
        PAGE_SIZE = 100

        try:
            while True:
                history = self.binance_client.get_c2c_trade_history(tradeType='BUY', page=current_page, rows=PAGE_SIZE, startTimestamp=start_ms, endTimestamp=end_ms)
                trades_in_page = history.get('data', [])
                if not trades_in_page: break

                for trade in trades_in_page:
                    # Step 1: Filter for only completed trades. The status is 'COMPLETED'.
                    if trade.get('orderStatus') == 'COMPLETED':
                        all_completed_trades.append(trade)

                if len(trades_in_page) < PAGE_SIZE: break
                current_page += 1
                time.sleep(0.5)
            # Step 2: De-duplicate the COMPLETED trades by orderNumber to be 100% safe.
            # This handles the case where the API might send the same completed order twice.
            unique_trades = {trade['orderNumber']: trade for trade in all_completed_trades}
            final_trades = list(unique_trades.values())

            raw_transactions = []
            for trade in final_trades:
                raw_transactions.append({
                    'tx_type': 'P2P_BUY',
                    'timestamp': pd.to_datetime(trade.get('createTime'), unit='ms', utc=True),
                    'source': 'Binance P2P Buy',
                    'transaction_hash': trade.get('orderNumber'),
                    'raw_data': {
                        'asset': 'USDT',
                        'quantity': float(trade.get('amount', 0)),
                        'fiat_currency': p2p_fiat_currency,
                        'fiat_amount': float(trade.get('totalPrice', 0))
                    }
                })

            self.logger.debug(f"Found {len(raw_transactions)} COMPLETED and de-duplicated P2P buy transactions.")
            return raw_transactions
        except BinanceAPIException as e:
            if getattr(e, 'code', None) == -1021:
                self.logger.warning("BinanceAPIException -1021: Timestamp out of recvWindow. Attempting to re-sync time and retry once.")
                self._resync_time()
                try:
                    current_page = 1
                    all_completed_trades = []
                    while True:
                        history = self.binance_client.get_c2c_trade_history(tradeType='BUY', page=current_page, rows=PAGE_SIZE, startTimestamp=start_ms, endTimestamp=end_ms)
                        trades_in_page = history.get('data', [])
                        if not trades_in_page: break
                        for trade in trades_in_page:
                            if trade.get('orderStatus') == 'COMPLETED':
                                all_completed_trades.append(trade)
                        if len(trades_in_page) < PAGE_SIZE: break
                        current_page += 1
                    # Step 2: De-duplicate the COMPLETED trades by orderNumber to be 100% safe.
                    # This handles the case where the API might send the same completed order twice.
                    unique_trades = {trade['orderNumber']: trade for trade in all_completed_trades}
                    final_trades = list(unique_trades.values())

                    raw_transactions = []
                    for trade in final_trades:
                        raw_transactions.append({
                            'tx_type': 'P2P_BUY',
                            'timestamp': pd.to_datetime(trade.get('createTime'), unit='ms', utc=True),
                            'source': 'Binance P2P Buy',
                            'transaction_hash': trade.get('orderNumber'),
                            'raw_data': {
                                'asset': 'USDT',
                                'quantity': float(trade.get('amount', 0)),
                                'fiat_currency': p2p_fiat_currency,
                                'fiat_amount': float(trade.get('totalPrice', 0))
                            }
                        })

                    self.logger.debug(f"Found {len(raw_transactions)} COMPLETED and de-duplicated P2P buy transactions.")
                    return raw_transactions
                except BinanceAPIException as e2:
                    if getattr(e2, 'code', None) == -1021:
                        self.logger.error("Failed after time re-sync: Timestamp still out of recvWindow. Your system clock may be out of sync with Binance servers. Please check your time settings.")
                        print("❌ Binance API error: Your system clock may be out of sync with Binance servers. Please check your time settings.")
                    else:
                        self.logger.error(f"Error fetching P2P USDT buys after retry: {e2}", exc_info=True)
                        print(f"❌ Binance API error: {e2}")
            else:
                self.logger.error(f"Error fetching P2P USDT buys: {e}", exc_info=True)
                print(f"❌ Binance API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching P2P USDT buys: {e}", exc_info=True)
            print(f"❌ Unexpected error: {e}")

    def fetch_spot_convert_history(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ms, end_ms = time_window
        raw_transactions = []
        try:
            history = self.binance_client.get_convert_trade_history(startTime=start_ms, endTime=end_ms, limit=1000)
            for trade in history.get('list', []):
                if trade.get('orderStatus') == "SUCCESS":
                    timestamp = pd.to_datetime(trade.get('createTime'), unit='ms', utc=True)
                    if pd.isna(timestamp): continue
                    from_asset = self.symbol_mappings.normalize_symbol(trade['fromAsset'].upper())
                    to_asset = self.symbol_mappings.normalize_symbol(trade['toAsset'].upper())
                    if not (from_asset in self.target_assets_for_sync or to_asset in self.target_assets_for_sync): continue
                    raw_transactions.append({'tx_type': 'CONVERT', 'timestamp': timestamp, 'source': 'Binance Convert', 'transaction_hash': f"convert_{trade.get('quoteId')}", 'raw_data': {'from_asset': from_asset, 'from_quantity': float(trade['fromAmount']), 'to_asset': to_asset, 'to_quantity': float(trade['toAmount'])}})
            return raw_transactions
        except BinanceAPIException as e:
            if getattr(e, 'code', None) == -1021:
                self.logger.warning("BinanceAPIException -1021: Timestamp out of recvWindow. Attempting to re-sync time and retry once.")
                self._resync_time()
                try:
                    history = self.binance_client.get_convert_trade_history(startTime=start_ms, endTime=end_ms, limit=1000)
                    for trade in history.get('list', []):
                        if trade.get('orderStatus') == "SUCCESS":
                            timestamp = pd.to_datetime(trade.get('createTime'), unit='ms', utc=True)
                            if pd.isna(timestamp): continue
                            from_asset = self.symbol_mappings.normalize_symbol(trade['fromAsset'].upper())
                            to_asset = self.symbol_mappings.normalize_symbol(trade['toAsset'].upper())
                            if not (from_asset in self.target_assets_for_sync or to_asset in self.target_assets_for_sync): continue
                            raw_transactions.append({'tx_type': 'CONVERT', 'timestamp': timestamp, 'source': 'Binance Convert', 'transaction_hash': f"convert_{trade.get('quoteId')}", 'raw_data': {'from_asset': from_asset, 'from_quantity': float(trade['fromAmount']), 'to_asset': to_asset, 'to_quantity': float(trade['toAmount'])}})
                    return raw_transactions
                except BinanceAPIException as e2:
                    if getattr(e2, 'code', None) == -1021:
                        self.logger.error("Failed after time re-sync: Timestamp still out of recvWindow. Your system clock may be out of sync with Binance servers. Please check your time settings.")
                        print("❌ Binance API error: Your system clock may be out of sync with Binance servers. Please check your time settings.")
                    else:
                        self.logger.error(f"Error fetching spot convert history after retry: {e2}", exc_info=True)
                        print(f"❌ Binance API error: {e2}")
            else:
                self.logger.error(f"Error fetching spot convert history: {e}", exc_info=True)
                print(f"❌ Binance API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching spot convert history: {e}", exc_info=True)
            print(f"❌ Unexpected error: {e}")

    def fetch_simple_earn_balances(self, spot_balances_df: pd.DataFrame) -> Dict[str, float]:
        """
        Efficiently fetch balances from Binance Simple Earn Flexible products
        by only checking for assets that exist in the provided spot balances dataframe.
        """
        if not self.binance_client or spot_balances_df.empty:
            self.logger.info("Binance client not available or no spot balances to check for Earn positions.")
            return {}

        earn_balances_aggregated: Dict[str, float] = {}
        assets_to_check_api = spot_balances_df['symbol'].unique().tolist()
        self.logger.info(f"Fetching Simple Earn Flexible balances for the {len(assets_to_check_api)} assets found in your spot wallet...")

        for asset_api_name in assets_to_check_api:
            try:
                if not asset_api_name: continue
                positions = self.binance_client.get_simple_earn_flexible_product_position(asset=asset_api_name)
                if positions and isinstance(positions.get('rows'), list) and positions['rows']:
                    total_amount_for_asset = sum(float(pos.get('totalAmount', 0.0)) for pos in positions['rows'])
                    if total_amount_for_asset > 0:
                        earn_balances_aggregated[asset_api_name] = total_amount_for_asset

                # A small delay to be polite to the API
                time.sleep(0.2)
            except BinanceAPIException as e:
                if e.code in [-6001, -11001] or "not supported" in str(e).lower() or "invalid asset" in str(e).lower():
                     self.logger.debug(f"No active Simple Earn product for {asset_api_name} or asset not supported.")
                else:
                     self.logger.error(f"API Error checking Simple Earn for {asset_api_name}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error checking Simple Earn for {asset_api_name}: {e}")

        self.logger.info(f"Finished checking Earn balances. Found holdings for {len(earn_balances_aggregated)} asset(s).")
        return earn_balances_aggregated

    def fetch_simple_earn_rewards(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """Fetches raw Simple Earn flexible reward data with robust pagination."""
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ms, end_ms = time_window

        raw_transactions = []
        current_page = 1
        endpoint_path = 'simple-earn/flexible/history/rewardRecord'

        while True:
            try:
                params = {"type": "FLEXIBLE", "startTime": start_ms, "endTime": end_ms, "current": current_page, "size": 100}
                response = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                rows = response.get('rows', [])
                if not rows: break

                for item in rows:
                    timestamp = pd.to_datetime(item.get('time'), unit='ms', utc=True)
                    if pd.isna(timestamp): continue

                    normalized_symbol = self.symbol_mappings.normalize_symbol(item.get('asset','').upper())
                    if normalized_symbol not in self.target_assets_for_sync: continue

                    raw_transactions.append({'tx_type': 'EARN_REWARD', 'timestamp': timestamp, 'source': f"Binance Simple Earn Reward ({item.get('type')})", 'transaction_hash': str(item.get('tranId')), 'raw_data': {'symbol': normalized_symbol, 'quantity': float(item.get('rewards', 0.0))}})

                if len(rows) < 100: break
                current_page += 1
            except Exception as e:
                self.logger.error(f"Error fetching Simple Earn rewards (Page {current_page}): {e}")
                break
        return raw_transactions

    def fetch_simple_earn_subscriptions(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """Fetches raw Simple Earn flexible subscription data with robust pagination."""
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ms, end_ms = time_window

        raw_transactions = []
        current_page = 1
        endpoint_path = 'simple-earn/flexible/history/subscriptionRecord'

        while True:
            try:
                params = {"startTime": start_ms, "endTime": end_ms, "current": current_page, "size": 100}
                response = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                rows = response.get('rows', [])
                if not rows: break

                for item in rows:
                    timestamp = pd.to_datetime(item.get('time'), unit='ms', utc=True)
                    if pd.isna(timestamp): continue
                    normalized_symbol = self.symbol_mappings.normalize_symbol(item.get('asset','').upper())
                    if normalized_symbol not in self.target_assets_for_sync: continue
                    raw_transactions.append({'tx_type': 'EARN_SUBSCRIPTION', 'timestamp': timestamp, 'source': 'Binance Simple Earn Subscription', 'transaction_hash': str(item.get('purchaseId')), 'raw_data': {'symbol': normalized_symbol, 'quantity': float(item.get('amount', 0.0))}})

                if len(rows) < 100: break
                current_page += 1
            except Exception as e:
                self.logger.error(f"Error fetching Simple Earn subscriptions (Page {current_page}): {e}")
                break
        return raw_transactions

    def fetch_simple_earn_redemptions(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """Fetches raw Simple Earn flexible redemption data with robust pagination."""
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ms, end_ms = time_window

        raw_transactions = []
        current_page = 1
        endpoint_path = 'simple-earn/flexible/history/redemptionRecord'

        while True:
            try:
                params = {"startTime": start_ms, "endTime": end_ms, "current": current_page, "size": 100}
                response = self.binance_client._request_margin_api('get', endpoint_path, True, data=params)
                rows = response.get('rows', [])
                if not rows: break

                for item in rows:
                    timestamp = pd.to_datetime(item.get('time'), unit='ms', utc=True)
                    if pd.isna(timestamp): continue
                    normalized_symbol = self.symbol_mappings.normalize_symbol(item.get('asset','').upper())
                    if normalized_symbol not in self.target_assets_for_sync: continue
                    raw_transactions.append({'tx_type': 'EARN_REDEMPTION', 'timestamp': timestamp, 'source': 'Binance Simple Earn Redemption', 'transaction_hash': str(item.get('redeemId')), 'raw_data': {'symbol': normalized_symbol, 'quantity': float(item.get('amount', 0.0))}})

                if len(rows) < 100: break
                current_page += 1
            except Exception as e:
                self.logger.error(f"Error fetching Simple Earn redemptions (Page {current_page}): {e}")
                break
        return raw_transactions

    def fetch_dividend_history(self, source_name: str, days_back: int = 90, latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts)
        if not time_window: return []
        start_ms, end_ms = time_window
        raw_transactions = []
        try:
            for item in self.binance_client.get_asset_dividend_history(startTime=start_ms, endTime=end_ms, limit=500).get('rows', []):
                timestamp = pd.to_datetime(item.get('divTime'), unit='ms', utc=True)
                if pd.isna(timestamp): continue
                normalized_symbol = self.symbol_mappings.normalize_symbol(item.get('asset','').upper())
                if normalized_symbol not in self.target_assets_for_sync: continue
                raw_transactions.append({'tx_type': 'DIVIDEND', 'timestamp': timestamp, 'source': 'Binance Dividend', 'transaction_hash': str(item.get('tranId')), 'raw_data': {'symbol': normalized_symbol, 'quantity': float(item.get('amount', 0.0)), 'notes': item.get('enInfo', '')}})
        except Exception as e:
            self.logger.error(f"Error fetching dividend history: {e}")
        return raw_transactions

    def fetch_staking_history(self, source_name: str, days_back: int = 90, latest_known_ts_map: Optional[Dict[str, datetime.datetime]] = None) -> List[Dict[str, Any]]:
        if latest_known_ts_map is None: latest_known_ts_map = {}
        all_txs = []
        transaction_types = {
            "SUBSCRIPTION": {'source': "Binance Staking Subscription", 'tx_type': "STAKING_SUBSCRIBE"},
            "REDEMPTION": {'source': "Binance Staking Redemption", 'tx_type': "STAKING_REDEMPTION"},
            "INTEREST": {'source': "Binance Staking Interest", 'tx_type': "STAKING_INTEREST"}
        }
        for txn_type, details in transaction_types.items():
            source_name = details['source']
            time_window = self._get_start_end_timestamps(source_name, days_back, latest_known_ts_map.get(source_name))
            if not time_window: continue
            start_ms, end_ms = time_window
            try:
                for item in self._fetch_paginated_history('staking/history', {"product": "STAKING", "txnType": txn_type, "startTime": start_ms, "endTime": end_ms}):
                    timestamp = pd.to_datetime(item.get('time'), unit='ms', utc=True)
                    if pd.isna(timestamp): continue
                    normalized_symbol = self.symbol_mappings.normalize_symbol(item.get('asset','').upper())
                    if normalized_symbol not in self.target_assets_for_sync: continue
                    all_txs.append({'tx_type': details['tx_type'], 'timestamp': timestamp, 'source': source_name, 'transaction_hash': str(item.get('txnId')), 'raw_data': {'symbol': normalized_symbol, 'quantity': float(item.get('amount', 0.0))}})
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
            self.logger.error(f"Error fetching futures account balance: {e}")
            return []

    def fetch_funding_balance(self) -> List[Dict[str, Any]]:
        """Fetches funding wallet balance by calling the SAPI endpoint directly."""
        self.logger.info("Fetching funding wallet balances...")
        try:
            funding_wallet = self.binance_client._request_margin_api(
                'post', 'asset/get-funding-asset', signed=True, data={}
            )
            return funding_wallet
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error fetching funding wallet: {e}")
            return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred fetching funding wallet: {e}")
            return []

    def _resync_time(self):
        """Synchronize local time with Binance server."""
        try:
            self.logger.info("Re-synchronizing time with Binance server for fresh data...")
            server_time = self.binance_client.get_server_time()['serverTime']
            local_time = int(time.time() * 1000)
            time_offset = server_time - local_time
            self.binance_client._server_time_offset = time_offset
            self.logger.info(f"Time re-synchronized for balance fetch. Offset: {time_offset}ms.")
        except Exception as e:
            self.logger.error(f"Failed to sync time with Binance: {e}", exc_info=True)
