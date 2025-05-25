"""
Crypto Portfolio Tracker - Main Class
Handles API interactions, data processing, analysis, and orchestration.
"""
import logging
import time
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
    NOTE: Simplified version - ignores fees, non-USD pairs, deposits/withdrawals.
    """
    buy_lots = deque()  # Queue to store {'qty': quantity, 'price': price_usd_per_unit}
    current_quantity = 0.0
    total_cost_basis = 0.0

    if transactions_df.empty:
        return 0.0, 0.0

    transactions_df = transactions_df.sort_values(by='timestamp').reset_index(drop=True)

    for _, row in transactions_df.iterrows():
        tx_type = row['type']
        quantity = row['quantity']
        price = row.get('price_usd')

        if tx_type == 'BUY':
            if price is None or price <= 0:
                logger.warning(f"BUY tx for {row.get('symbol', 'N/A')} missing price: {row.get('transaction_hash', 'N/A')}. Adding qty only.")
                current_quantity += quantity
                buy_lots.append({'qty': quantity, 'price': 0.0})
                continue

            buy_lots.append({'qty': quantity, 'price': price})
            current_quantity += quantity
            total_cost_basis += quantity * price

        elif tx_type == 'SELL':
            sell_qty = quantity
            current_quantity -= sell_qty

            while sell_qty > 0 and buy_lots:
                oldest_buy = buy_lots[0]
                if oldest_buy['qty'] <= sell_qty:
                    qty_to_remove = oldest_buy['qty']
                    cost_to_remove = oldest_buy['qty'] * oldest_buy['price']
                    total_cost_basis -= cost_to_remove
                    sell_qty -= qty_to_remove
                    buy_lots.popleft()
                else:
                    cost_to_remove = sell_qty * oldest_buy['price']
                    total_cost_basis -= cost_to_remove # Corrected: Should be cost_to_remove
                    oldest_buy['qty'] -= sell_qty
                    sell_qty = 0

            if sell_qty > 0:
                logger.warning(f"Sold {sell_qty} {row.get('symbol', 'N/A')} more than BUY history. Check for missing txs.")

        total_cost_basis = max(0, total_cost_basis)

    average_cost_basis = total_cost_basis / current_quantity if current_quantity > 0 else 0.0
    return current_quantity, average_cost_basis


def calculate_fifo_cost_basis(transactions_df: pd.DataFrame):
    """
    Calculates current quantity and average cost basis using FIFO.
    Assumes transactions_df is for a single asset, sorted by timestamp.
    NOTE: Simplified version - ignores fees, non-USD pairs, deposits/withdrawals.
    """
    buy_lots = deque()  # Queue to store {'qty': quantity, 'price': price_usd_per_unit}
    current_quantity = 0.0
    total_cost_basis = 0.0
    logger = logging.getLogger(__name__) # Use logger if available

    if transactions_df.empty:
        return 0.0, 0.0

    # Ensure it's sorted
    transactions_df = transactions_df.sort_values(by='timestamp').reset_index(drop=True)

    for _, row in transactions_df.iterrows():
        tx_type = row['type']
        quantity = row['quantity']
        price = row.get('price_usd') # Using .get() for safety

        if tx_type == 'BUY':
            if price is None or price <= 0:
                logger.warning(f"BUY tx for {row['symbol']} missing price: {row['transaction_hash']}. Adding qty only.")
                current_quantity += quantity
                # Optionally add a '0 cost' lot if needed, or handle as an error
                buy_lots.append({'qty': quantity, 'price': 0.0}) # Add as 0 cost for now
                continue

            buy_lots.append({'qty': quantity, 'price': price})
            current_quantity += quantity
            total_cost_basis += quantity * price

        elif tx_type == 'SELL':
            sell_qty = quantity
            current_quantity -= sell_qty

            while sell_qty > 0 and buy_lots:
                oldest_buy = buy_lots[0]

                if oldest_buy['qty'] <= sell_qty:
                    # This lot is fully (or exactly) consumed
                    qty_to_remove = oldest_buy['qty']
                    cost_to_remove = oldest_buy['qty'] * oldest_buy['price']
                    total_cost_basis -= cost_to_remove
                    sell_qty -= qty_to_remove
                    buy_lots.popleft() # Remove the oldest lot

                else:
                    # This lot is partially consumed
                    cost_to_remove = sell_qty * oldest_buy['price']
                    total_cost_basis -= sell_qty
                    oldest_buy['qty'] -= sell_qty
                    sell_qty = 0 # All sold quantity accounted for

            if sell_qty > 0:
                logger.warning(f"Sold {sell_qty} {row['symbol']} more than BUY history suggests. Check for missing deposits/txs.")
                # We can't reduce cost basis below zero

        # Ensure cost basis isn't negative due to rounding or warnings
        total_cost_basis = max(0, total_cost_basis)


    average_cost_basis = total_cost_basis / current_quantity if current_quantity > 0 else 0.0

    # Return total quantity and average cost basis
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
        self.symbol_mappings = self.config.get("symbol_mappings", {}).get("coingecko_ids", {})

    def _init_binance_client(self) -> Optional[Client]:
        """Initialize and return Binance client."""
        api_keys = self.config.get("api_keys", {}); api_key = api_keys.get("binance_key"); api_secret = api_keys.get("binance_secret")
        if not api_key or not api_secret: logger.warning("Binance API key/secret not found. Binance disabled."); return None
        try:
            client = Client(api_key, api_secret, requests_params={'timeout': 60})
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

    def fetch_binance_balances(self) -> pd.DataFrame:
        """Fetch current balances from Binance."""
        if not self.binance_client: return pd.DataFrame()
        try:
            account_info = self.binance_client.get_account()
            balances = [b for b in account_info.get('balances', []) if float(b['free']) > 0 or float(b['locked']) > 0]
            df = pd.DataFrame(balances); df['quantity'] = df['free'].astype(float) + df['locked'].astype(float)
            df = df[df['quantity'] > 0][['asset', 'quantity']].rename(columns={'asset': 'symbol'})
            logger.info(f"Fetched {len(df)} balances from Binance."); return df
        except Exception as e: logger.error(f"Error fetching Binance balances: {e}"); return pd.DataFrame()

    def fetch_binance_transactions(self) -> List[Dict[str, Any]]:
        """Fetch trade history from Binance for known symbols."""
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch transactions.")
            return []

        transactions = []
        norm_map = self.config.get("symbol_normalization_map", {})
        known_symbols = list(self.config.get("symbol_mappings", {}).get("coingecko_ids", {}).keys())
        known_quotes = ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH']

        logger.info(f"Attempting to fetch trades for {len(known_symbols)} base symbols...")

        processed_pairs = set()

        for symbol_to_find in known_symbols:
            # Skip stablecoins or LD assets when they are the primary symbol we're looking for
            if symbol_to_find.upper() in ['USDT', 'USDC', 'BUSD', 'DAI'] or symbol_to_find.upper().startswith('LD'):
                continue

            # Try common trading pairs
            for quote in known_quotes:
                pair = f"{symbol_to_find.upper()}{quote.upper()}"

                # Avoid self-trading pairs like BTCBTC, and already processed pairs
                if symbol_to_find.upper() == quote.upper() or pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                try:
                    logger.debug(f"Fetching trades for {pair}...")
                    trades = self.binance_client.get_my_trades(symbol=pair)

                    if not trades:
                        continue

                    logger.info(f"Fetched {len(trades)} trades for {pair}.")

                    for trade in trades:
                        # Determine base and quote (we know it based on how we built 'pair')
                        base_asset = symbol_to_find.upper()
                        quote_asset = quote.upper()

                        # ** Apply normalization **
                        normalized_symbol = norm_map.get(base_asset, base_asset)

                        tx_type = 'BUY' if trade['isBuyer'] else 'SELL'
                        quantity = float(trade['qty'])
                        price = float(trade['price'])
                        fee = float(trade['commission'])
                        fee_asset = trade['commissionAsset']

                        price_usd = price if quote_asset in ['USDT', 'BUSD', 'USDC'] else None
                        if price_usd is None:
                             logger.warning(f"Trade {trade['id']} for {pair} is non-USD. Cost basis needs historical price lookup (TODO).")

                        transactions.append({
                            "symbol": normalized_symbol,
                            "timestamp": pd.to_datetime(trade['time'], unit='ms').to_pydatetime(),
                            "type": tx_type,
                            "quantity": quantity,
                            "price_usd": price_usd,
                            "fee_usd": None,
                            "source": "Binance",
                            "transaction_hash": f"binance_{trade['id']}",
                            "notes": f"Pair: {pair}, Fee: {fee} {fee_asset}, Price:{price} {quote_asset}"
                        })

                except BinanceAPIException as e:
                    if e.status_code != 400 and e.code != -1121:
                         logger.error(f"API Error fetching trades for {pair}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error fetching trades for {pair}: {e}")

                time.sleep(0.5) # Be nice to the API

        logger.info(f"Fetched a total of {len(transactions)} trades.")
        return transactions

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
            # Ensure necessary columns are numeric and handle N/A
            group_df['price_usd'] = pd.to_numeric(group_df['price_usd'], errors='coerce').fillna(0)
            group_df['quantity'] = pd.to_numeric(group_df['quantity'], errors='coerce').fillna(0)

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
        """Synchronize data from APIs and update holdings."""
        logger.info("Starting data synchronization...")

        # 1. Fetch Binance Transactions & insert
        logger.info("Fetching new transactions...")
        binance_txs = self.fetch_binance_transactions()
        if binance_txs:
            self.db_manager.bulk_insert_transactions(binance_txs)
        else:
            logger.info("No new transactions fetched (or client disabled/error).")

        # 2. Recalculate holdings based on ALL transactions in DB
        self.update_holdings_from_transactions()

        logger.info("Data synchronization finished.")

    def get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for a list of symbols using CoinGecko (Batched)."""
        prices = {}
        coingecko_ids = []
        symbol_to_id_map = {} # To map results back

        logger.info(f"Mapping {len(symbols)} symbols to CoinGecko IDs...")
        for symbol in set(symbols): # Use set to avoid duplicates
            coin_id = self.symbol_mappings.get(symbol.upper())
            if not coin_id:
                logger.warning(f"No CoinGecko ID mapping found for {symbol}. Skipping.")
                prices[symbol] = 0.0 # Set to 0 if no ID
            else:
                coingecko_ids.append(coin_id)
                symbol_to_id_map[coin_id] = symbol # Store mapping

        if not coingecko_ids:
            return prices

        ids_string = ",".join(set(coingecko_ids))
        base_url = self.coingecko_api.get("base_url")
        timeout = self.coingecko_api.get("timeout", 30)
        url = f"{base_url}/simple/price?ids={ids_string}&vs_currencies=usd"

        logger.info(f"Fetching {len(set(coingecko_ids))} prices from CoinGecko in one batch...")
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            for coin_id, price_data in data.items():
                symbol = symbol_to_id_map.get(coin_id)
                if symbol and "usd" in price_data:
                    prices[symbol] = price_data["usd"]
                elif symbol:
                    prices[symbol] = 0.0 # Set to 0 if price not found

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching batch prices from CoinGecko: {e}")
            for cid in coingecko_ids:
                symbol = symbol_to_id_map.get(cid)
                if symbol: prices[symbol] = 0.0 # Set all to 0 on error

        logger.info("Price fetching complete.")
        # Ensure all originally requested symbols (even those without ID) have an entry
        for symbol in symbols:
            if symbol not in prices:
                 prices[symbol] = 0.0

        return prices

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate key portfolio metrics based on DB data."""
        logger.info("Calculating portfolio metrics..."); holdings_df = self.db_manager.get_holdings()
        if holdings_df.empty:
            holdings_df = self.fetch_binance_balances()
            if holdings_df.empty: return {"error": "No holdings data."}
            holdings_df['average_cost_basis'] = 0.0
        prices = self.get_current_prices(holdings_df['symbol'].tolist())
        holdings_df['current_price'] = holdings_df['symbol'].map(prices).fillna(0)
        holdings_df['value_usd'] = holdings_df['quantity'] * holdings_df['current_price']
        holdings_df['cost_basis'] = holdings_df['quantity'] * holdings_df['average_cost_basis']
        holdings_df['unrealized_pl_usd'] = holdings_df['value_usd'] - holdings_df['cost_basis']
        holdings_df['unrealized_pl_percent'] = (holdings_df['unrealized_pl_usd'] / holdings_df['cost_basis'].replace(0, 1)) * 100
        total_value = holdings_df['value_usd'].sum(); total_cost = holdings_df['cost_basis'].sum()
        holdings_df['allocation'] = (holdings_df['value_usd'] / total_value) if total_value > 0 else 0
        metrics = {"total_value_usd": total_value, "total_cost_basis_usd": total_cost, "unrealized_pl_usd": total_value - total_cost, "unrealized_pl_percent": ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0, "holdings_df": holdings_df, "timestamp": datetime.datetime.now()}
        logger.info(f"Metrics Calculated: Total Value = ${total_value:,.2f}"); return metrics

    def run_full_sync(self) -> Dict[str, Any]:
        """Run full sync and analysis process."""
        self.sync_data(); return self.calculate_portfolio_metrics()

    def get_rebalance_suggestions_by_cost(self) -> Optional[pd.DataFrame]:
        """
        Calculates buy/sell amounts to reach target allocation based on cost basis.
        """
        logger.info("Calculating rebalance suggestions by cost basis...")
        holdings_df = self.db_manager.get_holdings() # Needs accurate cost_basis!
        strategy_config = self.config.get("rebalancing_strategy", {})
        target_allocation = self.config.get("target_allocation", {})

        allow_selling = strategy_config.get("allow_selling", True)
        # Ensure comparison is case-insensitive and handles potential None
        never_sell = [s.upper() for s in strategy_config.get("never_sell_symbols", [])]

        if holdings_df.empty or not target_allocation:
            logger.warning("Need holdings with cost basis and target allocation for suggestions.")
            return None

        # Ensure cost_basis column exists
        if 'average_cost_basis' not in holdings_df.columns or 'quantity' not in holdings_df.columns:
             logger.error("Cannot calculate: Holdings data missing quantity or average_cost_basis.")
             logger.warning("Please ensure transaction history is synced and cost basis calculated.")
             return None

        holdings_df['current_cost_basis'] = holdings_df['quantity'] * holdings_df['average_cost_basis']

        total_portfolio_cost_basis = holdings_df['current_cost_basis'].sum()

        if total_portfolio_cost_basis == 0:
            logger.warning("Total cost basis is zero. Cannot calculate suggestions.")
            logger.warning("This usually means transaction history/cost basis is missing.")
            return None

        logger.info(f"Total Portfolio Cost Basis: ${total_portfolio_cost_basis:,.2f}")

        suggestions = []
        # Ensure we iterate through all assets, both in holdings and target
        all_symbols = set(list(target_allocation.keys()) + holdings_df['symbol'].tolist())

        for symbol in all_symbols:
            target_pct = target_allocation.get(symbol, 0) # Get target, default to 0 if not in target
            target_cost_basis = total_portfolio_cost_basis * target_pct

            current_row = holdings_df[holdings_df['symbol'] == symbol]
            current_cost = current_row['current_cost_basis'].iloc[0] if not current_row.empty else 0

            rebalance_amount = target_cost_basis - current_cost

            amount_to_buy = 0.0
            amount_to_sell = 0.0

            if rebalance_amount > 0:
                amount_to_buy = rebalance_amount
            elif rebalance_amount < 0:
                if allow_selling and symbol.upper() not in never_sell:
                    amount_to_sell = abs(rebalance_amount)

            # Only add to suggestions if there's a target or current cost
            if target_pct > 0 or current_cost > 0:
                suggestions.append({
                    "Symbol": symbol,
                    "Target %": f"{target_pct * 100:.2f}%",
                    "Current Cost (USD)": f"${current_cost:,.2f}",
                    "Target Cost (USD)": f"${target_cost_basis:,.2f}",
                    "Buy (USD)": f"${amount_to_buy:,.2f}",
                    "Sell (USD)": f"${amount_to_sell:,.2f}"
                })

        return pd.DataFrame(suggestions)

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
        print(pd.io.json.dumps(safe_config, indent=2) + "\n" + "="*50)

    def cleanup_old_data(self): self.db_manager.cleanup_old_data()

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
