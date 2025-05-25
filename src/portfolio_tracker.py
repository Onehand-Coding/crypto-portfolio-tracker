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
    logger = logging.getLogger(__name__) # Use logger if available

    if transactions_df.empty:
        return 0.0, 0.0

    # Ensure it's sorted
    transactions_df = transactions_df.sort_values(by='timestamp').reset_index(drop=True)

    for _, row in transactions_df.iterrows():
        tx_type = row['type']
        quantity = row['quantity']
        price = row.get('price_usd') # Already uses .get()

        if tx_type == 'BUY' or tx_type == 'DEPOSIT':
            if price is None or price < 0: # Allow $0 price for gifts/airdrops
                logger.warning(f"{tx_type} tx for {row.get('symbol', 'N/A')} has invalid/missing price: {row.get('transaction_hash', 'N/A')}. Using $0 cost for this lot.")
                price = 0.0 # Treat as $0 cost

            buy_lots.append({'qty': quantity, 'price': price})
            current_quantity += quantity
            total_cost_basis += quantity * price

        elif tx_type == 'SELL' or tx_type == 'WITHDRAWAL':
            sell_qty = quantity # For withdrawals, it's the quantity withdrawn
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
                    # This was a bug in a previous version from the diff, it should be cost_to_remove
                    total_cost_basis -= cost_to_remove
                    oldest_buy['qty'] -= sell_qty
                    sell_qty = 0

            if sell_qty > 0:
                logger.warning(f"{tx_type} of {sell_qty} {row.get('symbol', 'N/A')} more than BUY/DEPOSIT history. Check txs.")

        total_cost_basis = max(0, total_cost_basis)

    average_cost_basis = total_cost_basis / current_quantity if current_quantity > 0 else 0.0
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
        """Synchronize data from APIs and update holdings."""
        logger.info("Starting data synchronization...")

        # 1. Fetch Binance Transactions, Deposits, Withdrawals & insert
        logger.info("Fetching new trades...")
        binance_trades = self.fetch_binance_transactions()

        logger.info("Fetching new deposits...")
        binance_deposits = self.fetch_deposit_history()

        logger.info("Fetching new withdrawals...")
        binance_withdrawals = self.fetch_withdrawal_history()

        all_new_transactions = binance_trades + binance_deposits + binance_withdrawals

        if all_new_transactions:
            # Sort by timestamp before inserting to help FIFO later if DB doesn't sort perfectly
            all_new_transactions.sort(key=lambda x: x['timestamp'])
            self.db_manager.bulk_insert_transactions(all_new_transactions)
            logger.info(f"Inserted/Ignored {len(all_new_transactions)} new records.")
        else:
            logger.info("No new trades, deposits, or withdrawals fetched.")

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
        Calculates buy/sell USD amounts and coin quantities to reach target allocation
        based on cost basis.
        """
        logger.info("Calculating rebalance suggestions by cost basis...")
        holdings_df = self.db_manager.get_holdings()
        strategy_config = self.config.get("rebalancing_strategy", {})
        target_allocation = self.config.get("target_allocation", {})
        # For normalization consistency in rebalancing output
        norm_map = self.config.get("symbol_normalization_map", {})

        allow_selling = strategy_config.get("allow_selling", True)
        never_sell = [s.upper() for s in strategy_config.get("never_sell_symbols", [])]

        if holdings_df.empty or not target_allocation:
            logger.warning("Need holdings with cost basis and target allocation for suggestions.")
            return None
        if 'average_cost_basis' not in holdings_df.columns or 'quantity' not in holdings_df.columns:
            logger.error("Cannot calculate: Holdings data missing quantity or average_cost_basis.")
            logger.warning("Please ensure transaction history is synced and cost basis calculated.")
            return None

        holdings_df['current_cost_basis'] = holdings_df['quantity'] * holdings_df['average_cost_basis']
        total_portfolio_cost_basis = holdings_df['current_cost_basis'].sum()

        if total_portfolio_cost_basis == 0:
            logger.warning("Total cost basis is zero. Cannot calculate suggestions.")
            return None

        logger.info(f"Total Portfolio Cost Basis: ${total_portfolio_cost_basis:,.2f}")

        suggestions = []
        # Create a consistent list of symbols, normalizing those from target_allocation
        target_symbols_normalized = {norm_map.get(s.upper(), s.upper()): p for s, p in target_allocation.items()}
        holdings_symbols_normalized = holdings_df['symbol'].unique().tolist() # Already normalized from DB

        all_display_symbols = sorted(list(set(list(target_symbols_normalized.keys()) + holdings_symbols_normalized)))


        # Get current prices for all relevant symbols for quantity calculation
        current_prices = self.get_current_prices(all_display_symbols)

        for symbol_display in all_display_symbols: # Iterate using normalized symbols
            target_pct = target_symbols_normalized.get(symbol_display, 0)
            target_cost_basis = total_portfolio_cost_basis * target_pct

            current_row = holdings_df[holdings_df['symbol'] == symbol_display]
            current_cost = current_row['current_cost_basis'].iloc[0] if not current_row.empty else 0.0

            rebalance_amount_usd = target_cost_basis - current_cost

            amount_to_buy_usd = 0.0
            amount_to_sell_usd = 0.0
            buy_qty_coin = 0.0
            sell_qty_coin = 0.0
            current_price = current_prices.get(symbol_display, 0.0)

            if rebalance_amount_usd > 0:
                amount_to_buy_usd = rebalance_amount_usd
                if current_price and current_price > 0:
                    buy_qty_coin = amount_to_buy_usd / current_price
            elif rebalance_amount_usd < 0:
                if allow_selling and symbol_display.upper() not in never_sell:
                    amount_to_sell_usd = abs(rebalance_amount_usd)
                    if current_price and current_price > 0:
                        sell_qty_coin = amount_to_sell_usd / current_price

            # Only add to suggestions if there's a target or current cost
            if target_pct > 0 or current_cost > 0:
                suggestions.append({
                    "Symbol": symbol_display,
                    "Target %": f"{target_pct * 100:.2f}%",
                    "Cost (USD)": f"${current_cost:,.2f}", # Renamed for brevity
                    "Target Cost (USD)": f"${target_cost_basis:,.2f}",
                    "Buy (USD)": f"${amount_to_buy_usd:,.2f}",
                    "Buy (Qty)": f"{buy_qty_coin:,.6f}".rstrip('0').rstrip('.'),
                    "Sell (USD)": f"${amount_to_sell_usd:,.2f}",
                    "Sell (Qty)": f"{sell_qty_coin:,.6f}".rstrip('0').rstrip('.')
                })

        if not suggestions:
            return pd.DataFrame()

        df = pd.DataFrame(suggestions)
        # Define column order
        cols = ["Symbol", "Target %", "Cost (USD)", "Target Cost (USD)",
                "Buy (USD)", "Buy (Qty)", "Sell (USD)", "Sell (Qty)"]
        # Ensure all columns are present, add if missing (e.g., if no buys/sells happen)
        for col in cols:
            if col not in df.columns:
                df[col] = "$0.00" if "USD" in col else "0" # Default values
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


    def fetch_deposit_history(self) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch deposit history.")
            return []

        deposits_data = []
        norm_map = self.config.get("symbol_normalization_map", {})

        # Get PEPE gift details from config
        pepe_gift_config = self.config.get("pepe_gift_details", {})
        pepe_gift_symbol = pepe_gift_config.get("symbol", "PEPE").upper()
        try:
            # Ensure the amount from config is treated as float
            your_pepe_gift_amount = float(pepe_gift_config.get("amount", "0"))
        except ValueError:
            your_pepe_gift_amount = 0.0
            logger.warning(f"PEPE gift amount '{pepe_gift_config.get('amount')}' is invalid. Will not apply special $0 cost.")

        try:
            logger.info("Fetching recent deposit history (last 90 days)...")
            endTime = int(datetime.datetime.now().timestamp() * 1000)
            startTime = int((datetime.datetime.now() - datetime.timedelta(days=90)).timestamp() * 1000)
            all_deposits = self.binance_client.get_deposit_history(startTime=startTime, endTime=endTime, status=1)
            logger.info(f"Fetched {len(all_deposits)} successful deposit records.")

            for i, deposit in enumerate(all_deposits): # Use enumerate for unique logging
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

                deposit_timestamp_obj = None # Renamed to avoid confusion with pd.Timestamp
                try:
                    deposit_timestamp_obj = pd.to_datetime(insert_time_raw, unit='ms')
                    logger.debug(f"For {normalized_symbol} (TxID: {deposit.get('txId')}): Raw insertTime: {insert_time_raw}, Parsed Timestamp: {deposit_timestamp_obj}")
                    if pd.isna(deposit_timestamp_obj):
                        logger.error(f"Parsed timestamp is NaT for {normalized_symbol} (TxID: {deposit.get('txId')}) from raw value '{insert_time_raw}'. Skipping.")
                        continue
                except Exception as e:
                    logger.error(f"Could not parse insertTime '{insert_time_raw}' for {normalized_symbol} (TxID: {deposit.get('txId')}): {e}. Skipping.")
                    continue

                price_usd_at_deposit = 0.0

                # Heuristic for your PEPE gift
                # Compare amounts carefully using a small tolerance for float comparisons
                if (normalized_symbol == pepe_gift_symbol and
                    abs(float(deposit.get('amount', 0)) - your_pepe_gift_amount) < 1e-9 and
                    your_pepe_gift_amount > 0): # Make sure configured gift amount is valid
                    logger.info(f"Identified PEPE gift deposit ({deposit.get('amount')} {normalized_symbol}). Assigning $0 cost.")
                    price_usd_at_deposit = 0.0
                else:
                    coin_id = self.symbol_mappings.get(normalized_symbol)
                    if coin_id:
                        date_str = deposit_timestamp_obj.strftime('%d-%m-%Y')
                        historical_price = self._get_coingecko_historical_price(coin_id, date_str)
                        if historical_price is not None:
                            price_usd_at_deposit = historical_price
                            logger.info(f"Fetched historical price for {normalized_symbol} on {date_str}: ${price_usd_at_deposit:.6f}")
                        else:
                            logger.warning(f"Could not fetch historical price for {normalized_symbol} on {date_str}. Cost for deposit will be $0.")
                            price_usd_at_deposit = 0.0
                    else:
                        logger.warning(f"No CoinGecko ID for {normalized_symbol}. Cannot fetch historical price. Cost for deposit will be $0.")
                        price_usd_at_deposit = 0.0

                deposits_data.append({
                    "symbol": normalized_symbol,
                    "timestamp": deposit_timestamp_obj.to_pydatetime(), # Convert to Python datetime
                    "type": "DEPOSIT",
                    "quantity": float(deposit.get('amount', 0)), # Use .get() for safety
                    "price_usd": price_usd_at_deposit,
                    "fee_usd": 0.0,
                    "source": "Binance Deposit",
                    "transaction_hash": deposit.get('txId', f"binance_deposit_{deposit.get('id', i)}"), # Safer txId
                    "notes": f"Status: {deposit.get('status', 'N/A')}"
                })
                time.sleep(1)
        except BinanceAPIException as e:
            logger.error(f"API Error fetching deposit history: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching deposit history: {e}")
        return deposits_data

    def fetch_withdrawal_history(self) -> List[Dict[str, Any]]:
        if not self.binance_client:
            logger.warning("Binance client not initialized. Cannot fetch withdrawal history.")
            return []

        withdrawals_data = []
        try:
            logger.info("Fetching recent withdrawal history (last 90 days)...")
            endTime = int(datetime.datetime.now().timestamp() * 1000)
            startTime = int((datetime.datetime.now() - datetime.timedelta(days=90)).timestamp() * 1000)

            # Status: 0:Email Sended,1:Cancelled 2:Awaiting Approval 3:Rejected 4:Processing 5:Failure 6:Completed
            all_withdrawals = self.binance_client.get_withdraw_history(startTime=startTime, endTime=endTime, status=6) # status 6 is 'Completed'
            logger.info(f"Fetched {len(all_withdrawals)} completed withdrawal records.")

            for withdrawal in all_withdrawals:
                symbol = withdrawal['coin']
                norm_map = self.config.get("symbol_normalization_map", {})
                normalized_symbol = norm_map.get(symbol.upper(), symbol.upper())

                # For withdrawals, 'price_usd' is not directly relevant for cost basis reduction via FIFO
                # as FIFO just reduces quantity and the cost of the oldest lots.
                # However, for tax purposes, the market value at withdrawal might be needed.
                # We'll set it to None as it doesn't affect FIFO cost calculation directly.
                price_usd_at_withdrawal = None
                fee = float(withdrawal.get('transactionFee', 0.0)) # Withdrawal fee

                withdrawals_data.append({
                    "symbol": normalized_symbol,
                    "timestamp": pd.to_datetime(withdrawal['applyTime'], unit='ms').to_pydatetime(),
                    "type": "WITHDRAWAL",
                    "quantity": float(withdrawal['amount']),
                    "price_usd": price_usd_at_withdrawal,
                    "fee_usd": None, # TODO: Need to get fee asset and convert to USD if not already
                    "source": "Binance Withdrawal",
                    "transaction_hash": withdrawal.get('txId', f"binance_withdraw_{withdrawal['id']}"),
                    "notes": f"Network: {withdrawal['network']}, Fee: {fee} {symbol}"
                })
        except BinanceAPIException as e:
            logger.error(f"API Error fetching withdrawal history: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching withdrawal history: {e}")
        return withdrawals_data

    def _get_coingecko_historical_price(self, coin_id: str, date_str: str) -> Optional[float]:
        """
        Fetches the historical price of a coin from CoinGecko for a specific date.
        Date string should be in 'dd-mm-yyyy' format.
        """
        if not coin_id or not date_str:
            return None

        base_url = self.coingecko_api.get("base_url")
        timeout = self.coingecko_api.get("timeout", 30)
        # CoinGecko API requires date in dd-mm-yyyy format for historical data
        url = f"{base_url}/coins/{coin_id}/history?date={date_str}&localization=false"
        logger.debug(f"Fetching historical price for {coin_id} on {date_str} from URL: {url}")

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if "market_data" in data and "current_price" in data["market_data"] and "usd" in data["market_data"]["current_price"]:
                return float(data["market_data"]["current_price"]["usd"])
            else:
                logger.warning(f"Could not find USD price in historical data for {coin_id} on {date_str}. Response: {data}")
                return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404: # Not Found
                 logger.warning(f"No historical data found on CoinGecko for {coin_id} on {date_str}.")
            elif e.response.status_code == 429: # Too Many Requests
                 logger.error(f"Rate limited by CoinGecko fetching historical price for {coin_id} on {date_str}. Try again later.")
            else:
                 logger.error(f"HTTP error fetching historical price for {coin_id} on {date_str}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching historical price for {coin_id} on {date_str}: {e}")
            return None
