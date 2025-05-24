"""
Crypto Portfolio Tracker - Main Class
Handles API interactions, data processing, analysis, and orchestration.
"""
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import requests
import time
import datetime

# Make sure to import your modules correctly
from config import ConfigManager
from database import DatabaseManager
from exporters import ExcelExporter, HtmlExporter, CsvExporter
from visualizations import Visualizer

logger = logging.getLogger(__name__)

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
        """Fetch transaction history from Binance (placeholder)."""
        logger.warning("Binance transaction fetching is a placeholder."); return []

    def get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices using CoinGecko."""
        prices = {}
        for symbol in symbols:
            coin_id = self.symbol_mappings.get(symbol.upper())
            if not coin_id: logger.warning(f"No CoinGecko ID for {symbol}."); prices[symbol] = None; continue
            prices[symbol] = self._get_coingecko_price(coin_id); time.sleep(1) # Rate limit
        return prices

    def sync_data(self):
        """Synchronize data from APIs to the database."""
        logger.info("Starting data synchronization..."); binance_balances = self.fetch_binance_balances(); binance_txs = self.fetch_binance_transactions()
        if binance_txs: self.db_manager.bulk_insert_transactions(binance_txs)
        if not binance_balances.empty: logger.warning("Holdings update based only on balance - Cost basis inaccurate.")
        logger.info("Data synchronization finished.")

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
        print("\n🔧 Testing API Connections...")
        if self.binance_client:
            try: self.binance_client.ping(); print("✅ Binance Connection: SUCCESS")
            except Exception as e: print(f"❌ Binance Connection: FAILED ({e})")
        else: print("⚠️ Binance Connection: SKIPPED (No API keys)")
        test_id = list(self.symbol_mappings.values())[0] if self.symbol_mappings else 'bitcoin'
        price = self._get_coingecko_price(test_id)
        if price: print(f"✅ CoinGecko Connection: SUCCESS ({test_id.capitalize()} price: ${price})")
        else: print("❌ CoinGecko Connection: FAILED")
        print("-" * 30)
