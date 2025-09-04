"""
Refactored Crypto Portfolio Tracker - Facade Pattern Implementation
This class maintains the original public API while delegating work to specialized classes.
"""

import os
import logging
import datetime
from pathlib import Path
from diskcache import Cache
from typing import Dict, Any, Optional, List, Callable, Tuple, Union

from binance.client import Client

from .config import ConfigManager
from .database import DatabaseManager
from .visualizations import Visualizer
from .price_enricher import PriceEnricher
from .binance_fetcher import BinanceFetcher
from .exporters import ExcelExporter, HtmlExporter, CsvExporter
from .exceptions import NetworkOperationError, NetworkUnavailableError

# Import the new specialized classes
from .data_synchronizer import DataSynchronizer
from .portfolio_analyzer import PortfolioAnalyzer
from .trade_executor import TradeExecutor
from .dca_manager import DCAManager
from .report_generator import ReportGenerator
from .data_manager import DataManager
from .models import TradeResult, ExecutionMode


class CryptoPortfolioTracker:
    """
    Refactored main class for the crypto portfolio tracker using the facade pattern.

    This class orchestrates all portfolio tracking functionality by delegating to
    specialized components while maintaining the original public API for compatibility.
    """

    def __init__(self, config_manager: ConfigManager, force_offline: bool = False):
        """
        Initialize the tracker and its specialist components.

        Args:
            config_manager: Configuration manager instance
            force_offline: Whether to force offline mode regardless of network status
        """
        self.config_manager = config_manager
        self.config = self.config_manager.config
        self.logger = logging.getLogger(__name__)

        self.offline_mode = force_offline

        # Initialize database with explicit path
        db_path = self.config_manager.get_database_path()
        backup_dir = self.config_manager.get_backup_dir()
        connection_timeout = self.config.get("database", {}).get("connection_timeout", 30)
        cleanup_days = self.config.get("database", {}).get("cleanup_days", 90)

        self.db_manager = DatabaseManager(
            db_path=db_path,
            backup_dir=backup_dir,
            connection_timeout=connection_timeout,
            cleanup_days=cleanup_days,
        )

        self.symbol_mappings = self.config_manager.symbol_mapper

        # Initialize cache system
        self.cache_dir = Path(self.config.get("cache", {}).get("path", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.coingecko_price_cache = Cache(str(self.cache_dir / "coingecko_historical"))
        self.yfinance_disk_cache = Cache(str(self.cache_dir / "yfinance_historical"))
        self.fiat_exchange_rate_cache = Cache(str(self.cache_dir / "fiat_exchange_rates"))

        # Initialize Binance client
        if not self.offline_mode:
            try:
                self.binance_client = self._init_binance_client()
            except NetworkUnavailableError:
                # Let the main entrypoint handle prompting and re-instantiation
                raise
        else:
            self.binance_client = None

        # Initialize original specialized classes (keeping existing ones)
        if not self.offline_mode:
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

        # Initialize strategy state
        self.strategy_state_path = self.cache_dir.parent / "strategy_state.json"

        # --- Initialize NEW specialized classes ---
        self.data_synchronizer = DataSynchronizer(
            config_manager=self.config_manager,
            db_manager=self.db_manager,
            binance_client=self.binance_client,
            fetcher=self.fetcher,
            cache_dir=self.cache_dir,
            coingecko_price_cache=self.coingecko_price_cache,
            yfinance_disk_cache=self.yfinance_disk_cache,
            fiat_exchange_rate_cache=self.fiat_exchange_rate_cache,
            symbol_mappings=self.symbol_mappings,
            offline_mode=self.offline_mode
        )

        self.data_manager = DataManager(
            config=self.config,
            db_manager=self.db_manager,
            strategy_state_path=self.strategy_state_path
        )

        self.portfolio_analyzer = PortfolioAnalyzer(
            config=self.config,
            db_manager=self.db_manager,
            binance_client=self.binance_client,
            fetcher=self.fetcher,
            enricher=self.enricher,
            offline_mode=self.offline_mode
        )

        self.trade_executor = TradeExecutor(
            config=self.config,
            binance_client=self.binance_client,
            config_manager=self.config_manager,
            yfinance_disk_cache=self.yfinance_disk_cache,
            data_synchronizer=self.data_synchronizer,
            data_manager=self.data_manager
        )

        self.dca_manager = DCAManager(
            config=self.config,
            binance_client=self.binance_client,
            config_manager=self.config_manager,
            fetcher=self.fetcher,
            trade_executor=self.trade_executor,
            portfolio_analyzer=self.portfolio_analyzer,
            data_manager=self.data_manager
        )

        self.report_generator = ReportGenerator(
            config=self.config,
            db_manager=self.db_manager,
            visualizer=self.visualizer,
            excel_exporter=self.excel_exporter,
            html_exporter=self.html_exporter,
            csv_exporter=self.csv_exporter
        )

        self.logger.info("Refactored tracker and all components initialized.")

    def _init_binance_client(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> Optional[Client]:
        """Initialize and return Binance client - delegates to DataSynchronizer."""
        return self.data_synchronizer._init_binance_client(api_key, api_secret)

    def _sync_binance_client_time(self, client: Client, context: str = "general"):
        """Synchronize Binance client time - delegates to DataSynchronizer."""
        return self.data_synchronizer._sync_binance_client_time(client, context)

    def _get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices - delegates to DataSynchronizer."""
        return self.data_synchronizer._get_current_prices(symbols)

    def _get_coingecko_historical_price(self, symbol: str, target_date: datetime.date,
                                       coin_id: Optional[str] = None, fallback_to_nearest: bool = True) -> Optional[float]:
        """Get historical price - delegates to DataSynchronizer."""
        return self.data_synchronizer._get_coingecko_historical_price(symbol, target_date, coin_id, fallback_to_nearest)

    def _get_historical_fiat_exchange_rate(self, target_date: datetime.date,
                                         from_currency: str = "USD", to_currency: str = "CAD") -> Optional[float]:
        """Get historical exchange rate - delegates to DataSynchronizer."""
        return self.data_synchronizer._get_historical_fiat_exchange_rate(target_date, from_currency, to_currency)

    def _get_yfinance_ticker(self, symbol: str):
        """Get yFinance ticker - delegates to DataSynchronizer."""
        return self.data_synchronizer._get_yfinance_ticker(symbol)

    def _load_strategy_state(self) -> Dict[str, Any]:
        """Load strategy state - delegates to DataManager."""
        return self.data_manager._load_strategy_state()

    def _save_strategy_state(self):
        """Save strategy state - delegates to DataManager."""
        return self.data_manager._save_strategy_state()

    def _get_symbol_filters(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol filters - delegates to TradeExecutor."""
        return self.trade_executor._get_symbol_filters(symbol)

    def _adjust_quantity_to_lot_size(self, symbol: str, quantity: float) -> Optional[float]:
        """Adjust quantity to lot size - delegates to TradeExecutor."""
        return self.trade_executor._adjust_quantity_to_lot_size(symbol, quantity)

    def _redeem_from_earn_if_needed(self, asset: str, required_amount: float, is_live: bool = False) -> Tuple[bool, List[str]]:
        """Check/redeem from earn - delegates to TradeExecutor."""
        return self.trade_executor._redeem_from_earn_if_needed(asset, required_amount, is_live)

    def _execute_directional_trade(self, trade: Dict[str, Any], client: Client) -> TradeResult:
        """Execute directional trade - delegates to TradeExecutor."""
        return self.trade_executor._execute_directional_trade(trade, client)

    async def get_profit_taking_opportunities(self, core_holdings_df: Optional = None,
                                            rebalance_suggestions_df: Optional = None,
                                            cached_trend_data: Optional[Dict[str, Any]] = None):
        """Get profit taking opportunities - delegates to PortfolioAnalyzer."""
        return await self.portfolio_analyzer.get_profit_taking_opportunities(
            core_holdings_df, rebalance_suggestions_df, cached_trend_data
        )

    async def execute_profit_taking_trades(self, profit_trades: List[Dict[str, Any]], is_live: bool = False) -> TradeResult:
        """Execute profit taking trades - delegates to TradeExecutor."""
        return await self.trade_executor.execute_profit_taking_trades(profit_trades, is_live)

    def _make_tx_hash(self, source: str, batch_id: str, symbol: str, side: str,
                     ts: datetime.datetime, order: Optional[dict] = None) -> str:
        """Make transaction hash - delegates to DataManager."""
        return self.data_manager._make_tx_hash(source, batch_id, symbol, side, ts, order)

    def _record_trade_transaction(self, *, symbol: str, side: str, quantity: float, price_usd: float,
                                 source: str, mode: str, batch_id: str, order: Optional[dict] = None,
                                 error: Optional[str] = None) -> None:
        """Record trade transaction - delegates to DataManager."""
        return self.data_manager._record_trade_transaction(
            symbol=symbol, side=side, quantity=quantity, price_usd=price_usd,
            source=source, mode=mode, batch_id=batch_id, order=order, error=error
        )

    async def execute_rebalancing_trades_core(self, suggestions_df, earn_balances,
                                            confirmation_callback=None, execution_mode=ExecutionMode.CONFIRM):
        """Execute rebalancing trades - delegates to TradeExecutor."""
        return await self.trade_executor.execute_rebalancing_trades_core(
            suggestions_df, earn_balances, confirmation_callback, execution_mode
        )

    async def execute_manual_trade_core(self, trade_type, symbol, trade_ticker, amount, is_quote_qty, is_live):
        """Execute manual trade - delegates to TradeExecutor."""
        return await self.trade_executor.execute_manual_trade_core(
            trade_type, symbol, trade_ticker, amount, is_quote_qty, is_live
        )

    async def get_core_portfolio_rebalance_suggestions_technical(self):
        """Get rebalancing suggestions - delegates to PortfolioAnalyzer."""
        return await self.portfolio_analyzer.get_core_portfolio_rebalance_suggestions_technical()

    async def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio metrics - delegates to PortfolioAnalyzer."""
        return await self.portfolio_analyzer.calculate_portfolio_metrics()

    async def sync_data(self):
        """Sync data - delegates to DataSynchronizer."""
        return await self.data_synchronizer.sync_data(enricher=self.enricher)

    async def run_full_sync(self) -> Dict[str, Any]:
        """Run full sync - combines DataSynchronizer and PortfolioAnalyzer."""
        await self.data_synchronizer.sync_data(enricher=self.enricher)
        return await self.portfolio_analyzer.calculate_portfolio_metrics()

    def fetch_binance_balances(self):
        """Fetch Binance balances - delegates to original fetcher for compatibility."""
        if self.fetcher:
            return self.fetcher.fetch_binance_balances()
        else:
            return self.data_synchronizer.fetch_binance_balances()

    def update_holdings_from_transactions(self):
        """Update holdings from transactions - delegates to DataManager."""
        return self.data_manager.update_holdings_from_transactions()

    def create_portfolio_charts(self, chart_type: str, metrics: Dict[str, Any]) -> bool:
        """Create portfolio charts - delegates to ReportGenerator."""
        return self.report_generator.create_portfolio_charts(chart_type, metrics)

    def create_portfolio_charts_all(self, metrics: Dict[str, Any]):
        """Create all portfolio charts - delegates to ReportGenerator."""
        return self.report_generator.create_portfolio_charts_all(metrics)

    def export_portfolio_summary(self, metrics: Dict[str, Any], format: str):
        """Export portfolio summary - delegates to ReportGenerator."""
        return self.report_generator.export_portfolio_summary(metrics, format)

    def export_portfolio_summary_all_formats(self, metrics: Dict[str, Any]):
        """Export portfolio summary in all formats - delegates to ReportGenerator."""
        return self.report_generator.export_portfolio_summary_all_formats(metrics)

    def export_data_backup(self, data_type: str, format: str):
        """Export data backup - delegates to ReportGenerator."""
        return self.report_generator.export_data_backup(data_type, format)

    def export_all_data_backups(self):
        """Export all data backups - delegates to ReportGenerator."""
        return self.report_generator.export_all_data_backups()

    def export_trend_report(self, report: Dict[str, Any], timeframe: str, export_format: str = "HTML") -> Optional[Path]:
        """Export trend report - delegates to ReportGenerator."""
        return self.report_generator.export_trend_report(report, timeframe, export_format)

    def export_trend_report_all_formats(self, report: Dict[str, Any], timeframe: str) -> Dict[str, Optional[Path]]:
        """Export trend report in all formats - delegates to ReportGenerator."""
        return self.report_generator.export_trend_report_all_formats(report, timeframe)

    def cleanup_old_data(self):
        """Cleanup old data - delegates to DataManager."""
        return self.data_manager.cleanup_old_data()

    def calculate_proportional_dca(self, new_funds: float, target_allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate proportional DCA - delegates to DCAManager."""
        return self.dca_manager.calculate_proportional_dca(new_funds, target_allocation)

    def calculate_target_weight_dca(self, new_funds: float, current_portfolio, target_allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate target weight DCA - delegates to DCAManager."""
        return self.dca_manager.calculate_target_weight_dca(new_funds, current_portfolio, target_allocation)

    def get_available_usdt_balance(self) -> Dict[str, float]:
        """Get available USDT balance - delegates to DCAManager."""
        return self.dca_manager.get_available_usdt_balance()

    def validate_dca_amount(self, amount: float) -> Tuple[bool, str]:
        """Validate DCA amount - delegates to DCAManager."""
        return self.dca_manager.validate_dca_amount(amount)

    async def get_dca_suggestions(self, new_funds: float, method: str = "both") -> Dict[str, Any]:
        """Get DCA suggestions - delegates to DCAManager."""
        return await self.dca_manager.get_dca_suggestions(new_funds, method)

    async def execute_dca_trades(self, selected_trades: List[Dict[str, Any]], method: str, is_live: bool = False) -> TradeResult:
        """Execute DCA trades - delegates to DCAManager."""
        return await self.dca_manager.execute_dca_trades(selected_trades, method, is_live)

    def validate_dca_execution(self, selected_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate DCA execution - delegates to DCAManager."""
        return self.dca_manager.validate_dca_execution(selected_trades)

    def save_snapshot(self, metrics: Dict[str, Any]):
        """Save portfolio snapshot - delegates to DataManager."""
        return self.data_manager.save_snapshot(metrics)

    async def transfer_funding_to_spot(self, amount: float, asset: str = "USDT") -> bool:
        """Transfer funding to spot - delegates to DataSynchronizer."""
        return await self.data_synchronizer.transfer_funding_to_spot(amount, asset)

    # Strategy state management methods - delegates to DataManager
    def get_strategy_state(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy state."""
        return self.data_manager.get_strategy_state(strategy_name)

    def set_strategy_state(self, strategy_name: str, state: Dict[str, Any]):
        """Set strategy state."""
        return self.data_manager.set_strategy_state(strategy_name, state)

    def update_strategy_state(self, strategy_name: str, updates: Dict[str, Any]):
        """Update strategy state."""
        return self.data_manager.update_strategy_state(strategy_name, updates)

    def clear_strategy_state(self, strategy_name: str):
        """Clear strategy state."""
        return self.data_manager.clear_strategy_state(strategy_name)

    @property
    def strategy_states(self):
        """Get all strategy states."""
        return self.data_manager.get_all_strategy_states()

    @strategy_states.setter
    def strategy_states(self, value):
        """Set all strategy states."""
        self.data_manager.restore_strategy_states(value)
