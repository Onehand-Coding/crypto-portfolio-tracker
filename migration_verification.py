#!/usr/bin/env python3
"""
Migration Verification Script for Crypto Portfolio Tracker Refactoring

This script verifies that the refactored CryptoPortfolioTracker facade
maintains full compatibility with the original implementation.

Usage:
    python migration_verification.py [--dry-run] [--verbose]
"""

import sys
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List
import traceback

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from crypto_portfolio_tracker.config import ConfigManager
    from crypto_portfolio_tracker.crypto_portfolio_tracker_refactored import CryptoPortfolioTracker
    from crypto_portfolio_tracker.models import TradeResult, ExecutionMode
    from crypto_portfolio_tracker.exceptions import NetworkUnavailableError
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)


class MigrationTester:
    """Test suite for verifying the refactored implementation."""

    def __init__(self, dry_run: bool = True, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log_test(self, test_name: str, success: bool, message: str = "", error: Exception = None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

        if message:
            print(f"    {message}")

        if error and self.verbose:
            print(f"    Error: {str(error)}")
            if self.verbose:
                print(f"    Traceback: {traceback.format_exc()}")

        self.test_results.append({
            'name': test_name,
            'success': success,
            'message': message,
            'error': str(error) if error else None
        })

        if success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    def test_initialization(self) -> bool:
        """Test that the tracker can be initialized successfully."""
        try:
            # Try to create a basic configuration
            config_path = Path("default_config.json.example")
            if not config_path.exists():
                config_path = Path("config/config.json.example")

            if not config_path.exists():
                self.log_test(
                    "Initialization",
                    False,
                    "No config file found. Create a config file to test initialization."
                )
                return False

            config_manager = ConfigManager(config_path=str(config_path))

            # Test offline initialization (safer for testing)
            tracker = CryptoPortfolioTracker(config_manager, force_offline=True)

            # Verify all components were initialized
            components = [
                'data_synchronizer', 'data_manager', 'portfolio_analyzer',
                'trade_executor', 'dca_manager', 'report_generator'
            ]

            missing_components = []
            for component in components:
                if not hasattr(tracker, component):
                    missing_components.append(component)

            if missing_components:
                self.log_test(
                    "Initialization",
                    False,
                    f"Missing components: {missing_components}"
                )
                return False

            self.log_test("Initialization", True, "All components initialized successfully")
            return True

        except Exception as e:
            self.log_test("Initialization", False, error=e)
            return False

    def test_method_presence(self, tracker: CryptoPortfolioTracker) -> bool:
        """Test that all expected methods are present."""
        expected_methods = [
            # Data synchronization methods
            '_get_current_prices',
            '_get_coingecko_historical_price',
            '_get_historical_fiat_exchange_rate',
            '_get_yfinance_ticker',
            'sync_data',
            'run_full_sync',
            'fetch_binance_balances',
            'transfer_funding_to_spot',

            # Portfolio analysis methods
            'calculate_portfolio_metrics',
            'get_profit_taking_opportunities',
            'get_core_portfolio_rebalance_suggestions_technical',

            # Trade execution methods
            '_get_symbol_filters',
            '_adjust_quantity_to_lot_size',
            '_redeem_from_earn_if_needed',
            'execute_profit_taking_trades',
            'execute_rebalancing_trades_core',
            'execute_manual_trade_core',

            # DCA methods
            'calculate_proportional_dca',
            'calculate_target_weight_dca',
            'get_available_usdt_balance',
            'validate_dca_amount',
            'get_dca_suggestions',
            'execute_dca_trades',
            'validate_dca_execution',

            # Reporting methods
            'create_portfolio_charts',
            'create_portfolio_charts_all',
            'export_portfolio_summary',
            'export_portfolio_summary_all_formats',
            'export_data_backup',
            'export_all_data_backups',
            'export_trend_report',
            'export_trend_report_all_formats',

            # Data management methods
            '_record_trade_transaction',
            'update_holdings_from_transactions',
            'save_snapshot',
            'cleanup_old_data',
        ]

        missing_methods = []
        for method_name in expected_methods:
            if not hasattr(tracker, method_name):
                missing_methods.append(method_name)
            elif not callable(getattr(tracker, method_name)):
                missing_methods.append(f"{method_name} (not callable)")

        if missing_methods:
            self.log_test(
                "Method Presence",
                False,
                f"Missing methods: {missing_methods}"
            )
            return False

        self.log_test(
            "Method Presence",
            True,
            f"All {len(expected_methods)} expected methods are present"
        )
        return True

    def test_component_delegation(self, tracker: CryptoPortfolioTracker) -> bool:
        """Test that methods properly delegate to specialized components."""
        try:
            # Test data synchronizer delegation
            if not hasattr(tracker.data_synchronizer, '_get_current_prices'):
                self.log_test("Component Delegation", False, "DataSynchronizer missing _get_current_prices")
                return False

            # Test portfolio analyzer delegation
            if not hasattr(tracker.portfolio_analyzer, 'calculate_portfolio_metrics'):
                self.log_test("Component Delegation", False, "PortfolioAnalyzer missing calculate_portfolio_metrics")
                return False

            # Test trade executor delegation
            if not hasattr(tracker.trade_executor, 'execute_rebalancing_trades_core'):
                self.log_test("Component Delegation", False, "TradeExecutor missing execute_rebalancing_trades_core")
                return False

            # Test DCA manager delegation
            if not hasattr(tracker.dca_manager, 'calculate_proportional_dca'):
                self.log_test("Component Delegation", False, "DCAManager missing calculate_proportional_dca")
                return False

            # Test report generator delegation
            if not hasattr(tracker.report_generator, 'create_portfolio_charts'):
                self.log_test("Component Delegation", False, "ReportGenerator missing create_portfolio_charts")
                return False

            # Test data manager delegation
            if not hasattr(tracker.data_manager, '_record_trade_transaction'):
                self.log_test("Component Delegation", False, "DataManager missing _record_trade_transaction")
                return False

            self.log_test("Component Delegation", True, "All components have expected methods")
            return True

        except Exception as e:
            self.log_test("Component Delegation", False, error=e)
            return False

    async def test_portfolio_metrics_calculation(self, tracker: CryptoPortfolioTracker) -> bool:
        """Test portfolio metrics calculation (offline mode)."""
        try:
            metrics = await tracker.calculate_portfolio_metrics()

            # Verify expected keys are present
            expected_keys = [
                'total_value_usd', 'spot_earn_value_usd', 'futures_value_usd',
                'funding_value_usd', 'total_cost_basis_usd', 'unrealized_pl_usd',
                'unrealized_pl_percent', 'holdings_df', 'core_holdings_df',
                'other_holdings_df', 'timestamp'
            ]

            missing_keys = [key for key in expected_keys if key not in metrics]

            if missing_keys:
                self.log_test(
                    "Portfolio Metrics",
                    False,
                    f"Missing keys: {missing_keys}"
                )
                return False

            self.log_test("Portfolio Metrics", True, "Metrics calculated successfully")
            return True

        except Exception as e:
            self.log_test("Portfolio Metrics", False, error=e)
            return False

    def test_dca_calculations(self, tracker: CryptoPortfolioTracker) -> bool:
        """Test DCA calculation methods."""
        try:
            # Test proportional DCA
            target_allocation = {"BTC": 0.5, "ETH": 0.3, "ADA": 0.2}
            new_funds = 1000.0

            proportional_trades = tracker.calculate_proportional_dca(new_funds, target_allocation)

            if not isinstance(proportional_trades, dict):
                self.log_test("DCA Calculations", False, "Proportional DCA didn't return dict")
                return False

            expected_btc_amount = 1000.0 * 0.5
            if proportional_trades.get("BTC") != expected_btc_amount:
                self.log_test(
                    "DCA Calculations",
                    False,
                    f"Expected BTC amount {expected_btc_amount}, got {proportional_trades.get('BTC')}"
                )
                return False

            # Test validation
            valid, message = tracker.validate_dca_amount(100.0)
            if not isinstance(valid, bool) or not isinstance(message, str):
                self.log_test("DCA Calculations", False, "DCA validation returned wrong types")
                return False

            self.log_test("DCA Calculations", True, "DCA calculations working correctly")
            return True

        except Exception as e:
            self.log_test("DCA Calculations", False, error=e)
            return False

    def test_trade_result_structure(self, tracker: CryptoPortfolioTracker) -> bool:
        """Test that TradeResult objects have the expected structure."""
        try:
            # Create a test TradeResult
            result = TradeResult(success=True)
            result.messages.append("Test message")
            result.errors.append("Test error")
            result.data["test_key"] = "test_value"

            # Verify structure
            if not hasattr(result, 'success'):
                self.log_test("TradeResult Structure", False, "Missing success attribute")
                return False

            if not hasattr(result, 'messages') or not isinstance(result.messages, list):
                self.log_test("TradeResult Structure", False, "Missing or invalid messages attribute")
                return False

            if not hasattr(result, 'errors') or not isinstance(result.errors, list):
                self.log_test("TradeResult Structure", False, "Missing or invalid errors attribute")
                return False

            if not hasattr(result, 'data') or not isinstance(result.data, dict):
                self.log_test("TradeResult Structure", False, "Missing or invalid data attribute")
                return False

            self.log_test("TradeResult Structure", True, "TradeResult structure is correct")
            return True

        except Exception as e:
            self.log_test("TradeResult Structure", False, error=e)
            return False

    def test_execution_mode_enum(self, tracker: CryptoPortfolioTracker) -> bool:
        """Test that ExecutionMode enum has expected values."""
        try:
            expected_modes = ["AUTO", "BULK", "INTERACTIVE", "CONFIRM"]

            for mode_name in expected_modes:
                if not hasattr(ExecutionMode, mode_name):
                    self.log_test(
                        "ExecutionMode Enum",
                        False,
                        f"Missing mode: {mode_name}"
                    )
                    return False

            self.log_test("ExecutionMode Enum", True, "All execution modes present")
            return True

        except Exception as e:
            self.log_test("ExecutionMode Enum", False, error=e)
            return False

    async def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print("🧪 Starting Migration Verification Tests")
        print("=" * 60)

        # Test 1: Initialization
        success = self.test_initialization()
        if not success:
            print("❌ Cannot continue - initialization failed")
            return False

        # Create tracker instance for remaining tests
        try:
            config_path = Path("default_config.json.example")
            if not config_path.exists():
                config_path = Path("config/config.json.example")

            config_manager = ConfigManager(config_path=str(config_path))
            tracker = CryptoPortfolioTracker(config_manager, force_offline=True)
        except Exception as e:
            print(f"❌ Failed to create tracker for testing: {e}")
            return False

        # Test 2: Method presence
        self.test_method_presence(tracker)

        # Test 3: Component delegation
        self.test_component_delegation(tracker)

        # Test 4: Portfolio metrics (async)
        await self.test_portfolio_metrics_calculation(tracker)

        # Test 5: DCA calculations
        self.test_dca_calculations(tracker)

        # Test 6: TradeResult structure
        self.test_trade_result_structure(tracker)

        # Test 7: ExecutionMode enum
        self.test_execution_mode_enum(tracker)

        # Summary
        print("=" * 60)
        print("📊 Test Summary:")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📈 Success Rate: {(self.passed_tests / (self.passed_tests + self.failed_tests)) * 100:.1f}%")

        if self.failed_tests == 0:
            print("\n🎉 All tests passed! The refactored implementation is ready for migration.")
            return True
        else:
            print(f"\n⚠️  {self.failed_tests} test(s) failed. Please fix these issues before migration.")
            return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify CryptoPortfolioTracker refactoring")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    tester = MigrationTester(dry_run=args.dry_run, verbose=args.verbose)
    success = await tester.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
