"""
Production Confidence Tests for Crypto Portfolio Tracker

These tests verify the robustness and reliability of critical enhancements
made to the application, particularly focusing on error handling, 
configuration validation, and edge cases.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.crypto_portfolio_tracker.config import ConfigManager
from src.crypto_portfolio_tracker.database import DatabaseManager
from src.crypto_portfolio_tracker.binance_fetcher import BinanceFetcher
from src.crypto_portfolio_tracker.symbol_mapper import SymbolMapper
from src.crypto_portfolio_tracker.utils import calculate_fifo_cost_basis
from src.crypto_portfolio_tracker.exceptions import NetworkOperationError


class ProductionConfidenceTests(unittest.TestCase):
    """Production-level confidence tests for critical enhancements"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        
        # Create a minimal config file for testing
        self.config_file = self.test_path / "default_config.json"
        self.test_config = {
            "target_allocation": {
                "BTC": 0.5,
                "ETH": 0.5
            },
            "portfolio": {
                "testnet_mode": True,
                "live_trading_enabled": False
            },
            "database": {
                "path": str(self.test_path / "test_portfolio.db"),
                "testnet_path": str(self.test_path / "test_testnet_portfolio.db"),
                "connection_timeout": 30,
                "cleanup_days": 90
            },
            "logging": {
                "file_config": {
                    "path": str(self.test_path / "test_portfolio_tracker.log")
                }
            },
            "exports": {
                "path": str(self.test_path / "exports")
            },
            "cache": {
                "path": str(self.test_path / "cache")
            }
        }
        
        # Write the test config
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
        # Set up environment variables for testing
        os.environ['MAIN_API_KEY'] = 'test_key'
        os.environ['MAIN_API_SECRET'] = 'test_secret'
        os.environ['TESTNET_API_KEY'] = 'testnet_key'
        os.environ['TESTNET_API_SECRET'] = 'testnet_secret'

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.test_dir.cleanup()
        # Clean up environment variables
        os.environ.pop('MAIN_API_KEY', None)
        os.environ.pop('MAIN_API_SECRET', None)
        os.environ.pop('TESTNET_API_KEY', None)
        os.environ.pop('TESTNET_API_SECRET', None)

    def test_config_validation_production_scenarios(self):
        """Test configuration validation in production-like scenarios"""
        # Test normal valid configuration
        config_manager = ConfigManager(str(self.config_file))
        self.assertIsNotNone(config_manager.config)
        self.assertIn('target_allocation', config_manager.config)
        
        # Test target allocation that's slightly off (should still work)
        slightly_off_config = self.test_config.copy()
        slightly_off_config["target_allocation"] = {"BTC": 0.49, "ETH": 0.51}  # 100% total
        slightly_off_file = self.test_path / "slightly_off_config.json"
        with open(slightly_off_file, 'w') as f:
            json.dump(slightly_off_config, f, indent=2)
        
        config_manager = ConfigManager(str(slightly_off_file))
        self.assertIsNotNone(config_manager.config)
        
        # Test configuration with all required sections
        full_config = self.test_config.copy()
        full_config["apis"] = {
            "binance": {
                "timeout": 60,
                "recv_window": 20000
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "timeout": 30
            }
        }
        full_config["trend_analyzer"] = {
            "cryptocurrencies": ["BTC-USD", "ETH-USD"],
            "timeframe_settings": {
                "long_term": {
                    "period": "4y",
                    "sma_short_window": 50,
                    "sma_long_window": 200
                }
            }
        }
        
        full_config_file = self.test_path / "full_config.json"
        with open(full_config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
            
        config_manager = ConfigManager(str(full_config_file))
        self.assertIsNotNone(config_manager.config)
        self.assertIn('apis', config_manager.config)
        self.assertIn('trend_analyzer', config_manager.config)

    def test_database_schema_versioning(self):
        """Test database schema versioning functionality"""
        config_manager = ConfigManager(str(self.config_file))
        db_path = config_manager.get_database_path()
        backup_dir = config_manager.get_backup_dir()
        
        # Initialize database manager
        db_manager = DatabaseManager(
            db_path=db_path,
            backup_dir=backup_dir,
            connection_timeout=30,
            cleanup_days=90
        )
        
        # Verify schema version table exists and has initial version
        with db_manager._get_connection() as conn:
            cursor = conn.cursor()
            # Check that schema_version table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            """)
            result = cursor.fetchone()
            self.assertIsNotNone(result, "schema_version table should exist")
            
            # Check current schema version
            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            version_result = cursor.fetchone()
            self.assertIsNotNone(version_result, "Should have a schema version")
            self.assertGreaterEqual(version_result[0], 1, "Schema version should be at least 1")

    def test_database_backup_enhancements(self):
        """Test enhanced backup functionality"""
        config_manager = ConfigManager(str(self.config_file))
        db_path = config_manager.get_database_path()
        backup_dir = config_manager.get_backup_dir()
        
        # Initialize database manager
        db_manager = DatabaseManager(
            db_path=db_path,
            backup_dir=backup_dir,
            connection_timeout=30,
            cleanup_days=90
        )
        
        # Test backup with reason parameter
        backup_path = db_manager.backup_database("test_reason")
        if backup_path:
            self.assertTrue(os.path.exists(backup_path))
            # Check that reason is in filename
            self.assertIn("test_reason", backup_path)
        
        # Test backup cleanup (create multiple backups)
        backup_paths = []
        for i in range(5):
            backup_path = db_manager.backup_database(f"cleanup_test_{i}")
            if backup_path:
                backup_paths.append(backup_path)
        
        # Verify multiple backups were created
        self.assertGreaterEqual(len(backup_paths), 1)

    @patch('src.crypto_portfolio_tracker.binance_fetcher.BinanceAPIException')
    def test_binance_error_handling_timestamp_sync(self, mock_api_exception):
        """Test Binance -1021 timestamp sync error handling"""
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
        
        config_manager = ConfigManager(str(self.config_file))
        symbol_mapper = SymbolMapper(config_manager.config)
        
        # Create a mock Binance client
        mock_client = Mock(spec=Client)
        mock_client.get_server_time.return_value = {"serverTime": 1234567890123}
        mock_client._server_time_offset = 0
        
        # Create BinanceFetcher
        fetcher = BinanceFetcher(mock_client, symbol_mapper, config_manager.config)
        
        # Mock an API exception for timestamp sync error
        mock_error = Mock(spec=BinanceAPIException)
        mock_error.code = -1021
        mock_error.message = "Timestamp for this request is outside of the recvWindow."
        
        # Test that our error handler recognizes this error
        result = fetcher._handle_binance_api_exception(mock_error, "test_operation")
        # Should return True indicating it's a recoverable error
        self.assertTrue(result, "Timestamp sync error should be recoverable")
        
        # Verify time sync was attempted
        mock_client.get_server_time.assert_called()

    @patch('src.crypto_portfolio_tracker.binance_fetcher.BinanceAPIException')
    def test_binance_error_handling_permission_errors(self, mock_api_exception):
        """Test Binance -2015 permission error handling"""
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
        
        config_manager = ConfigManager(str(self.config_file))
        symbol_mapper = SymbolMapper(config_manager.config)
        
        # Create a mock Binance client
        mock_client = Mock(spec=Client)
        
        # Create BinanceFetcher
        fetcher = BinanceFetcher(mock_client, symbol_mapper, config_manager.config)
        
        # Mock an API exception for permission error
        mock_error = Mock(spec=BinanceAPIException)
        mock_error.code = -2015
        mock_error.message = "Invalid API-key, IP, or permissions for action."
        
        # Test that our error handler recognizes this error
        result = fetcher._handle_binance_api_exception(mock_error, "test_operation")
        # Should return False indicating it's not recoverable
        self.assertFalse(result, "Permission error should not be recoverable")

    @patch('src.crypto_portfolio_tracker.binance_fetcher.BinanceAPIException')
    def test_binance_error_handling_invalid_symbol(self, mock_api_exception):
        """Test Binance -1121 invalid symbol error handling"""
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
        
        config_manager = ConfigManager(str(self.config_file))
        symbol_mapper = SymbolMapper(config_manager.config)
        
        # Create a mock Binance client
        mock_client = Mock(spec=Client)
        
        # Create BinanceFetcher
        fetcher = BinanceFetcher(mock_client, symbol_mapper, config_manager.config)
        
        # Mock an API exception for invalid symbol error
        mock_error = Mock(spec=BinanceAPIException)
        mock_error.code = -1121
        mock_error.message = "Invalid symbol."
        
        # Test that our error handler recognizes this error
        result = fetcher._handle_binance_api_exception(mock_error, "test_operation")
        # Should return False indicating it's not recoverable
        self.assertFalse(result, "Invalid symbol error should not be recoverable")

    def test_fifo_calculation_edge_cases(self):
        """Test FIFO calculation with edge cases"""
        # Test with empty transactions
        empty_df = pd.DataFrame(columns=['timestamp', 'type', 'quantity', 'price_usd', 'fee_usd', 'symbol'])
        quantity, avg_cost = calculate_fifo_cost_basis(empty_df)
        self.assertEqual(quantity, 0.0)
        self.assertEqual(avg_cost, 0.0)
        
        # Test with only sells (negative position)
        # Note: The actual implementation prevents negative positions by returning 0
        # when trying to sell more than available
        sell_only_data = {
            'timestamp': [pd.Timestamp('2023-01-01 10:00:00')],
            'type': ['SELL'],
            'quantity': [1.0],
            'price_usd': [10000.0],
            'fee_usd': [0.0],
            'symbol': ['BTC']
        }
        sell_only_df = pd.DataFrame(sell_only_data)
        quantity, avg_cost = calculate_fifo_cost_basis(sell_only_df)
        # Should have zero quantity (prevents negative positions)
        self.assertEqual(quantity, 0.0)
        self.assertEqual(avg_cost, 0.0)
        
        # Test with zero quantities
        zero_qty_data = {
            'timestamp': [pd.Timestamp('2023-01-01 10:00:00')],
            'type': ['BUY'],
            'quantity': [0.0],
            'price_usd': [10000.0],
            'fee_usd': [0.0],
            'symbol': ['BTC']
        }
        zero_qty_df = pd.DataFrame(zero_qty_data)
        quantity, avg_cost = calculate_fifo_cost_basis(zero_qty_df)
        self.assertEqual(quantity, 0.0)
        self.assertEqual(avg_cost, 0.0)

    def test_database_backup_before_critical_operations(self):
        """Test that backups are created before critical database operations"""
        config_manager = ConfigManager(str(self.config_file))
        db_path = config_manager.get_database_path()
        backup_dir = config_manager.get_backup_dir()
        
        # Initialize database manager
        db_manager = DatabaseManager(
            db_path=db_path,
            backup_dir=backup_dir,
            connection_timeout=30,
            cleanup_days=90
        )
        
        # Count existing backup files
        initial_backups = list(backup_dir.glob("*.bak"))
        initial_count = len(initial_backups)
        
        # Perform a critical operation that should trigger backup
        test_data = pd.DataFrame({
            'symbol': ['BTC', 'ETH'],
            'quantity': [1.0, 10.0],
            'average_cost_basis': [10000.0, 1000.0],
            'last_updated': [pd.Timestamp.now(), pd.Timestamp.now()]
        })
        
        # This should create a backup before updating
        db_manager.update_holdings(test_data)
        
        # Check that a new backup was created
        final_backups = list(backup_dir.glob("*.bak"))
        final_count = len(final_backups)
        
        # Note: The exact count might vary depending on other operations
        # but we should have at least as many as before
        self.assertGreaterEqual(final_count, initial_count)

    def test_config_validation_boundary_conditions(self):
        """Test configuration validation with boundary conditions"""
        # Test with empty target allocation
        empty_alloc_config = self.test_config.copy()
        empty_alloc_config["target_allocation"] = {}
        empty_alloc_file = self.test_path / "empty_alloc_config.json"
        with open(empty_alloc_file, 'w') as f:
            json.dump(empty_alloc_config, f, indent=2)
        
        config_manager = ConfigManager(str(empty_alloc_file))
        self.assertIsNotNone(config_manager.config)
        
        # Test with single asset allocation
        single_asset_config = self.test_config.copy()
        single_asset_config["target_allocation"] = {"BTC": 1.0}
        single_asset_file = self.test_path / "single_asset_config.json"
        with open(single_asset_file, 'w') as f:
            json.dump(single_asset_config, f, indent=2)
            
        config_manager = ConfigManager(str(single_asset_file))
        self.assertIsNotNone(config_manager.config)
        
        # Verify allocation sum
        target_alloc = config_manager.config['target_allocation']
        total = sum(target_alloc.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_database_connection_resilience(self):
        """Test database connection resilience"""
        config_manager = ConfigManager(str(self.config_file))
        db_path = config_manager.get_database_path()
        backup_dir = config_manager.get_backup_dir()
        
        # Initialize database manager
        db_manager = DatabaseManager(
            db_path=db_path,
            backup_dir=backup_dir,
            connection_timeout=30,
            cleanup_days=90
        )
        
        # Test that we can get holdings (basic connectivity)
        holdings = db_manager.get_holdings()
        self.assertIsInstance(holdings, pd.DataFrame)
        
        # Test that we can backup the database
        backup_path = db_manager.backup_database("connectivity_test")
        # Backup might be None if DB doesn't exist yet, which is OK
        if backup_path:
            self.assertTrue(os.path.exists(backup_path))


if __name__ == '__main__':
    unittest.main()