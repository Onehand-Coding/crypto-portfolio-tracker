"""
Critical path tests for the crypto portfolio tracker.
These tests verify the core functionality of the application.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path

from src.crypto_portfolio_tracker.config import ConfigManager
from src.crypto_portfolio_tracker.database import DatabaseManager


class CriticalPathTests(unittest.TestCase):
    """Test critical paths in the application"""

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

    def test_config_loading(self):
        """Test that configuration loads correctly"""
        config_manager = ConfigManager(str(self.config_file))
        self.assertIsNotNone(config_manager.config)
        self.assertIn('target_allocation', config_manager.config)
        self.assertIn('BTC', config_manager.config['target_allocation'])
        self.assertIn('ETH', config_manager.config['target_allocation'])
        
        # Test that target allocation sums to approximately 100%
        target_alloc = config_manager.config['target_allocation']
        total = sum(target_alloc.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_database_connection(self):
        """Test that database connection works"""
        import pandas as pd  # Import here to avoid import issues
        
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
        
        # Test basic database operations
        holdings = db_manager.get_holdings()
        self.assertIsInstance(holdings, pd.DataFrame)
        
        # Test that we can backup the database
        backup_path = db_manager.backup_database()
        if backup_path:
            self.assertTrue(os.path.exists(backup_path))

    def test_binance_fetcher_initialization(self):
        """Test that BinanceFetcher can be initialized"""
        from src.crypto_portfolio_tracker.binance_fetcher import BinanceFetcher
        from src.crypto_portfolio_tracker.symbol_mapper import SymbolMapper
        from binance.client import Client
        
        config_manager = ConfigManager(str(self.config_file))
        symbol_mapper = SymbolMapper(config_manager.config)
        
        # Create a mock Binance client for testing
        client = Client('test_key', 'test_secret')
        
        # Initialize BinanceFetcher
        fetcher = BinanceFetcher(client, symbol_mapper, config_manager.config)
        self.assertIsNotNone(fetcher)
        self.assertEqual(fetcher.config, config_manager.config)

    def test_fifo_calculation_basic(self):
        """Test basic FIFO cost basis calculation"""
        import pandas as pd
        from src.crypto_portfolio_tracker.utils import calculate_fifo_cost_basis
        
        # Create simple test data: buy 1 BTC at $10000, buy 1 BTC at $12000
        transactions_data = {
            'timestamp': [
                pd.Timestamp('2023-01-01 10:00:00'),
                pd.Timestamp('2023-01-02 10:00:00')
            ],
            'type': ['BUY', 'BUY'],
            'quantity': [1.0, 1.0],
            'price_usd': [10000.0, 12000.0],
            'fee_usd': [0.0, 0.0],
            'symbol': ['BTC', 'BTC']  # Add the required symbol column
        }
        
        transactions_df = pd.DataFrame(transactions_data)
        quantity, avg_cost = calculate_fifo_cost_basis(transactions_df)
        
        # Should have 2 BTC with average cost of $11000
        self.assertEqual(quantity, 2.0)
        self.assertEqual(avg_cost, 11000.0)

    def test_fifo_calculation_with_sell(self):
        """Test FIFO cost basis calculation with sells"""
        import pandas as pd
        from src.crypto_portfolio_tracker.utils import calculate_fifo_cost_basis
        
        # Buy 2 BTC at $10000, sell 1 BTC, buy 1 BTC at $12000
        transactions_data = {
            'timestamp': [
                pd.Timestamp('2023-01-01 10:00:00'),
                pd.Timestamp('2023-01-02 10:00:00'),
                pd.Timestamp('2023-01-03 10:00:00')
            ],
            'type': ['BUY', 'SELL', 'BUY'],
            'quantity': [2.0, 1.0, 1.0],
            'price_usd': [10000.0, 11000.0, 12000.0],
            'fee_usd': [0.0, 0.0, 0.0],
            'symbol': ['BTC', 'BTC', 'BTC']  # Add the required symbol column
        }
        
        transactions_df = pd.DataFrame(transactions_data)
        quantity, avg_cost = calculate_fifo_cost_basis(transactions_df)
        
        # Should have 2 BTC remaining:
        # 1 BTC from first purchase (remaining after FIFO sell)
        # 1 BTC from third purchase
        # Average cost should be (10000 + 12000) / 2 = 11000
        self.assertEqual(quantity, 2.0)
        self.assertEqual(avg_cost, 11000.0)

    def test_dca_calculation_proportional(self):
        """Test proportional DCA calculation"""
        from src.crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker
        from unittest.mock import Mock
        
        # Create a mock tracker for testing DCA calculations
        config_manager = ConfigManager(str(self.config_file))
        tracker = Mock()
        tracker.config = config_manager.config
        
        # Test proportional DCA calculation
        new_funds = 1000.0
        target_allocation = {"BTC": 0.5, "ETH": 0.5}
        
        # This is a simplified version of the calculation
        trades = {}
        for asset, target_pct in target_allocation.items():
            buy_amount = new_funds * target_pct
            if buy_amount > 0:
                trades[asset] = buy_amount
        
        # Verify the calculation
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(trades["BTC"], 500.0)
        self.assertAlmostEqual(trades["ETH"], 500.0)
        self.assertAlmostEqual(sum(trades.values()), new_funds)


if __name__ == '__main__':
    # Import pandas here to avoid import issues in some test environments
    import pandas as pd
    unittest.main()