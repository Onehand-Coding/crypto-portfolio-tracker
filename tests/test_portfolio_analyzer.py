"""
Tests for portfolio analyzer operations.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from crypto_portfolio_tracker.portfolio_analyzer import PortfolioAnalyzer


class TestPortfolioAnalyzer:
    """Test suite for PortfolioAnalyzer."""

    def test_calculate_total_invested_capital(self):
        """Test total invested capital calculation in PortfolioAnalyzer."""
        # Create a mock database manager
        mock_db_manager = Mock()
        
        # Mock the get_invested_capital_transactions method to return test data
        mock_db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
            {
                "source": "Binance P2P Buy",
                "type": "BUY",
                "quantity": 0.1,
                "price_usd": 50000.0
            },
            {
                "source": "Binance P2P Buy",
                "type": "BUY",
                "quantity": 0.5,
                "price_usd": 2000.0
            },
            {
                "source": None,
                "type": "WITHDRAWAL",
                "quantity": 1.0,
                "price_usd": 3000.0
            }
        ])
        
        # Create a PortfolioAnalyzer instance with the mock
        config = {}
        analyzer = PortfolioAnalyzer(
            config=config,
            db_manager=mock_db_manager
        )
        
        # Calculate total invested capital
        total = analyzer.calculate_total_invested_capital()
        
        # Expected calculation:
        # P2P Buys: (0.1 * 50000) + (0.5 * 2000) = 5000 + 1000 = 6000
        # Withdrawals: (1.0 * 3000) = 3000
        # Net: 6000 - 3000 = 3000
        assert total == 3000.0

    def test_calculate_total_invested_capital_empty_transactions(self):
        """Test total invested capital calculation with no transactions."""
        # Create a mock database manager
        mock_db_manager = Mock()
        
        # Mock the get_invested_capital_transactions method to return empty DataFrame
        mock_db_manager.get_invested_capital_transactions.return_value = pd.DataFrame()
        
        # Create a PortfolioAnalyzer instance with the mock
        config = {}
        analyzer = PortfolioAnalyzer(
            config=config,
            db_manager=mock_db_manager
        )
        
        # Calculate total invested capital
        total = analyzer.calculate_total_invested_capital()
        
        # Should be 0 when no transactions
        assert total == 0.0

    def test_calculate_total_invested_capital_only_p2p_buys(self):
        """Test total invested capital calculation with only P2P buys."""
        # Create a mock database manager
        mock_db_manager = Mock()
        
        # Mock the get_invested_capital_transactions method to return only P2P buys
        mock_db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
            {
                "source": "Binance P2P Buy",
                "type": "BUY",
                "quantity": 0.1,
                "price_usd": 50000.0
            },
            {
                "source": "Binance P2P Buy",
                "type": "BUY",
                "quantity": 0.5,
                "price_usd": 2000.0
            }
        ])
        
        # Create a PortfolioAnalyzer instance with the mock
        config = {}
        analyzer = PortfolioAnalyzer(
            config=config,
            db_manager=mock_db_manager
        )
        
        # Calculate total invested capital
        total = analyzer.calculate_total_invested_capital()
        
        # Expected calculation:
        # P2P Buys: (0.1 * 50000) + (0.5 * 2000) = 5000 + 1000 = 6000
        # No withdrawals
        # Net: 6000
        assert total == 6000.0

    def test_calculate_total_invested_capital_only_withdrawals(self):
        """Test total invested capital calculation with only withdrawals."""
        # Create a mock database manager
        mock_db_manager = Mock()
        
        # Mock the get_invested_capital_transactions method to return only withdrawals
        mock_db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
            {
                "source": None,
                "type": "WITHDRAWAL",
                "quantity": 1.0,
                "price_usd": 3000.0
            },
            {
                "source": None,
                "type": "WITHDRAWAL",
                "quantity": 0.5,
                "price_usd": 2000.0
            }
        ])
        
        # Create a PortfolioAnalyzer instance with the mock
        config = {}
        analyzer = PortfolioAnalyzer(
            config=config,
            db_manager=mock_db_manager
        )
        
        # Calculate total invested capital
        total = analyzer.calculate_total_invested_capital()
        
        # Expected calculation:
        # No P2P Buys
        # Withdrawals: (1.0 * 3000) + (0.5 * 2000) = 3000 + 1000 = 4000
        # Net: 0 - 4000 = -4000
        assert total == -4000.0