import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from src.crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker
from src.crypto_portfolio_tracker.data_synchronizer import DataSynchronizer
from src.crypto_portfolio_tracker.config import ConfigManager


class TestTransferFundingToSpot:
    """Test the transfer_funding_to_spot method with is_live parameter."""

    @pytest.fixture
    def mock_config_manager(self):
        config_manager = Mock(spec=ConfigManager)
        config_manager.config = {}
        config_manager.is_live = False
        config_manager.get_database_path = Mock(return_value=":memory:")
        config_manager.get_backup_dir = Mock(return_value=".")
        config_manager.symbol_mapper = {}
        return config_manager

    @pytest.fixture
    def portfolio_tracker(self, mock_config_manager):
        with patch('src.crypto_portfolio_tracker.portfolio_tracker.DatabaseManager'), \
             patch('src.crypto_portfolio_tracker.portfolio_tracker.DataSynchronizer') as mock_data_synchronizer_class, \
             patch('src.crypto_portfolio_tracker.portfolio_tracker.BinanceFetcher'):
            
            # Create mock data synchronizer
            mock_data_synchronizer = AsyncMock(spec=DataSynchronizer)
            mock_data_synchronizer.transfer_funding_to_spot = AsyncMock(return_value={
                "success": True, 
                "messages": [], 
                "errors": [], 
                "transfer_id": None
            })
            mock_data_synchronizer_class.return_value = mock_data_synchronizer
            
            # Create tracker
            tracker = CryptoPortfolioTracker(
                config_manager=mock_config_manager,
                force_offline=True
            )
            
            # Replace the data_synchronizer with our mock
            tracker.data_synchronizer = mock_data_synchronizer
            
            return tracker, mock_data_synchronizer

    @pytest.mark.asyncio
    async def test_transfer_funding_to_spot_with_is_live_false(self, portfolio_tracker):
        """Test transfer_funding_to_spot with is_live=False."""
        tracker, mock_data_synchronizer = portfolio_tracker
        
        # Set up mock return value
        mock_data_synchronizer.transfer_funding_to_spot.return_value = {
            "success": True,
            "messages": ["(Dry Run) Would transfer 100.0 USDT from funding to spot wallet"],
            "errors": [],
            "transfer_id": None
        }
        
        result = await tracker.transfer_funding_to_spot(
            amount=100.0, 
            asset="USDT", 
            is_live=False
        )
        
        # Verify the method was called with the correct parameters
        mock_data_synchronizer.transfer_funding_to_spot.assert_awaited_once_with(
            "USDT", 100.0, False
        )
        
        # Verify the result
        assert result["success"] is True
        assert "(Dry Run) Would transfer 100.0 USDT from funding to spot wallet" in result["messages"]

    @pytest.mark.asyncio
    async def test_transfer_funding_to_spot_with_is_live_true(self, portfolio_tracker):
        """Test transfer_funding_to_spot with is_live=True."""
        tracker, mock_data_synchronizer = portfolio_tracker
        
        # Set up mock return value
        mock_data_synchronizer.transfer_funding_to_spot.return_value = {
            "success": True,
            "messages": ["Successfully transferred 50.0 USDT to spot wallet. Transaction ID: 12345"],
            "errors": [],
            "transfer_id": "12345"
        }
        
        result = await tracker.transfer_funding_to_spot(
            amount=50.0, 
            asset="USDT", 
            is_live=True
        )
        
        # Verify the method was called with the correct parameters
        mock_data_synchronizer.transfer_funding_to_spot.assert_awaited_once_with(
            "USDT", 50.0, True
        )
        
        # Verify the result
        assert result["success"] is True
        assert "Successfully transferred 50.0 USDT to spot wallet. Transaction ID: 12345" in result["messages"]

    @pytest.mark.asyncio
    async def test_transfer_funding_to_spot_default_parameters(self, portfolio_tracker):
        """Test transfer_funding_to_spot with default parameters."""
        tracker, mock_data_synchronizer = portfolio_tracker
        
        # Set up mock return value
        mock_data_synchronizer.transfer_funding_to_spot.return_value = {
            "success": True,
            "messages": ["(Dry Run) Would transfer 75.0 USDT from funding to spot wallet"],
            "errors": [],
            "transfer_id": None
        }
        
        result = await tracker.transfer_funding_to_spot(
            amount=75.0
        )
        
        # Verify the method was called with the correct parameters (default asset="USDT", is_live=False)
        mock_data_synchronizer.transfer_funding_to_spot.assert_awaited_once_with(
            "USDT", 75.0, False
        )
        
        # Verify the result
        assert result["success"] is True
        assert "(Dry Run) Would transfer 75.0 USDT from funding to spot wallet" in result["messages"]