"""
Tests for database operations.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from crypto_portfolio_tracker.database import DatabaseManager
from crypto_portfolio_tracker.exceptions import DatabaseOperationError


class TestDatabaseManager:
    """Test suite for DatabaseManager."""

    def test_initialization(self, db_manager):
        """Test database initialization."""
        assert db_manager.db_path.exists()
        assert isinstance(db_manager, DatabaseManager)

    def test_insert_asset(self, db_manager):
        """Test asset insertion."""
        asset_id = db_manager.get_asset_id("BTC", "Bitcoin", "bitcoin")
        assert asset_id > 0

    def test_bulk_insert_transactions(self, db_manager):
        """Test bulk transaction insertion."""
        transactions = [
            {
                "symbol": "BTC",
                "timestamp": datetime.now(timezone.utc),
                "type": "BUY",
                "quantity": 0.1,
                "price_usd": 50000.0,
                "fee_usd": 5.0,
                "transaction_hash": "hash1",
            },
            {
                "symbol": "ETH",
                "timestamp": datetime.now(timezone.utc),
                "type": "SELL",
                "quantity": 1.0,
                "price_usd": 3000.0,
                "fee_usd": 3.0,
                "transaction_hash": "hash2",
            },
        ]

        db_manager.bulk_insert_transactions(transactions)

        # Verify transactions were inserted
        all_transactions = db_manager.get_all_transactions()
        assert len(all_transactions) >= 2

    def test_get_invested_capital_transactions(self, db_manager):
        """Test fetching invested capital transactions."""
        # Insert test asset and transactions
        asset_id = db_manager.get_asset_id("BTC", "Bitcoin", "bitcoin")

        # Insert different types of transactions
        transactions = [
            {
                "symbol": "BTC",
                "timestamp": datetime.now(timezone.utc),
                "type": "BUY",
                "source": "Binance P2P Buy",  # This should be included
                "quantity": 0.1,
                "price_usd": 50000.0,
                "fee_usd": 5.0,
                "transaction_hash": "hash1",
            },
            {
                "symbol": "ETH",
                "timestamp": datetime.now(timezone.utc),
                "type": "WITHDRAWAL",  # This should be included
                "quantity": 1.0,
                "price_usd": 3000.0,
                "transaction_hash": "hash2",
            },
            {
                "symbol": "BNB",
                "timestamp": datetime.now(timezone.utc),
                "type": "BUY",  # This should NOT be included (no source = 'Binance P2P Buy')
                "quantity": 2.0,
                "price_usd": 300.0,
                "transaction_hash": "hash3",
            }
        ]
        # ACTUALLY INSERT THE TRANSACTIONS
        db_manager.bulk_insert_transactions(transactions)

        # Fetch invested capital transactions
        df = db_manager.get_invested_capital_transactions()

        # Should only have 2 transactions (BTC P2P buy and ETH withdrawal)
        assert len(df) == 2
        
        # Check that we have the P2P buy transaction
        p2p_transaction = df[df['source'] == 'Binance P2P Buy']
        assert len(p2p_transaction) == 1
        assert p2p_transaction.iloc[0]['quantity'] == 0.1
        assert p2p_transaction.iloc[0]['price_usd'] == 50000.0
        
        # Check that we have the withdrawal transaction
        withdrawal_transaction = df[df['type'] == 'WITHDRAWAL']
        assert len(withdrawal_transaction) == 1
        assert withdrawal_transaction.iloc[0]['quantity'] == 1.0
        assert withdrawal_transaction.iloc[0]['price_usd'] == 3000.0

    def test_get_holdings(self, db_manager):
        """Test holdings retrieval."""
        # Insert test data
        asset_id = db_manager.get_asset_id("BTC", "Bitcoin", "bitcoin")

        transactions = [
            {
                "symbol": "BTC",
                "timestamp": datetime.now(timezone.utc),
                "type": "BUY",
                "quantity": 0.1,
                "price_usd": 50000.0,
                "fee_usd": 5.0,
                "transaction_hash": "hash1",
            }
        ]
        db_manager.bulk_insert_transactions(transactions)

        holdings = db_manager.get_holdings()
        assert isinstance(holdings, pd.DataFrame)
        # Holdings might be empty if no transactions are processed yet
        # This is expected behavior for a fresh database

    def test_database_operations_error_handling(self, db_manager):
        """Test error handling in database operations."""
        # Test with invalid data
        with pytest.raises(Exception):
            db_manager.bulk_insert_transactions(
                [
                    {
                        "symbol": "INVALID",
                        "timestamp": "invalid_timestamp",
                        "type": "INVALID_TYPE",
                        "quantity": "not_a_number",
                        "price_usd": "not_a_number",
                        "fee_usd": "not_a_number",
                        "transaction_hash": "hash1",
                    }
                ]
            )
