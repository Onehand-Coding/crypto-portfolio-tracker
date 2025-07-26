import datetime

import pytest
import pandas as pd

from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.portfolio_tracker import (
    CryptoPortfolioTracker,
    calculate_fifo_cost_basis,
)


def test_calculate_fifo_cost_basis():
    """
    Tests the FIFO cost basis calculation with a simple buy-sell-buy scenario.
    """
    # 1. Arrange: Create sample transaction data
    transactions = [
        {
            "timestamp": "2025-01-01",
            "type": "BUY",
            "quantity": 2,
            "price_usd": 100,
            "fee_usd": 5,
            "symbol": "TEST",
        },  # Total cost: 2*100 + 5 = 205
        {
            "timestamp": "2025-01-05",
            "type": "BUY",
            "quantity": 1,
            "price_usd": 110,
            "fee_usd": 2,
            "symbol": "TEST",
        },  # Total cost: 1*110 + 2 = 112
        {
            "timestamp": "2025-01-10",
            "type": "SELL",
            "quantity": 1.5,
            "price_usd": 120,
            "fee_usd": 3,
            "symbol": "TEST",
        },
    ]
    transactions_df = pd.DataFrame(transactions)

    # 2. Act: Run the function being tested
    current_quantity, avg_cost_basis = calculate_fifo_cost_basis(transactions_df)

    # 3. Assert: Verify the results are correct
    # After selling 1.5 units, we have 1.5 units left.
    # - 0.5 units from the first lot (cost: 0.5 * (205/2) = 51.25)
    # - 1 unit from the second lot (cost: 1 * 112 = 112)
    # Total remaining cost basis = 51.25 + 112 = 163.25
    # Average cost basis = 163.25 / 1.5 = 108.8333...
    expected_quantity = 1.5
    expected_avg_cost_basis = 108.83333333

    assert current_quantity == pytest.approx(expected_quantity)
    assert avg_cost_basis == pytest.approx(expected_avg_cost_basis)


def test_calculate_fifo_cost_basis_sell_all():
    """
    Tests that quantity and cost basis are zero after selling all holdings.
    """
    # 1. Arrange: Create transactions where we sell exactly what we bought
    transactions = [
        {
            "timestamp": "2025-01-01",
            "type": "BUY",
            "quantity": 2,
            "price_usd": 100,
            "fee_usd": 5,
            "symbol": "TEST",
        },
        {
            "timestamp": "2025-01-05",
            "type": "SELL",
            "quantity": 2,
            "price_usd": 110,
            "fee_usd": 2,
            "symbol": "TEST",
        },
    ]
    transactions_df = pd.DataFrame(transactions)

    # 2. Act: Run the function
    current_quantity, avg_cost_basis = calculate_fifo_cost_basis(transactions_df)

    # 3. Assert: Verify that everything is zeroed out
    assert current_quantity == 0.0
    assert avg_cost_basis == 0.0
