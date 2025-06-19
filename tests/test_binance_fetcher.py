import pytest
import pandas as pd
from unittest.mock import MagicMock

from crypto_portfolio_tracker.binance_fetcher import BinanceFetcher
from crypto_portfolio_tracker.symbol_mapper import SymbolMapper


@pytest.fixture
def mock_binance_client():
    """Provides a mock binance client."""
    client = MagicMock()
    # Configure the mock to return a sample trade for BTCUSDT
    mock_trade_data = [
        {'symbol': 'BTCUSDT', 'id': 123, 'time': 1620000000000, 'isBuyer': True,
         'qty': '0.1', 'price': '50000.0', 'commission': '0.0001', 'commissionAsset': 'BNB'}
    ]

    # Use a side effect to only return data when asked for BTCUSDT
    def get_trades_side_effect(symbol, **kwargs):
        if symbol == 'BTCUSDT':
            return mock_trade_data
        return []

    client.get_my_trades.side_effect = get_trades_side_effect
    return client


@pytest.fixture
def mock_symbol_mapper():
    """Provides a mock symbol mapper."""
    mapper = MagicMock(spec=SymbolMapper)
    mapper.normalize_symbol.side_effect = lambda s: s.upper()
    return mapper


def test_fetch_binance_transactions_returns_raw_data(mock_binance_client, mock_symbol_mapper):
    """
    Tests that the fetcher correctly calls the client and shapes the raw transaction data.
    """
    # 1. Arrange
    config = {
        "portfolio": {"crypto_quotes": ["USDT"], "stablecoin_symbols": ["USDT"]},
        "target_allocation": {"BTC": 1.0, "USDT": 1.0} # Ensure USDT is a target asset for synthetic tx
    }
    fetcher = BinanceFetcher(client=mock_binance_client, symbol_mapper=mock_symbol_mapper, config=config)

    # 2. Act
    raw_transactions = fetcher.fetch_binance_transactions(days_back=1)

    # 3. Assert
    # --- FIX: Expect 2 transactions (primary + synthetic) ---
    assert len(raw_transactions) == 2

    # Verify the primary transaction
    tx_btc = next(t for t in raw_transactions if t['raw_data'].get('base_asset') == 'BTC')
    assert tx_btc['tx_type'] == 'TRADE'
    assert tx_btc['raw_data']['price'] == 50000.0
