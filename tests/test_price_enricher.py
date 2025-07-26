import datetime
from unittest.mock import MagicMock

import pytest
import pandas as pd
from diskcache import Cache

from crypto_portfolio_tracker.price_enricher import PriceEnricher
from crypto_portfolio_tracker.symbol_mapper import SymbolMapper

pytestmark = pytest.mark.usefixtures("mocker")


@pytest.fixture
def mock_symbol_mapper():
    """Provides a mock symbol mapper."""
    mapper = MagicMock(spec=SymbolMapper)
    mapper.get.side_effect = lambda symbol: f"{symbol.lower()}-id" if symbol else None
    return mapper


@pytest.fixture
def mock_disk_cache(tmp_path):
    """Provides a temporary mock disk cache."""
    return Cache(str(tmp_path))


@pytest.mark.asyncio
async def test_enrich_trade_transaction(mock_symbol_mapper, mock_disk_cache, mocker):
    """
    Tests that the enricher correctly processes a raw TRADE transaction.
    """
    # 1. Arrange
    sample_raw_tx = [
        {
            "tx_type": "TRADE",
            "timestamp": pd.to_datetime("2025-06-15 12:00:00", utc=True),
            "source": "Binance Trade",
            "transaction_hash": "binance_trade_123",
            "raw_data": {
                "base_asset": "BTC",
                "quote_asset": "USDT",
                "is_buyer": True,
                "quantity": 0.1,
                "price": 50000.0,
                "fee_quantity": 0.0001,
                "fee_currency": "BNB",
            },
        }
    ]

    config = {
        "apis": {"coingecko": {}, "yfinance": {}},
        "portfolio": {"stablecoin_symbols": ["USDT"]},
        "target_allocation": {"BTC": 1.0, "USDT": 1.0},
    }
    enricher = PriceEnricher(mock_symbol_mapper, config, mock_disk_cache)

    date_str = "15-06-2025"
    enricher.price_cache = {
        f"{mock_symbol_mapper.get('BNB')}_{date_str}": 300.0,
    }

    # 2. Act
    enriched_transactions = await enricher.enrich_transactions(sample_raw_tx)

    # 3. Assert
    assert len(enriched_transactions) == 2

    tx_btc = next(t for t in enriched_transactions if t["symbol"] == "BTC")
    tx_usdt = next(t for t in enriched_transactions if t["symbol"] == "USDT")

    # --- FIX: Use pytest.approx() for all floating-point comparisons ---
    assert tx_btc["type"] == "BUY"
    assert tx_btc["price_usd"] == pytest.approx(50000.0)
    assert tx_btc["fee_usd"] == pytest.approx(0.03)

    assert tx_usdt["type"] == "SELL"
    assert tx_usdt["quantity"] == pytest.approx(5000.0)
    assert tx_usdt["price_usd"] == pytest.approx(1.0)
