"""Regression tests for the holdings FIFO refresh.

Background: the `holdings` table (qty + average_cost_basis per asset) feeds
Streamlit's FIFO pair, snapshot cost basis, and now the API. It went stale
because `update_holdings_from_transactions()` existed but nothing called it.
These tests pin (1) the updater's behaviour and (2) that `run_full_sync`
actually calls it, in the right order.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from crypto_portfolio_tracker.data_manager import DataManager
from crypto_portfolio_tracker.database import DatabaseManager
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker


def _make_db(tmp_path):
    return DatabaseManager(
        db_path=tmp_path / "test_holdings.db",
        backup_dir=tmp_path / "backups",
        cleanup_days=7,
    )


def _tx(symbol, tx_type, quantity, price, source, tx_hash, ts):
    return {
        "symbol": symbol,
        "timestamp": ts,
        "type": tx_type,
        "quantity": quantity,
        "price_usd": price,
        "fee_usd": 0.0,
        "source": source,
        "transaction_hash": tx_hash,
    }


def test_update_holdings_computes_fifo_and_excludes_earn(tmp_path):
    """BTC BUY 2 @100 then SELL 1 @150 leaves qty 1 @100; Earn-only SOL leaves no row."""
    db_manager = _make_db(tmp_path)
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db_manager.bulk_insert_transactions([
        _tx("BTC", "BUY", 2.0, 100.0, "Binance Trade", "btc-buy-1", ts),
        _tx("BTC", "SELL", 1.0, 150.0, "Binance Trade", "btc-sell-1",
            datetime(2025, 1, 2, tzinfo=timezone.utc)),
        _tx("SOL", "BUY", 5.0, 10.0, "Binance Simple Earn Redemption",
            "sol-earn-1", ts),
    ])

    data_manager = DataManager(config={}, db_manager=db_manager)
    data_manager.update_holdings_from_transactions()

    holdings = db_manager.get_holdings()
    by_symbol = {row["symbol"]: row for _, row in holdings.iterrows()}

    assert "BTC" in by_symbol
    assert by_symbol["BTC"]["quantity"] == pytest.approx(1.0)
    assert by_symbol["BTC"]["average_cost_basis"] == pytest.approx(100.0)
    # SOL's only activity is Earn-sourced, so cost_basis_tx_df is empty and the
    # updater `continue`s without writing a row.
    assert "SOL" not in by_symbol


def test_all_earn_symbol_leaves_no_stale_row(tmp_path):
    """A DB whose only activity is Earn-sourced gets no holdings rows at all."""
    db_manager = _make_db(tmp_path)
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db_manager.bulk_insert_transactions([
        _tx("SOL", "BUY", 5.0, 10.0, "Binance Simple Earn Redemption",
            "sol-earn-1", ts),
    ])

    data_manager = DataManager(config={}, db_manager=db_manager)
    data_manager.update_holdings_from_transactions()

    holdings = db_manager.get_holdings()
    assert holdings.empty


@pytest.mark.asyncio
async def test_run_full_sync_refreshes_holdings_between_sync_and_metrics():
    """run_full_sync must call update_holdings AFTER sync lands, BEFORE metrics."""
    tracker = CryptoPortfolioTracker.__new__(CryptoPortfolioTracker)
    tracker.enricher = Mock()
    tracker.data_synchronizer = Mock()
    tracker.data_synchronizer.sync_data = AsyncMock(return_value=None)
    tracker.portfolio_analyzer = Mock()
    tracker.portfolio_analyzer.calculate_portfolio_metrics = AsyncMock(
        return_value={}
    )
    tracker.data_manager = Mock()

    parent = Mock()
    parent.attach_mock(tracker.data_synchronizer.sync_data, "sync_data")
    parent.attach_mock(
        tracker.data_manager.update_holdings_from_transactions,
        "update_holdings_from_transactions",
    )
    parent.attach_mock(
        tracker.portfolio_analyzer.calculate_portfolio_metrics,
        "calculate_portfolio_metrics",
    )

    await tracker.run_full_sync()

    assert [c[0] for c in parent.mock_calls] == [
        "sync_data",
        "update_holdings_from_transactions",
        "calculate_portfolio_metrics",
    ]
