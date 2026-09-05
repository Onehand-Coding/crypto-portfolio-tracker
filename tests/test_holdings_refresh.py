"""Regression tests for the holdings FIFO refresh.

Background: the `holdings` table (qty + average_cost_basis per asset) feeds
Streamlit's FIFO pair, snapshot cost basis, and now the API. It went stale
because `update_holdings_from_transactions()` existed but nothing called it.
These tests pin (1) the updater's behaviour and (2) that `run_full_sync`
actually calls it, in the right order.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from binance.exceptions import BinanceAPIException

from crypto_portfolio_tracker.binance_fetcher import BinanceFetcher
from crypto_portfolio_tracker.data_manager import DataManager
from crypto_portfolio_tracker.data_synchronizer import DataSynchronizer
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


def test_update_holdings_clears_an_asset_after_it_is_fully_sold(tmp_path):
    """A later sell-all refresh must replace the previously positive FIFO row."""
    db_manager = _make_db(tmp_path)
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db_manager.bulk_insert_transactions([
        _tx("BTC", "BUY", 1.0, 100.0, "Binance Trade", "btc-buy-1", ts),
    ])
    data_manager = DataManager(config={}, db_manager=db_manager)
    data_manager.update_holdings_from_transactions()

    assert set(db_manager.get_holdings()["symbol"]) == {"BTC"}

    db_manager.bulk_insert_transactions([
        _tx("BTC", "SELL", 1.0, 150.0, "Binance Trade", "btc-sell-1",
            datetime(2025, 1, 2, tzinfo=timezone.utc)),
    ])
    data_manager.update_holdings_from_transactions()

    assert db_manager.get_holdings().empty


def test_update_holdings_clears_a_fully_sold_fractional_lot(tmp_path):
    """Floating-point rounding must not retain a lot whose sells equal its buy."""
    db_manager = _make_db(tmp_path)
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db_manager.bulk_insert_transactions([
        _tx("BTC", "BUY", 0.3, 100.0, "Binance Trade", "btc-buy-1", ts),
    ])
    data_manager = DataManager(config={}, db_manager=db_manager)
    data_manager.update_holdings_from_transactions()

    db_manager.bulk_insert_transactions([
        _tx("BTC", "SELL", 0.1, 150.0, "Binance Trade", "btc-sell-1",
            datetime(2025, 1, 2, tzinfo=timezone.utc)),
        _tx("BTC", "SELL", 0.2, 150.0, "Binance Trade", "btc-sell-2",
            datetime(2025, 1, 3, tzinfo=timezone.utc)),
    ])
    data_manager.update_holdings_from_transactions()

    assert db_manager.get_holdings().empty


def test_update_holdings_preserves_manual_holding_after_unmatched_sell(tmp_path):
    """An unmatched sell cannot prove that an imported holding was fully sold."""
    db_manager = _make_db(tmp_path)
    ts = datetime(2025, 1, 2, tzinfo=timezone.utc)
    db_manager.bulk_insert_transactions([
        _tx("BTC", "SELL", 1.0, 150.0, "Binance Trade", "btc-sell-1", ts),
    ])
    db_manager.update_holdings(pd.DataFrame([
        {"symbol": "BTC", "quantity": 0.5, "average_cost_basis": 200.0},
    ]))

    DataManager(config={}, db_manager=db_manager).update_holdings_from_transactions()

    holding = db_manager.get_holdings().iloc[0]
    assert holding["symbol"] == "BTC"
    assert holding["quantity"] == pytest.approx(0.5)
    assert holding["average_cost_basis"] == pytest.approx(200.0)


def test_update_holdings_replaces_manual_holding_after_matched_sell(tmp_path):
    """A successful sync makes matched Binance history the source of truth."""
    db_manager = _make_db(tmp_path)
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db_manager.bulk_insert_transactions([
        _tx("BTC", "BUY", 1.0, 100.0, "Binance Trade", "btc-buy-1", ts),
        _tx("BTC", "SELL", 1.0, 150.0, "Binance Trade", "btc-sell-1",
            datetime(2025, 1, 2, tzinfo=timezone.utc)),
    ])
    db_manager.update_holdings(pd.DataFrame([
        {"symbol": "BTC", "quantity": 0.5, "average_cost_basis": 200.0},
    ]))

    DataManager(config={}, db_manager=db_manager).update_holdings_from_transactions()

    assert db_manager.get_holdings().empty


@pytest.mark.asyncio
async def test_run_full_sync_refreshes_holdings_between_sync_and_metrics():
    """run_full_sync must call update_holdings AFTER sync lands, BEFORE metrics."""
    tracker = CryptoPortfolioTracker.__new__(CryptoPortfolioTracker)
    tracker.offline_mode = False
    tracker.enricher = Mock()
    tracker.data_synchronizer = Mock()
    tracker.data_synchronizer.sync_data = AsyncMock(return_value=True)
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


@pytest.mark.asyncio
async def test_run_full_sync_does_not_refresh_holdings_after_failed_sync():
    """A failed fetch must leave the existing FIFO table intact."""
    tracker = CryptoPortfolioTracker.__new__(CryptoPortfolioTracker)
    tracker.offline_mode = False
    tracker.enricher = Mock()
    tracker.data_synchronizer = Mock()
    tracker.data_synchronizer.sync_data = AsyncMock(return_value=False)
    tracker.portfolio_analyzer = Mock()
    tracker.portfolio_analyzer.calculate_portfolio_metrics = AsyncMock(
        return_value={}
    )
    tracker.data_manager = Mock()

    await tracker.run_full_sync()

    tracker.data_manager.update_holdings_from_transactions.assert_not_called()
    tracker.portfolio_analyzer.calculate_portfolio_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_full_sync_does_not_refresh_holdings_offline():
    """Offline metrics must not supersede an imported holding from stale history."""
    tracker = CryptoPortfolioTracker.__new__(CryptoPortfolioTracker)
    tracker.offline_mode = True
    tracker.enricher = Mock()
    tracker.data_synchronizer = Mock()
    tracker.data_synchronizer.sync_data = AsyncMock(return_value=True)
    tracker.portfolio_analyzer = Mock()
    tracker.portfolio_analyzer.calculate_portfolio_metrics = AsyncMock(
        return_value={}
    )
    tracker.data_manager = Mock()

    await tracker.run_full_sync()

    tracker.data_manager.update_holdings_from_transactions.assert_not_called()
    tracker.portfolio_analyzer.calculate_portfolio_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_data_returns_false_after_a_fetch_failure():
    """A partial fetch must not be reported as a successful FIFO refresh source."""
    synchronizer = DataSynchronizer.__new__(DataSynchronizer)
    synchronizer.offline_mode = False
    synchronizer.logger = Mock()
    synchronizer.config = {"history_lookback_days": {}}
    synchronizer.config_manager = Mock()
    synchronizer.config_manager.is_testnet_mode = True
    synchronizer.db_manager = Mock()
    synchronizer.db_manager.get_latest_timestamp_for_source.return_value = None
    synchronizer.fetcher = Mock()
    synchronizer.fetcher.fetch_binance_transactions.side_effect = RuntimeError(
        "Binance unavailable"
    )

    result = await synchronizer.sync_data(enricher=Mock())

    assert result is False


@pytest.mark.asyncio
async def test_sync_data_returns_false_when_a_fetcher_returns_none():
    """Fetchers that log their own errors signal failure by returning None."""
    synchronizer = DataSynchronizer.__new__(DataSynchronizer)
    synchronizer.offline_mode = False
    synchronizer.logger = Mock()
    synchronizer.config = {"history_lookback_days": {}}
    synchronizer.config_manager = Mock()
    synchronizer.config_manager.is_testnet_mode = True
    synchronizer.db_manager = Mock()
    synchronizer.db_manager.get_latest_timestamp_for_source.return_value = None
    synchronizer.fetcher = Mock()
    synchronizer.fetcher.fetch_binance_transactions.return_value = None
    enricher = Mock()
    enricher.enrich_transactions = AsyncMock()

    result = await synchronizer.sync_data(enricher=enricher)

    assert result is False
    enricher.enrich_transactions.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_data_returns_false_after_a_critical_source_marks_failure():
    """A source that retained partial rows cannot authorize a FIFO refresh."""
    synchronizer = DataSynchronizer.__new__(DataSynchronizer)
    synchronizer.offline_mode = False
    synchronizer.logger = Mock()
    synchronizer.config = {"history_lookback_days": {}}
    synchronizer.config_manager = Mock()
    synchronizer.config_manager.is_testnet_mode = True
    synchronizer.db_manager = Mock()
    synchronizer.db_manager.get_latest_timestamp_for_source.return_value = None
    synchronizer.fetcher = Mock()
    synchronizer.fetcher._sync_fetch_failed = False

    def partial_fetch():
        synchronizer.fetcher._sync_fetch_failed = True
        return []

    synchronizer.fetcher.fetch_binance_transactions.side_effect = partial_fetch

    result = await synchronizer.sync_data(enricher=Mock())

    assert result is False


def test_fetch_deposit_history_returns_none_after_an_unrecoverable_error():
    """A failed source cannot return partial history as a successful result."""
    fetcher = BinanceFetcher.__new__(BinanceFetcher)
    fetcher.logger = Mock()
    fetcher.binance_client = Mock()
    fetcher.binance_client.get_deposit_history.side_effect = BinanceAPIException(
        Mock(), 500, '{"code": -1000, "msg": "failure"}'
    )
    fetcher._get_start_end_timestamps = Mock(return_value=(1, 2))
    fetcher._handle_binance_api_exception = Mock(return_value=False)

    result = fetcher.fetch_deposit_history("Binance Deposit")

    assert result is None


def test_simple_earn_subscription_failure_marks_the_sync_incomplete():
    """Subscription history is accounting-critical even when no rows are retained."""
    fetcher = BinanceFetcher.__new__(BinanceFetcher)
    fetcher.logger = Mock()
    fetcher.binance_client = Mock()
    fetcher.binance_client._request_margin_api.side_effect = RuntimeError("unavailable")
    fetcher._get_start_end_timestamps = Mock(return_value=(1, 2))
    fetcher._sync_fetch_failed = False

    fetcher.fetch_simple_earn_subscriptions("Binance Simple Earn Subscription")

    assert fetcher._sync_fetch_failed is True


def test_simple_earn_reward_failure_does_not_mark_the_sync_incomplete():
    """Binance's deprecated rewards endpoint is deliberately non-blocking."""
    fetcher = BinanceFetcher.__new__(BinanceFetcher)
    fetcher.logger = Mock()
    fetcher.binance_client = Mock()
    fetcher.binance_client._request_margin_api.side_effect = RuntimeError("unavailable")
    fetcher._get_start_end_timestamps = Mock(return_value=(1, 2))
    fetcher._sync_fetch_failed = False

    fetcher.fetch_simple_earn_rewards("Binance Simple Earn Reward")

    assert fetcher._sync_fetch_failed is False


@pytest.mark.asyncio
async def test_sync_data_returns_false_when_enrichment_saves_nothing():
    """Fetched rows that cannot be enriched are not a complete sync result."""
    synchronizer = DataSynchronizer.__new__(DataSynchronizer)
    synchronizer.offline_mode = False
    synchronizer.logger = Mock()
    synchronizer.config = {"history_lookback_days": {}}
    synchronizer.config_manager = Mock()
    synchronizer.config_manager.is_testnet_mode = True
    synchronizer.db_manager = Mock()
    synchronizer.db_manager.get_latest_timestamp_for_source.return_value = None
    synchronizer.fetcher = Mock()
    synchronizer.fetcher.fetch_binance_transactions.return_value = [{"id": "raw"}]
    enricher = Mock()
    enricher.enrich_transactions = AsyncMock(return_value=[])

    result = await synchronizer.sync_data(enricher=enricher)

    assert result is False
