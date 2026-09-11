"""SELL proceeds must fund BUYs within one rebalance execution.

Regression test for the testnet observation where "run all" executed only
the SELL legs: execute_rebalancing_trades snapshots spot balances up front
but never credits SELL proceeds to the simulated USDT balance, so later BUY
legs are wrongly skipped as "exceeds simulated available balance" even
though the real exchange balance would cover them (sells run first).

Runs the DRY-RUN path (live_trading_enabled off): no orders leave the box,
but the skip guards and balance bookkeeping are identical to live.
"""

import asyncio
from types import SimpleNamespace

import pandas as pd

from src.crypto_portfolio_tracker.models import ExecutionMode
from src.crypto_portfolio_tracker.trade_executor import TradeExecutor


class _FakeBinance:
    def __init__(self, balances):
        self._balances = balances

    def get_asset_balance(self, asset):
        return {"free": self._balances.get(asset, 0.0)}

    def get_symbol_info(self, symbol):
        base = symbol.removesuffix("USDT")
        return {
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.00001000"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "5.00000000"},
            ],
            "baseAsset": base,
        }


class _FakeSync:
    def __init__(self, prices):
        self._prices = prices

    def _get_current_prices(self, symbols):
        return {s: self._prices[s] for s in symbols if s in self._prices}


class _FakeCache:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value, expire=None):
        self._store[key] = value


def _executor():
    return TradeExecutor(
        config={"portfolio": {"minimum_trade_usd": 5.0, "live_trading_enabled": False}},
        binance_client=_FakeBinance({"BTC": 1.0, "ETH": 0.0, "USDT": 100.0}),
        config_manager=SimpleNamespace(is_testnet_mode=True),
        yfinance_disk_cache=_FakeCache(),
        data_synchronizer=_FakeSync({"BTC": 50_000.0, "ETH": 2_500.0}),
        data_manager=None,
    )


def _plan():
    # SELL sorts first, so its $25k proceeds are really available for the BUY.
    return pd.DataFrame(
        [
            {
                "Symbol": "BTC",
                "Signal": "SELL",
                "Drift (pts)": 40.0,
                "action_usd_value": 25_000.0,
                "action_coin_quantity": 0.5,
                "Suggested Action Detail": "Sell ~$25,000 worth of BTC",
            },
            {
                "Symbol": "ETH",
                "Signal": "BUY",
                "Drift (pts)": -25.0,
                "action_usd_value": 10_000.0,
                "action_coin_quantity": 4.0,
                "Suggested Action Detail": "Buy ~$10,000 worth of ETH",
            },
        ]
    )


def test_sell_proceeds_fund_later_buys():
    result = asyncio.run(
        _executor().execute_rebalancing_trades(
            _plan(), {}, execution_mode=ExecutionMode.AUTO
        )
    )
    assert result.success is True
    assert result.data["trades_executed"] == 2
    assert not any("SKIPPING BUY" in m for m in result.messages)
