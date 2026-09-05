"""Disposal-kind classification in calculate_fifo_realized_gains.

Every realized row must carry the economic kind of the disposal (trade leg,
conversion, Earn move, cash-out, or other) so the UI can explain gross
proceeds instead of presenting one scary lump sum. Classification reads the
transaction source; a missing or unrecognized source is OTHER, never an
error -- and the FIFO figures themselves must be byte-identical with or
without the column.
"""

import pandas as pd

from src.crypto_portfolio_tracker.utils import calculate_fifo_realized_gains


def _txns(rows):
    base = []
    for r in rows:
        row = {"timestamp": "2025-01-01T00:00:00Z", "symbol": "BTC",
               "type": "BUY", "quantity": 1.0, "price_usd": 100.0,
               "fee_usd": 0.0, "source": "Binance Trade"}
        row.update(r)
        base.append(row)
    return pd.DataFrame(base)


def _kinds(df):
    assert len(df) == 1
    return df.iloc[0]["kind"]


def test_trade_sell_is_trade():
    df = calculate_fifo_realized_gains(_txns([
        {}, {"type": "SELL", "price_usd": 150.0, "source": "Binance Trade"},
    ]))
    assert _kinds(df) == "TRADE"


def test_synthetic_leg_is_trade():
    df = calculate_fifo_realized_gains(_txns([
        {}, {"type": "SELL", "price_usd": 150.0, "source": "Binance Synthetic",
             "notes": "Synthetic for BTC/USDT trade"},
    ]))
    assert _kinds(df) == "TRADE"


def test_earn_subscription_is_earn():
    df = calculate_fifo_realized_gains(_txns([
        {}, {"type": "SELL", "price_usd": 150.0,
             "source": "Binance Simple Earn Subscription"},
    ]))
    assert _kinds(df) == "EARN"


def test_convert_is_convert():
    df = calculate_fifo_realized_gains(_txns([
        {}, {"type": "SELL", "price_usd": 150.0, "source": "Binance Convert"},
    ]))
    assert _kinds(df) == "CONVERT"


def test_p2p_sell_is_cashout():
    df = calculate_fifo_realized_gains(_txns([
        {}, {"type": "SELL", "price_usd": 150.0, "source": "Binance P2P Sell"},
    ]))
    assert _kinds(df) == "CASHOUT"


def test_unknown_source_is_other_not_an_error():
    df = calculate_fifo_realized_gains(_txns([
        {}, {"type": "SELL", "price_usd": 150.0, "source": "Something New"},
    ]))
    assert _kinds(df) == "OTHER"


def test_missing_source_column_is_other():
    df = pd.DataFrame([
        {"timestamp": "2025-01-01T00:00:00Z", "symbol": "BTC", "type": "BUY",
         "quantity": 1.0, "price_usd": 100.0, "fee_usd": 0.0},
        {"timestamp": "2025-02-01T00:00:00Z", "symbol": "BTC", "type": "SELL",
         "quantity": 1.0, "price_usd": 150.0, "fee_usd": 0.0},
    ])
    out = calculate_fifo_realized_gains(df)
    assert out.iloc[0]["kind"] == "OTHER"
    assert out.iloc[0]["gain_usd"] == 50.0


def test_kind_does_not_change_fifo_figures():
    df = calculate_fifo_realized_gains(_txns([
        {}, {"type": "SELL", "quantity": 0.4, "price_usd": 150.0,
             "source": "Binance Simple Earn Subscription"},
    ]))
    row = df.iloc[0]
    assert row["proceeds_usd"] == 60.0
    assert row["cost_basis_usd"] == 40.0
    assert row["gain_usd"] == 20.0
