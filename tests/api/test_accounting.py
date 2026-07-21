import pandas as pd

from api.accounting import portfolio_fifo_cost_basis


def test_fifo_cost_basis_sums_across_symbols():
    txs = pd.DataFrame([
        {"symbol": "BTC", "timestamp": "2026-01-01", "type": "BUY",
         "quantity": 1.0, "price_usd": 100.0, "fee_usd": 0.0},
        {"symbol": "ETH", "timestamp": "2026-01-02", "type": "BUY",
         "quantity": 2.0, "price_usd": 50.0, "fee_usd": 0.0},
    ])
    assert portfolio_fifo_cost_basis(txs) == 200.0


def test_fifo_cost_basis_excludes_sold_lots():
    txs = pd.DataFrame([
        {"symbol": "BTC", "timestamp": "2026-01-01", "type": "BUY",
         "quantity": 2.0, "price_usd": 100.0, "fee_usd": 0.0},
        {"symbol": "BTC", "timestamp": "2026-01-02", "type": "SELL",
         "quantity": 1.0, "price_usd": 150.0, "fee_usd": 0.0},
    ])
    assert portfolio_fifo_cost_basis(txs) == 100.0


def test_fifo_cost_basis_on_empty_frame_is_zero():
    assert portfolio_fifo_cost_basis(pd.DataFrame()) == 0.0
