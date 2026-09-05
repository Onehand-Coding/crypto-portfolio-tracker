import pandas as pd

from api.accounting import portfolio_fifo_cost_basis


def test_fifo_cost_basis_sums_across_symbols():
    holdings = pd.DataFrame([
        {"symbol": "BTC", "quantity": 1.0, "average_cost_basis": 100.0},
        {"symbol": "ETH", "quantity": 2.0, "average_cost_basis": 50.0},
    ])
    assert portfolio_fifo_cost_basis(holdings) == 200.0


def test_fifo_cost_basis_zero_qty_dust_contributes_zero():
    holdings = pd.DataFrame([
        {"symbol": "BTC", "quantity": 0.0, "average_cost_basis": 100.0},
        {"symbol": "ETH", "quantity": 2.0, "average_cost_basis": 50.0},
    ])
    assert portfolio_fifo_cost_basis(holdings) == 100.0


def test_fifo_cost_basis_on_empty_frame_is_zero():
    assert portfolio_fifo_cost_basis(pd.DataFrame()) == 0.0


def test_fifo_cost_basis_skips_nan_average_row():
    holdings = pd.DataFrame([
        {"symbol": "BTC", "quantity": 1.0, "average_cost_basis": float("nan")},
        {"symbol": "ETH", "quantity": 2.0, "average_cost_basis": 50.0},
    ])
    assert portfolio_fifo_cost_basis(holdings) == 100.0
