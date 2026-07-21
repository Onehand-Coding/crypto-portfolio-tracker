"""Portfolio-wide accounting derived from the core's per-asset primitives.

calculate_fifo_cost_basis operates on one asset at a time and returns
(quantity, average_cost_basis). Aggregating it is this module's job; the
core is not modified to do it.
"""

import pandas as pd

from crypto_portfolio_tracker.utils import calculate_fifo_cost_basis


def portfolio_fifo_cost_basis(transactions_df: pd.DataFrame) -> float:
    """Sum the FIFO cost basis of remaining lots across every symbol."""
    if transactions_df is None or transactions_df.empty:
        return 0.0
    if "symbol" not in transactions_df.columns:
        return 0.0

    total = 0.0
    for _symbol, group in transactions_df.groupby("symbol"):
        quantity, average_cost = calculate_fifo_cost_basis(group)
        total += quantity * average_cost
    return float(total)
