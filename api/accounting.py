"""Portfolio-wide accounting derived from the maintained holdings table.

The core's `update_holdings_from_transactions()` (run on every sync) writes
one row per asset with FIFO quantity and average cost. Aggregating that table
is this module's job; recomputing lots here would be a second methodology
drifting from the core's forever.
"""

import pandas as pd


def portfolio_fifo_cost_basis(holdings_df: pd.DataFrame) -> float:
    """Sum quantity * average_cost_basis across every held symbol."""
    if holdings_df is None or holdings_df.empty:
        return 0.0
    if "quantity" not in holdings_df.columns:
        return 0.0
    if "average_cost_basis" not in holdings_df.columns:
        return 0.0

    frame = holdings_df.copy()
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["average_cost_basis"] = pd.to_numeric(
        frame["average_cost_basis"], errors="coerce"
    )
    # A NaN average is unknown, not zero: fabricating 0 would understate the
    # basis. The updater only writes positive averages, so this is defensive.
    frame = frame.dropna(subset=["quantity", "average_cost_basis"])
    if frame.empty:
        return 0.0
    return float((frame["quantity"] * frame["average_cost_basis"]).sum())
