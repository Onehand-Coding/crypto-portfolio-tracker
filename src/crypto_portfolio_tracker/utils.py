import logging
from collections import deque

import pandas as pd

logger = logging.getLogger(__name__)


def parse_df_string(df_string):
    """Parse a DataFrame from its .to_string() output."""
    return pd.read_fwf(io.StringIO(df_string))


def format_percent(val):
    return f"{val:.2f}%" if pd.notnull(val) else ""


def format_usd(val):
    return f"${val:,.2f}" if pd.notnull(val) else ""


def format_qty(val):
    return f"{val:,.8f}" if pd.notnull(val) else ""


def build_holdings_table(df, alloc_col="allocation"):
    # Columns: Asset, Total Qty, Spot Qty, Earn Qty, Value (USD), Cost Basis, P/L (USD), Alloc.
    cols = [
        ("Asset", "symbol"),
        ("Total Qty", "total_quantity_api"),
        ("Spot Qty", "spot_quantity"),
        ("Earn Qty", "earn_quantity"),
        ("Value (USD)", "value_usd"),
        ("Cost Basis", "cost_basis_total"),
        ("P/L (USD)", "unrealized_pl_usd"),
        ("Alloc.", alloc_col),
    ]
    out = pd.DataFrame()
    for label, col in cols:
        if col in df.columns:
            if "alloc" in col:
                out[label] = (df[col] * 100).map(format_percent)
            elif (
                "usd" in col or "cost_basis" in col or "pl_usd" in col or "value" in col
            ):
                out[label] = df[col].map(format_usd)
            elif "qty" in col or "quantity" in col:
                out[label] = df[col].map(format_qty)
            else:
                out[label] = df[col]
        else:
            out[label] = ""
    return out


def clean_futures_balances(df):
    # Only show nonzero balances
    if "balance" in df.columns:
        df = df[df["balance"].astype(float) > 0]
    if df.empty:
        return None
    keep = ["asset", "balance", "availableBalance", "crossUnPnl"]
    df = df[[col for col in keep if col in df.columns]].copy()
    df.rename(
        columns={
            "asset": "Asset",
            "balance": "Balance",
            "availableBalance": "Available",
            "crossUnPnl": "Unrealized P/L",
        },
        inplace=True,
    )
    for col in ["Balance", "Available", "Unrealized P/L"]:
        if col in df.columns:
            df[col] = df[col].astype(float).map("{:,.8f}".format)
    return df


def clean_funding_balances(df):
    # Only show nonzero balances
    if "balance" in df.columns:
        df = df[df["balance"].astype(float) > 0]
    if df.empty:
        return None
    return df


def clean_export_df(df):
    col_map = {
        "symbol": "Asset",
        "total_quantity_api": "Total Qty",
        "earn_quantity": "Earn Qty",
        "total_quantity": "Total Qty",
        "spot_quantity": "Spot Qty",
        "average_cost_basis": "Avg Cost Basis",
        "current_price": "Current Price",
        "value_usd": "Value (USD)",
        "cost_basis_total": "Cost Basis",
        "unrealized_pl_usd": "Unrealized P/L (USD)",
        "unrealized_pl_percent": "Unrealized P/L (%)",
        "allocation": "Allocation (%)",
        "is_core": "Core Asset",
    }
    desired_order = [
        "Asset",
        "Total Qty",
        "Spot Qty",
        "Earn Qty",
        "Current Price",
        "Value (USD)",
        "Avg Cost Basis",
        "Cost Basis",
        "Unrealized P/L (USD)",
        "Unrealized P/L (%)",
        "Allocation (%)",
        "Core Asset",
    ]
    df = df.rename(columns=col_map)
    cols = [col for col in desired_order if col in df.columns]
    df = df[cols]
    for col in df.columns:
        try:
            col_dtype = df[col].dtype
            if col_dtype == bool:
                df[col] = df[col].map({True: "Yes", False: "No"})
            elif col == "Allocation (%)":
                # Format as percentage with 2 decimals
                df[col] = (df[col] * 100).map(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
                )
            elif (
                "USD" in col or "Cost Basis" in col or "Value" in col or "Price" in col
            ):
                df[col] = df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
            elif "Qty" in col:
                df[col] = df[col].map(lambda x: f"{x:,.6f}" if pd.notnull(x) else "")
            elif "P/L (%)" in col:
                df[col] = df[col].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
            elif pd.api.types.is_numeric_dtype(col_dtype):
                df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
        except Exception:
            continue
    return df


def calculate_fifo_realized_gains(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates realized gains/losses for each SELL using FIFO.
    Returns a DataFrame with columns:
    ['date', 'symbol', 'quantity', 'proceeds_usd', 'cost_basis_usd', 'gain_usd']
    """
    results = []
    for symbol, group in transactions_df.groupby("symbol"):
        # Sort by timestamp
        group = group.sort_values("timestamp").reset_index(drop=True)
        buy_lots = deque()
        for idx, row in group.iterrows():
            tx_type = row["type"]
            quantity = float(row["quantity"])
            price = float(row["price_usd"])
            fee_usd = (
                float(row.get("fee_usd", 0.0))
                if not pd.isna(row.get("fee_usd", 0.0))
                else 0.0
            )
            date = pd.to_datetime(row["timestamp"])
            if tx_type in ["BUY", "DEPOSIT"]:
                # Add to FIFO queue
                total_cost = (quantity * price) + fee_usd
                effective_price = total_cost / quantity if quantity > 0 else 0.0
                buy_lots.append({"qty": quantity, "price": effective_price})
            elif tx_type in ["SELL", "WITHDRAWAL"]:
                sell_qty = quantity
                proceeds = quantity * price - fee_usd  # Net proceeds after fee
                cost_basis = 0.0
                lots_used = []
                while sell_qty > 0 and buy_lots:
                    lot = buy_lots[0]
                    lot_qty = lot["qty"]
                    lot_price = lot["price"]
                    if lot_qty <= sell_qty:
                        used_qty = lot_qty
                        cost_basis += used_qty * lot_price
                        sell_qty -= used_qty
                        buy_lots.popleft()
                    else:
                        used_qty = sell_qty
                        cost_basis += used_qty * lot_price
                        lot["qty"] -= used_qty
                        sell_qty = 0
                # If not enough buys, cost_basis for missing qty is 0 (could warn)
                gain = proceeds - cost_basis
                results.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "quantity": quantity,
                        "proceeds_usd": proceeds,
                        "cost_basis_usd": cost_basis,
                        "gain_usd": gain,
                    }
                )
    return pd.DataFrame(results)


def calculate_fifo_cost_basis(transactions_df: pd.DataFrame):
    """
    Calculates current quantity and average cost basis using FIFO.
    Assumes transactions_df is for a single asset, sorted by timestamp.
    Incorporates fee_usd into the cost of BUY/DEPOSIT transactions.
    
    Args:
        transactions_df: DataFrame containing transaction history for a single asset,
                        with columns ['timestamp', 'type', 'quantity', 'price_usd', 'fee_usd']
                        
    Returns:
        tuple: (current_quantity, average_cost_basis) representing the current holdings
               and their average cost basis
    """
    buy_lots = (
        deque()
    )  # Queue to store {"qty": quantity, "price": effective_price_usd_per_unit}
    current_quantity = 0.0
    total_cost_basis = 0.0  # This will track the cost basis of *remaining* lots

    if transactions_df.empty:
        symbol = (
            transactions_df["symbol"].iloc[0]
            if not transactions_df.empty and "symbol" in transactions_df.columns
            else "Unknown"
        )
        logger.debug(
            f"No transactions provided to calculate_fifo_cost_basis for symbol {symbol}. Returning 0, 0."
        )
        return 0.0, 0.0

    # Ensure correct dtypes and sorting
    transactions_df = transactions_df.copy()
    transactions_df["timestamp"] = pd.to_datetime(
        transactions_df["timestamp"], errors="coerce"
    )
    transactions_df = transactions_df.sort_values(by="timestamp").reset_index(drop=True)

    if transactions_df["timestamp"].isna().any():
        logger.warning(
            f"Found NaT timestamps for symbol {transactions_df['symbol'].iloc[0] if not transactions_df.empty else 'Unknown'} after coercion. Dropping these rows for FIFO."
        )
        transactions_df.dropna(subset=["timestamp"], inplace=True)
        if transactions_df.empty:
            logger.warning(
                f"All transactions dropped for symbol due to NaT timestamps. Returning 0, 0."
            )
            return 0.0, 0.0

    for _, row in transactions_df.iterrows():
        tx_type = row.get("type")

        try:
            quantity = float(row.get("quantity", 0.0))
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid quantity '{row.get('quantity')}' for tx {row.get('transaction_hash')}. Using 0.0."
            )
            quantity = 0.0

        if quantity == 0.0 and tx_type in ["BUY", "SELL", "DEPOSIT", "WITHDRAWAL"]:
            logger.debug(
                f"Skipping zero quantity {tx_type} transaction: {row.get('transaction_hash')}"
            )
            continue

        price = row.get("price_usd")

        fee_usd = row.get("fee_usd", 0.0)
        if pd.isna(fee_usd) or not isinstance(fee_usd, (int, float)):
            logger.debug(
                f"Invalid or missing fee_usd ('{row.get('fee_usd')}') for tx {row.get('transaction_hash')}. Using 0.0 for fee_usd."
            )
            fee_usd = 0.0
        else:
            try:
                fee_usd = float(fee_usd)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert fee_usd '{row.get('fee_usd')}' to float for tx {row.get('transaction_hash')}. Using 0.0 for fee_usd."
                )
                fee_usd = 0.0

        if tx_type == "BUY" or tx_type == "DEPOSIT":
            actual_tx_price_usd = 0.0
            if price is not None and isinstance(price, (int, float)) and price > 0:
                actual_tx_price_usd = float(price)
            elif price == 0.0 and tx_type == "DEPOSIT":
                actual_tx_price_usd = 0.0
            elif price is None or not isinstance(price, (int, float)) or price <= 0:
                logger.warning(
                    f"{tx_type} tx for {row.get('symbol', 'N/A')} (ID: {row.get('transaction_hash', 'N/A')}) "
                    f"has invalid/missing price ({price}). Using $0 price component for this lot."
                )
                actual_tx_price_usd = 0.0

            # <<< FEE INCORPORATION FOR BUYS/DEPOSITS >>>
            cost_for_this_transaction_lot = (quantity * actual_tx_price_usd) + fee_usd
            effective_price_per_unit = (
                cost_for_this_transaction_lot / quantity if quantity > 0 else 0.0
            )  # Price per unit *including* fee

            buy_lots.append({"qty": quantity, "price": effective_price_per_unit})
            current_quantity += quantity
            total_cost_basis += (
                cost_for_this_transaction_lot  # Add full cost (value + fee)
            )

        elif tx_type == "SELL" or tx_type == "WITHDRAWAL":
            sell_qty = quantity
            current_quantity -= sell_qty

            while sell_qty > 0 and buy_lots:
                oldest_buy = buy_lots[0]
                # oldest_buy["price"] is the effective_price_per_unit from its acquisition (already includes its buy fee)

                if oldest_buy["qty"] <= sell_qty:
                    qty_removed_from_lot = oldest_buy["qty"]
                    cost_basis_removed = qty_removed_from_lot * oldest_buy["price"]
                    total_cost_basis -= cost_basis_removed
                    sell_qty -= qty_removed_from_lot
                    buy_lots.popleft()
                else:
                    cost_basis_removed = sell_qty * oldest_buy["price"]
                    total_cost_basis -= cost_basis_removed
                    oldest_buy["qty"] -= sell_qty
                    sell_qty = 0

            if sell_qty > 0:
                logger.warning(
                    f"{tx_type} of {sell_qty} {row.get('symbol', 'N/A')} more than available from history. Cost basis may be affected."
                )

        total_cost_basis = max(0, total_cost_basis)

    average_cost_basis = (
        total_cost_basis / current_quantity if current_quantity > 0 else 0.0
    )
    current_quantity = max(0, current_quantity)

    logger.debug(
        f"FIFO Result for {transactions_df['symbol'].iloc[0] if not transactions_df.empty else 'Unknown'}: Qty={current_quantity:.8f}, AvgCost={average_cost_basis:.8f}, TotalCostBasisForAsset={total_cost_basis:.8f}"
    )
    return current_quantity, average_cost_basis
