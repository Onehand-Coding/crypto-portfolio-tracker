# src/crypto_portfolio_tracker/utils.py

def clean_export_df(df):
    import pandas as pd
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
        "is_core": "Core Asset"
    }
    desired_order = [
        "Asset", "Total Qty", "Spot Qty", "Earn Qty", "Current Price", "Value (USD)",
        "Avg Cost Basis", "Cost Basis", "Unrealized P/L (USD)", "Unrealized P/L (%)", "Allocation (%)", "Core Asset"
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
                df[col] = (df[col] * 100).map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
            elif "USD" in col or "Cost Basis" in col or "Value" in col or "Price" in col:
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