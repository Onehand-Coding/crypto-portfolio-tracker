import pandas as pd
import streamlit as st

from crypto_portfolio_tracker.dashboard import utils as ui_utils
from crypto_portfolio_tracker.dashboard.components.redeem_widget import (
    render_redeem_widget,
)
from crypto_portfolio_tracker.dashboard.components.trading_status_banner import (
    render_trading_status_banner,
)
from crypto_portfolio_tracker.dashboard.components.transfer_widget import (
    render_transfer_widget,
)
from crypto_portfolio_tracker.utils import format_usd, parse_df_string


def _display_wallet_balances(metrics, dashboard=None):
    """Display wallet balances in a structured format with clear separation."""

    # --- Wallet Value Summary ---
    st.markdown("### 💰 Wallet Value Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_value = (
            metrics.get("spot_earn_value_usd", 0)
            + metrics.get("futures_value_usd", 0)
            + metrics.get("funding_value_usd", 0)
        )
        st.metric(
            label="Total Portfolio Value",
            value=format_usd(total_value),
            help="Sum of all wallet values",
        )

    with col2:
        st.metric(
            label="Spot & Earn Value",
            value=format_usd(metrics.get("spot_earn_value_usd", 0)),
            help="Combined value of Spot and Earn wallets",
        )

    with col3:
        st.metric(
            label="Futures Value",
            value=format_usd(metrics.get("futures_value_usd", 0)),
            help="Value of Futures wallet",
        )

    with col4:
        st.metric(
            label="Funding Value",
            value=format_usd(metrics.get("funding_value_usd", 0)),
            help="Value of Funding wallet",
        )

    st.markdown("---")

    # --- Detailed Wallet Breakdown ---
    st.markdown("### 📊 Detailed Wallet Breakdown")

    # Spot & Earn Wallets
    st.markdown("#### 📍 Spot & Earn Wallets")
    holdings_df = (
        parse_df_string(metrics.get("holdings_df"))
        if isinstance(metrics.get("holdings_df"), str)
        else metrics.get("holdings_df")
    )

    if holdings_df is not None and not holdings_df.empty:
        # Filter for non-zero holdings
        non_zero_holdings = holdings_df[holdings_df["total_quantity"] > 1e-8].copy()

        if not non_zero_holdings.empty:
            # Separate Spot and Earn holdings for better clarity
            spot_holdings = non_zero_holdings[
                non_zero_holdings["spot_quantity"] > 1e-8
            ].copy()
            earn_holdings = non_zero_holdings[
                non_zero_holdings["earn_quantity"] > 1e-8
            ].copy()

            if not spot_holdings.empty:
                st.markdown("**Spot Wallet Holdings**")
                spot_display = spot_holdings[
                    ["symbol", "spot_quantity", "value_usd"]
                ].copy()
                spot_display = spot_display.rename(
                    columns={
                        "symbol": "Asset",
                        "spot_quantity": "Quantity",
                        "value_usd": "Value (USD)",
                    }
                )
                # Format the columns
                spot_display["Quantity"] = spot_display["Quantity"].apply(
                    lambda x: f"{float(x):,.8f}" if pd.notna(x) else "0.00000000"
                )
                spot_display["Value (USD)"] = spot_display["Value (USD)"].apply(
                    lambda x: f"${float(x):,.2f}" if pd.notna(x) else "$0.00"
                )
                st.dataframe(spot_display, use_container_width=True)

            if not earn_holdings.empty:
                st.markdown("**Earn Wallet Holdings**")
                earn_display = earn_holdings[
                    ["symbol", "earn_quantity", "value_usd"]
                ].copy()
                earn_display = earn_display.rename(
                    columns={
                        "symbol": "Asset",
                        "earn_quantity": "Quantity",
                        "value_usd": "Value (USD)",
                    }
                )
                # Format the columns
                earn_display["Quantity"] = earn_display["Quantity"].apply(
                    lambda x: f"{float(x):,.8f}" if pd.notna(x) else "0.00000000"
                )
                earn_display["Value (USD)"] = earn_display["Value (USD)"].apply(
                    lambda x: f"${float(x):,.2f}" if pd.notna(x) else "$0.00"
                )
                st.dataframe(earn_display, use_container_width=True)
        else:
            st.info("No non-zero holdings in Spot or Earn wallets.")
    else:
        st.info("No Spot or Earn holdings found.")

    # Futures Wallet
    st.markdown("#### 💹 Futures Wallet")
    futures_balances = metrics.get("futures_balances", [])
    if futures_balances:
        futures_df = pd.DataFrame(futures_balances)
        if not futures_df.empty:
            # Filter for non-zero balances
            futures_df["balance"] = pd.to_numeric(
                futures_df["balance"], errors="coerce"
            )
            non_zero_futures = futures_df[futures_df["balance"] > 1e-8]

            if not non_zero_futures.empty:
                futures_display = non_zero_futures[["asset", "balance"]].copy()
                futures_display = futures_display.rename(
                    columns={"asset": "Asset", "balance": "Balance"}
                )
                st.dataframe(futures_display, use_container_width=True)
            else:
                st.info("No non-zero balances in Futures wallet.")
        else:
            st.info("No Futures balances found.")
    else:
        st.info("No Futures balances found.")

    # Funding Wallet
    st.markdown("#### 💵 Funding Wallet")
    funding_balances = metrics.get("funding_balances", [])
    if funding_balances:
        funding_df = pd.DataFrame(funding_balances)
        if not funding_df.empty:
            # Filter for non-zero balances
            funding_df["free"] = pd.to_numeric(funding_df["free"], errors="coerce")
            non_zero_funding = funding_df[funding_df["free"] > 1e-8]

            if not non_zero_funding.empty:
                funding_display = non_zero_funding[["asset", "free"]].copy()
                funding_display = funding_display.rename(
                    columns={"asset": "Asset", "free": "Balance"}
                )
                st.dataframe(funding_display, use_container_width=True)
            else:
                st.info("No non-zero balances in Funding wallet.")
        else:
            st.info("No Funding balances found.")
    else:
        st.info("No Funding balances found.")


def _display_wallet_allocation(metrics):
    """Display wallet allocation visualization."""
    st.markdown("### 📈 Wallet Allocation")

    spot_value = metrics.get("spot_earn_value_usd", 0)
    futures_value = metrics.get("futures_value_usd", 0)
    funding_value = metrics.get("funding_value_usd", 0)
    total_value = spot_value + futures_value + funding_value

    if total_value > 0:
        # Create allocation data for visualization
        allocation_data = {
            "Wallet": ["Spot & Earn", "Futures", "Funding"],
            "Value": [spot_value, futures_value, funding_value],
            "Percentage": [
                (spot_value / total_value) * 100 if total_value > 0 else 0,
                (futures_value / total_value) * 100 if total_value > 0 else 0,
                (funding_value / total_value) * 100 if total_value > 0 else 0,
            ],
        }
        allocation_df = pd.DataFrame(allocation_data)

        # Filter out wallets with zero value for display
        non_zero_allocation = allocation_df[allocation_df["Value"] > 0]

        if not non_zero_allocation.empty:
            # Display as table
            allocation_display = non_zero_allocation.copy()
            allocation_display["Value"] = allocation_display["Value"].apply(
                lambda x: f"${x:,.2f}"
            )
            allocation_display["Percentage"] = allocation_display["Percentage"].apply(
                lambda x: f"{x:.2f}%"
            )
            allocation_display = allocation_display.rename(
                columns={"Value": "Value (USD)", "Percentage": "Allocation %"}
            )

            st.dataframe(allocation_display, use_container_width=True)

            # Visual representation (only if we have non-zero values)
            if len(non_zero_allocation) > 1 or non_zero_allocation.iloc[0]["Value"] > 0:
                st.bar_chart(non_zero_allocation.set_index("Wallet")["Value"])
        else:
            st.info("No wallet values to display.")
    else:
        st.info("No portfolio value to display allocation.")


def render_wallets_page(dashboard):
    """Render the wallets page with comprehensive wallet information."""
    tracker = dashboard.initialize_tracker()

    if not tracker:
        st.error("❌ Failed to initialize tracker.")
        return

    # Initialize page state
    ui_utils.initialize_page_state("wallets")

    st.markdown("## 💳 Wallets")

    # Display trading status banner
    is_live = dashboard.config_manager.is_live
    is_testnet = dashboard.config_manager.is_testnet_mode
    render_trading_status_banner(is_live, is_testnet)

    # Get portfolio metrics
    metrics = st.session_state.get("portfolio_metrics")
    if not metrics:
        st.info("No portfolio metrics available. Please run a full sync first.")
        if st.button("🔄 Run Full Sync Now"):
            if dashboard.offline_mode:
                st.error("You're offline")
                return
            dashboard.run_full_sync()
            st.rerun()
        return

    # Create tabs for different wallet views
    tab_overview, tab_details, tab_actions = st.tabs(
        ["📊 Overview", "🔍 Details", "⚡ Actions"]
    )

    # Overview Tab
    with tab_overview:
        st.header("Portfolio Wallet Overview")
        _display_wallet_balances(metrics, dashboard)
        _display_wallet_allocation(metrics)

    # Details Tab
    with tab_details:
        st.header("Wallet Details")

        # Spot & Earn Details
        st.markdown("### 📍 Spot & Earn Details")
        holdings_df = (
            parse_df_string(metrics.get("holdings_df"))
            if isinstance(metrics.get("holdings_df"), str)
            else metrics.get("holdings_df")
        )

        if holdings_df is not None and not holdings_df.empty:
            # Display full holdings table with all details
            st.markdown("**Complete Holdings Information**")

            # Filter for non-zero holdings
            non_zero_holdings = holdings_df[holdings_df["total_quantity"] > 1e-8].copy()

            if not non_zero_holdings.empty:
                # Create a comprehensive display DataFrame with only the most relevant columns
                display_columns = [
                    "symbol",
                    "total_quantity",
                    "spot_quantity",
                    "earn_quantity",
                    "value_usd",
                    "average_cost_basis",
                    "cost_basis_total",
                    "unrealized_pl_usd",
                    "unrealized_pl_percent",
                ]

                # Only include columns that exist in the DataFrame
                available_columns = [
                    col for col in display_columns if col in non_zero_holdings.columns
                ]
                display_df = non_zero_holdings[available_columns].copy()

                # Rename columns for clarity
                column_renames = {
                    "symbol": "Asset",
                    "total_quantity": "Total Quantity",
                    "spot_quantity": "Spot Quantity",
                    "earn_quantity": "Earn Quantity",
                    "value_usd": "Value (USD)",
                    "average_cost_basis": "Avg Cost Basis",
                    "cost_basis_total": "Total Cost Basis",
                    "unrealized_pl_usd": "Unrealized P/L (USD)",
                    "unrealized_pl_percent": "Unrealized P/L %",
                }
                display_df = display_df.rename(columns=column_renames)

                # Format numeric columns
                for col in display_df.columns:
                    if col != "Asset":  # Don't format the asset name column
                        if "Quantity" in col:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{float(x):,.8f}"
                                if pd.notna(x)
                                else "0.00000000"
                            )
                        elif "USD" in col or "Cost Basis" in col:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"${float(x):,.2f}"
                                if pd.notna(x)
                                else "$0.00"
                            )
                        elif "P/L %" in col:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{float(x):+.2f}%"
                                if pd.notna(x)
                                else "0.00%"
                            )
                        else:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{float(x):,.2f}" if pd.notna(x) else "0.00"
                            )

                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No non-zero holdings found.")
        else:
            st.info("No holdings data available.")

        # Futures Details
        st.markdown("### 💹 Futures Details")
        futures_balances = metrics.get("futures_balances", [])
        if futures_balances:
            futures_df = pd.DataFrame(futures_balances)
            if not futures_df.empty:
                # Filter for non-zero balances and select only relevant columns
                if "balance" in futures_df.columns:
                    futures_df["balance"] = pd.to_numeric(
                        futures_df["balance"], errors="coerce"
                    )
                    non_zero_futures = futures_df[futures_df["balance"] > 1e-8]
                else:
                    # If balance column doesn't exist, show all data
                    non_zero_futures = futures_df

                if not non_zero_futures.empty:
                    # Select only the most relevant columns for display
                    display_columns = ["asset", "balance"]
                    # Add other columns if they exist and have meaningful data
                    for col in ["crossWalletBalance", "availableBalance", "crossUnPnl"]:
                        if col in non_zero_futures.columns:
                            display_columns.append(col)

                    futures_display = non_zero_futures[display_columns].copy()

                    # Rename columns for clarity
                    column_renames = {
                        "asset": "Asset",
                        "balance": "Balance",
                        "crossWalletBalance": "Cross Wallet Balance",
                        "availableBalance": "Available Balance",
                        "crossUnPnl": "Cross Unrealized PnL",
                    }
                    futures_display = futures_display.rename(columns=column_renames)

                    # Format numeric columns
                    for col in futures_display.columns:
                        if col != "Asset":  # Don't format the asset name column
                            futures_display[col] = futures_display[col].apply(
                                lambda x: f"{float(x):,.8f}"
                                if pd.notna(x)
                                else "0.00000000"
                            )

                    st.dataframe(futures_display, use_container_width=True)
                else:
                    st.info("No non-zero balances in Futures wallet.")
            else:
                st.info("No Futures data available.")
        else:
            st.info("No Futures data available.")

        # Funding Details
        st.markdown("### 💵 Funding Details")
        funding_balances = metrics.get("funding_balances", [])
        if funding_balances:
            funding_df = pd.DataFrame(funding_balances)
            if not funding_df.empty:
                # Filter for non-zero balances
                if "free" in funding_df.columns:
                    funding_df["free"] = pd.to_numeric(
                        funding_df["free"], errors="coerce"
                    )
                    non_zero_funding = funding_df[funding_df["free"] > 1e-8]
                else:
                    # If free column doesn't exist, check the 'balance' column
                    if "balance" in funding_df.columns:
                        funding_df["balance"] = pd.to_numeric(
                            funding_df["balance"], errors="coerce"
                        )
                        non_zero_funding = funding_df[funding_df["balance"] > 1e-8]
                    else:
                        # If neither column exists, show all data
                        non_zero_funding = funding_df

                if not non_zero_funding.empty:
                    # Select only the most relevant columns for display
                    if (
                        "asset" in non_zero_funding.columns
                        and "free" in non_zero_funding.columns
                    ):
                        funding_display = non_zero_funding[["asset", "free"]].copy()
                        funding_display = funding_display.rename(
                            columns={"asset": "Asset", "free": "Available Balance"}
                        )
                    elif (
                        "asset" in non_zero_funding.columns
                        and "balance" in non_zero_funding.columns
                    ):
                        funding_display = non_zero_funding[["asset", "balance"]].copy()
                        funding_display = funding_display.rename(
                            columns={"asset": "Asset", "balance": "Available Balance"}
                        )
                    else:
                        # If we can't find the right columns, show what we have
                        funding_display = non_zero_funding.copy()

                    # Format numeric columns
                    for col in funding_display.columns:
                        if col != "Asset":  # Don't format the asset name column
                            funding_display[col] = funding_display[col].apply(
                                lambda x: f"{float(x):,.8f}"
                                if pd.notna(x)
                                else "0.00000000"
                            )

                    st.dataframe(funding_display, use_container_width=True)
                else:
                    st.info("No non-zero balances in Funding wallet.")
            else:
                st.info("No Funding data available.")
        else:
            st.info("No Funding data available.")

    # Actions Tab
    with tab_actions:
        st.header("Wallet Actions")

        # Transfer Funds Section
        st.markdown("### 🔄 Transfer Funds")
        st.markdown("Transfer funds between your Binance wallets.")

        # Use the enhanced transfer widget
        render_transfer_widget(dashboard, context="Wallets Actions")

        # Redeem from Earn Section
        st.markdown("### 💸 Redeem from Earn")
        st.markdown("Redeem your assets from Binance Earn products.")

        # Check if there are any earn holdings to display context
        holdings_df = (
            parse_df_string(metrics.get("holdings_df"))
            if isinstance(metrics.get("holdings_df"), str)
            else metrics.get("holdings_df")
        )

        if holdings_df is not None and not holdings_df.empty:
            earn_holdings = holdings_df[holdings_df["earn_quantity"] > 1e-8].copy()
            if not earn_holdings.empty:
                st.info(
                    f"You have {len(earn_holdings)} assets in Binance Earn products."
                )
            else:
                st.info("No assets currently in Binance Earn products.")
        else:
            st.info("Unable to determine Earn product holdings.")

        render_redeem_widget(dashboard, context="Wallets Actions")

        # Wallet History Section
        st.markdown("### 📜 Wallet History")
        st.markdown("View transaction history for each wallet.")
        st.info("Wallet history functionality will be implemented in a future update.")
