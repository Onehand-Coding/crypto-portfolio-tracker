import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from crypto_portfolio_tracker.visualizations import Visualizer
from crypto_portfolio_tracker.utils import format_percent, format_usd, build_holdings_table, calculate_fifo_realized_gains, parse_df_string, clean_futures_balances, clean_funding_balances


def _display_metrics(metrics):
    """
    Displays key portfolio metrics in a professional grid layout using st.metric.
    """
    if not metrics:
        st.info("No portfolio metrics available. Please sync first.")
        return

    # --- Value Breakdown ---
    st.markdown("#### 💳 Wallet & Capital Breakdown")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_wallet_value = (
            metrics.get("spot_earn_value_usd", 0)
            + metrics.get("futures_value_usd", 0)
            + metrics.get("funding_value_usd", 0)
        )
        st.metric(
            label=" Total Wallet Value",  # Spot & Earn Value + Futures + Funding
            value=format_usd(total_wallet_value),
            help="Sum of all wallet values",
        )

    with col2:
        st.metric(
            label="Spot Wallet Value",  # Note assets in earn is just assets from spot that are earning interest
            value=format_usd(metrics.get("spot_earn_value_usd", 0)),
            help="Spot and Earn value",
        )

    with col3:
        st.metric(
            label="Futures Wallet Value",
            value=format_usd(metrics.get("futures_value_usd", 0)),
            help="Futures value",
        )

    with col4:
        st.metric(
            label="Funding Wallet Value",
            value=format_usd(metrics.get("funding_value_usd", 0)),
            help="Funding value",
        )

    st.markdown("---")  # Visual separator

    # --- Holdings DataFrames ---
    # All Holdings
    st.markdown("#### 🗂️ All Holdings (Alloc. = % of spot/earn portfolio value)")
    if isinstance(metrics.get("holdings_df"), str):
        all_df = parse_df_string(metrics["holdings_df"])
    else:
        all_df = metrics.get("holdings_df")
    if all_df is not None:
        st.dataframe(
            build_holdings_table(all_df, alloc_col="allocation"),
            use_container_width=True,
        )
    else:
        st.info("No holdings data available.")

    # Core Holdings
    st.markdown("#### 🎯 Core Holdings (Alloc. = % of core portfolio value)")
    if isinstance(metrics.get("core_holdings_df"), str):
        core_df = parse_df_string(metrics["core_holdings_df"])
    else:
        core_df = metrics.get("core_holdings_df")
    if core_df is not None:
        st.dataframe(
            build_holdings_table(core_df, alloc_col="core_allocation"),
            use_container_width=True,
        )
    else:
        st.info("No core holdings data available.")

    # Other Holdings
    st.markdown("#### 📈 Other Holdings (Alloc. = % of spot/earn portfolio value)")
    if isinstance(metrics.get("other_holdings_df"), str):
        other_df = parse_df_string(metrics["other_holdings_df"])
    else:
        other_df = metrics.get("other_holdings_df")
    if other_df is not None:
        st.dataframe(
            build_holdings_table(other_df, alloc_col="allocation"),
            use_container_width=True,
        )
    else:
        st.info("No other holdings data available.")

    # Futures Wallet Summary
    st.markdown("#### 💹 Futures Wallet Summary")
    if metrics.get("futures_balances"):
        fut_df = pd.DataFrame(metrics["futures_balances"])
        fut_clean = clean_futures_balances(fut_df)
        if fut_clean is not None and not fut_clean.empty:
            st.dataframe(fut_clean, use_container_width=True)
        else:
            st.info("No futures balances found.")
    else:
        st.info("No futures balances found.")

    # Funding Wallet Summary
    st.markdown("#### 💰 Funding Wallet Summary")
    if metrics.get("funding_balances"):
        fund_df = pd.DataFrame(metrics["funding_balances"])
        fund_clean = clean_funding_balances(fund_df)
        if fund_clean is not None and not fund_clean.empty:
            st.dataframe(fund_clean, use_container_width=True)
        else:
            st.info("No funding balances found.")
    else:
        st.info("No funding balances found.")


def render_home_page(dashboard):
    tracker = dashboard.initialize_tracker()
    st.markdown("## 🏠 Home")

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

    # --- Row 1: Top-Level Metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Invested Capital",
            value=format_usd(metrics.get("total_invested_capital", 0)),
            help="The total amount of USD you have put into the portfolio.",
        )

    with col2:
        st.metric(
            label="Total Cost Basis",
            value=format_usd(metrics.get("total_cost_basis_usd", 0)),
            help="The total amount of USD you have put into the portfolio from FIFO.",
        )

    with col3:
        st.metric(
            label="Total Portfolio Value",
            value=format_usd(metrics.get("total_value_usd", 0)),
            help="Current portfolio total value.",
        )

    with col4:
        st.metric(
            label="Overall P/L",
            value=format_usd(metrics.get("overall_pl_usd", 0)),
            delta=format_percent(metrics.get("overall_pl_percent", 0)),
            help="Profit/Loss based on Total Invested Capital.",
        )

    with col5:
        st.metric(
            label="Unrealized P/L (FIFO)",
            value=format_usd(metrics.get("unrealized_pl_usd", 0)),
            delta=format_percent(metrics.get("unrealized_pl_percent", 0)),
            help="Profit/Loss based on FIFO Cost Basis.",
        )

    # --- Load data required for the home page ---
    holdings_df = (
        parse_df_string(metrics.get("holdings_df"))
        if isinstance(metrics.get("holdings_df"), str)
        else metrics.get("holdings_df")
    )
    target_allocation = dashboard.config_manager.config.get("target_allocation", {})
    try:
        snapshots = tracker.db_manager.get_all_snapshots()
        snapshots = snapshots.drop_duplicates(subset=["timestamp"])

        # Set timestamp as index for proper charting
        if not snapshots.empty and "timestamp" in snapshots.columns:
            # Convert timestamp to datetime first
            snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"])
            snapshots = snapshots.set_index("timestamp")
        else:
            st.write("No timestamp column found or empty snapshots")

    except Exception as e:
        st.error(f"Failed to load snapshots: {e}")
        snapshots = None

    # --- Tab layout ---
    tab_perf, tab_viz, tab_acts, tab_export = st.tabs(
        [
            "📊 Performance",
            "📈 Visualizations",
            "📖 Activities",
            "📂 View Exports",
        ]
    )

    # --- 1. Portfolio Performance Tab ---
    with tab_perf:
        st.header("📊 Portfolio Performance")
        _display_metrics(metrics)
        st.markdown("---")
        st.markdown("#### 📊 Export Portfolio Summary")
        export_format = st.radio(
            "Export Format",
            options=["Excel", "HTML", "CSV"],
            format_func=lambda x: {
                "Excel": "📗 Excel",
                "HTML": "🌐 HTML",
                "CSV": "📄 CSV",
            }[x],
            horizontal=True,
        )

        if st.button("Export Portfolio Summary"):
            if export_format == "Excel":
                tracker.export_portfolio_summary(metrics, "Excel")
                st.success("Portfolio Summary exported to Excel!")
            elif export_format == "HTML":
                tracker.export_portfolio_summary(metrics, "HTML")
                st.success("Portfolio Summary exported to HTML!")
            elif export_format == "CSV":
                tracker.export_portfolio_summary(metrics, "CSV")
                st.success("Portfolio Summary exported to CSV!")

    # --- 2. Visualizations Tab ---
    with tab_viz:
        st.header("📈 Visualizations")

        # Initialize visualizer
        visualizer = Visualizer(dashboard.config_manager.config)

        chart_options = [
            "Portfolio Allocation Pie",
            "Current vs. Target Allocation",
            "Unrealized P/L by Asset",
            "Portfolio Value History",
        ]

        chart_choice = st.selectbox("Select Chart Type", chart_options)

        if chart_choice == "Portfolio Allocation Pie":
            st.subheader("Portfolio Allocation Pie")
            if holdings_df is not None and not holdings_df.empty:
                # Interactive version
                fig_interactive = visualizer.create_interactive_allocation_pie(
                    holdings_df
                )
                st.plotly_chart(fig_interactive, use_container_width=True)

                # Add export button that saves to exports directory
                if st.button("Export Chart", key="export_allocation_pie"):
                    visualizer.create_portfolio_allocation_pie(
                        holdings_df, metrics, save_to_disk=True
                    )
                    st.success("✅ Portfolio Allocation Pie Chart exported!")
            else:
                st.info("No holdings data available for allocation pie chart.")

        elif chart_choice == "Current vs. Target Allocation":
            st.subheader("Current vs. Target Allocation")
            if (
                holdings_df is not None
                and not holdings_df.empty
                and target_allocation
            ):
                # Interactive version
                fig_interactive = (
                    visualizer.create_interactive_allocation_comparison(
                        holdings_df, target_allocation
                    )
                )
                st.plotly_chart(fig_interactive, use_container_width=True)

                # Add export button that saves to exports directory
                if st.button("Export Chart", key="export_allocation_comparison"):
                    visualizer.create_allocation_comparison_bar(
                        holdings_df, target_allocation, save_to_disk=True
                    )
                    st.success("✅ Current vs. Target Allocation Chart exported!")
            else:
                st.info("No data available for allocation comparison chart.")

        elif chart_choice == "Unrealized P/L by Asset":
            st.subheader("Unrealized Profit/Loss (P/L) by Asset")
            if (
                holdings_df is not None
                and not holdings_df.empty
                and "unrealized_pl_usd" in holdings_df.columns
            ):
                # Interactive version
                fig_interactive = visualizer.create_interactive_pl_by_asset(
                    holdings_df
                )
                st.plotly_chart(fig_interactive, use_container_width=True)

                # Add export button that saves to exports directory
                if st.button("Export Chart", key="export_pl_by_asset"):
                    visualizer.create_pl_by_asset_bar(
                        holdings_df, save_to_disk=True
                    )
                    st.success("✅ Unrealized P/L by Asset Chart exported!")
            else:
                st.info("No P/L data available for chart.")

        elif chart_choice == "Portfolio Value History":
            st.subheader("Portfolio Value History")
            if snapshots is not None and not snapshots.empty:
                # Interactive version
                fig_interactive = visualizer.create_interactive_value_history(
                    snapshots
                )
                st.plotly_chart(fig_interactive, use_container_width=True)

                # Add export button that saves to exports directory
                if st.button("Export Chart", key="export_value_history"):
                    visualizer.create_portfolio_value_history(
                        snapshots, save_to_disk=True
                    )
                    st.success("✅ Portfolio Value History Chart exported!")
            else:
                st.info("No snapshot data available for value history chart.")

    # --- 3. Activities Tab ---
    with tab_acts:
        st.header("📖 Activities")
        st.markdown("#### 📝 Tax Report")
        db_manager = tracker.db_manager
        try:
            tx_df = db_manager.get_all_transactions()
            if tx_df.empty:
                st.info("No transactions found to generate a report.")
            else:
                tax_df = calculate_fifo_realized_gains(tx_df)
                if tax_df.empty:
                    st.info("No taxable events (sales) found.")
                else:
                    # Remove timezone from datetime columns before processing
                    if "date" in tax_df.columns and hasattr(tax_df["date"], "dt"):
                        tax_df["date"] = tax_df["date"].dt.tz_localize(None)

                    tax_df["year"] = pd.to_datetime(tax_df["date"]).dt.year

                    # Rename columns for user-friendliness
                    tax_df_renamed = tax_df.rename(
                        columns={
                            "date": "Date",
                            "symbol": "Asset",
                            "quantity": "Quantity",
                            "proceeds_usd": "Proceeds (USD)",
                            "cost_basis_usd": "Cost Basis (USD)",
                            "gain_usd": "Gain/Loss (USD)",
                            "year": "Year",
                        }
                    )

                    st.dataframe(tax_df_renamed, use_container_width=True)

                    st.markdown("#### 📋 Summary of Realized Gains")
                    summary = (
                        tax_df.groupby("symbol")
                        .agg(
                            total_gain_usd=("gain_usd", "sum"),
                            total_proceeds_usd=("proceeds_usd", "sum"),
                            total_cost_basis_usd=("cost_basis_usd", "sum"),
                        )
                        .reset_index()
                    )

                    # Rename columns for summary
                    summary_renamed = summary.rename(
                        columns={
                            "symbol": "Asset",
                            "total_gain_usd": "Total Gain/Loss (USD)",
                            "total_proceeds_usd": "Total Proceeds (USD)",
                            "total_cost_basis_usd": "Total Cost Basis (USD)",
                        }
                    )

                    st.dataframe(summary_renamed, use_container_width=True)

                    st.markdown("---")
                    export_dir = Path(
                        dashboard.config_manager.config.get("exports", {}).get(
                            "path", "data/exports/"
                        )
                    )

                    export_dir.mkdir(parents=True, exist_ok=True)
                    if st.button("Export Full Tax Report to Excel"):
                        # Make sure all datetime columns are timezone-unaware before export
                        datetime_cols = tax_df.select_dtypes(
                            include=["datetime64[ns, UTC]"]
                        ).columns
                        for col in datetime_cols:
                            tax_df[col] = tax_df[col].dt.tz_localize(None)

                        filename = f"tax_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        tax_df.to_excel(export_dir / filename, index=False)
                        st.success(f"Excel tax report exported as {filename}!")
        except Exception as e:
            st.error(f"Failed to generate tax report: {e}")

        st.markdown("---")
        st.markdown("#### 🧾 Trade Log")
        db_manager = tracker.db_manager
        try:
            tx_df = db_manager.get_all_transactions()
            if tx_df.empty:
                st.info("No transactions found.")
            else:
                # Clean up columns for display
                col_map = {
                    "timestamp": "Date/Time",
                    "symbol": "Asset",
                    "type": "Type",
                    "quantity": "Quantity",
                    "price": "Price (USD)",
                    "fee_usd": "Fee (USD)",
                    "side": "Side",
                    "notes": "Notes",
                    "source": "Trade Source",
                }
                desired_order = [
                    "Date/Time",
                    "Asset",
                    "Type",
                    "Side",
                    "Quantity",
                    "Price (USD)",
                    "Fee (USD)",
                    "Trade Source",
                    "Notes",
                ]
                drop_cols = [
                    col for col in ["id", "asset_id"] if col in tx_df.columns
                ]
                display_df = tx_df.drop(columns=drop_cols, errors="ignore").rename(
                    columns=col_map
                )

                ordered_cols = [
                    col for col in desired_order if col in display_df.columns
                ]
                display_df = display_df[ordered_cols].reset_index(drop=True)

                st.dataframe(display_df, use_container_width=True)
                st.download_button(
                    "Download Trade Log (CSV)",
                    tx_df.to_csv(index=False),
                    "trade_log.csv",
                )
        except Exception as e:
            st.error(f"Failed to load transactions: {e}")

    # --- 4. View Exports Tab ---
    with tab_export:
        st.header("📂 View Exported Data")
        export_dir = Path(
            dashboard.config_manager.config.get("exports", {}).get(
                "path", "data/exports/"
            )
        )
        charts_dir = export_dir / "charts"
        export_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Create tabs for different export types
        export_tab1, export_tab2 = st.tabs(["📄 Data Exports", "📈 Chart Exports"])

        # --- Data Exports Tab ---
        with export_tab1:
            files = sorted(
                [
                    f for f in export_dir.glob("*.*") if f.parent == export_dir
                ],  # Only files in main export dir
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            if not files:
                st.info("No exported data files found.")
            else:
                file_names = [f.name for f in files]
                selected_file = st.selectbox(
                    "Select Exported File", file_names, key="data_files"
                )
                file_path = export_dir / selected_file

                # --- Preview logic ---
                try:
                    if selected_file.endswith((".csv", ".xlsx")):
                        df = (
                            pd.read_csv(file_path)
                            if selected_file.endswith(".csv")
                            else pd.read_excel(file_path)
                        )
                        st.dataframe(df)
                    elif selected_file.endswith(".html"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            html = f.read()
                        st.components.v1.html(html, height=600, scrolling=True)
                    elif selected_file.endswith(".json"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        st.json(data)
                    else:
                        st.info("Preview not supported for this file type.")
                except Exception as e:
                    st.error(f"Failed to preview file: {e}")

                # --- Action Buttons ---
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    with open(file_path, "rb") as f:
                        st.download_button(
                            "Download File",
                            f,
                            file_name=selected_file,
                            use_container_width=True,
                        )
                with col2:
                    if st.button(
                        "Delete File",
                        key=f"delete_data_{selected_file}",
                        use_container_width=True,
                        type="primary",
                    ):
                        try:
                            file_path.unlink()
                            st.success(f"Deleted {selected_file}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

        # --- Chart Exports Tab ---
        with export_tab2:
            chart_files = sorted(
                [f for f in charts_dir.glob("*.png")],  # Only PNG chart files
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            if not chart_files:
                st.info("No exported chart files found.")
            else:
                chart_names = [f.name for f in chart_files]
                selected_chart = st.selectbox(
                    "Select Chart File", chart_names, key="chart_files"
                )
                chart_path = charts_dir / selected_chart

                # --- Chart Preview ---
                st.markdown(f"### 📊 {selected_chart}")
                try:
                    st.image(chart_path, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to display chart: {e}")

                # --- Action Buttons ---
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    with open(chart_path, "rb") as f:
                        st.download_button(
                            "Download Chart",
                            f,
                            file_name=selected_chart,
                            use_container_width=True,
                        )
                with col2:
                    if st.button(
                        "Delete Chart",
                        key=f"delete_chart_{selected_chart}",
                        use_container_width=True,
                        type="primary",
                    ):
                        try:
                            chart_path.unlink()
                            st.success(f"Deleted {selected_chart}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
