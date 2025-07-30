#!/usr/bin/env python3
import io
import os
import json
import inspect
import asyncio
import logging
import warnings
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.dates as mdates
from jinja2 import Environment, FileSystemLoader

from crypto_portfolio_tracker import trading_strategies
from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.strategy_backtester import StrategyBacktester
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker
from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer
from crypto_portfolio_tracker.rebalancing_backtester import RebalancingBacktester
from crypto_portfolio_tracker.exceptions import (
    NetworkOperationError,
    NetworkUnavailableError,
)
from crypto_portfolio_tracker.utils import (
    parse_df_string,
    format_percent,
    format_usd,
    format_qty,
    build_holdings_table,
    clean_futures_balances,
    clean_funding_balances,
    clean_export_df,
    calculate_fifo_realized_gains,
    calculate_fifo_cost_basis,
)
from crypto_portfolio_tracker.visualizations import Visualizer

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

# Configure Streamlit page
st.set_page_config(
    page_title="Crypto Portfolio Tracker",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .status-online {
        color: #28a745;
    }
    .status-offline {
        color: #dc3545;
    }
    .sidebar-section {
        background: transparent;
        padding: 0.25rem 0;
        border-radius: 0;
        margin-bottom: 0.5rem;
    }
    .sidebar-section h3, .sidebar-section h4 {
        margin: 0;
        font-weight: 700;
        color: #fff;
        font-size: 1.15rem;
        letter-spacing: 0.01em;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .sidebar .stMarkdown hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class PortfolioDashboard:
    """Main Streamlit Dashboard class"""

    def __init__(self):
        self.config_manager = None
        self.tracker = None
        self.offline_mode = False
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "tracker_initialized" not in st.session_state:
            st.session_state.tracker_initialized = False
        if "offline_mode" not in st.session_state:
            st.session_state.offline_mode = False
        if "last_sync" not in st.session_state:
            st.session_state.last_sync = None
        if "portfolio_metrics" not in st.session_state:
            st.session_state.portfolio_metrics = None
        if "confirm_delete" not in st.session_state:
            st.session_state["confirm_delete"] = None

    def setup_logging(self, level_override: Optional[str] = None):
        """Setup logging configuration"""
        try:
            self.config_manager = ConfigManager()
            logging_config = self.config_manager.config.get("logging", {})
            log_level_str = (
                level_override or logging_config.get("level", "INFO").upper()
            )
            log_level = getattr(logging, log_level_str, logging.INFO)
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logging.getLogger("httpx").setLevel(logging.WARNING)
        except Exception as e:
            st.error(f"Failed to setup logging: {str(e)}")

    def initialize_tracker(self):
        """Initialize the portfolio tracker and always set self.tracker"""
        if (
            st.session_state.get("tracker_initialized", False)
            and self.tracker is not None
        ):
            return self.tracker
        try:
            with st.spinner("Initializing Portfolio Tracker..."):
                self.config_manager = ConfigManager()
                self.tracker = CryptoPortfolioTracker(self.config_manager)
                st.session_state.tracker_initialized = True
                st.session_state.offline_mode = False
                return self.tracker
        except NetworkUnavailableError:
            st.warning("⚠️ Network appears unavailable. Running in offline mode.")
            try:
                self.tracker = CryptoPortfolioTracker(
                    self.config_manager, force_offline=True
                )
                st.session_state.tracker_initialized = True
                st.session_state.offline_mode = True
                self.offline_mode = True
                return self.tracker
            except Exception as e:
                st.error(f"Failed to initialize tracker in offline mode: {str(e)}")
                return None
        except Exception as e:
            st.error(f"Failed to initialize tracker: {str(e)}")
            return None

    def render_header(self):
        """Render the main header"""
        st.markdown(
            """
        <div class="main-header">
            <h1>🚀 Crypto Portfolio Tracker v2.1.0</h1>
            <p>Professional Portfolio Management & Analysis Dashboard</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_status_indicator(self):
        """Render connection status indicator"""
        if st.session_state.offline_mode:
            st.markdown(
                """
            <div style="text-align: center; padding: 0.5rem; background: #f8d7da; border-radius: 8px; margin-bottom: 1rem;">
                <span class="status-offline">⚠️ OFFLINE MODE - Network features are disabled</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style="text-align: center; padding: 0.5rem; background: #d4edda; border-radius: 8px; margin-bottom: 1rem;">
                <span class="status-online">✅ ONLINE - All features available</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def render_sidebar(self):
        """Render the sidebar with navigation and controls"""
        st.sidebar.markdown(
            """
        <div class="sidebar-section">
            <h3>📊 Dashboard Controls</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            [
                "🏠 Home",
                "📈 Market",
                "⚖️ Rebalance",
                "💰 Trade",
                "🧪 Backtest",
                "🗄️ Database",
                "⚙️ Settings",
            ],
        )

        st.sidebar.markdown(
            """
        <div class="sidebar-section">
            <h4>⚡ Quick Actions</h4>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.sidebar.button("🔄 Sync Portfolio"):
            self.run_full_sync()

        if st.sidebar.button("💾 Save Snapshot"):
            if st.session_state.portfolio_metrics:
                self.save_snapshot()
            else:
                st.sidebar.warning("No metrics available to save")

        # Last sync info
        if st.session_state.last_sync:
            st.sidebar.markdown(
                f"""
            <div class="sidebar-section">
                <small>Last Sync: {st.session_state.last_sync}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        return page

    def run_full_sync(self):
        """Run full portfolio sync and update session state with real metrics."""
        tracker = self.initialize_tracker()
        if not tracker:
            st.error("Tracker not initialized")
            return
        try:
            with st.spinner("Running full sync and analysis..."):
                metrics = asyncio.run(tracker.run_full_sync())
                st.session_state.portfolio_metrics = metrics
                st.session_state.last_sync = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.success("✅ Full sync completed successfully!")
        except Exception as e:
            st.error(f"Sync failed: {str(e)}")

    def save_snapshot(self):
        tracker = self.initialize_tracker()
        try:
            if st.session_state.portfolio_metrics:
                tracker.save_snapshot(st.session_state.portfolio_metrics)
                st.success("📸 Portfolio snapshot saved!")
            else:
                st.warning("No portfolio metrics available to save")
        except Exception as e:
            st.error(f"Failed to save snapshot: {str(e)}")

    def display_metrics(self, metrics):
        """
        Displays key portfolio metrics in a professional grid layout using st.metric.
        """
        if not metrics:
            st.info("No portfolio metrics available. Please sync first.")
            return

        # --- Row 1: Top-Level Metrics ---
        st.markdown("---")
        st.markdown("#### 📊 Overall Performance")
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

        st.markdown("---")  # Visual separator

        # --- Row 2: Value Breakdown ---
        st.markdown("#### 💳 Wallet & Capital Breakdown")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Spot & Earn Value",
                value=format_usd(metrics.get("spot_earn_value_usd", 0)),
            )

        with col2:
            st.metric(
                label="Futures Wallet Value",
                value=format_usd(metrics.get("futures_value_usd", 0)),
            )

        with col3:
            st.metric(
                label="Funding Wallet Value",
                value=format_usd(metrics.get("funding_value_usd", 0)),
            )

        st.markdown("---")  # Visual separator

        # --- Holdings DataFrames ---
        # All Holdings
        st.markdown("#### 🗂️ All Holdings")
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
        st.markdown("#### 📈 Other Holdings (Alloc. = % of total portfolio value)")
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

    def render_home_page(self):
        tracker = self.initialize_tracker()
        st.markdown("## 🏠 Home")

        metrics = st.session_state.get("portfolio_metrics")
        if not metrics:
            st.info("No portfolio metrics available. Please run a full sync first.")
            if st.button("🔄 Run Full Sync Now"):
                if self.offline_mode:
                    st.error("You're offline")
                self.run_full_sync()
                st.rerun()
            return

        # --- Load data required for the home page ---
        holdings_df = (
            parse_df_string(metrics.get("holdings_df"))
            if isinstance(metrics.get("holdings_df"), str)
            else metrics.get("holdings_df")
        )
        target_allocation = self.config_manager.config.get("target_allocation", {})
        try:
            snapshots = tracker.db_manager.get_all_snapshots()
            snapshots = snapshots.drop_duplicates(subset=["timestamp"])
            
            # Set timestamp as index for proper charting
            if not snapshots.empty and "timestamp" in snapshots.columns:
                # Convert timestamp to datetime first
                snapshots['timestamp'] = pd.to_datetime(snapshots['timestamp'])
                snapshots = snapshots.set_index("timestamp")
                st.write("Successfully set timestamp as index")
            else:
                st.write("No timestamp column found or empty snapshots")
            
        except Exception as e:
            st.error(f"Failed to load snapshots: {e}")
            snapshots = None

        # --- Tab layout ---
        tab_perf, tab_viz, tab_tax, tab_log, tab_export = st.tabs(
            [
                "📊 Performance",
                "📈 Visualizations",
                "🧾 Tax Report",
                "📝 Trade Log",
                "📂 View Exports",
            ]
        )

        # --- 1. Portfolio Performance Tab ---
        with tab_perf:
            st.header("📊 Portfolio Performance")
            self.display_metrics(metrics)  # Using the improved metric card layout

            st.markdown("---")
            st.info("For a full, professional report, use the export buttons below.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export Portfolio Summary (Excel)", use_container_width=True):
                    tracker.export_portfolio_summary(metrics, "Excel")
                    st.success("Portfolio Summary exported to Excel!")
            with col2:
                if st.button("Export Portfolio Summary (HTML)", use_container_width=True):
                    tracker.export_portfolio_summary(metrics, "HTML")
                    st.success("Portfolio Summary exported to HTML!")

        # --- 2. Visualizations Tab ---
        with tab_viz:
            st.header("📈 Visualizations")
            
            # Initialize visualizer
            visualizer = Visualizer(self.config_manager.config)
            
            chart_options = [
                "Portfolio Allocation Pie",
                "Current vs. Target Allocation",
                "Unrealized P/L by Asset",
                "Portfolio Value History"
            ]

            chart_choice = st.selectbox("Select Chart Type", chart_options)
            
            if chart_choice == "Portfolio Allocation Pie":
                st.subheader("Portfolio Allocation Pie")
                if holdings_df is not None and not holdings_df.empty:
                    # Interactive version
                    fig_interactive = visualizer.create_interactive_allocation_pie(holdings_df)
                    st.plotly_chart(fig_interactive, use_container_width=True)
                    
                    # Download buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        chart_bytes = visualizer.get_chart_bytes("allocation_pie", {
                            "holdings_df": holdings_df
                        })
                        if chart_bytes:
                            st.download_button(
                                                "Download PNG",
                                                chart_bytes,
                                file_name="portfolio_allocation_pie.png",
                                                mime="image/png"
                            )
                else:
                    st.info("No holdings data available for allocation pie chart.")

            elif chart_choice == "Current vs. Target Allocation":
                st.subheader("Current vs. Target Allocation")
                if (holdings_df is not None and not holdings_df.empty and target_allocation):
                    # Interactive version
                    fig_interactive = visualizer.create_interactive_allocation_comparison(
                        holdings_df, target_allocation
                    )
                    st.plotly_chart(fig_interactive, use_container_width=True)
                    
                    # Download buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        chart_bytes = visualizer.get_chart_bytes("allocation_comparison", {
                            "holdings_df": holdings_df,
                            "target_allocation": target_allocation
                        })
                        if chart_bytes:
                            st.download_button(
                                                "Download PNG",
                                                chart_bytes,
                                                file_name="allocation_comparison.png",
                                                mime="image/png"
                                            )
                else:
                    st.info("No data available for allocation comparison chart.")

            elif chart_choice == "Unrealized P/L by Asset":
                st.subheader("Unrealized Profit/Loss (P/L) by Asset")
                if (holdings_df is not None and not holdings_df.empty and 
                    "unrealized_pl_usd" in holdings_df.columns):
                    # Interactive version
                    fig_interactive = visualizer.create_interactive_pl_by_asset(holdings_df)
                    st.plotly_chart(fig_interactive, use_container_width=True)
                    
                    # Download buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        chart_bytes = visualizer.get_chart_bytes("pl_by_asset", {
                            "holdings_df": holdings_df
                        })
                        if chart_bytes:
                            st.download_button(
                                                "Download PNG",
                                                chart_bytes,
                                                file_name="pl_by_asset.png",
                                                mime="image/png"
                                            )
                else:
                    st.info("No P/L data available for chart.")

            elif chart_choice == "Portfolio Value History":
                st.subheader("Portfolio Value History")
                if snapshots is not None and not snapshots.empty:
                    # Interactive version
                    fig_interactive = visualizer.create_interactive_value_history(snapshots)
                    st.plotly_chart(fig_interactive, use_container_width=True)
                    
                    # Download buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        chart_bytes = visualizer.get_chart_bytes("value_history", {
                            "snapshots_df": snapshots
                        })
                        if chart_bytes:
                            st.download_button(
                                                "Download PNG",
                                                chart_bytes,
                                file_name="portfolio_value_history.png",
                                                mime="image/png"
                            )
                else:
                    st.info("No snapshot data available for value history chart.")

        # --- 3. Tax Report Tab ---
        with tab_tax:
            st.header("🧾 Tax Report")
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

                        st.markdown("#### Summary of Realized Gains")
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
                            self.config_manager.config.get("exports", {}).get(
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

        # --- 2. Trade Log Tab ---
        with tab_log:
            st.header("📝 Trade Log")
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
                self.config_manager.config.get("exports", {}).get(
                    "path", "data/exports/"
                )
            )
            export_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(
                [f for f in export_dir.glob("*.*")],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            if not files:
                st.info("No exported files found.")
            else:
                file_names = [f.name for f in files]
                selected_file = st.selectbox("Select Exported File", file_names)
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
                        key=f"delete_{selected_file}",
                        use_container_width=True,
                        type="primary",
                    ):
                        try:
                            file_path.unlink()
                            st.success(f"Deleted {selected_file}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
    def render_market_trends(self):
        st.markdown("## 📈 Market Trends")
        export_dir = Path(
            self.config_manager.config.get("exports", {}).get("path", "data/exports/")
        )
        export_dir.mkdir(parents=True, exist_ok=True)

        # 1. Define get_trend_report
        @st.cache_data(show_spinner="Generating trend report...", ttl=300)
        def get_trend_report(config, _binance_client, timeframe):
            from crypto_portfolio_tracker.crypto_trend_analyzer import (
                CryptoTrendAnalyzer,
            )

            analyzer = CryptoTrendAnalyzer(
                config=config, binance_client=_binance_client
            )
            return asyncio.run(analyzer.generate_report(timeframe))

        # 2. Define the chart plotting function
        def plot_coin_chart(symbol, timeframe, config, binance_client):
            import matplotlib.pyplot as plt
            from crypto_portfolio_tracker.crypto_trend_analyzer import (
                CryptoTrendAnalyzer,
            )
            import numpy as np
            import pandas as pd

            analyzer = CryptoTrendAnalyzer(config=config, binance_client=binance_client)
            settings = analyzer.timeframe_settings.get(timeframe)
            if not settings:
                st.error("No settings for this timeframe.")
                return

            period = settings.get("period", "1mo")
            interval = "1wk" if timeframe == "long_term" else "1d"
            data = asyncio.run(
                analyzer.fetch_crypto_data_async(symbol, period, interval)
            )
            if data is None or data.empty:
                st.warning(f"No data available for {symbol}.")
                return

            indicator_df = analyzer._calculate_indicators(data, settings)
            # Defensive: ensure index is datetime, sorted, unique, and not NaT
            if indicator_df is None or indicator_df.empty:
                st.warning("Not enough data for indicators.")
                return
            if not pd.api.types.is_datetime64_any_dtype(indicator_df.index):
                st.warning("Data index is not datetime. Cannot plot.")
                return
            indicator_df = indicator_df[~indicator_df.index.duplicated(keep="first")]
            indicator_df = indicator_df[~indicator_df.index.isna()]
            indicator_df = indicator_df.sort_index()

            # Require a minimum number of rows for meaningful plotting
            min_required_rows = 30
            if len(indicator_df) < min_required_rows:
                st.warning(
                    f"Not enough data to plot a meaningful chart (need at least {min_required_rows} rows, got {len(indicator_df)})."
                )
                return

            # Check for all-empty rows (all columns are NaN)
            if indicator_df.dropna(how="all").empty:
                st.warning(
                    "No valid data to plot (all rows are empty or all columns are NaN)."
                )
                return

            # Check for all-NaN in the columns you want to plot
            plot_cols = ["Close"]
            if all(
                indicator_df[col].dropna().empty
                for col in plot_cols
                if col in indicator_df.columns
            ):
                st.warning(
                    "No valid data to plot (all values in plot columns are NaN)."
                )
                return

            # Limit to last 500 rows for safety
            if len(indicator_df) > 500:
                indicator_df = indicator_df.tail(500)

            # Only now is it safe to plot!
            fig, axs = plt.subplots(
                3,
                1,
                figsize=(10, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 1, 1]},
            )
            fig.suptitle(
                f"{symbol} Price & Indicators ({timeframe.replace('_', ' ').title()})",
                fontsize=16,
            )

            # --- Price and SMAs ---
            axs[0].plot(
                indicator_df.index, indicator_df["Close"], label="Close", color="blue"
            )
            sma_short = settings.get("sma_short_window")
            sma_long = settings.get("sma_long_window")
            if sma_short and f"SMA_{sma_short}" in indicator_df.columns:
                axs[0].plot(
                    indicator_df.index,
                    indicator_df[f"SMA_{sma_short}"],
                    label=f"SMA {sma_short}",
                    color="orange",
                )
            if sma_long and f"SMA_{sma_long}" in indicator_df.columns:
                axs[0].plot(
                    indicator_df.index,
                    indicator_df[f"SMA_{sma_long}"],
                    label=f"SMA {sma_long}",
                    color="green",
                )
            # Support/Resistance (last value)
            if "Low" in indicator_df.columns and "High" in indicator_df.columns:
                window = 30 if len(indicator_df) > 30 else len(indicator_df)
                support = indicator_df["Low"].tail(window).min()
                resistance = indicator_df["High"].tail(window).max()
                axs[0].axhline(support, color="red", linestyle="--", label="Support")
                axs[0].axhline(
                    resistance, color="purple", linestyle="--", label="Resistance"
                )
            axs[0].set_ylabel("Price")
            axs[0].legend()
            axs[0].grid(True)

            # --- RSI ---
            rsi_col = f"RSI_{analyzer.rsi_period}"
            if rsi_col in indicator_df.columns:
                axs[1].plot(
                    indicator_df.index,
                    indicator_df[rsi_col],
                    label="RSI",
                    color="magenta",
                )
                axs[1].axhline(70, color="red", linestyle="--", linewidth=1)
                axs[1].axhline(30, color="green", linestyle="--", linewidth=1)
                axs[1].set_ylabel("RSI")
                axs[1].legend()
                axs[1].grid(True)
            else:
                axs[1].text(0.5, 0.5, "No RSI data", ha="center", va="center")

            # --- MACD ---
            macd_col = "MACD_12_26_9"
            macds_col = "MACDs_12_26_9"
            if macd_col in indicator_df.columns and macds_col in indicator_df.columns:
                axs[2].plot(
                    indicator_df.index,
                    indicator_df[macd_col],
                    label="MACD",
                    color="blue",
                )
                axs[2].plot(
                    indicator_df.index,
                    indicator_df[macds_col],
                    label="Signal",
                    color="orange",
                )
                axs[2].set_ylabel("MACD")
                axs[2].legend()
                axs[2].grid(True)
            else:
                axs[2].text(0.5, 0.5, "No MACD data", ha="center", va="center")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            st.pyplot(fig)

        # 3. Timeframe selector
        timeframe_map = {
            "Long-term (4 Years)": "long_term",
            "Swing (3 Months)": "swing",
            "Day (1 Month)": "day",
        }
        timeframe_label = st.radio(
            "Select timeframe for trend analysis:",
            list(timeframe_map.keys()),
            horizontal=True,
        )
        timeframe = timeframe_map[timeframe_label]

        tracker = self.initialize_tracker()
        report = get_trend_report(
            self.config_manager.config,
            getattr(tracker, "binance_client", None),
            timeframe,
        )

        if not report:
            st.error("Could not generate trend report. Please try again later.")
            return

        btc = report.get("benchmark_analysis", {})
        coin_analyses = report.get("coin_analyses", {})

        # --- Unified Chart Section ---
        st.markdown("### 📊 Coin Chart Viewer")

        # Get all available symbols (including BTC)
        all_symbols = list(coin_analyses.keys())
        if all_symbols:
            if "current_chart_symbol" not in st.session_state:
                st.session_state.current_chart_symbol = all_symbols[
                    0
                ]  # Default to first coin

            # Dropdown for selecting coin
            selected_symbol = st.selectbox(
                "Select coin to view chart:",
                all_symbols,
                key="unified_chart_coin_select",
                index=all_symbols.index(st.session_state.current_chart_symbol),
            )

            # Update session state with selected symbol
            st.session_state.current_chart_symbol = selected_symbol

            # Always render the chart when coins are available
            plot_coin_chart(
                st.session_state.current_chart_symbol,
                timeframe,
                self.config_manager.config,
                getattr(tracker, "binance_client", None),
            )
        else:
            st.info("No coin data available for charting.")

        # 5. Coin-by-Coin Table
        st.markdown("### 🪙 Coin Analysis")
        rows = []
        for symbol, analysis in coin_analyses.items():
            rows.append(
                {
                    "Symbol": symbol,
                    "Price": f"${analysis.get('current_price', 0):,.2f}",
                    "Change (%)": f"{analysis.get('price_change_pct', 0):+.2f}",
                    "RSI": f"{analysis.get('rsi', 0):,.2f}",
                    "Support": f"${analysis.get('support_level', 0):,.2f}",
                    "Resistance": f"${analysis.get('resistance_level', 0):,.2f}",
                    "Active Conditions": ", ".join(
                        analysis.get("active_conditions", [])
                    ),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No coin analysis data available.")

        # --- Prepare exportable data ---
        df_export = pd.DataFrame(
            [
                {
                    "Symbol": symbol,
                    "Price": analysis.get("current_price", 0),
                    "Change (%)": analysis.get("price_change_pct", 0),
                    "RSI": analysis.get("rsi", 0),
                    "Support": analysis.get("support_level", 0),
                    "Resistance": analysis.get("resistance_level", 0),
                    "Active Conditions": ", ".join(
                        analysis.get("active_conditions", [])
                    ),
                }
                for symbol, analysis in coin_analyses.items()
            ]
        )

        # --- Export Section ---
        st.markdown("### 📤 Analysis Exports")

        # Export creation controls
        with st.container():
            st.markdown("#### Create New Export")
            col_format, col_action = st.columns([2, 3])

            with col_format:
                export_format = st.radio(
                    "Format",
                    options=["CSV", "JSON", "HTML"],
                    format_func=lambda x: {
                        "CSV": "📄 CSV",
                        "JSON": "🔣 JSON",
                        "HTML": "🌐 HTML",
                    }[x],
                    horizontal=True,
                )

            with col_action:
                if st.button("🚀 Generate Export", use_container_width=True):
                    with st.spinner(f"Creating {export_format} export..."):
                        try:
                            # Use the unified export method from tracker
                            exported_file = self.tracker.export_trend_report(
                                report, timeframe, export_format
                            )
                            
                            if exported_file:
                                st.success(f"Successfully created {exported_file.name}")
                                st.rerun()
                            else:
                                st.error("Export failed")
                                
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")

        # Consolidated exports dropdown
        st.markdown("---")
        st.markdown("#### My Exports")

        # Get all export files sorted by modification time (newest first)
        all_files = sorted(
            list(export_dir.glob("trend_report_*.*")),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not all_files:
            st.info("No exports found. Generate one above!")
            return

        # Dropdown to select export file
        selected_export = st.selectbox(
            "Select an export file:",
            options=all_files,
            format_func=lambda x: f"{x.name} ({x.stat().st_size / 1024:.1f} KB, {datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})",
            index=0,
            help="Select an export file to view, download or delete",
        )

        # Initialize preview state if not exists
        if "preview_file" not in st.session_state:
            st.session_state.preview_file = None

        # File actions
        col_preview, col_download, col_delete = st.columns(3)

        with col_preview:
            if st.button("👁️ Preview", use_container_width=True):
                st.session_state.preview_file = selected_export
                st.rerun()

        with col_download:
            with open(selected_export, "rb") as f:
                file_bytes = f.read()
                st.download_button(
                    label="⬇️ Download",
                    data=file_bytes,
                    file_name=selected_export.name,
                    mime=f"text/{selected_export.suffix[1:].lower()}"
                    if selected_export.suffix != ".json"
                    else "application/json",
                    use_container_width=True,
                )

        with col_delete:
            if st.button("🗑️ Delete", use_container_width=True):
                try:
                    selected_export.unlink()
                    if st.session_state.preview_file == selected_export:
                        st.session_state.preview_file = None
                    st.success(f"Deleted {selected_export.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")

        # Preview panel
        if st.session_state.preview_file:
            preview_file = st.session_state.preview_file
            filetype = preview_file.suffix[1:].upper()

            st.markdown("---")
            st.markdown(f"### 👁️ Preview: {preview_file.name}")

            try:
                if filetype == "CSV":
                    st.dataframe(pd.read_csv(preview_file), use_container_width=True)
                elif filetype == "JSON":
                    st.json(json.load(open(preview_file)))
                elif filetype == "HTML":
                    st.components.v1.html(
                        preview_file.read_text(), height=800, scrolling=True
                    )
            except Exception as e:
                st.error(f"Could not preview file: {str(e)}")

            if st.button("❌ Close Preview"):
                st.session_state.preview_file = None
                st.rerun()

    def render_rebalancing(self):
        import asyncio

        st.markdown("## ⚖️ Portfolio Rebalancing")

        # Initialize session state
        if "rebalance_metrics" not in st.session_state:
            st.session_state.rebalance_metrics = None
        if "rebalance_suggestions" not in st.session_state:
            st.session_state.rebalance_suggestions = None
        if "rebalance_usdt" not in st.session_state:
            st.session_state.rebalance_usdt = None
        if "rebalance_actionable" not in st.session_state:
            st.session_state.rebalance_actionable = False
        if "rebalance_executing" not in st.session_state:
            st.session_state.rebalance_executing = False
        if "rebalance_results" not in st.session_state:
            st.session_state.rebalance_results = None
        if "accepted_trades" not in st.session_state:
            st.session_state.accepted_trades = set()

        # --- Generate Suggestions Section ---
        st.markdown("### 🔄 Portfolio Analysis")

        col1, col2 = st.columns([2, 1])
        with col1:
            generate_clicked = st.button(
                "🔍 Analyze Portfolio & Generate Suggestions",
                key="rebalance_generate_btn",
                disabled=st.session_state.rebalance_executing,
                use_container_width=True,
                type="primary",
            )

        with col2:
            if st.session_state.rebalance_metrics:
                last_update = datetime.now().strftime("%H:%M:%S")
                st.success(f"✅ Last updated: {last_update}")

        if generate_clicked or st.session_state.rebalance_metrics is None:
            with st.spinner(
                "🔄 Analyzing portfolio and generating rebalancing suggestions..."
            ):
                tracker = self.initialize_tracker()

                # 1. Get current metrics
                metrics = asyncio.run(tracker.calculate_portfolio_metrics())
                st.session_state.rebalance_metrics = metrics

                # 2. Get USDT balance (Spot + Earn)
                usdt_balance = 0.0
                try:
                    spot_balance = float(
                        tracker.binance_client.get_asset_balance(asset="USDT").get(
                            "free", 0.0
                        )
                    )
                    usdt_balance += spot_balance
                    if not self.config_manager.is_testnet_mode:
                        earn_positions = tracker.fetcher.fetch_simple_earn_balances(
                            pd.DataFrame([{"symbol": "USDT"}])
                        )
                        earn_balance = earn_positions.get("USDT", 0.0)
                        usdt_balance += earn_balance
                except Exception:
                    pass
                st.session_state.rebalance_usdt = usdt_balance

                # 3. Get rebalancing suggestions
                suggestions_df = asyncio.run(
                    tracker.get_core_portfolio_rebalance_suggestions_technical()
                )
                st.session_state.rebalance_suggestions = suggestions_df

                # 4. Check for actionable trades
                actionable = False
                if suggestions_df is not None and not suggestions_df.empty:
                    actionable = any(
                        s in ["BUY", "SELL"] for s in suggestions_df["Signal"]
                    )
                st.session_state.rebalance_actionable = actionable

                # Clear previous results and accepted trades
                st.session_state.rebalance_results = None
                st.session_state.accepted_trades = set()

        # --- Display Core Portfolio Status ---
        st.markdown("### 🎯 Current Core Portfolio Status")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Current Allocation")
            suggestions_df = st.session_state.rebalance_suggestions
            if suggestions_df is not None and not suggestions_df.empty:
                # Build the status table from suggestions for uniformity
                alloc_table = suggestions_df[
                    [
                        "Symbol",
                        "Current %",
                        "Target %",
                        "Drift (pts)",
                        "Current Value (USD)",
                    ]
                ].copy()
                alloc_table["Current %"] = alloc_table["Current %"].round(2)
                alloc_table["Target %"] = alloc_table["Target %"].round(2)
                alloc_table["Drift (pts)"] = alloc_table["Drift (pts)"].round(2)
                alloc_table["Current Value (USD)"] = alloc_table[
                    "Current Value (USD)"
                ].apply(lambda x: f"${x:,.2f}")
                alloc_table.columns = [
                    "Asset",
                    "Current %",
                    "Target %",
                    "Drift (pts)",
                    "Value (USD)",
                ]
                st.dataframe(alloc_table, use_container_width=True, hide_index=True)
                st.caption(
                    "Allocations are shown as a percentage of your core portfolio (assets being rebalanced)."
                )
            else:
                st.info("Portfolio analysis required")

        with col2:
            st.markdown("#### 💰 Available Liquidity")
            usdt = st.session_state.rebalance_usdt
            if usdt is not None:
                st.metric("USDT Balance (Spot + Earn)", f"${usdt:,.2f}")
                if usdt < 100:
                    st.warning("⚠️ Low USDT balance may limit rebalancing options")
            else:
                st.success("✅ Sufficient liquidity for rebalancing")

        # --- Display Interactive Rebalancing Suggestions ---
        st.markdown("### 📝 Rebalancing Suggestions & Trade Selection")

        suggestions_df = st.session_state.rebalance_suggestions
        if suggestions_df is not None and not suggestions_df.empty:
            # Separate actionable and non-actionable trades
            actionable_trades = suggestions_df[
                suggestions_df["Signal"].isin(["BUY", "SELL"])
            ].copy()
            hold_trades = suggestions_df[suggestions_df["Signal"] == "HOLD"].copy()

            # Show summary stats first
            st.markdown("#### 📈 Analysis Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actionable Trades", len(actionable_trades))
            with col2:
                st.metric("Hold Positions", len(hold_trades))
            with col3:
                selected_count = len(st.session_state.accepted_trades)
                st.metric("Selected for Execution", selected_count)

            # Show actionable trades with enhanced selection interface
            if not actionable_trades.empty:
                st.markdown("#### 🎯 Actionable Trades")
                st.markdown(
                    "*Select the trades you want to execute by checking the boxes below:*"
                )

                # Add select all / deselect all buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                with col1:
                    if st.button(
                        "✅ Select All",
                        key="select_all_trades",
                        use_container_width=True,
                    ):
                        for idx, row in actionable_trades.iterrows():
                            trade_key = f"{row['Symbol']}_{row['Signal']}"
                            st.session_state.accepted_trades.add(trade_key)
                        st.rerun()

                with col2:
                    if st.button(
                        "❌ Deselect All",
                        key="deselect_all_trades",
                        use_container_width=True,
                    ):
                        st.session_state.accepted_trades.clear()
                        st.rerun()

                with col3:
                    if st.button(
                        "🔄 Refresh", key="refresh_selection", use_container_width=True
                    ):
                        st.rerun()

                st.markdown("---")

                # Enhanced trade cards
                for idx, row in actionable_trades.iterrows():
                    trade_key = f"{row['Symbol']}_{row['Signal']}"

                    # Create a card-like container for each trade
                    with st.container():
                        # Use colored border based on signal type
                        if row["Signal"] == "BUY":
                            st.markdown(
                                """
                            <div style="border-left: 4px solid #28a745; padding-left: 15px; margin: 10px 0;">
                            """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                            <div style="border-left: 4px solid #dc3545; padding-left: 15px; margin: 10px 0;">
                            """,
                                unsafe_allow_html=True,
                            )

                        col1, col2, col3, col4 = st.columns([1, 4, 2, 1])

                        with col1:
                            # Enhanced checkbox with styling
                            accepted = st.checkbox(
                                "Execute",
                                key=f"accept_{trade_key}",
                                value=trade_key in st.session_state.accepted_trades,
                                help=f"Check to include this {row['Signal']} order in execution",
                            )
                            if accepted:
                                st.session_state.accepted_trades.add(trade_key)
                            else:
                                st.session_state.accepted_trades.discard(trade_key)

                        with col2:
                            st.markdown(f"**{row['Symbol']}**")
                            # Enhanced trade details
                            drift_color = "🔴" if abs(row["Drift (pts)"]) > 5 else "🟡"
                            st.markdown(
                                f"**Drift:** {drift_color} {row['Drift (pts)']:.2f} pts"
                            )
                            st.markdown(
                                f"**Allocation:** {row['Current %']:.1f}% → {row['Target %']:.1f}%"
                            )

                            # Show the reason for the trade
                            if row["Signal"] == "BUY":
                                st.markdown(
                                    f"*📈 Underweight by {abs(row['Drift (pts)']):,.2f} percentage points*"
                                )
                            else:
                                st.markdown(
                                    f"*📉 Overweight by {abs(row['Drift (pts)']):,.2f} percentage points*"
                                )

                    with col3:
                        st.markdown("**Current Value**")
                        st.markdown(f"${row['Current Value (USD)']:,.0f}")

                        # Show the actual USDT amount for the trade
                        if row["Signal"] in ["BUY", "SELL"] and "action_usd_value" in row:
                            trade_amount = row["action_usd_value"]
                            if trade_amount > 0:
                                st.markdown("**Trade Amount**")
                                if row["Signal"] == "BUY":
                                    st.markdown(f"💵 **${trade_amount:,.2f} USDT**")
                                else:  # SELL
                                    st.markdown(f"💵 **${trade_amount:,.2f} USDT**")
                                
                                # Also show the coin quantity for SELL orders
                                if row["Signal"] == "SELL" and "action_coin_quantity" in row:
                                    coin_qty = row["action_coin_quantity"]
                                    if coin_qty > 0:
                                        st.markdown(f" **{coin_qty:.8f} {row['Symbol']}**")

                    with col4:
                        # Enhanced signal display
                        if row["Signal"] == "BUY":
                            st.markdown(
                                """
                                <div style="background: #d4edda; color: #155724; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;">
                                    📈 BUY
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div style="background: #f8d7da; color: #721c24; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;">
                                    📉 SELL
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

            # Show HOLD trades in a more compact format
            if not hold_trades.empty:
                with st.expander(
                    f"📊 Hold Positions ({len(hold_trades)} assets) - No Action Required",
                    expanded=False,
                ):
                    for idx, row in hold_trades.iterrows():
                        col1, col2, col3 = st.columns([1, 4, 1])
                        with col1:
                            st.markdown(
                                """
                            <div style="background: #e2e3e5; color: #495057; padding: 4px 8px; border-radius: 4px; text-align: center; font-size: 0.8em;">
                                HOLD
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col2:
                            st.markdown(f"**{row['Symbol']}** - Within target range")
                            st.markdown(
                                f"Drift: {row['Drift (pts)']:.2f} pts | Current: {row['Current %']:.1f}% (Target: {row['Target %']:.1f}%)"
                            )
                        with col3:
                            st.markdown(f"${row['Current Value (USD)']:,.0f}")

            # Enhanced selection summary
            st.markdown("#### 📋 Execution Summary")
            accepted_count = len(st.session_state.accepted_trades)
            if accepted_count > 0:
                # Show selected trades summary
                selected_trades = []
                for idx, row in actionable_trades.iterrows():
                    trade_key = f"{row['Symbol']}_{row['Signal']}"
                    if trade_key in st.session_state.accepted_trades:
                        selected_trades.append(f"**{row['Symbol']}** ({row['Signal']})")

                st.success(f"✅ **{accepted_count} trade(s) selected for execution:**")
                st.markdown(" • ".join(selected_trades))

                # Risk warning
                if accepted_count > 3:
                    st.warning(
                        "⚠️ **Large Rebalancing Operation**: You're about to execute multiple trades. Please review carefully."
                    )
            else:
                st.info(
                    "ℹ️ No trades selected. Check the boxes above to select trades for execution."
                )

        else:
            st.info("📊 Run portfolio analysis to see rebalancing suggestions here.")

        # --- Enhanced Execute Button Section ---
        st.markdown("### ⚡ Execute Selected Trades")

        # Execution controls
        col1, col2 = st.columns([3, 1])

        with col1:
            execute_enabled = (
                len(st.session_state.accepted_trades) > 0
                and not st.session_state.rebalance_executing
            )

            execute_clicked = st.button(
                f"🚀 Execute {len(st.session_state.accepted_trades)} Selected Trade(s)",
                key="rebalance_execute_btn",
                disabled=not execute_enabled,
                use_container_width=True,
                type="primary" if execute_enabled else "secondary",
            )

            # Show execution status
            if st.session_state.rebalance_executing:
                st.info("⏳ Execution in progress... Please wait.")
            elif len(st.session_state.accepted_trades) == 0:
                st.info("ℹ️ Select trades above to enable execution.")

        with col2:
            # Emergency stop button (if executing)
            if st.session_state.rebalance_executing:
                if st.button("🛑 Cancel", use_container_width=True, type="secondary"):
                    st.session_state.rebalance_executing = False
                    st.warning("⚠️ Execution cancelled by user.")
                    st.rerun()

        # --- Enhanced Execution Logic ---
        if execute_clicked:
            st.session_state.rebalance_executing = True
            st.session_state.rebalance_results = ""

            # Show execution progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("⚡ Executing selected rebalancing trades..."):
                try:
                    status_text.text("🔄 Initializing execution...")
                    progress_bar.progress(10)

                    tracker = self.initialize_tracker()
                    suggestions_df = st.session_state.rebalance_suggestions

                    # Filter to only accepted trades
                    accepted_suggestions = []
                    for idx, row in suggestions_df.iterrows():
                        if row["Signal"] in ["BUY", "SELL"]:
                            trade_key = f"{row['Symbol']}_{row['Signal']}"
                            if trade_key in st.session_state.accepted_trades:
                                accepted_suggestions.append(row)

                    progress_bar.progress(30)
                    status_text.text("📊 Preparing trade execution...")

                    if accepted_suggestions:
                        # Create filtered DataFrame
                        filtered_df = pd.DataFrame(accepted_suggestions)

                        progress_bar.progress(50)
                        status_text.text("💰 Fetching earn balances...")

                        # Fetch Earn balances for execution
                        earn_balances = {}
                        if not self.config_manager.is_testnet_mode:
                            spot_balances_df = tracker.fetch_binance_balances()
                            earn_balances = tracker.fetcher.fetch_simple_earn_balances(
                                spot_balances_df
                            )

                        progress_bar.progress(70)
                        status_text.text("🚀 Executing trades...")

                        # Run execution
                        import asyncio

                        result = asyncio.run(
                            tracker.execute_rebalancing_trades_core(
                                filtered_df,
                                earn_balances,
                                interactive=False,
                                auto_confirm=True,
                            )
                        )
                        output = "\n".join(result.messages)
                        errors_output = "\n".join(result.errors) if result.errors else ""

                        if result.success:
                            st.session_state.rebalance_results = f"""=== REBALANCING EXECUTION COMPLETED ===
Executed {result.data.get("trades_executed", 0)} trade(s)
    Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{output}
    === END EXECUTION LOG ===
    """
                        else:
                            # Show both messages and errors
                            st.session_state.rebalance_results = f"""=== REBALANCING EXECUTION RESULTS ===
Executed {result.data.get("trades_executed", 0)} trade(s)
    Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{output}
{errors_output if errors_output else ""}
    === END EXECUTION LOG ===
    """
                    else:
                        st.session_state.rebalance_results = (
                            f"❌ Rebalancing failed:\n" + "\n".join(result.errors)
                        )
                    progress_bar.progress(100)
                    status_text.text(
                        "✅ Execution completed!"
                        if result.success
                        else "❌ Execution failed!"
                    )
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text("❌ Setup failed!")
                    st.session_state.rebalance_results = f"❌ **Setup Error**: {str(e)}"

                finally:
                    st.session_state.rebalance_executing = False
                    # Auto-refresh to show results
                    st.rerun()

        # --- Enhanced Execution Results ---
        st.markdown("### 📋 Execution Results & Logs")

        if st.session_state.rebalance_results:
            # Parse results to show summary first
            results = st.session_state.rebalance_results

            if "EXECUTION COMPLETED" in results:
                st.success("✅ **Execution Successful!**")

                # Show execution summary in an expandable section
                with st.expander("📊 View Detailed Execution Log", expanded=True):
                    st.code(results, language="text")

                # Clear selection after successful execution
                if st.button(
                    "🔄 Clear Selection & Refresh Portfolio", type="secondary"
                ):
                    st.session_state.accepted_trades.clear()
                    st.session_state.rebalance_metrics = None
                    st.session_state.rebalance_suggestions = None
                    st.session_state.rebalance_results = None
                    st.success(
                        "✅ Selection cleared. Run analysis again to see updated portfolio."
                    )
                    st.rerun()

            elif "Error" in results:
                st.error("❌ **Execution Failed**")
                st.code(results, language="text")
            else:
                st.info("ℹ️ **Execution Information**")
                st.code(results, language="text")
        else:
            st.info(
                "📝 Execution results and detailed logs will appear here after running trades."
            )

            # Show helpful tips
            with st.expander("💡 Execution Tips", expanded=False):
                st.markdown("""
                **Before executing trades:**
                - ✅ Ensure sufficient USDT balance for BUY orders
                - ✅ Verify you have the assets available for SELL orders  
                - ✅ Check current market conditions
                - ✅ Review each selected trade carefully
                
                **During execution:**
                - ⏳ Please be patient - trades are executed sequentially
                - 🔄 Do not refresh the page during execution
                - 📊 Monitor the progress indicators
                
                **After execution:**
                - 📈 Results will show order confirmations
                - 🔄 Re-run analysis to see updated portfolio
                - 💾 Consider saving a portfolio snapshot
                """)

    def render_trading(self):
        # (Removed debug: session state at start of trading page render)
        st.markdown("## 💰 Trading")

        # Initialize session state
        if "trading_mode" not in st.session_state:
            st.session_state.trading_mode = "manual"
        if "trading_results" not in st.session_state:
            st.session_state.trading_results = None
        if "trading_executing" not in st.session_state:
            st.session_state.trading_executing = False
        if "manual_trade_data" not in st.session_state:
            st.session_state.manual_trade_data = {}
        if "strategy_signals" not in st.session_state:
            st.session_state.strategy_signals = None

        # Check live trading status
        is_live = self.config_manager.config.get("portfolio", {}).get(
            "live_trading_enabled", False
        )
        is_testnet = self.config_manager.is_testnet_mode

        # Trading Status Banner
        col1, col2, col3 = st.columns(3)
        with col1:
            if is_live:
                st.error(" LIVE TRADING ENABLED")
            else:
                st.warning("🟡 DRY RUN MODE")
        with col2:
            if is_testnet:
                st.info(" TESTNET CONNECTION")
            else:
                st.info(" MAINNET CONNECTION")
        with col3:
            if is_live and not is_testnet:
                st.error("⚠️ REAL ORDERS WILL BE PLACED")
            else:
                st.success("✅ SIMULATION MODE")

        # Trading Mode Selection with Radio
        st.markdown("### 🎯 Select Trading Mode")
        mode = st.radio(
            "Choose trading mode:",
            ["📝 Manual Trade", "🤖 Live Strategy"],
            index=0 if st.session_state.trading_mode == "manual" else 1,
            key="trading_mode_radio",
            horizontal=True,
        )

        if mode == "📝 Manual Trade":
            if st.session_state.trading_mode != "manual":
                st.session_state.trading_mode = "manual"
                st.session_state.strategy_signals = None
            self._render_manual_trading()
        else:
            if st.session_state.trading_mode != "strategy":
                st.session_state.trading_mode = "strategy"
                st.session_state.manual_trade_data = {}
                if (
                    "trading_results" in st.session_state
                    and st.session_state.trading_results
                ):
                    st.session_state.trading_results = None
            self._render_live_strategy_trading()

        # --- Always show debug output at the end of the trading page ---
        # (Removed debug output at the end of the trading page)
        pass

    def _render_manual_trading(self):
        # (Removed debug: session state at start of manual trading render)
        st.markdown("### 📝 Manual Trading")
        # Fetch and display available USDT balance
        usdt_balance = None
        tracker = self.initialize_tracker()
        try:
            usdt_balance = float(
                tracker.binance_client.get_asset_balance(asset="USDT").get("free", 0.0)
            )
        except Exception:
            usdt_balance = None

        st.markdown("#### 💰 Available USDT")
        if usdt_balance is not None:
            st.success(f"${usdt_balance:,.2f}")
        else:
            st.info("USDT balance unavailable.")

        # --- SHOW EXECUTION RESULTS IF AVAILABLE ---
        if st.session_state.trading_results:
            st.markdown("### 📋 Execution Results")
            st.code(st.session_state.trading_results, language="text")
            # Post-execution actions
            if "EXECUTION COMPLETED" in st.session_state.trading_results:
                st.success("✅ **Trade executed successfully!**")
                st.markdown("### 🔄 Portfolio Update")
                st.info(
                    "💡 **Recommendation**: After executing a trade, it's a good practice to sync your portfolio to see the updated balances and positions."
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "🔄 Sync Portfolio", type="primary", use_container_width=True
                    ):
                        self._sync_portfolio_and_reset()
                with col2:
                    if st.button(
                        "🆕 New Trade", type="secondary", use_container_width=True
                    ):
                        st.session_state.trading_results = None
                        st.session_state.manual_trade_data = {}
                        st.rerun()
                with st.expander("💡 Post-Trade Tips", expanded=False):
                    st.markdown("""
                    **After executing a trade:**
                    - 🔄 **Sync Portfolio**: Update your portfolio data to see new balances
                    - 📊 **Check Dashboard**: Use the sidebar to navigate to Dashboard for updated portfolio metrics
                    - 📈 **Monitor Performance**: Track how the trade affects your portfolio
                    - 💾 **Save Snapshot**: Consider saving a portfolio snapshot for record keeping

                    **Next Steps:**
                    - Use the sidebar "🔄 Sync Portfolio" button for a full portfolio update
                    - Visit the Dashboard to see your updated portfolio summary
                    - Check the Rebalancing page to see if any rebalancing is now needed
                    """)
            elif "Error" in st.session_state.trading_results:
                st.error("❌ **Trade execution failed**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "🔄 Try Again", type="primary", use_container_width=True
                    ):
                        st.session_state.trading_results = None
                        st.session_state.manual_trade_data = {}
                        st.rerun()
                with col2:
                    if st.button(
                        "❌ Cancel", type="secondary", use_container_width=True
                    ):
                        st.session_state.trading_results = None
                        st.session_state.manual_trade_data = {}
                        st.rerun()
            return  # <-- Prevents showing confirmation or form

        # --- SHOW TRADE CONFIRMATION IF DATA IS STORED ---
        if st.session_state.manual_trade_data:
            self._show_trade_confirmation()
            return  # <-- Prevents showing the form

        # --- OTHERWISE, SHOW THE TRADE FORM ---
        # (PASTE YOUR FORM CODE HERE)
        core_coins = list(
            self.config_manager.config.get("target_allocation", {}).keys()
        )
        core_coins_upper = [c.upper() for c in core_coins]
        symbol_mapper = self.config_manager.symbol_mapper
        all_symbols = list(symbol_mapper.get_all_mappings().keys())
        if not all_symbols:
            all_coin_dicts = symbol_mapper._fetch_master_coins_list()
            all_symbols = [
                coin["symbol"].upper() for coin in all_coin_dicts if "symbol" in coin
            ]
        non_core_symbols = sorted(set(all_symbols) - set(core_coins_upper))
        coin_options = core_coins_upper + non_core_symbols

        col1, col2 = st.columns(2)
        with col1:
            trade_type = st.selectbox(
                "Trade Action",
                ["BUY", "SELL"],
                help="Choose whether to buy or sell the asset",
            )
            symbol = st.selectbox(
                "Asset Symbol", coin_options, help="Select the asset symbol"
            )
        with col2:
            amount_input = st.text_input(
                f"Amount to {trade_type}",
                placeholder="e.g., 0.1 BTC or 100 USDT",
                help="Enter amount in asset units or USDT",
            )
            is_quote_qty = st.checkbox(
                "Amount is in USDT",
                help="Check if the amount is in USDT, uncheck if in asset units",
            )

        st.markdown("---")
        is_valid = bool(symbol and amount_input)
        button_text = f"🚀 {trade_type} {symbol if symbol else 'ASSET'}"
        if st.button(
            button_text, type="primary", disabled=not is_valid, use_container_width=True
        ):
            st.session_state.manual_trade_data = {
                "trade_type": trade_type,
                "symbol": symbol,
                "amount_input": amount_input,
                "is_quote_qty": is_quote_qty,
            }
            st.rerun()

        if not symbol:
            st.warning("⚠️ Please enter an asset symbol")
        if not amount_input:
            st.warning("⚠️ Please enter an amount")
        if symbol and amount_input:
            st.success("✅ Form is ready for submission")

    def _show_trade_confirmation(self):
        """Shows the trade confirmation interface."""
        data = st.session_state.manual_trade_data

        st.markdown("### 📝 Trade Confirmation")

        # Use Streamlit containers instead of HTML for better styling
        with st.container():
            st.markdown("**Please review your trade details:**")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Action:** {data['trade_type']}")
                st.markdown(f"**Asset:** {data['symbol']}")
            with col2:
                if data["is_quote_qty"]:
                    st.markdown(f"**Amount:** ${data['amount_input']} USDT")
                else:
                    st.markdown(f"**Amount:** {data['amount_input']} {data['symbol']}")

        # Confirmation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ EXECUTE TRADE", type="primary", use_container_width=True):
                self._confirm_manual_trade(
                    data["trade_type"],
                    data["symbol"],
                    data["amount_input"],
                    data["is_quote_qty"],
                )
        with col2:
            if st.button("❌ CANCEL", type="secondary", use_container_width=True):
                st.session_state.manual_trade_data = {}
                st.session_state.trading_results = "Trade cancelled by user."
                st.rerun()

    def _confirm_manual_trade(self, trade_type, symbol, amount_input, is_quote_qty):
        st.session_state.trading_executing = True
        try:
            with st.spinner(f"Executing {trade_type} order for {symbol}..."):
                try:
                    tracker = self.initialize_tracker()
                    is_live = self.config_manager.config.get("portfolio", {}).get(
                        "live_trading_enabled", False
                    )
                    # Parse amount
                    import re

                    numeric_part = re.search(r"[\d\.]+", amount_input)
                    if not numeric_part:
                        st.error(
                            "❌ Invalid amount format. Please enter a valid number."
                        )
                        # Set debug output even on error
                        st.session_state.trade_debug_output = {
                            "error": "Invalid amount format",
                            "params": {
                                "trade_type": trade_type,
                                "symbol": symbol,
                                "amount_input": amount_input,
                                "is_quote_qty": is_quote_qty,
                            },
                        }
                        return
                    amount = float(numeric_part.group(0))
                    trade_ticker = f"{symbol}USDT"
                    import asyncio

                    result = asyncio.run(
                        tracker.execute_manual_trade_core(
                            trade_type,
                            symbol,
                            trade_ticker,
                            amount,
                            is_quote_qty,
                            is_live,
                        )
                    )
                    # Set debug output with result
                    st.session_state.trade_debug_output = {
                        "result": result,
                        "params": {
                            "trade_type": trade_type,
                            "symbol": symbol,
                            "trade_ticker": trade_ticker,
                            "amount": amount,
                            "is_quote_qty": is_quote_qty,
                            "is_live": is_live,
                        },
                    }
                    output = "\n".join(result.messages)
                    if result.success:
                        st.session_state.trading_results = f"=== MANUAL TRADE EXECUTION COMPLETED ===\n{output}\n=== END EXECUTION LOG ==="
                    else:
                        st.session_state.trading_results = (
                            f"❌ Trade failed:\n" + "\n".join(result.errors)
                        )

                    # CRITICAL: Clear manual_trade_data so results are shown instead of confirmation
                    st.session_state.manual_trade_data = {}

                except Exception as e:
                    st.session_state.trade_debug_output = {
                        "exception": str(e),
                        "params": {
                            "trade_type": trade_type,
                            "symbol": symbol,
                            "amount_input": amount_input,
                            "is_quote_qty": is_quote_qty,
                        },
                    }
                    st.session_state.trading_results = (
                        f"❌ **Execution Error**: {str(e)}"
                    )
                    # CRITICAL: Clear manual_trade_data even on error
                    st.session_state.manual_trade_data = {}
        finally:
            st.session_state.trading_executing = False
            st.rerun()

    def _sync_portfolio_and_reset(self):
        with st.spinner("🔄 Syncing portfolio and updating data..."):
            try:
                tracker = self.initialize_tracker()
                metrics = asyncio.run(tracker.run_full_sync())
                st.session_state.portfolio_metrics = metrics
                st.session_state.last_sync = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Get current trading mode to determine what to reset
                current_trading_mode = st.session_state.get("trading_mode", "manual")

                if current_trading_mode == "strategy":
                    # Reset only strategy-related state
                    st.session_state.trading_results = None
                    st.session_state.strategy_signals = None
                    st.session_state.strategy_reset_flag = True

                    # Clean up strategy parameter session state
                    keys_to_delete = []
                    for key in st.session_state.keys():
                        if key.startswith("strategy_param_"):
                            keys_to_delete.append(key)

                    for key in keys_to_delete:
                        del st.session_state[key]

                    st.success("✅ Portfolio synced successfully! Strategy trading page reset.")

                else:  # manual or any other mode
                    # Reset only manual trading state
                    st.session_state.trading_results = None
                    st.session_state.manual_trade_data = {}

                    # Clean up any manual trading related keys if they exist
                    manual_keys = ["manual_trade_symbol", "manual_trade_amount", "manual_trade_type"]
                    for key in manual_keys:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.success("✅ Portfolio synced successfully! Manual trading page reset.")

                st.rerun()

            except Exception as e:
                st.error(f"❌ Portfolio sync failed: {str(e)}")

        # Also update the reset flag handling in the main method
    def _render_live_strategy_trading(self):
        if st.session_state.get("strategy_reset_flag"):
            st.session_state.strategy_signals = None
            st.session_state.trading_results = None
            st.session_state.strategy_reset_flag = False

            # Reset widget-tied session state by removing them entirely
            # This allows the widgets to reinitialize with default values
            widget_keys_to_reset = [
                "strategy_selected_coins",
                "strategy_trade_amount_mode",
                "strategy_trade_pct",
                "strategy_trade_amount",
                "strategy_account",
                "strategy_name"
            ]

            for key in widget_keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]

            # Also clean up any remaining strategy parameter keys
            keys_to_delete = []
            for key in st.session_state.keys():
                if key.startswith("strategy_param_"):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del st.session_state[key]

        st.markdown("### 🤖 Live Strategy Trading")

        # ... rest of the method remains the same ...

    # Also update the reset flag handling in the main method
    def _render_live_strategy_trading(self):
        if st.session_state.get("strategy_reset_flag"):
            st.session_state.strategy_signals = None
            st.session_state.trading_results = None
            st.session_state.strategy_reset_flag = False

            # Reset widget-tied session state by removing them entirely
            # This allows the widgets to reinitialize with default values
            widget_keys_to_reset = [
                "strategy_selected_coins",
                "strategy_trade_amount_mode",
                "strategy_trade_pct",
                "strategy_trade_amount",
                "strategy_account",
                "strategy_name"
            ]

            for key in widget_keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]

            # Also clean up any remaining strategy parameter keys
            keys_to_delete = []
            for key in st.session_state.keys():
                if key.startswith("strategy_param_"):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del st.session_state[key]

        st.markdown("### 🤖 Live Strategy Trading")

        # ... rest of the method remains the same ...

    def _render_live_strategy_trading(self):
        if st.session_state.get("strategy_reset_flag"):
            st.session_state.strategy_signals = None
            st.session_state.trading_results = None
            st.session_state.strategy_reset_flag = False

        st.markdown("### 🤖 Live Strategy Trading")

        # --- 1. Account Selection ---
        tracker = self.initialize_tracker()
        config = self.config_manager.config
        main_account = {
            "name": "Main Account",
            "type": "main",
            **config.get("main_api_keys", {}),
        }
        sub_accounts = config.get("sub_accounts", [])
        accounts = [main_account] + sub_accounts

        # Defensive: No API keys
        if not any(acc.get("api_key") or acc.get("binance_key") for acc in accounts):
            st.error("❌ No API keys found for any account. Cannot run live strategies.")
            return

        account_names = [acc["name"] for acc in accounts]
        selected_account_name = st.selectbox(
            "Select Account", account_names, key="strategy_account"
        )
        selected_account = next(
            acc for acc in accounts if acc["name"] == selected_account_name
        )
        account_type = selected_account.get("type", "main")

        # --- Show Available USDT Balance ---
        usdt_balance = self._get_usdt_balance(tracker, selected_account)

        st.markdown("#### 💰 Available USDT")
        if usdt_balance is not None:
            st.success(f"${usdt_balance:,.2f}")
        else:
            st.info("USDT balance unavailable.")

        # --- 2. Strategy Selection ---
        available_strategies = self._get_available_strategies(account_type)

        if not available_strategies:
            st.error(f"❌ No suitable strategies found for account type '{account_type}'.")
            return

        strategy_names = list(available_strategies.keys())
        selected_strategy_name = st.selectbox(
            "Select Strategy", strategy_names, key="strategy_name"
        )
        strategy_class = available_strategies[selected_strategy_name]

        # --- 3. Coin Selection ---
        coin_options = self._get_coin_options(config)

        if "strategy_selected_coins" not in st.session_state:
            core_coins = list(config.get("target_allocation", {}).keys())
            core_coins_upper = [f"{c.upper()}-USD" for c in core_coins]
            st.session_state.strategy_selected_coins = core_coins_upper

        st.markdown("#### 🪙 Select Coins for Strategy Trading")
        selected_coins = st.multiselect(
            "Coins to include",
            options=coin_options,
            default=st.session_state.get("strategy_selected_coins", []),
            key="strategy_selected_coins",
        )

        # Defensive: Require at least one coin
        if not selected_coins:
            st.warning("Please select at least one coin to run the strategy on.")
            return

        # --- 4. Strategy Parameters and Signal Generation ---
        param_specs = getattr(strategy_class, "strategy_param_specs", {})

        with st.form("strategy_params_form"):
            st.markdown("#### ⚙️ Strategy Parameters")

            param_inputs = {}
            for param, spec in param_specs.items():
                param_inputs[param] = self._render_parameter_input(param, spec)

            # Trade amount configuration
            trade_config = self._render_trade_amount_config()

            submitted = st.form_submit_button("Generate Signals")

        # --- 5. Signal Generation ---
        if submitted:
            signals = self._generate_signals(
                tracker, selected_account, config, strategy_class,
                selected_coins, param_inputs, param_specs
            )
            st.session_state.strategy_signals = signals

        # --- 6. Signal Review ---
        signals = st.session_state.get("strategy_signals", [])
        if not signals:
            if submitted:
                st.info("No actionable BUY or SELL signals generated by the strategy.")
            return

        st.markdown("#### 🚨 Proposed Trades")
        st.dataframe(signals, use_container_width=True)

        # --- 7. Execution ---
        if st.button(
            "🚀 Execute All Signals",
            type="primary",
            use_container_width=True,
            disabled=not signals,
        ):
            self._execute_strategy_signals(
                tracker, selected_account, config, signals, trade_config, usdt_balance
            )

        # --- 8. Show Execution Results ---
        self._display_execution_results()

    def _get_usdt_balance(self, tracker, selected_account):
        """Get USDT balance for the selected account."""
        try:
            if selected_account["name"] == "Main Account":
                balance = tracker.binance_client.get_asset_balance(asset="USDT")
                return float(balance.get("free", 0.0))
            else:
                live_client = tracker._init_binance_client(
                    api_key=selected_account.get("api_key") or selected_account.get("binance_key"),
                    api_secret=selected_account.get("api_secret") or selected_account.get("binance_secret"),
                )
                balance = live_client.get_asset_balance(asset="USDT")
                return float(balance.get("free", 0.0))
        except Exception as e:
            st.info(f"USDT balance unavailable. Error: {e}")
            return None

    def _get_available_strategies(self, account_type):
        """Get strategies available for the account type."""
        all_strategies = {
            name: obj
            for name, obj in inspect.getmembers(trading_strategies, inspect.isclass)
            if issubclass(obj, trading_strategies.Strategy)
            and obj is not trading_strategies.Strategy
        }

        if account_type == "main":
            return all_strategies
        elif account_type == "swing":
            return {
                k: v for k, v in all_strategies.items()
                if getattr(v, "strategy_type", None) in ["swing", "general"]
            }
        elif account_type == "day":
            return {
                k: v for k, v in all_strategies.items()
                if getattr(v, "strategy_type", None) in ["day", "general"]
            }
        else:
            return all_strategies

    def _get_coin_options(self, config):
        """Get available coin options."""
        symbol_mapper = self.config_manager.symbol_mapper
        core_coins = list(config.get("target_allocation", {}).keys())
        core_coins_upper = [f"{c.upper()}-USD" for c in core_coins]

        all_symbols = list(symbol_mapper.get_all_mappings().keys())
        if not all_symbols:
            all_coin_dicts = symbol_mapper._fetch_master_coins_list()
            all_symbols = [
                f"{coin['symbol'].upper()}-USD"
                for coin in all_coin_dicts
                if "symbol" in coin
            ]

        non_core_symbols = sorted(set(all_symbols) - set(core_coins_upper))
        return core_coins_upper + non_core_symbols

    def _render_parameter_input(self, param, spec):
        """Render input field for a strategy parameter."""
        label = spec.get("label", param.replace("_", " ").capitalize())
        default = spec.get("default", "")
        key = f"strategy_param_{param}"

        if spec.get("type") == "float":
            return st.number_input(
                label,
                value=float(default),
                min_value=spec.get("min_value", 0.0),
                max_value=spec.get("max_value", 100.0),
                key=key,
            )
        elif spec.get("type") == "int":
            return st.number_input(
                label,
                value=int(default),
                min_value=spec.get("min_value", 0),
                max_value=spec.get("max_value", 100),
                step=1,
                key=key,
            )
        elif spec.get("type") == "bool":
            return st.checkbox(label, value=bool(default), key=key)
        else:
            return st.text_input(label, value=str(default), key=key)

    def _render_trade_amount_config(self):
        """Render trade amount configuration."""
        trade_amount_mode = st.radio(
            "How do you want to specify trade size?",
            ["% of USDT balance", "Fixed USDT amount"],
            horizontal=True,
            key="strategy_trade_amount_mode",
        )

        if trade_amount_mode == "% of USDT balance":
            trade_pct = st.number_input(
                "Percent of available USDT to use per trade",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                key="strategy_trade_pct",
            )
            return {"mode": "percentage", "value": trade_pct}
        else:
            trade_amount = st.number_input(
                "Fixed USDT amount to use per trade",
                min_value=1.0,
                value=50.0,
                step=1.0,
                key="strategy_trade_amount",
            )
            return {"mode": "fixed", "value": trade_amount}

    def _generate_signals(self, tracker, selected_account, config, strategy_class,
                     selected_coins, param_inputs, param_specs):
        """Generate trading signals for selected coins."""
        st.session_state.strategy_signals = None  # Reset

        with st.spinner("Generating signals..."):
            try:
                live_client = tracker._init_binance_client(
                    api_key=selected_account.get("api_key") or selected_account.get("binance_key"),
                    api_secret=selected_account.get("api_secret") or selected_account.get("binance_secret"),
                )
                analyzer = CryptoTrendAnalyzer(config=config, binance_client=live_client)
                signals = []

                for coin in selected_coins:
                    try:
                        signal_data = self._generate_coin_signal(
                            analyzer, strategy_class, coin, param_inputs, param_specs
                        )
                        if signal_data and signal_data["Signal"] in ["BUY", "SELL"]:
                            signals.append(signal_data)
                    except Exception as e:
                        st.warning(f"Failed to generate signal for {coin}: {e}")
                        continue

                return signals

            except Exception as e:
                st.error(f"Signal generation failed: {e}")
                return []

    def _generate_coin_signal(self, analyzer, strategy_class, coin, param_inputs, param_specs):
        """Generate signal for a single coin."""
        # Extract just the coin name from "BTC-USD" format
        coin_name = coin.split('-')[0] if '-' in coin else coin
        yf_ticker = f"{coin_name}-USD"
        analyzer.set_symbol(yf_ticker)

        # Convert percent params to fraction if needed
        strategy_kwargs = param_inputs.copy()
        for k, v in param_specs.items():
            if (v.get("type") == "float" and "pct" in k and
                strategy_kwargs[k] is not None and strategy_kwargs[k] > 1):
                strategy_kwargs[k] = strategy_kwargs[k] / 100.0

        strategy_instance = strategy_class(analyzer=analyzer, **strategy_kwargs)
        interval = getattr(strategy_instance, "valid_intervals", ["1d"])[0]
        period = "7d" if "m" in interval or "h" in interval else "1y"

        data = asyncio.run(
            analyzer.fetch_crypto_data_async(yf_ticker, period=period, interval=interval)
        )

        if data is None or data.empty:
            return None

        signal, size, reason = asyncio.run(strategy_instance.generate_signal(data))

        if signal in ["BUY", "SELL"]:
            return {
                "Symbol": coin_name,  # Store just the coin name (BTC, ETH, etc.)
                "Signal": signal,
                "Size": size,
                "Reason": reason,
            }
        return None

    def _execute_strategy_signals(self, tracker, selected_account, config, signals,
                             trade_config, usdt_balance):
        """Execute all trading signals."""
        with st.spinner("Executing trades..."):
            try:
                results = []
                min_trade_usd = config.get("portfolio", {}).get("minimum_trade_usd", 10.0)
                is_live = config.get("portfolio", {}).get("live_trading_enabled", False)

                for trade in signals:
                    try:
                        result = self._execute_single_trade(
                            tracker, trade, trade_config, usdt_balance, min_trade_usd, is_live
                        )
                        results.append(result)
                    except Exception as e:
                        results.append(f"{trade['Signal']} {trade['Symbol']}: ❌ Execution Error: {e}")

                st.session_state.trading_results = "\n".join(results)
                st.success("✅ All signals executed. See results below.")

            except Exception as e:
                st.session_state.trading_results = f"❌ Execution Error: {e}"
                st.error(f"Execution failed: {e}")

    def _execute_single_trade(self, tracker, trade, trade_config, usdt_balance, min_trade_usd, is_live):
        """Execute a single trade."""
        trade_type = trade["Signal"]
        symbol = trade["Symbol"]

        if trade_config["mode"] == "percentage":
            usdt_to_spend = max((trade_config["value"] / 100.0) * usdt_balance, min_trade_usd)
        else:
            usdt_to_spend = max(trade_config["value"], min_trade_usd)

        trade_ticker = f"{symbol}USDT"

        # Execute trade using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            # Get the actual result from execute_manual_trade_core
            result = loop.run_until_complete(
                tracker.execute_manual_trade_core(
                    trade_type, symbol, trade_ticker, usdt_to_spend, True, is_live
                )
            )

            sys.stdout = old_stdout
            execution_output = mystdout.getvalue()

            # Check if the trade was successful
            if result.success:
                return (f"{trade_type} {symbol}: \n"
                        f"Preparing MARKET {trade_type} for {symbol}...\n"
                        f"Amount: {usdt_to_spend:.2f} USDT\n"
                        f"�� PLACING LIVE ORDER...\n"
                        f"{execution_output}")
            else:
                # Return error message
                error_msg = "\n".join(result.errors) if result.errors else "Unknown error occurred"
                return f"{trade_type} {symbol}: ❌ FAILED - {error_msg}"

        finally:
            loop.close()

    def _display_execution_results(self):
        """Display execution results and post-execution options."""
        if not st.session_state.get("trading_results"):
            return

        st.markdown("### 📋 Execution Results")
        st.code(st.session_state.trading_results, language="text")

        # Check for failed trades
        if "❌" in st.session_state.trading_results:
            st.warning("⚠️ Some trades failed. Please review the log above.")
        else:
            st.success("✅ All strategy trades executed successfully!")

        st.markdown("### 🔄 Portfolio Update")
        st.info("💡 Recommendation: After executing trades, sync your portfolio to see updated balances and positions.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Sync Portfolio", type="primary", use_container_width=True):
                self._sync_portfolio_and_reset()
        with col2:
            if st.button("🆕 New Strategy Run", type="secondary", use_container_width=True):
                st.session_state.trading_results = None
                st.session_state.strategy_signals = None
                st.rerun()

    def render_backtesting(self):
        import inspect
        import pandas as pd
        import asyncio
        from crypto_portfolio_tracker.strategy_backtester import StrategyBacktester
        from crypto_portfolio_tracker.rebalancing_backtester import (
            RebalancingBacktester,
        )
        from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer

        st.markdown("## 🧪 Backtesting")
        st.markdown(
            "Test your trading strategies and rebalancing approaches with historical data"
        )

        tab1, tab2 = st.tabs(["⚖️ Rebalancing Backtest", "🧪 Strategy Backtest"])

        # --- Rebalancing Backtest Tab ---
        with tab1:
            st.header("⚖️ Rebalancing Backtest")
            st.markdown(
                "Test portfolio rebalancing strategies with customizable drift thresholds and market conditions"
            )

            # Basic Settings Section
            with st.container():
                st.subheader("💰 Basic Configuration")
                col1, col2 = st.columns(2)

                with col1:
                    initial_capital = st.number_input(
                        "Initial Capital (USD)",
                        min_value=100.0,
                        value=10000.0,
                        step=500.0,
                        key="rebalance_initial_capital",
                        help="Starting capital for the rebalancing backtest",
                    )

                with col2:
                    period_options = ["1y", "2y", "3y", "4y", "5y", "Custom"]
                    selected_period_option = st.selectbox(
                        "Backtest Period",
                        period_options,
                        index=2,  # Default to 3y
                        key="rebalance_period_dropdown",
                        help="Historical period for rebalancing analysis",
                    )

                    if selected_period_option == "Custom":
                        period = st.text_input(
                            "Custom Period (e.g., 7y)",
                            value="6y",
                            key="rebalance_period_custom",
                            help="Enter custom period: Xy for years",
                        )
                    else:
                        period = selected_period_option

            # Advanced Parameters Section
            with st.expander("🔧 Advanced Rebalancing Parameters", expanded=False):
                # Create tabs within the expander for better organization
                param_tab1, param_tab2, param_tab3, param_tab4 = st.tabs(
                    [
                        "📊 Drift Settings",
                        "💱 Trade Control",
                        "🐻 Market Rules",
                        "🎯 Asset Allocation",
                    ]
                )

                with param_tab1:
                    st.markdown(
                        "#### Set allocation drift thresholds that trigger rebalancing"
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🪙 Major Coins (BTC/ETH)**")
                        majors_drift_threshold = st.slider(
                            "Drift Threshold (%)",
                            min_value=1.0,
                            max_value=20.0,
                            value=3.0,
                            step=0.5,
                            help="Drift threshold for BTC/ETH before rebalancing triggers",
                            key="majors_drift",
                        )
                    with col2:
                        st.markdown("**🚀 Altcoins**")
                        alts_drift_threshold = st.slider(
                            "Drift Threshold (%)",
                            min_value=1.0,
                            max_value=20.0,
                            value=5.0,
                            step=0.5,
                            help="Drift threshold for altcoins before rebalancing triggers",
                            key="alts_drift",
                        )

                with param_tab2:
                    st.markdown("#### Control trade aggressiveness during rebalancing")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🪙 Major Coins (BTC/ETH)**")
                        majors_sell_multiplier = st.slider(
                            "Sell Multiplier",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.5,
                            step=0.1,
                            help="Portion of overweight amount to sell (0.5 = sell 50%)",
                            key="majors_sell",
                        )
                        majors_buy_multiplier = st.slider(
                            "Buy Multiplier",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.75,
                            step=0.1,
                            help="Portion of underweight amount to buy (0.75 = buy 75%)",
                            key="majors_buy",
                        )
                    with col2:
                        st.markdown("**🚀 Altcoins**")
                        alts_sell_multiplier = st.slider(
                            "Sell Multiplier",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.5,
                            step=0.1,
                            help="Portion of overweight amount to sell for altcoins",
                            key="alts_sell",
                        )
                        alts_buy_multiplier = st.slider(
                            "Buy Multiplier",
                            min_value=0.1,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            help="Portion of underweight amount to buy for altcoins",
                            key="alts_buy",
                        )

                with param_tab3:
                    st.markdown("#### Configure market condition responses")
                    suppress_buys_in_bear = st.checkbox(
                        "🐻 Suppress Buys in Bear Market",
                        value=True,
                        help="Avoid buying during bearish market conditions (except oversold scenarios)",
                    )

                with param_tab4:
                    st.markdown("#### Customize your portfolio allocation")
                    st.info(
                        "💡 Modify allocations below or leave as-is to use your default configuration"
                    )

                    # Get all available symbols (e.g., from symbol_mapper)
                    all_symbols = list(
                        self.config_manager.symbol_mapper.get_all_mappings().keys()
                    )
                    default_allocation = self.config_manager.config.get(
                        "target_allocation", {}
                    )

                    # Use session state to persist custom allocation edits
                    if (
                        "custom_allocation" not in st.session_state
                        or not st.session_state.custom_allocation
                    ):
                        st.session_state.custom_allocation = default_allocation.copy()

                    custom_allocation = st.session_state.custom_allocation

                    # Current assets section
                    if custom_allocation:
                        st.markdown("**Current Asset Allocation:**")
                        assets_to_remove = []
                        for asset, alloc in list(custom_allocation.items()):
                            cols = st.columns([2, 2, 1.5, 1])
                            with cols[0]:
                                st.markdown(f"**{asset.upper()}**")
                            with cols[1]:

                                def update_allocation(asset_name):
                                    def callback():
                                        new_value = st.session_state[
                                            f"alloc_{asset_name}"
                                        ]
                                        st.session_state.custom_allocation[
                                            asset_name
                                        ] = new_value / 100.0

                                    return callback

                                new_alloc = st.number_input(
                                    f"Allocation",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=alloc * 100,
                                    step=0.1,
                                    key=f"alloc_{asset}",
                                    label_visibility="collapsed",
                                    on_change=update_allocation(asset),
                                )
                            with cols[2]:
                                st.markdown(f"*{new_alloc:.1f}%*")
                            with cols[3]:
                                if st.button(
                                    "❌",
                                    key=f"remove_{asset}",
                                    help=f"Remove {asset.upper()}",
                                ):
                                    assets_to_remove.append(asset)

                        for asset in assets_to_remove:
                            del custom_allocation[asset]
                            st.rerun()  # Force rerun to update the UI immediately

                    # Recalculate total_alloc after updates
                    total_alloc = sum(st.session_state.custom_allocation.values())

                    # Add asset functionality - PROPER ALIGNMENT
                    st.markdown("---")
                    st.markdown("**Add New Asset:**")
                    available_to_add = [
                        s for s in all_symbols if s not in custom_allocation
                    ]
                    if available_to_add:
                        # Use a container for better layout control
                        with st.container():
                            # Create a custom layout with proper alignment
                            st.markdown("Select Asset to Add:")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                add_asset = st.selectbox(
                                    "",  # Empty label to avoid extra spacing
                                    ["Select an asset..."] + available_to_add,
                                    key=f"add_asset_select_{st.session_state.get('add_asset_counter', 0)}",
                                    label_visibility="collapsed",  # Hide the label completely
                                )
                            with col2:
                                # Show button only when a valid asset is selected
                                if add_asset and add_asset != "Select an asset...":
                                    if st.button("➕ Add", key="add_asset_btn"):
                                        if (
                                            add_asset
                                            and add_asset != "Select an asset..."
                                        ):
                                            custom_allocation[add_asset] = (
                                                0.0  # Default to 0% allocation
                                            )
                                            # Increment counter to force selectbox reset
                                            st.session_state["add_asset_counter"] = (
                                                st.session_state.get(
                                                    "add_asset_counter", 0
                                                )
                                                + 1
                                            )
                                            st.rerun()  # Force rerun to update the UI
                                else:
                                    # Show disabled button when no asset selected
                                    st.button(
                                        "➕ Add",
                                        key="add_asset_btn_disabled",
                                        disabled=True,
                                    )
                    else:
                        st.info("All available assets are already in your allocation")

                    # Validation section
                    st.markdown("---")
                    total_alloc = sum(custom_allocation.values())
                    col1, col2 = st.columns(2)
                    with col1:
                        if abs(total_alloc - 1.0) > 0.0001:
                            st.warning(f"⚠️ Total: {total_alloc:.2%} (must be 100%)")
                        else:
                            st.success(f"✅ Total: {total_alloc:.2%}")
                    with col2:
                        if st.button("🔄 Reset to Default", key="reset_allocation"):
                            st.session_state.custom_allocation = (
                                default_allocation.copy()
                            )
                            st.rerun()

            # Run Rebalancing Backtest Button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_rebalance_backtest = st.button(
                    "🚀 Run Rebalancing Backtest",
                    type="primary",
                    use_container_width=True,
                )

            if run_rebalance_backtest:
                # Validate allocation before running backtest
                total_alloc = (
                    sum(custom_allocation.values()) if custom_allocation else 0
                )
                if total_alloc > 1.0:
                    st.error(
                        f"❌ Cannot run backtest: Total allocation is {total_alloc:.2%} which exceeds 100%. Please adjust your allocations."
                    )
                else:
                    with st.spinner("🔄 Running rebalancing backtest analysis..."):
                        try:
                            # Create custom config with user parameters
                            custom_config = self.config_manager.config.copy()

                            # Update rebalancing parameters
                            custom_config["rebalance_technical"] = {
                                "market_regime_rules": {
                                    "suppress_buys_in_bear": suppress_buys_in_bear
                                },
                                "majors": {
                                    "allocation_drift_threshold_pct": majors_drift_threshold,
                                    "sell_percentage_multiplier": majors_sell_multiplier,
                                    "buy_amount_multiplier": majors_buy_multiplier,
                                },
                                "alts": {
                                    "allocation_drift_threshold_pct": alts_drift_threshold,
                                    "sell_percentage_multiplier": alts_sell_multiplier,
                                    "buy_amount_multiplier": alts_buy_multiplier,
                                },
                            }

                            # Update target allocation if user customized it
                            if abs(sum(custom_allocation.values()) - 1.0) < 0.001:
                                custom_config["target_allocation"] = custom_allocation

                            backtester = RebalancingBacktester(config=custom_config)
                            backtester.run(
                                initial_capital=initial_capital, period=period
                            )

                            st.success(
                                "✅ Rebalancing backtest completed successfully!"
                            )

                            # Results Section
                            if hasattr(backtester, "summary_stats"):
                                st.markdown("## 📊 Rebalancing Results")

                                # Key metrics in columns
                                stats = backtester.summary_stats
                                metric_cols = st.columns(4)

                                with metric_cols[0]:
                                    st.metric(
                                        "Strategy Return",
                                        f"{stats['Strategy Total Return']:.1%}",
                                        delta=f"{stats['Strategy Outperformance']:+.1%} vs B&H",
                                    )

                                with metric_cols[1]:
                                    st.metric(
                                        "Final Value",
                                        f"${stats['Final Portfolio Value']:,.0f}",
                                        delta=f"${stats['Final Portfolio Value'] - stats['Initial Capital']:+,.0f}",
                                    )

                                with metric_cols[2]:
                                    st.metric(
                                        "Max Drawdown",
                                        f"{stats['Maximum Drawdown']:.1%}",
                                    )

                                with metric_cols[3]:
                                    st.metric(
                                        "Total Trades",
                                        f"{stats['Total Trades Executed']}",
                                    )

                                # Detailed Results Table
                                with st.expander(
                                    "📋 Detailed Performance Metrics", expanded=True
                                ):
                                    st.table(
                                        {
                                            "Metric": [
                                                "Initial Capital",
                                                "Final Portfolio Value",
                                                "Strategy Total Return",
                                                "Buy & Hold Return",
                                                "Strategy Outperformance",
                                                "Maximum Drawdown",
                                                "Annualized Volatility",
                                                "Sharpe Ratio",
                                                "Total Trades Executed",
                                            ],
                                            "Value": [
                                                f"${stats['Initial Capital']:,.2f}",
                                                f"${stats['Final Portfolio Value']:,.2f}",
                                                f"{stats['Strategy Total Return']:.2%}",
                                                f"{stats['Buy & Hold Return']:.2%}",
                                                f"{stats['Strategy Outperformance']:+.2%}",
                                                f"{stats['Maximum Drawdown']:.2%}",
                                                f"{stats['Annualized Volatility']:.2%}",
                                                f"{stats['Sharpe Ratio']:.2f}",
                                                f"{stats['Total Trades Executed']}",
                                            ],
                                        }
                                    )

                            # Equity Curve
                            if hasattr(backtester, "portfolio_value_history"):
                                st.markdown("### 📈 Portfolio Equity Curve")
                                if (
                                    isinstance(backtester.portfolio_value_history, list)
                                    and backtester.portfolio_value_history
                                ):
                                    # Convert to DataFrame with proper date index
                                    df = pd.DataFrame(
                                        backtester.portfolio_value_history
                                    )
                                    df["date"] = pd.to_datetime(df["date"])
                                    df = df.set_index("date")
                                    df = df.rename(
                                        columns={"value": "Portfolio Value ($)"}
                                    )

                                    # Create the chart with proper date axis
                                    st.line_chart(df, use_container_width=True)

                            # Trade Log
                            if hasattr(backtester, "trade_log"):
                                with st.expander("📝 Rebalancing Trade Log"):
                                    st.code(
                                        "\n".join(backtester.trade_log), language="text"
                                    )

                        except Exception as e:
                            st.error(f"❌ Rebalancing backtest failed: {str(e)}")
                            st.info(
                                "💡 Try adjusting the parameters or check your allocation settings"
                            )

        # --- Strategy Backtest Tab ---
        with tab2:
            st.header("🧪 Strategy Backtest")
            st.markdown(
                "Select a trading strategy and customize its parameters to test performance against historical data"
            )

            # Strategy Selection Section
            with st.container():
                st.subheader("📈 Strategy Selection")
                all_strategies = {
                    name: obj
                    for name, obj in inspect.getmembers(
                        trading_strategies, inspect.isclass
                    )
                    if issubclass(obj, trading_strategies.Strategy)
                    and obj is not trading_strategies.Strategy
                }
                strategy_names = list(all_strategies.keys())
                selected_strategy_name = st.selectbox(
                    "Choose Trading Strategy",
                    strategy_names,
                    help="Select the trading strategy you want to backtest",
                )
                strategy_class = all_strategies[selected_strategy_name]

            # Parameters Section
            param_specs = getattr(strategy_class, "strategy_param_specs", {})
            if param_specs:
                with st.expander("🔧 Strategy Parameters", expanded=True):
                    st.markdown("##### Configure Strategy Parameters")
                    param_inputs = {}

                    # Create columns for better parameter layout
                    param_cols = st.columns(2)
                    col_idx = 0

                    for param, spec in param_specs.items():
                        with param_cols[col_idx % 2]:
                            label = spec.get("label", param.replace("_", " ").title())
                            default = spec.get("default", "")
                            key = f"backtest_param_{param}"
                            help_text = spec.get("help", f"Configure {label.lower()}")

                            if spec.get("type") == "float":
                                param_inputs[param] = st.number_input(
                                    label, value=float(default), key=key, help=help_text
                                )
                            elif spec.get("type") == "int":
                                param_inputs[param] = st.number_input(
                                    label, value=int(default), key=key, help=help_text
                                )
                            elif spec.get("type") == "bool":
                                param_inputs[param] = st.checkbox(
                                    label, value=bool(default), key=key, help=help_text
                                )
                            else:
                                param_inputs[param] = st.text_input(
                                    label, value=str(default), key=key, help=help_text
                                )
                        col_idx += 1
            else:
                param_inputs = {}

            # Backtest Configuration Section
            with st.container():
                st.subheader("⚙️ Backtest Configuration")

                col1, col2, col3 = st.columns(3)

                with col1:
                    valid_intervals = getattr(strategy_class, "valid_intervals", ["1d"])
                    interval = st.selectbox(
                        "Data Interval",
                        valid_intervals,
                        help="Time interval for price data (higher frequency = more granular analysis)",
                    )

                with col2:
                    symbol_mapper = self.config_manager.symbol_mapper
                    config = self.config_manager.config
                    core_coins = list(config.get("target_allocation", {}).keys())
                    core_coins_upper = [f"{c.upper()}-USD" for c in core_coins]
                    all_symbols = list(symbol_mapper.get_all_mappings().keys())
                    if not all_symbols:
                        all_coin_dicts = symbol_mapper._fetch_master_coins_list()
                        all_symbols = [
                            f"{coin['symbol'].upper()}-USD"
                            for coin in all_coin_dicts
                            if "symbol" in coin
                        ]
                    non_core_symbols = sorted(set(all_symbols) - set(core_coins_upper))
                    coin_options = core_coins_upper + non_core_symbols

                    selected_coin = st.selectbox(
                        "Select Cryptocurrency",
                        coin_options,
                        help="Choose the cryptocurrency to backtest the strategy on",
                    )

                with col3:
                    period_options = ["1y", "3y", "5y", "60d", "90d", "180d", "Custom"]
                    selected_period_option = st.selectbox(
                        "Backtest Period",
                        period_options,
                        index=1,  # Default to 3y
                        help="Historical period to run the backtest over",
                    )

                    if selected_period_option == "Custom":
                        period = st.text_input(
                            "Custom Period (e.g., 7y, 30d)",
                            value="30d",
                            key="strategy_period_custom",
                            help="Enter custom period: Xy for years, Xd for days",
                        )
                    else:
                        period = selected_period_option

                # --- Warn about yfinance data limits for high-frequency intervals ---
                if ("15m" in interval and "y" in period) or (
                    "1h" in interval
                    and "y" in period
                    and int(period.replace("y", "")) > 2
                ):
                    st.warning(
                        "⚠️ For 15m interval, use a period ≤ 60d. For 1h interval, use a period ≤ 729d. Adjust your selection to avoid data errors."
                    )

                # Capital input in its own row for better visibility
                st.markdown("##### 💰 Initial Investment")
                initial_capital = st.number_input(
                    "Initial Capital (USD)",
                    min_value=100.0,
                    value=10000.0,
                    step=500.0,
                    help="Starting capital for the backtest simulation",
                )

            # Run Backtest Button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_backtest = st.button(
                    "🚀 Run Strategy Backtest", type="primary", use_container_width=True
                )

            if run_backtest:
                with st.spinner("🔄 Running backtest analysis..."):
                    try:
                        analyzer = CryptoTrendAnalyzer(
                            config=config, binance_client=None
                        )
                        backtester = StrategyBacktester(
                            config=config, analyzer=analyzer
                        )
                        strategy_instance = strategy_class(
                            analyzer=analyzer, **param_inputs
                        )
                        asyncio.run(
                            backtester.run(
                                strategy=strategy_instance,
                                symbol=selected_coin,
                                initial_capital=initial_capital,
                                period=period,
                                interval=interval,
                            )
                        )

                        backtester.generate_report()

                        st.success("✅ Backtest completed successfully!")

                        # Results Section
                        if hasattr(backtester, "summary_stats"):
                            st.markdown("## 📊 Backtest Results")

                            # Key metrics in columns
                            stats = backtester.summary_stats
                            metric_cols = st.columns(4)

                            with metric_cols[0]:
                                st.metric(
                                    "Strategy Return",
                                    f"{stats['Strategy Total Return']:.1%}",
                                    delta=f"{stats['Strategy Outperformance']:+.1%} vs B&H",
                                )

                            with metric_cols[1]:
                                st.metric(
                                    "Final Value",
                                    f"${stats['Final Portfolio Value']:,.0f}",
                                    delta=f"${stats['Final Portfolio Value'] - stats['Initial Capital']:+,.0f}",
                                )

                            with metric_cols[2]:
                                st.metric(
                                    "Max Drawdown", f"{stats['Maximum Drawdown']:.1%}"
                                )

                            with metric_cols[3]:
                                st.metric(
                                    "Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}"
                                )

                            # Detailed Results Table
                            with st.expander(
                                "📋 Detailed Performance Metrics", expanded=True
                            ):
                                st.table(
                                    {
                                        "Metric": [
                                            "Initial Capital",
                                            "Final Portfolio Value",
                                            "Strategy Total Return",
                                            "Buy & Hold Return",
                                            "Strategy Outperformance",
                                            "Maximum Drawdown",
                                            "Annualized Volatility",
                                            "Sharpe Ratio",
                                            "Total Trades Executed",
                                        ],
                                        "Value": [
                                            f"${stats['Initial Capital']:,.2f}",
                                            f"${stats['Final Portfolio Value']:,.2f}",
                                            f"{stats['Strategy Total Return']:.2%}",
                                            f"{stats['Buy & Hold Return']:.2%}",
                                            f"{stats['Strategy Outperformance']:+.2%}",
                                            f"{stats['Maximum Drawdown']:.2%}",
                                            f"{stats['Annualized Volatility']:.2%}",
                                            f"{stats['Sharpe Ratio']:.2f}",
                                            f"{stats['Total Trades Executed']}",
                                        ],
                                    }
                                )

                        # Equity Curve
                        if hasattr(backtester, "portfolio_value_history"):
                            st.markdown("### 📈 Portfolio Equity Curve")
                            if (
                                isinstance(backtester.portfolio_value_history, list)
                                and backtester.portfolio_value_history
                                and hasattr(backtester, "data")
                                and backtester.data is not None
                            ):
                                # Use the price data's index as the x-axis
                                df = pd.DataFrame(
                                    {
                                        "Portfolio Value ($)": backtester.portfolio_value_history
                                    },
                                    index=backtester.data.index[
                                        : len(backtester.portfolio_value_history)
                                    ],
                                )
                                st.line_chart(df, use_container_width=True)

                        # Trade Log
                        if hasattr(backtester, "trade_log"):
                            with st.expander("📝 Trade Execution Log"):
                                st.code(
                                    "\n".join(backtester.trade_log), language="text"
                                )

                    except Exception as e:
                        st.error(f"❌ Backtest failed: {str(e)}")
                        st.info(
                            "💡 Try adjusting the parameters or selecting a different time period"
                        )

    def render_data_management(self):
        import pandas as pd
        from pathlib import Path

        st.markdown("## 🗄️ Data Management")

        tracker = self.initialize_tracker()
        db_path = tracker.db_manager.db_path
        backup_dir = tracker.db_manager.backup_dir

        tab1, tab2, tab3, tab4 = st.tabs(
            ["💾 Backup & Restore", "⬆️ Import / ⬇️ Export", "📸 Snapshots", "ℹ️ Info"]
        )

        # --- 1. Backup & Restore ---
        with tab1:
            st.header("💾 Backup & Restore")
            if st.button("Create Backup"):
                backup_path = tracker.db_manager.backup_database()
                if backup_path:
                    st.success(f"Backup created: {backup_path}")
                else:
                    st.error("Backup failed. See logs for details.")

            backups = tracker.db_manager.list_backups()
            if not backups:
                st.info("No backups found.")
            else:
                backup_names = [b.name for b in backups]
                selected_backup = st.selectbox("Select backup", backup_names)
                backup_path = backup_dir / selected_backup

                col1, col2, col3 = st.columns(3)
                with col1:
                    with open(backup_path, "rb") as f:
                        st.download_button(
                            "Download",
                            f,
                            file_name=selected_backup,
                            use_container_width=True,
                        )
                with col2:
                    if st.button(
                        "Delete",
                        key=f"delete_{selected_backup}",
                        use_container_width=True,
                    ):
                        try:
                            backup_path.unlink()
                            st.success(f"Deleted {selected_backup}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
                with col3:
                    if st.button(
                        "Restore",
                        key=f"restore_{selected_backup}",
                        use_container_width=True,
                    ):
                        st.warning(
                            "⚠️ This will overwrite your current database. This action is irreversible!"
                        )
                        if st.button(
                            "Confirm Restore", key=f"confirm_restore_{selected_backup}"
                        ):
                            success = tracker.db_manager.restore_from_backup(
                                backup_path
                            )
                            if success:
                                st.success(
                                    "Database restored. Please restart the app to use the restored data."
                                )
                            else:
                                st.error("Restore failed. See logs for details.")

        # --- 2. Import/Export ---
        with tab2:
            st.header("⬆️ Import / ⬇️ Export Raw Data")
            export_dir = Path(
                self.config_manager.config.get("exports", {}).get(
                    "path", "data/exports/"
                )
            )
            data_export_dir = export_dir / "data_management"
            data_export_dir.mkdir(parents=True, exist_ok=True)

            st.markdown("#### Create New Export")

            col1, col2 = st.columns(2)
            with col1:
                export_type = st.selectbox(
                    "Select data to export", ["Holdings", "Transactions"]
                )
            with col2:
                export_format = st.radio(
                    "Select format", ["CSV", "Excel"], horizontal=True
                )

            if st.button(
                "🚀 Generate Export", use_container_width=True, type="primary"
            ):
                with st.spinner(f"Generating {export_type} export..."):
                    df = None
                    if export_type == "Holdings":
                        df = tracker.db_manager.get_holdings()
                    else:  # Transactions
                        df = tracker.db_manager.get_all_transactions()

                    if df is not None and not df.empty:
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_extension = "xlsx" if export_format == "Excel" else "csv"
                        filename = (
                            f"{export_type.lower()}_export_{now_str}.{file_extension}"
                        )
                        filepath = data_export_dir / filename

                        # Create a copy and handle timezones for Excel compatibility
                        export_df = df.copy()
                        for col in export_df.select_dtypes(["datetimetz"]).columns:
                            export_df[col] = export_df[col].dt.tz_localize(None)

                        if export_format == "CSV":
                            export_df.to_csv(filepath, index=False)
                        else:  # Excel
                            export_df.to_excel(filepath, index=False, engine="openpyxl")

                        st.success(f"Successfully created export: `{filename}`")
                        st.rerun()
                    else:
                        st.warning(f"No {export_type} data available to export.")

            st.markdown("---")
            st.markdown("#### My Exports")

            all_files = sorted(
                list(data_export_dir.glob("*_export_*.*")),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            if not all_files:
                st.info("No exports found. Generate one above!")
            else:
                selected_export = st.selectbox(
                    "Select an export file:",
                    options=all_files,
                    format_func=lambda x: f"{x.name} ({x.stat().st_size / 1024:.1f} KB, {datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})",
                )

                if "preview_file" not in st.session_state:
                    st.session_state.preview_file = None

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👁️ Preview", use_container_width=True):
                        st.session_state.preview_file = selected_export
                        st.rerun()
                with col2:
                    with open(selected_export, "rb") as f:
                        file_bytes = f.read()
                        st.download_button(
                            label="⬇️ Download",
                            data=file_bytes,
                            file_name=selected_export.name,
                            mime=f"text/{selected_export.suffix[1:].lower()}"
                            if selected_export.suffix == ".csv"
                            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
                with col3:
                    if st.button("🗑️ Delete", use_container_width=True):
                        try:
                            selected_export.unlink()
                            if st.session_state.preview_file == selected_export:
                                st.session_state.preview_file = None
                            st.success(f"Deleted {selected_export.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

                if st.session_state.preview_file:
                    st.markdown("---")
                    st.markdown(f"### 👁️ Preview: {st.session_state.preview_file.name}")
                    try:
                        if st.session_state.preview_file.name.endswith(".csv"):
                            st.dataframe(pd.read_csv(st.session_state.preview_file))
                        else:
                            st.dataframe(pd.read_excel(st.session_state.preview_file))
                    except Exception as e:
                        st.error(f"Could not preview file: {e}")

                    if st.button("❌ Close Preview"):
                        st.session_state.preview_file = None
                        st.rerun()

            st.markdown("---")
            st.markdown("### Import")
            uploaded_holdings = st.file_uploader(
                "Import Holdings (CSV/Excel)",
                type=["csv", "xlsx"],
                key="import_holdings",
            )
            if uploaded_holdings:
                try:
                    if uploaded_holdings.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_holdings)
                    else:
                        df = pd.read_excel(uploaded_holdings)
                    
                    # Validate required columns for holdings
                    required_holdings_cols = ["symbol", "quantity", "average_cost_basis"]
                    missing_cols = [col for col in required_holdings_cols if col not in df.columns]
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                        st.info("💡 Required columns: symbol, quantity, average_cost_basis")
                        st.info("💡 All other columns will be imported if present")
                        st.stop()
                    
                    # Import ALL columns - let the database methods handle what they need
                    st.dataframe(df)
                    if st.button("Import Holdings"):
                        tracker.db_manager.update_holdings(df)
                        st.success("Holdings imported successfully!")
                except Exception as e:
                    st.error(f"Failed to import holdings: {e}")

            uploaded_tx = st.file_uploader(
                "Import Transactions (CSV/Excel)", type=["csv", "xlsx"], key="import_tx"
            )
            if uploaded_tx:
                try:
                    if uploaded_tx.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_tx)
                    else:
                        df = pd.read_excel(uploaded_tx)
                    
                    # Validate required columns but import everything
                    required_tx_cols = ["symbol", "timestamp", "type", "quantity"]
                    missing_cols = [col for col in required_tx_cols if col not in df.columns]
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                        st.info("💡 Required columns: symbol, timestamp, type, quantity")
                        st.info("💡 All other columns will be imported if present")
                        st.stop()
                    
                    # Validate transaction types
                    valid_types = ['BUY', 'SELL', 'DEPOSIT', 'WITHDRAWAL', 'TRANSFER']
                    invalid_types = df[~df['type'].isin(valid_types)]['type'].unique()
                    if len(invalid_types) > 0:
                        st.error(f"❌ Invalid transaction types found: {invalid_types}")
                        st.info(f"💡 Valid types: {valid_types}")
                        st.stop()
                    
                    # Import ALL columns - let the database methods handle what they need
                    st.dataframe(df)
                    if st.button("Import Transactions"):
                        st.warning("⚠️ This will update existing transactions and may create duplicates if transaction_hash is missing.")
                        if st.button("Confirm Import"):
                            try:
                                transactions_list = df.to_dict(orient="records")
                                rows_affected = tracker.db_manager.bulk_insert_transactions(transactions_list)
                                st.success(f"✅ Successfully imported {rows_affected} transactions!")
                            except Exception as e:
                                st.error(f"❌ Failed to import transactions: {e}")
                except Exception as e:
                    st.error(f"Failed to read transaction file: {e}")

        # --- 3. Manage Snapshots ---
        with tab3:
            st.header("📸 Snapshots")
            try:
                snapshots = tracker.db_manager.get_all_snapshots()
                # Don't filter out snapshots with NaN timestamps - let the chart handle it
                # snapshots = snapshots[~pd.isna(snapshots["timestamp"])].copy()
                snapshots = snapshots.drop_duplicates(subset=["timestamp"])
                
                # Set timestamp as index for proper charting
                if not snapshots.empty and "timestamp" in snapshots.columns:
                    # Convert timestamp to datetime first
                    snapshots['timestamp'] = pd.to_datetime(snapshots['timestamp'])
                    snapshots = snapshots.set_index("timestamp")
                    st.write("Successfully set timestamp as index")
                else:
                    st.write("No timestamp column found or empty snapshots")
                
            except Exception as e:
                st.error(f"Failed to load snapshots: {e}")
                snapshots = None
            
            if snapshots.empty:
                st.info("No snapshots found. Take a snapshot from the dashboard first.")
            else:
                # Sort by timestamp, handling None values properly
                snapshots = snapshots.sort_values("timestamp", na_position='first')

                # Include ALL snapshots (including problematic ones) so users can delete them
                snapshot_labels = []
                for idx, row in snapshots.iterrows():
                    # Check for invalid snapshots: no timestamp OR all zero values
                    # Use row.name to access the timestamp since it's now the index
                    timestamp_value = row.name
                    is_no_timestamp = pd.isna(timestamp_value) or timestamp_value is None or str(timestamp_value).lower() in ['none', 'nan', 'nat', 'null', '']
                    is_zero_values = (
                        row["total_value_usd"] == 0.0
                        and row["total_cost_basis_usd"] == 0.0
                        and row["unrealized_pl_usd"] == 0.0
                        and row["unrealized_pl_percent"] == 0.0
                    )

                    if is_no_timestamp:
                        label = f"⚠️ Invalid Snapshot (No Timestamp) | Value: ${row['total_value_usd']:,.2f}"
                    elif is_zero_values:
                        label = f"⚠️ Invalid Snapshot (Zero Values) | {timestamp_value} | Value: ${row['total_value_usd']:,.2f}"
                    else:
                        label = f"{timestamp_value} | Value: ${row['total_value_usd']:,.2f}"
                    snapshot_labels.append(label)

                selected_idx = st.selectbox(
                    "Select Snapshot",
                    range(len(snapshots)),
                    format_func=lambda i: snapshot_labels[i],
                )
                selected_row = snapshots.iloc[selected_idx]

                st.write("### Snapshot Details")
                display_dict = {
                    "Timestamp": str(selected_row.name),  # Use .name to access the index
                    "Total Value (USD)": f"${selected_row['total_value_usd']:,.2f}",
                    "Total Cost Basis (USD)": f"${selected_row['total_cost_basis_usd']:,.2f}",
                    "Unrealized P/L (USD)": f"${selected_row['unrealized_pl_usd']:,.2f}",
                    "Unrealized P/L (%)": f"{selected_row['unrealized_pl_percent']:.2f}%",
                }
                st.table(pd.DataFrame([display_dict]))

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Snapshot (CSV)",
                        selected_row.to_frame().T.to_csv(index=False),
                        "portfolio_snapshot.csv",
                    )
                with col2:
                    if st.button("Delete Selected Snapshot"):
                        try:
                            rows_deleted = tracker.db_manager.delete_snapshot(
                                selected_row.name,  # Use .name to access the timestamp index
                                selected_row["total_value_usd"],
                                selected_row["total_cost_basis_usd"],
                                selected_row["unrealized_pl_usd"],
                                selected_row["unrealized_pl_percent"],
                            )
                            if rows_deleted > 0:
                                st.success("✅ Successfully deleted snapshot.")
                                st.rerun()
                            else:
                                st.warning(
                                    "⚠️ No rows were deleted. The snapshot may have already been removed or the query didn't match any records."
                                )
                        except Exception as e:
                            st.error(f"❌ Failed to delete snapshot: {e}")

            st.markdown("---")

            # --- Improved Data Cleanup with Context and Confirmation ---
            st.subheader("🗑️ Data Cleanup")

            # Get cleanup configuration
            cleanup_days = tracker.config.get("database", {}).get("cleanup_days", 90)

            # Show current configuration
            st.info(f"**Current Retention Period:** {cleanup_days} days")
            if cleanup_days <= 0:
                st.warning("⚠️ Data cleanup is currently disabled (cleanup_days = 0)")
                st.stop()

            # Calculate what would be deleted
            cutoff_date = datetime.now() - timedelta(days=cleanup_days)

            # Get cleanup statistics
            stats = tracker.db_manager.get_cleanup_statistics()

            if not stats["cleanup_enabled"]:
                st.warning("⚠️ Data cleanup is currently disabled (cleanup_days = 0)")
                st.stop()

            if "error" in stats:
                st.error(f"Could not analyze database: {stats['error']}")
                st.stop()

            old_transactions = stats["old_transactions"]
            old_snapshots = stats["old_snapshots"]
            total_transactions = stats["total_transactions"]
            total_snapshots = stats["total_snapshots"]
            cutoff_date = stats["cutoff_date"]

            # Display what will be deleted
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "📊 Old Transactions",
                    f"{old_transactions:,}",
                    f"of {total_transactions:,} total",
                )
            with col2:
                st.metric(
                    "📸 Old Snapshots",
                    f"{old_snapshots:,}",
                    f"of {total_snapshots:,} total",
                )

            # Show cutoff date
            st.write(f"**Cutoff Date:** {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")

            # Warning about what this affects
            if old_transactions > 0 or old_snapshots > 0:
                st.warning("""
                ⚠️ **This will permanently delete:**
                - Historical transaction data older than the retention period
                - Portfolio snapshots older than the retention period
                - **Impact:** This may affect tax reporting, historical analysis, and portfolio tracking
                """)

                # Confirmation section
                st.markdown("---")
                st.subheader("🔐 Confirmation Required")

                # Two-step confirmation
                if "cleanup_confirmed" not in st.session_state:
                    st.session_state.cleanup_confirmed = False

                if not st.session_state.cleanup_confirmed:
                    if st.button(
                        "🗑️ I understand - Show Final Confirmation", type="secondary"
                    ):
                        st.session_state.cleanup_confirmed = True
                        st.rerun()
                else:
                    st.error("""
                    🚨 **FINAL WARNING:**
                    - This action is **IRREVERSIBLE**
                    - No backup will be created automatically
                    - Consider creating a backup first
                    """)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ CONFIRM DELETION", type="primary"):
                            try:
                                # Create backup before deletion
                                backup_path = tracker.db_manager.backup_database()
                                if backup_path:
                                    st.success(f"✅ Backup created: {backup_path}")

                                # Perform cleanup
                                tracker.cleanup_old_data()
                                st.success("✅ Data cleanup completed successfully!")
                                st.session_state.cleanup_confirmed = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Cleanup failed: {e}")
                                st.session_state.cleanup_confirmed = False

                    with col2:
                        if st.button("❌ Cancel", type="secondary"):
                            st.session_state.cleanup_confirmed = False
                            st.rerun()
            else:
                st.success("✅ No old data to clean up!")
                st.info("All your data is within the retention period.")

        # --- 4. Database Info ---
        with tab4:
            st.header("ℹ️ Database Info")
            st.write(f"**Database Path:** `{db_path}`")
            if os.path.exists(db_path):
                st.write(f"**Size:** {os.path.getsize(db_path) / 1024:.1f} KB")
                st.write(
                    f"**Last Modified:** {pd.to_datetime(os.path.getmtime(db_path), unit='s')}"
                )
            else:
                st.warning("Database file not found.")
            st.write(f"**Backup Directory:** `{backup_dir}`")
            if backup_dir.exists():
                st.write(
                    f"**Backups Available:** {len(list(backup_dir.glob('*.bak')))}"
                )
            else:
                st.warning("Backup directory not found.")

        # Add this debug section in the snapshots tab, right after loading snapshots:

        # Debug section - add this after loading snapshots
        with st.expander("🔍 Debug: Raw Database Data"):
            try:
                # Get raw data from database
                with tracker.db_manager._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM portfolio_snapshots ORDER BY timestamp")
                    raw_rows = cursor.fetchall()
                    
                st.write(f"**Total snapshots in database:** {len(raw_rows)}")
                
                if raw_rows:
                    st.write("**Raw database rows:**")
                    for i, row in enumerate(raw_rows):
                        st.write(f"Row {i+1}: timestamp='{row[0]}', value={row[1]}, cost={row[2]}, pl={row[3]}, pl_pct={row[4]}")
                
                st.write("**Pandas DataFrame after parsing:**")
                st.dataframe(snapshots)
                
            except Exception as e:
                st.error(f"Debug failed: {e}")

    def render_settings_and_status(self):
        import os
        import psutil
        import platform
        import getpass
        import json
        import re
        from pathlib import Path
        from datetime import datetime

        st.markdown("## ⚙️ Settings")

        tab_config, tab_status = st.tabs(["⚙️ Configuration", "🔧 System Status"])

        # --- Configuration Tab ---
        with tab_config:
            st.header("⚙️ Configuration")
            config = self.config_manager.config

            # Configuration file locations
            st.markdown("---")
            st.subheader(f"📁 Configuration locations")
            st.success(f"📁 Config file: `{self.config_manager.config_file_path}`\n\n")
            st.success(f"📁 Environment file: `{self.config_manager.env_path}`")
            st.markdown("---")

            # Create responsive columns for configuration sections
            col_portfolio, col_api = st.columns(2)

            # --- Portfolio Settings ---
            with col_portfolio:
                with st.container():
                    st.subheader("💼 Portfolio Settings")

                    with st.expander("💰 Trading Configuration", expanded=False):
                        min_trade_usd = config.get("portfolio", {}).get(
                            "minimum_trade_usd", 5.0
                        )
                        new_min_trade_usd = st.number_input(
                            "💵 Minimum Trade (USD)",
                            min_value=1.0,
                            max_value=10000.0,
                            value=float(min_trade_usd),
                            step=1.0,
                            help="💵 Minimum USD value for trades. Smaller trades are ignored.",
                        )

                    with st.expander("⚙️ Trading Mode", expanded=False):
                        live_enabled_config = config.get("portfolio", {}).get(
                            "live_trading_enabled", False
                        )
                        new_live = st.toggle(
                            "🔴 Enable Live Trading",
                            value=live_enabled_config,
                            help="🔴 Enable real trades; otherwise, trades are simulated (dry run).",
                        )

                        testnet_config = config.get("portfolio", {}).get(
                            "testnet_mode", False
                        )
                        new_testnet = st.toggle(
                            "🧪 Enable Binance Testnet Mode",
                            value=testnet_config,
                            help="🧪 Switch between Binance mainnet and testnet.",
                        )

                    with st.expander("💱 Currency Settings", expanded=False):
                        p2p_fiat = config.get("portfolio", {}).get(
                            "p2p_fiat_currency", "USD"
                        )
                        new_p2p_fiat = st.text_input(
                            "💱 P2P Fiat Currency",
                            value=p2p_fiat,
                            help="💱 Fiat currency code (e.g., USD, EUR, PHP) for P2P trades.",
                        )

                        crypto_quotes = config.get("portfolio", {}).get(
                            "crypto_quotes", []
                        )
                        new_crypto_quotes = st.text_input(
                            "₿ Crypto Quotes (comma-separated)",
                            value=", ".join(crypto_quotes),
                            help="₿ Crypto quote symbols (e.g., BTC, ETH, USDT), comma-separated.",
                        )

                        stablecoins = config.get("portfolio", {}).get(
                            "stablecoin_symbols", ["USDT"]
                        )
                        new_stablecoins = st.text_input(
                            "🟢 Stablecoin Symbols (comma-separated)",
                            value=", ".join(stablecoins),
                            help="🟢 Stablecoin symbols (e.g., USDT, USDC), comma-separated.",
                        )

                    # Save Portfolio Settings
                    st.markdown("---")
                    if st.button(
                        "💾 Save Portfolio Settings",
                        use_container_width=True,
                        type="primary",
                    ):
                        try:
                            config["portfolio"]["minimum_trade_usd"] = new_min_trade_usd
                            config["portfolio"]["live_trading_enabled"] = new_live
                            config["portfolio"]["testnet_mode"] = new_testnet
                            config["portfolio"]["p2p_fiat_currency"] = (
                                new_p2p_fiat.strip().upper()
                            )
                            config["portfolio"]["crypto_quotes"] = [
                                s.strip().upper()
                                for s in new_crypto_quotes.split(",")
                                if s.strip()
                            ]
                            config["portfolio"]["stablecoin_symbols"] = [
                                s.strip().upper()
                                for s in new_stablecoins.split(",")
                                if s.strip()
                            ]
                            self.config_manager.save_config()
                            st.success("✅ Portfolio settings updated and applied!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to save portfolio settings: {str(e)}")

            # --- API Settings ---
            with col_api:
                with st.container():
                    st.subheader("🔌 API Settings")

                    with st.expander("⏱️ Timeout Settings", expanded=False):
                        coingecko_timeout = (
                            config.get("apis", {})
                            .get("coingecko", {})
                            .get("timeout", 30)
                        )
                        binance_timeout = (
                            config.get("apis", {}).get("binance", {}).get("timeout", 60)
                        )

                        new_cg_timeout = st.number_input(
                            "CoinGecko Timeout (s)",
                            min_value=5,
                            max_value=120,
                            value=coingecko_timeout,
                            key="cg_timeout",
                            help="⏰ Maximum time (in seconds) to wait for a response from CoinGecko API.",
                        )
                        new_bi_timeout = st.number_input(
                            "Binance Timeout (s)",
                            min_value=5,
                            max_value=120,
                            value=binance_timeout,
                            key="bi_timeout",
                            help="⏰ Maximum time (in seconds) to wait for a response from Binance API.",
                        )

                    with st.expander("⚡ Performance Settings", expanded=False):
                        binance_recv_window = (
                            config.get("apis", {})
                            .get("binance", {})
                            .get("recv_window", 20000)
                        )
                        new_bi_recv_window = st.number_input(
                            "Binance Recv Window (ms)",
                            min_value=1000,
                            max_value=120000,
                            value=binance_recv_window,
                            step=1000,
                            key="bi_recv_window",
                            help="📡 Binance API recvWindow parameter (ms). Increase if timestamp errors occur.",
                        )

                        binance_delay = (
                            config.get("apis", {})
                            .get("binance", {})
                            .get("request_delay_ms", 500)
                        )
                        coingecko_delay = (
                            config.get("apis", {})
                            .get("coingecko", {})
                            .get("request_delay_ms", 1500)
                        )

                        new_bi_delay = st.number_input(
                            "Binance Request Delay (ms)",
                            min_value=0,
                            max_value=10000,
                            value=binance_delay,
                            step=100,
                            key="bi_delay",
                            help="⚡ Delay (ms) between Binance API requests to avoid rate limits.",
                        )
                        new_cg_delay = st.number_input(
                            "CoinGecko Request Delay (ms)",
                            min_value=0,
                            max_value=10000,
                            value=coingecko_delay,
                            step=100,
                            key="cg_delay",
                            help="⚡ Delay (ms) between CoinGecko API requests to avoid rate limits.",
                        )

                    with st.expander("📅 Historical Data Settings", expanded=False):
                        st.markdown("**Data Lookback Periods**")
                        lookback = config.get("history_lookback_days", {})
                        lookback_types = [
                            ("trades", 90, "💼"),
                            ("deposits", 90, "💰"),
                            ("withdrawals", 90, "💸"),
                            ("p2p_buys", 90, "🤝"),
                            ("internal_transfers", 90, "🔄"),
                            ("spot_futures_transfers", 90, "📈"),
                            ("spot_convert_history", 90, "🔄"),
                            ("simple_earn_rewards", 90, "💎"),
                            ("simple_earn_subscriptions", 90, "📊"),
                            ("simple_earn_redemptions", 90, "💰"),
                            ("dividend_history", 90, "📈"),
                            ("staking_history", 90, "🏦"),
                        ]

                        new_lookback = {}
                        for key, default, emoji in lookback_types:
                            display_name = f"{emoji} {key.replace('_', ' ').title()}"
                            new_lookback[key] = st.number_input(
                                display_name,
                                min_value=1,
                                max_value=3650,
                                value=int(lookback.get(key, default)),
                                key=f"lookback_{key}",
                                help=f"📅 Days of {key.replace('_', ' ')} history to fetch from API.",
                            )

                    # Save API Settings with improved styling
                    st.markdown("---")
                    if st.button(
                        "💾 Save API Settings", use_container_width=True, type="primary"
                    ):
                        try:
                            config["apis"]["coingecko"]["timeout"] = new_cg_timeout
                            config["apis"]["binance"]["timeout"] = new_bi_timeout
                            config["apis"]["binance"]["recv_window"] = (
                                new_bi_recv_window
                            )
                            config["apis"]["binance"]["request_delay_ms"] = new_bi_delay
                            config["apis"]["coingecko"]["request_delay_ms"] = (
                                new_cg_delay
                            )
                            config["history_lookback_days"] = new_lookback
                            self.config_manager.save_config()
                            st.success("✅ API settings updated and applied!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to save API settings: {str(e)}")

            col_trend, col_logging = st.columns(2)

            # --- Logging Settings ---
            with col_logging:
                with st.container():
                    st.subheader("📝 Logging Settings")

                    with st.expander("📊 Log Configuration", expanded=False):
                        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                        level_icons = {
                            "DEBUG": "🔍",
                            "INFO": "ℹ️",
                            "WARNING": "⚠️",
                            "ERROR": "❌",
                            "CRITICAL": "🚨",
                        }
                        current_level = config.get("logging", {}).get("level", "INFO")

                        new_level = st.selectbox(
                            "📊 Log Level",
                            log_levels,
                            index=log_levels.index(current_level),
                            format_func=lambda x: f"{level_icons.get(x, '')} {x}",
                            key="log_level",
                            help="📊 Set verbosity of logs. DEBUG is most verbose, CRITICAL is least.",
                        )

                        file_config = config.get("logging", {}).get("file_config", {})
                        file_enabled = file_config.get("enabled", True)
                        new_file_enabled = st.toggle(
                            "📄 Enable File Logging",
                            value=file_enabled,
                            key="file_logging_toggle",
                            help="📄 Write logs to a file for persistent storage.",
                        )

                        log_path = file_config.get("path", "logs/portfolio_tracker.log")
                        new_log_path = st.text_input(
                            "📁 Log File Path",
                            value=log_path,
                            key="log_file_path",
                            help="📁 Path to store log file. Ensure directory is writable.",
                        )

                        console_config = config.get("logging", {}).get(
                            "console_config", {}
                        )
                        console_enabled = console_config.get("enabled", True)
                        new_console_enabled = st.toggle(
                            "🖥️ Enable Console Logging",
                            value=console_enabled,
                            key="console_logging_toggle",
                            help="🖥️ Output logs to console for real-time monitoring.",
                        )

                    with st.expander("👀 Log Preview", expanded=False):
                        preview_lines = st.number_input(
                            "📄 Log Preview Lines",
                            min_value=1,
                            max_value=100,
                            value=10,
                            key="log_preview_lines",
                            help="📄 Number of recent log lines to display in preview.",
                        )

                        if os.path.exists(log_path):
                            try:
                                with open(log_path, "r") as f:
                                    lines = f.readlines()[-preview_lines:]
                                if lines:
                                    st.code("".join(lines), language="text")
                                else:
                                    st.info("📄 Log file is empty")
                            except Exception as e:
                                st.error(f"❌ Failed to read log file: {e}")
                        else:
                            st.info("📄 Log file not found")

                    # Save Logging Settings
                    st.markdown("---")
                    # col_save, col_clear = st.columns(2)
                    # with col_save:
                    if st.button(
                        "💾 Save Logging Settings",
                        use_container_width=True,
                        type="primary",
                    ):
                        try:
                            # Validate log file path
                            log_path_obj = Path(new_log_path).parent
                            log_path_obj.mkdir(parents=True, exist_ok=True)
                            if not os.access(log_path_obj, os.W_OK):
                                st.error("❌ Log file directory is not writable!")
                            else:
                                config["logging"]["level"] = new_level
                                config["logging"]["file_config"]["enabled"] = (
                                    new_file_enabled
                                )
                                config["logging"]["file_config"]["path"] = new_log_path
                                config["logging"]["console_config"]["enabled"] = (
                                    new_console_enabled
                                )
                                self.config_manager.save_config()
                                self.setup_logging(level_override=new_level)
                                st.success(
                                    f"✅ Logging settings updated! Level: {level_icons.get(new_level, '')} {new_level}"
                                )
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to save logging settings: {str(e)}")

            # --- Trend Analyzer Settings ---
            with col_trend:
                with st.container():
                    st.subheader("📈 Trend Analyzer")
                    trend_config = config.get("trend_analyzer", {})

                    with st.expander("₿ Cryptocurrencies", expanded=False):
                        cryptocurrencies = trend_config.get("cryptocurrencies", [])
                        new_cryptocurrencies = st.text_input(
                            "₿ Cryptocurrencies (comma-separated)",
                            value=", ".join(cryptocurrencies),
                            key="trend_cryptos",
                            help="₿ Cryptocurrencies to analyze (e.g., BTC, ETH, SOL).",
                        )

                    with st.expander("📊 RSI Settings", expanded=False):
                        rsi_period = trend_config.get("rsi_period", 14)
                        new_rsi_period = st.number_input(
                            "⏱️ RSI Period",
                            min_value=1,
                            max_value=50,
                            value=rsi_period,
                            key="rsi_period",
                            help="⏱️ Period for Relative Strength Index calculation.",
                        )

                        rsi_oversold = trend_config.get("rsi_oversold", 30)
                        new_rsi_oversold = st.number_input(
                            "📉 RSI Oversold Threshold",
                            min_value=0,
                            max_value=100,
                            value=rsi_oversold,
                            key="rsi_oversold",
                            help="📉 RSI value below which an asset is considered oversold.",
                        )

                        rsi_overbought = trend_config.get("rsi_overbought", 70)
                        new_rsi_overbought = st.number_input(
                            "📈 RSI Overbought Threshold",
                            min_value=0,
                            max_value=100,
                            value=rsi_overbought,
                            key="rsi_overbought",
                            help="📈 RSI value above which an asset is considered overbought.",
                        )

                    with st.expander("⏰ Timeframe Settings", expanded=False):
                        timeframe_settings = trend_config.get("timeframe_settings", {})
                        timeframe_icons = {
                            "long_term": "📅",
                            "swing": "📊",
                            "day": "⚡",
                        }

                        for timeframe in ["long_term", "swing", "day"]:
                            # Ensure default settings exist
                            if timeframe not in timeframe_settings:
                                timeframe_settings[timeframe] = {
                                    "period": "1y"
                                    if timeframe == "long_term"
                                    else "90d"
                                    if timeframe == "swing"
                                    else "7d",
                                    "sma_short_window": 10,
                                    "sma_long_window": 30,
                                }

                            settings = timeframe_settings[timeframe]
                            current_period = settings.get(
                                "period",
                                "1y"
                                if timeframe == "long_term"
                                else "90d"
                                if timeframe == "swing"
                                else "7d",
                            )

                            with st.expander(
                                f"{timeframe_icons.get(timeframe, '📊')} {timeframe.replace('_', ' ').title()}",
                                expanded=False,
                            ):
                                new_period = st.text_input(
                                    f"⏰ Period",
                                    value=current_period,
                                    key=f"timeframe_{timeframe}_period",
                                    help="⏰ Period for analysis (e.g., 4y for years, 60d for days, 3mo for months).",
                                )

                                col_sma1, col_sma2 = st.columns(2)
                                with col_sma1:
                                    new_sma_short = st.number_input(
                                        f"📊 SMA Short",
                                        min_value=1,
                                        max_value=200,
                                        value=settings.get("sma_short_window", 10),
                                        key=f"timeframe_{timeframe}_sma_short",
                                        help="📊 Short-term Simple Moving Average window.",
                                    )
                                with col_sma2:
                                    new_sma_long = st.number_input(
                                        f"📈 SMA Long",
                                        min_value=1,
                                        max_value=200,
                                        value=settings.get("sma_long_window", 30),
                                        key=f"timeframe_{timeframe}_sma_long",
                                        help="📈 Long-term Simple Moving Average window.",
                                    )

                                timeframe_settings[timeframe] = {
                                    "period": new_period.strip(),
                                    "sma_short_window": new_sma_short,
                                    "sma_long_window": new_sma_long,
                                }

                    # Validation and Save
                    validation_errors = []

                    if new_rsi_oversold >= new_rsi_overbought:
                        validation_errors.append(
                            "RSI Oversold must be less than RSI Overbought!"
                        )

                    for timeframe, settings in timeframe_settings.items():
                        period = settings.get("period", "").strip()
                        if not period:
                            validation_errors.append(
                                f"Period for {timeframe} cannot be empty!"
                            )
                        elif not re.match(r"^\d+(y|d|mo)$", period):
                            validation_errors.append(
                                f"Invalid period format for {timeframe}: Use Xy, Xd, or Xmo"
                            )

                        if settings["sma_short_window"] >= settings["sma_long_window"]:
                            validation_errors.append(
                                f"Short SMA must be less than Long SMA for {timeframe}!"
                            )

                    if validation_errors:
                        for error in validation_errors:
                            st.error(f"❌ {error}")

                    st.markdown("---")
                    if st.button(
                        "💾 Save Trend Analyzer Settings",
                        use_container_width=True,
                        type="primary",
                        disabled=bool(validation_errors),
                    ):
                        try:
                            config["trend_analyzer"]["cryptocurrencies"] = [
                                s.strip().upper()
                                for s in new_cryptocurrencies.split(",")
                                if s.strip()
                            ]
                            config["trend_analyzer"]["rsi_period"] = new_rsi_period
                            config["trend_analyzer"]["rsi_oversold"] = new_rsi_oversold
                            config["trend_analyzer"]["rsi_overbought"] = (
                                new_rsi_overbought
                            )
                            config["trend_analyzer"]["timeframe_settings"] = (
                                timeframe_settings
                            )
                            self.config_manager.save_config()
                            st.success(
                                "✅ Trend Analyzer settings updated and applied!"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(
                                f"❌ Failed to save trend analyzer settings: {str(e)}"
                            )

            # --- Import/Export Config ---
            st.markdown("---")
            st.subheader("📁 Import/Export Configuration")

            col_export, col_import = st.columns(2)

            with col_export:
                with st.expander("📤 Export Configuration", expanded=False):
                    export_dir = config.get("exports", {}).get("path", "data/exports")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_export_path = (
                        Path(export_dir) / f"config_export_{timestamp}.json"
                    )

                    export_path = st.text_input(
                        "📁 Export Path",
                        value=str(default_export_path),
                        help="📁 Path to save the exported JSON config.",
                    )

                    if st.button("📤 Export Configuration", use_container_width=True):
                        try:
                            export_config = config.copy()
                            # Remove sensitive data
                            export_config.pop("main_api_keys", None)
                            export_config.pop("sub_accounts", None)

                            export_path_obj = Path(export_path)
                            export_path_obj.parent.mkdir(parents=True, exist_ok=True)

                            with open(export_path_obj, "w") as f:
                                json.dump(export_config, f, indent=2)

                            st.success(
                                f"✅ Configuration exported to `{export_path_obj}`"
                            )
                        except Exception as e:
                            st.error(f"❌ Failed to export configuration: {str(e)}")

            with col_import:
                with st.expander("📥 Import Configuration", expanded=False):
                    uploaded_file = st.file_uploader(
                        "📥 Upload Configuration (JSON)",
                        type="json",
                        help="📥 Upload a previously exported configuration file",
                    )

                    if uploaded_file:
                        try:
                            new_config = json.load(uploaded_file)

                            # Validation
                            default_config = self.config_manager.config
                            required_keys = {
                                key: type(value)
                                for key, value in default_config.items()
                                if key not in ["main_api_keys", "sub_accounts"]
                            }

                            validation_passed = True
                            validation_messages = []

                            # Validate structure
                            for key, expected_type in required_keys.items():
                                if key not in new_config:
                                    validation_messages.append(
                                        f"❌ Missing required key: '{key}'"
                                    )
                                    validation_passed = False
                                elif not isinstance(new_config[key], expected_type):
                                    validation_messages.append(
                                        f"❌ Invalid type for '{key}': expected {expected_type.__name__}"
                                    )
                                    validation_passed = False

                            # Display validation results
                            if validation_passed:
                                st.success("✅ Configuration file is valid!")

                                # Preview configuration
                                with st.expander(
                                    "👀 Preview Configuration", expanded=False
                                ):
                                    st.json(new_config)

                                confirm_import = st.checkbox(
                                    "⚠️ Confirm Import (will overwrite current config)",
                                    help="⚠️ This will replace your current settings with the imported configuration.",
                                )

                                if confirm_import and st.button(
                                    "📥 Apply Configuration",
                                    use_container_width=True,
                                    type="primary",
                                ):
                                    try:
                                        # Preserve sensitive data
                                        new_config["main_api_keys"] = config.get(
                                            "main_api_keys", {}
                                        )
                                        new_config["sub_accounts"] = config.get(
                                            "sub_accounts", {}
                                        )

                                        config.update(new_config)
                                        self.config_manager.save_config()

                                        st.success(
                                            "✅ Configuration imported successfully! Please restart the application."
                                        )
                                        st.rerun()
                                    except Exception as e:
                                        st.error(
                                            f"❌ Failed to apply configuration: {str(e)}"
                                        )
                            else:
                                st.error("❌ Configuration file validation failed:")
                                for msg in validation_messages:
                                    st.error(msg)

                        except json.JSONDecodeError as e:
                            st.error(f"❌ Invalid JSON file: {str(e)}")
                        except Exception as e:
                            st.error(
                                f"❌ Failed to process configuration file: {str(e)}"
                            )

        # --- System Status Tab ---
        with tab_status:
            st.header("🔧 System Status")
            st.markdown("---")

            # Create responsive columns for status sections
            col_system, col_services = st.columns(2)

            # --- System Information Column ---
            with col_system:
                with st.container():
                    st.subheader("🖥️ System Information")

                    # Application Info Expander
                    with st.expander("📱 Application Details", expanded=True):
                        app_metrics = {
                            "Version": config.get("version", "Unknown"),
                            "User": getpass.getuser(),
                            "Python": platform.python_version(),
                            "OS": platform.platform(terse=True),
                            "System Boot": datetime.fromtimestamp(
                                psutil.boot_time()
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        for label, value in app_metrics.items():
                            st.text(f"{label}: {value}")

                    # System Resources Expander
                    with st.expander("📊 System Resources", expanded=True):
                        res_col_cpu, res_col_ram, res_col_disk = st.columns(3)
                        with res_col_cpu:
                            cpu_percent = psutil.cpu_percent(interval=1)
                            st.metric("🖥️ CPU", f"{cpu_percent:.1f}%")
                        with res_col_ram:
                            ram_info = psutil.virtual_memory()
                            st.metric(
                                "💾 RAM",
                                f"{ram_info.percent}%",
                                f"{ram_info.used / (1024**3):.1f} GB",
                            )
                        with res_col_disk:
                            disk_info = psutil.disk_usage("/")
                            st.metric(
                                "💿 Disk",
                                f"{disk_info.percent}%",
                                f"{disk_info.used / (1024**3):.1f} GB",
                            )

            # --- Services & Storage Column ---
            with col_services:
                with st.container():
                    st.subheader("🔌 Services & Storage")

                    # API Status Expander
                    with st.expander("🔗 API Status", expanded=True):
                        if self.tracker.binance_client:
                            try:
                                self.tracker.binance_client.ping()
                                st.success("✅ Binance API: Connected")
                            except Exception as e:
                                st.error(f"❌ Binance API: Connection failed")
                        else:
                            st.warning("⚠️ Binance API: Not connected")

                        # Other API info
                        test_id = "BTC"
                        prices = self.tracker._get_current_prices([test_id])
                        price = prices.get(test_id)
                        if price:
                            st.success(f"🦎 CoinGecko: Available")
                        else:
                            st.warning(f"❌ CoinGecko: Not Available")
                        st.info("📈 YFinance: API available")

                    # Database & Storage Expander
                    with st.expander("💾 Database & Storage", expanded=True):
                        db_path = self.tracker.db_manager.db_path
                        if os.path.exists(db_path):
                            db_size_kb = os.path.getsize(db_path) / 1024
                            db_modified = datetime.fromtimestamp(
                                os.path.getmtime(db_path)
                            )
                            st.success(f"Database: `{os.path.basename(db_path)}`")
                            st.text(f"Size: {db_size_kb:.1f} KB")
                            st.text(
                                f"Last Modified: {db_modified.strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                        else:
                            st.error("❌ Database file not found")

                        st.markdown("---")
                        export_dir = config.get("exports", {}).get(
                            "path", "data/exports/"
                        )
                        cache_dir = config.get("cache", {}).get("path", "data/cache")
                        st.info(f"📁 Export Path: `{export_dir}`")
                        st.info(f"🗂️ Cache Path: `{cache_dir}`")

            # --- Logging Status (Full Width) ---
            st.markdown("---")
            with st.expander("📝 Logging Status", expanded=False):
                log_path = (
                    config.get("logging", {})
                    .get("file_config", {})
                    .get("path", "logs/portfolio_tracker.log")
                )

                log_level = config.get("logging", {}).get("level", "INFO")
                level_icons = {
                    "DEBUG": "🔍",
                    "INFO": "ℹ️",
                    "WARNING": "⚠️",
                    "ERROR": "❌",
                    "CRITICAL": "🚨",
                }
                icon = level_icons.get(log_level, "📝")
                st.info(f"**Current Log Level:** {icon} {log_level}")
                st.info(f"**Log File Path:** `{log_path}`")

                if os.path.exists(log_path):
                    try:
                        with open(log_path, "r") as f:
                            lines = f.readlines()[
                                -15:
                            ]  # Increased lines for better context
                        if lines:
                            st.code("".join(lines), language="log")
                        else:
                            st.info("📄 Log file is empty.")
                    except Exception as e:
                        st.error(f"❌ Failed to read log file: {e}")
                else:
                    st.warning("📄 Log file not found at the configured path.")

    def run(self):
        """Main application runner"""
        self.setup_logging()
        self.initialize_tracker()
        self.render_header()
        self.render_status_indicator()
        selected_page = self.render_sidebar()
        page_mapping = {
            "🏠 Home": self.render_home_page,
            "📈 Market": self.render_market_trends,
            "⚖️ Rebalance": self.render_rebalancing,
            "💰 Trade": self.render_trading,
            "🧪 Backtest": self.render_backtesting,
            "🗄️ Database": self.render_data_management,
            "⚙️ Settings": self.render_settings_and_status,
        }
        if selected_page in page_mapping:
            page_mapping[selected_page]()
        else:
            st.error("Page not found")


# Run the dashboard
if __name__ == "__main__":
    dashboard = PortfolioDashboard()
    dashboard.run()
