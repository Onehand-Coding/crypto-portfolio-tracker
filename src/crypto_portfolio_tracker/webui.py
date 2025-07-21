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
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.dates as mdates

from crypto_portfolio_tracker import trading_strategies
from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.strategy_backtester import StrategyBacktester
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker
from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer
from crypto_portfolio_tracker.rebalancing_backtester import RebalancingBacktester
from crypto_portfolio_tracker.exceptions import NetworkOperationError, NetworkUnavailableError
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

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

# Configure Streamlit page
st.set_page_config(
    page_title="Crypto Portfolio Tracker",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
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
""", unsafe_allow_html=True)


class PortfolioDashboard:
    """Main Streamlit Dashboard class"""

    def __init__(self):
        self.config_manager = None
        self.tracker = None
        self.offline_mode = False
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'tracker_initialized' not in st.session_state:
            st.session_state.tracker_initialized = False
        if 'offline_mode' not in st.session_state:
            st.session_state.offline_mode = False
        if 'last_sync' not in st.session_state:
            st.session_state.last_sync = None
        if 'portfolio_metrics' not in st.session_state:
            st.session_state.portfolio_metrics = None
        if "confirm_delete" not in st.session_state:
            st.session_state["confirm_delete"] = None

    def setup_logging(self, level_override: Optional[str] = None):
        """Setup logging configuration"""
        try:
            self.config_manager = ConfigManager()
            logging_config = self.config_manager.config.get("logging", {})
            log_level_str = level_override or logging_config.get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logging.getLogger("httpx").setLevel(logging.WARNING)
        except Exception as e:
            st.error(f"Failed to setup logging: {str(e)}")

    def initialize_tracker(self):
        """Initialize the portfolio tracker and always set self.tracker"""
        if st.session_state.get('tracker_initialized', False) and self.tracker is not None:
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
                self.tracker = CryptoPortfolioTracker(self.config_manager, force_offline=True)
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
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Crypto Portfolio Tracker v2.1.0</h1>
            <p>Professional Portfolio Management & Analysis Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

    def render_status_indicator(self):
        """Render connection status indicator"""
        if st.session_state.offline_mode:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background: #f8d7da; border-radius: 8px; margin-bottom: 1rem;">
                <span class="status-offline">⚠️ OFFLINE MODE - Network features are disabled</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background: #d4edda; border-radius: 8px; margin-bottom: 1rem;">
                <span class="status-online">✅ ONLINE - All features available</span>
            </div>
            """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with navigation and controls"""
        st.sidebar.markdown("""
        <div class="sidebar-section">
            <h3>📊 Dashboard Controls</h3>
        </div>
        """, unsafe_allow_html=True)

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
                "⚙️ Configuration",
                "🔧 System Status"
            ]
        )

        st.sidebar.markdown("""
        <div class="sidebar-section">
            <h4>⚡ Quick Actions</h4>
        </div>
        """, unsafe_allow_html=True)

        if st.sidebar.button("🔄 Sync Portfolio"):
            self.run_full_sync()

        if st.sidebar.button("💾 Save Snapshot"):
            if st.session_state.portfolio_metrics:
                self.save_snapshot()
            else:
                st.sidebar.warning("No metrics available to save")

        # Last sync info
        if st.session_state.last_sync:
            st.sidebar.markdown(f"""
            <div class="sidebar-section">
                <small>Last Sync: {st.session_state.last_sync}</small>
            </div>
            """, unsafe_allow_html=True)

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
                st.session_state.last_sync = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        if not metrics:
            st.info("No portfolio metrics available. Please sync first.")
            return

        # st.markdown("### 📊 Portfolio Summary")
        st.write(f"**Timestamp:** {metrics.get('timestamp', '')}")
        st.write(f"**Database:** {metrics.get('database', 'portfolio.db')}")
        st.write(f"**Total Portfolio Value:** {format_usd(metrics.get('total_value_usd', 0))}")
        st.write(f"**Spot & Earn Value:** {format_usd(metrics.get('spot_earn_value_usd', 0))}")
        st.write(f"**Futures Wallet Value:** {format_usd(metrics.get('futures_value_usd', 0))}")
        st.write(f"**Funding Wallet Value:** {format_usd(metrics.get('funding_value_usd', 0))}")
        st.write(f"**Total Invested Capital:** {format_usd(metrics.get('total_invested_capital', 0))}")
        st.write(f"**Overall P/L:** {format_usd(metrics.get('overall_pl_usd', 0))} ({format_percent(metrics.get('overall_pl_percent', 0))})")
        st.write(f"**Total Cost Basis (FIFO):** {format_usd(metrics.get('total_cost_basis_usd', 0))}")
        st.write(f"**Unrealized P/L (FIFO):** {format_usd(metrics.get('unrealized_pl_usd', 0))} ({format_percent(metrics.get('unrealized_pl_percent', 0))})")
        st.markdown("---")

        # All Holdings
        st.markdown("#### 🗂️ All Holdings")
        if isinstance(metrics["holdings_df"], str):
            all_df = parse_df_string(metrics["holdings_df"])
        else:
            all_df = metrics["holdings_df"]
        st.dataframe(build_holdings_table(all_df, alloc_col="allocation"), use_container_width=True)

        # Core Holdings
        st.markdown("#### 🎯 Core Holdings (Alloc. = % of core portfolio value)")
        if isinstance(metrics["core_holdings_df"], str):
            core_df = parse_df_string(metrics["core_holdings_df"])
        else:
            core_df = metrics["core_holdings_df"]
        st.dataframe(build_holdings_table(core_df, alloc_col="core_allocation"), use_container_width=True)

        # Other Holdings
        st.markdown("#### 📈 Other Holdings (Alloc. = % of total portfolio value)")
        if isinstance(metrics["other_holdings_df"], str):
            other_df = parse_df_string(metrics["other_holdings_df"])
        else:
            other_df = metrics["other_holdings_df"]
        st.dataframe(build_holdings_table(other_df, alloc_col="allocation"), use_container_width=True)

        # Futures Wallet Summary
        st.markdown("#### 💹 Futures Wallet Summary")
        if metrics.get("futures_balances"):
            fut_df = pd.DataFrame(metrics["futures_balances"])
            fut_clean = clean_futures_balances(fut_df)
            if fut_clean is not None:
                st.dataframe(fut_clean, use_container_width=True)
            else:
                st.info("No balances found.")
        else:
            st.info("No balances found.")

        # Funding Wallet Summary
        st.markdown("#### 💰 Funding Wallet Summary")
        if metrics.get("funding_balances"):
            fund_df = pd.DataFrame(metrics["funding_balances"])
            fund_clean = clean_funding_balances(fund_df)
            if fund_clean is not None:
                st.dataframe(fund_clean, use_container_width=True)
            else:
                st.info("No balances found.")
        else:
            st.info("No balances found.")

    def render_home_page(self):
        import io
        from pathlib import Path
        import pandas as pd
        import matplotlib.pyplot as plt
        import plotly.express as px
        import plotly.graph_objects as go

        tracker = self.initialize_tracker()
        st.markdown("## 🏠 Home")

        tracker = self.initialize_tracker()
        metrics = st.session_state.get("portfolio_metrics")
        if not metrics:
            st.info("No portfolio metrics available. Please run a full sync first.")
            return

        holdings_df = metrics.get("holdings_df")
        target_allocation = self.config_manager.config.get("target_allocation", {})
        snapshots = None
        try:
            snapshots = tracker.db_manager.get_all_snapshots()
        except Exception:
            snapshots = None

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Performance",
            "📝 Trade Log",
            "🧾 Tax Report",
            "📂 View Exports",
            "🟣 Allocation Pie",
            "🟡 Allocation Comparison",
            "🟢 P/L by Asset",
            "🔵 Value Over Time"
        ])

        # --- 1. Portfolio Performance ---
        with tab1:
            st.header("📊 Portfolio Performance")
            if st.button("🔄 Calculate Portfolio Metrics"):
                if not st.session_state.offline_mode:
                    with st.spinner("Calculating portfolio metrics..."):
                        metrics = asyncio.run(tracker.calculate_portfolio_metrics())
                        st.session_state.portfolio_metrics = metrics
                        st.success("Portfolio metrics calculated!")
                else:
                    st.error("Portfolio metrics calculation unavailable in offline mode")
            if not metrics:
                st.info("No portfolio metrics available. Please run a full sync first.")
            else:
                self.display_metrics(metrics)
                st.info("For a full, professional report, use the export buttons below.")
                export_dir = Path(self.config_manager.config.get("exports", {}).get("path", "data/exports/"))
                export_dir.mkdir(parents=True, exist_ok=True)
                if st.button("Export to Excel"):
                    tracker.export_to_excel(metrics)
                    st.success("Excel report exported!")
                if st.button("Export to HTML"):
                    tracker.export_to_html(metrics)
                    st.success("HTML report exported!")

        # --- 2. Trade Log ---
        with tab2:
            st.header("📝 Trade Log")
            db_manager = tracker.db_manager
            try:
                tx_df = db_manager.get_all_transactions()
            except Exception as e:
                st.error(f"Failed to load transactions: {e}")
                tx_df = pd.DataFrame()
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
                    "Date/Time", "Asset", "Type", "Side", "Quantity", "Price (USD)",
                    "Fee (USD)", "Trade Source", "Notes"
                ]
                drop_cols = [col for col in ["id", "asset_id"] if col in tx_df.columns]
                display_df = tx_df.drop(columns=drop_cols, errors="ignore")
                display_df = display_df.rename(columns=col_map)

                # Explicitly set column order for display
                ordered_cols = [col for col in desired_order if col in display_df.columns]
                display_df = display_df[ordered_cols]

                display_df = display_df.reset_index(drop=True)
                st.dataframe(display_df)
                st.download_button("Download Trade Log (CSV)", tx_df.to_csv(index=False), "trade_log.csv")

        # --- 4. Tax Report ---
        with tab3:
            st.header("🧾 Tax Report")
            try:
                tx_df = db_manager.get_all_transactions()
            except Exception as e:
                st.error(f"Failed to load transactions: {e}")
                tx_df = pd.DataFrame()
            if tx_df.empty:
                st.info("No transactions found.")
            else:
                tax_df = calculate_fifo_realized_gains(tx_df)
                if tax_df.empty:
                    st.info("No taxable events found.")
                else:
                    tax_df["year"] = pd.to_datetime(tax_df["date"]).dt.year
                    col1, col2 = st.columns(2)
                    with col1:
                        year = st.selectbox("Year", ["All"] + sorted(tax_df["year"].unique().astype(str)), key="tax_year")
                    with col2:
                        asset = st.selectbox("Asset", ["All"] + sorted(tax_df["symbol"].unique()), key="tax_asset")
                    filtered = tax_df.copy()
                    if year != "All":
                        filtered = filtered[filtered["year"].astype(str) == year]
                    if asset != "All":
                        filtered = filtered[filtered["symbol"] == asset]
                    tax_col_map = {
                        "date": "Date",
                        "symbol": "Asset",
                        "quantity": "Quantity",
                        "proceeds_usd": "Proceeds (USD)",
                        "cost_basis_usd": "Cost Basis (USD)",
                        "gain_usd": "Realized Gain (USD)",
                        "year": "Year"
                    }
                    filtered = filtered.rename(columns=tax_col_map)
                    for col in ["Proceeds (USD)", "Cost Basis (USD)", "Realized Gain (USD)"]:
                        if col in filtered.columns:
                            filtered[col] = filtered[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
                    if "Quantity" in filtered.columns:
                        filtered["Quantity"] = filtered["Quantity"].map(lambda x: f"{x:,.6f}" if pd.notnull(x) else "")
                    st.dataframe(filtered)
                    st.markdown("#### Summary")
                    # 1. Group and aggregate on raw numeric columns
                    summary = tax_df.copy()
                    if year != "All":
                        summary = summary[summary["year"].astype(str) == year]
                    if asset != "All":
                        summary = summary[summary["symbol"] == asset]

                    summary = summary.groupby(["year", "symbol"]).agg(
                        total_gain_usd=("gain_usd", "sum"),
                        total_proceeds_usd=("proceeds_usd", "sum"),
                        total_cost_basis_usd=("cost_basis_usd", "sum"),
                        total_quantity=("quantity", "sum"),
                    ).reset_index()

                    # 2. Format the summary for display
                    summary = summary.rename(columns={
                        "year": "Year",
                        "symbol": "Asset",
                        "total_gain_usd": "Total Realized Gain (USD)",
                        "total_proceeds_usd": "Total Proceeds (USD)",
                        "total_cost_basis_usd": "Total Cost Basis (USD)",
                        "total_quantity": "Total Quantity"
                    })
                    for col in ["Total Realized Gain (USD)", "Total Proceeds (USD)", "Total Cost Basis (USD)"]:
                        summary[col] = summary[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
                    if "Total Quantity" in summary.columns:
                        summary["Total Quantity"] = summary["Total Quantity"].map(lambda x: f"{x:,.6f}" if pd.notnull(x) else "")

                    st.dataframe(summary)
                    st.info("For a full, professional tax report, use the export buttons below.")
                    export_dir = Path(self.config_manager.config.get("exports", {}).get("path", "data/exports/"))
                    export_dir.mkdir(parents=True, exist_ok=True)
                    if st.button("Export Tax Report to Excel"):
                        # Remove timezone info from all datetime columns
                        for col in tax_df.select_dtypes(include=["datetimetz"]).columns:
                            tax_df[col] = tax_df[col].dt.tz_localize(None)
                        filename = f"tax_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        tax_df.to_excel(export_dir / filename, index=False)
                        st.success(f"Excel tax report exported as {filename}!")
                    if st.button("Export Tax Report to HTML"):
                        filename = f"tax_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        template_path = Path(__file__).parent / "templates"
                        env = Environment(loader=FileSystemLoader(template_path))
                        template = env.get_template("tax_report_template.html")
                        html_content = template.render(
                            tax_table=tax_df.to_html(index=False, classes='table table-striped'),
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        with open(export_dir / filename, "w") as f:
                            f.write(html_content)
                        st.success(f"HTML tax report exported as {filename}!")

        # --- 5. View Exported Data ---
        with tab4:
            st.header("📂 View Exported Data")
            export_dir = Path(self.config_manager.config.get("exports", {}).get("path", "data/exports/"))
            export_dir.mkdir(parents=True, exist_ok=True)

            # Filter out trend report exports
            files = sorted(
                [f for f in export_dir.glob("*.*") if not f.name.startswith("trend_report_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if not files:
                st.info("No exported files found.")
            else:
                file_names = [f.name for f in files]
                selected_file = st.selectbox("Select Exported File", file_names)
                file_path = export_dir / selected_file
                st.markdown(f"**File:** `{selected_file}`")

                # --- Preview logic ---
                previewed = False
                if selected_file.endswith(".csv"):
                    try:
                        df = pd.read_csv(file_path)
                        st.dataframe(df)
                        previewed = True
                    except Exception as e:
                        st.error(f"Failed to preview CSV: {e}")
                elif selected_file.endswith(".xlsx"):
                    try:
                        df = pd.read_excel(file_path)
                        st.dataframe(df)
                        previewed = True
                    except Exception as e:
                        st.error(f"Failed to preview Excel: {e}")
                elif selected_file.endswith(".html"):
                    try:
                        with open(file_path, "r") as f:
                            html = f.read()
                        st.components.v1.html(html, height=600, scrolling=True)
                        previewed = True
                    except Exception as e:
                        st.error(f"Failed to preview HTML: {e}")
                elif selected_file.endswith(".json"):
                    try:
                        import json
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        st.json(data)
                        previewed = True
                    except Exception as e:
                        st.error(f"Failed to preview JSON: {e}")
                else:
                    st.info("Preview not supported for this file type.")

                # --- Buttons below preview ---
                col1, col2 = st.columns(2)
                with col1:
                    with open(file_path, "rb") as f:
                        st.download_button("Download", f, file_name=selected_file, use_container_width=True)
                with col2:
                    if st.button("Delete", key=f"delete_{selected_file}", use_container_width=True):
                        try:
                            file_path.unlink()
                            st.success(f"Deleted {selected_file}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

        # --- 1. Portfolio Allocation Pie Chart ---
        with tab5:
            st.subheader("Portfolio Allocation (by Value)")
            if holdings_df is not None and not holdings_df.empty:
                pie_df = holdings_df.copy()
                pie_df = pie_df[pie_df["value_usd"] > 0]
                # Plotly chart for interactivity
                fig_plotly = px.pie(
                    pie_df,
                    names="symbol",
                    values="value_usd",
                    title="Portfolio Allocation by Value",
                    hole=0.4
                )
                st.plotly_chart(fig_plotly, use_container_width=True)
                # Matplotlib chart for universal PNG download
                fig, ax = plt.subplots()
                ax.pie(pie_df["value_usd"], labels=pie_df["symbol"], autopct='%1.1f%%', startangle=90)
                ax.set_title("Portfolio Allocation by Value")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    "Download Pie Chart (PNG, universal)",
                    buf,
                    file_name="portfolio_allocation_pie.png",
                    mime="image/png"
                )
                plt.close(fig)
            else:
                st.info("No holdings data available for allocation pie chart.")

        # --- 2. Current vs. Target Allocation Bar Chart ---
        with tab6:
            st.subheader("Current vs. Target Allocation")
            if holdings_df is not None and not holdings_df.empty and target_allocation:
                current_alloc = holdings_df.set_index('symbol')['allocation'] * 100
                target_alloc = pd.Series(target_allocation) * 100
                comparison_df = pd.DataFrame({
                    'Current (%)': current_alloc,
                    'Target (%)': target_alloc
                }).fillna(0)
                # Plotly chart for interactivity
                fig_plotly = go.Figure()
                fig_plotly.add_trace(go.Bar(
                    x=comparison_df.index,
                    y=comparison_df['Current (%)'],
                    name='Current (%)',
                    marker_color='indigo'
                ))
                fig_plotly.add_trace(go.Bar(
                    x=comparison_df.index,
                    y=comparison_df['Target (%)'],
                    name='Target (%)',
                    marker_color='orange'
                ))
                fig_plotly.update_layout(
                    barmode='group',
                    title="Current vs. Target Portfolio Allocation",
                    xaxis_title="Asset",
                    yaxis_title="Allocation (%)"
                )
                st.plotly_chart(fig_plotly, use_container_width=True)
                # Matplotlib chart for universal PNG download
                fig, ax = plt.subplots()
                width = 0.35
                x = range(len(comparison_df.index))
                ax.bar([i - width/2 for i in x], comparison_df['Current (%)'], width=width, label='Current (%)', color='indigo')
                ax.bar([i + width/2 for i in x], comparison_df['Target (%)'], width=width, label='Target (%)', color='orange')
                ax.set_xticks(list(x))
                ax.set_xticklabels(comparison_df.index, rotation=45)
                ax.set_ylabel("Allocation (%)")
                ax.set_title("Current vs. Target Portfolio Allocation")
                ax.legend()
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    "Download Allocation Comparison (PNG, universal)",
                    buf,
                    file_name="allocation_comparison_bar.png",
                    mime="image/png"
                )
                plt.close(fig)
            else:
                st.info("No data available for allocation comparison chart.")

        # --- 3. Unrealized P/L by Asset Bar Chart ---
        with tab7:
            st.subheader("Unrealized Profit/Loss (P/L) by Asset")
            if holdings_df is not None and not holdings_df.empty and "unrealized_pl_usd" in holdings_df.columns:
                pl_df = holdings_df.set_index('symbol')["unrealized_pl_usd"].sort_values()
                colors = ['red' if x < 0 else 'green' for x in pl_df]
                # Plotly chart for interactivity
                fig_plotly = go.Figure([go.Bar(
                    x=pl_df.index,
                    y=pl_df.values,
                    marker_color=colors
                )])
                fig_plotly.update_layout(
                    title="Unrealized P/L by Asset",
                    xaxis_title="Asset",
                    yaxis_title="Unrealized P/L (USD)"
                )
                st.plotly_chart(fig_plotly, use_container_width=True)
                # Matplotlib chart for universal PNG download
                fig, ax = plt.subplots()
                bar_colors = ['green' if v >= 0 else 'red' for v in pl_df.values]
                ax.bar(pl_df.index, pl_df.values, color=bar_colors)
                ax.set_ylabel("Unrealized P/L (USD)")
                ax.set_title("Unrealized P/L by Asset")
                ax.axhline(0, color='black', linewidth=0.8)
                ax.tick_params(axis='x', rotation=45)  # <-- Use this line
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    "Download P/L Chart (PNG, universal)",
                    buf,
                    file_name="pl_by_asset_bar.png",
                    mime="image/png"
                )
                plt.close(fig)
            else:
                st.info("No P/L data available for this chart.")

        # --- 4. Portfolio Value Over Time Line Chart ---
        with tab8:
            st.subheader("Portfolio Value Over Time")
            if snapshots is not None and not snapshots.empty:
                snapshots = snapshots.copy()
                snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"])
                snapshots = snapshots.dropna(subset=["timestamp"])
                snapshots = snapshots[snapshots["timestamp"].dt.year > 2000]  # Filter out weird old dates

                # Drop duplicate timestamps, keeping the last entry for each timestamp
                snapshots = snapshots.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

                # Debug output (optional, for development)
                # st.write("Snapshot Data Preview", snapshots.head(20))
                # st.write("Unique timestamps:", snapshots["timestamp"].nunique(), "Total rows:", len(snapshots))
                # duplicates = snapshots[snapshots.duplicated(subset=["timestamp"], keep=False)]
                # if not duplicates.empty:
                #     st.warning("Duplicate timestamps found in snapshots!")
                #     st.dataframe(duplicates)

                # Plotly chart for interactivity
                fig_plotly = px.line(
                    snapshots,
                    x="timestamp",
                    y="total_value_usd",
                    title="Portfolio Value Over Time",
                    markers=True,
                    labels={"total_value_usd": "Total Value (USD)", "timestamp": "Date"}
                )
                fig_plotly.update_xaxes(
                    tickformat="%Y-%m-%d",  # Format as date
                    ticklabelmode="period"
                )
                st.plotly_chart(fig_plotly, use_container_width=True)
                # Matplotlib chart for universal PNG download
                fig, ax = plt.subplots()
                ax.plot(snapshots["timestamp"], snapshots["total_value_usd"], marker='o')
                ax.set_xlabel("Date")
                ax.set_ylabel("Total Value (USD)")
                ax.set_title("Portfolio Value Over Time")
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    "Download Value History (PNG, universal)",
                    buf,
                    file_name="portfolio_value_history.png",
                    mime="image/png"
                )
                plt.close(fig)
            else:
                st.info("No snapshot data available for value history chart.")

    def render_market_trends(self):
        st.markdown("## 📈 Market Trends")
        export_dir = Path(self.config_manager.config.get("exports", {}).get("path", "data/exports/"))
        export_dir.mkdir(parents=True, exist_ok=True)

        # 1. Define get_trend_report
        @st.cache_data(show_spinner="Generating trend report...", ttl=300)
        def get_trend_report(config, _binance_client, timeframe):
            from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer
            analyzer = CryptoTrendAnalyzer(config=config, binance_client=_binance_client)
            return asyncio.run(analyzer.generate_report(timeframe))

        # 2. Define the chart plotting function
        def plot_coin_chart(symbol, timeframe, config, binance_client):
            import matplotlib.pyplot as plt
            from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer
            import numpy as np
            import pandas as pd

            analyzer = CryptoTrendAnalyzer(config=config, binance_client=binance_client)
            settings = analyzer.timeframe_settings.get(timeframe)
            if not settings:
                st.error("No settings for this timeframe.")
                return

            period = settings.get("period", "1mo")
            interval = "1wk" if timeframe == "long_term" else "1d"
            data = asyncio.run(analyzer.fetch_crypto_data_async(symbol, period, interval))
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
            indicator_df = indicator_df[~indicator_df.index.duplicated(keep='first')]
            indicator_df = indicator_df[~indicator_df.index.isna()]
            indicator_df = indicator_df.sort_index()

            # Require a minimum number of rows for meaningful plotting
            min_required_rows = 30
            if len(indicator_df) < min_required_rows:
                st.warning(f"Not enough data to plot a meaningful chart (need at least {min_required_rows} rows, got {len(indicator_df)}).")
                return

            # Check for all-empty rows (all columns are NaN)
            if indicator_df.dropna(how='all').empty:
                st.warning("No valid data to plot (all rows are empty or all columns are NaN).")
                return

            # Check for all-NaN in the columns you want to plot
            plot_cols = ['Close']
            if all(indicator_df[col].dropna().empty for col in plot_cols if col in indicator_df.columns):
                st.warning("No valid data to plot (all values in plot columns are NaN).")
                return

            # Limit to last 500 rows for safety
            if len(indicator_df) > 500:
                indicator_df = indicator_df.tail(500)

            # Only now is it safe to plot!
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
            fig.suptitle(f"{symbol} Price & Indicators ({timeframe.replace('_', ' ').title()})", fontsize=16)

            # --- Price and SMAs ---
            axs[0].plot(indicator_df.index, indicator_df['Close'], label='Close', color='blue')
            sma_short = settings.get('sma_short_window')
            sma_long = settings.get('sma_long_window')
            if sma_short and f"SMA_{sma_short}" in indicator_df.columns:
                axs[0].plot(indicator_df.index, indicator_df[f"SMA_{sma_short}"], label=f"SMA {sma_short}", color='orange')
            if sma_long and f"SMA_{sma_long}" in indicator_df.columns:
                axs[0].plot(indicator_df.index, indicator_df[f"SMA_{sma_long}"], label=f"SMA {sma_long}", color='green')
            # Support/Resistance (last value)
            if 'Low' in indicator_df.columns and 'High' in indicator_df.columns:
                window = 30 if len(indicator_df) > 30 else len(indicator_df)
                support = indicator_df['Low'].tail(window).min()
                resistance = indicator_df['High'].tail(window).max()
                axs[0].axhline(support, color='red', linestyle='--', label='Support')
                axs[0].axhline(resistance, color='purple', linestyle='--', label='Resistance')
            axs[0].set_ylabel("Price")
            axs[0].legend()
            axs[0].grid(True)

            # --- RSI ---
            rsi_col = f"RSI_{analyzer.rsi_period}"
            if rsi_col in indicator_df.columns:
                axs[1].plot(indicator_df.index, indicator_df[rsi_col], label='RSI', color='magenta')
                axs[1].axhline(70, color='red', linestyle='--', linewidth=1)
                axs[1].axhline(30, color='green', linestyle='--', linewidth=1)
                axs[1].set_ylabel("RSI")
                axs[1].legend()
                axs[1].grid(True)
            else:
                axs[1].text(0.5, 0.5, "No RSI data", ha='center', va='center')

            # --- MACD ---
            macd_col = "MACD_12_26_9"
            macds_col = "MACDs_12_26_9"
            if macd_col in indicator_df.columns and macds_col in indicator_df.columns:
                axs[2].plot(indicator_df.index, indicator_df[macd_col], label='MACD', color='blue')
                axs[2].plot(indicator_df.index, indicator_df[macds_col], label='Signal', color='orange')
                axs[2].set_ylabel("MACD")
                axs[2].legend()
                axs[2].grid(True)
            else:
                axs[2].text(0.5, 0.5, "No MACD data", ha='center', va='center')

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            st.pyplot(fig)

        # 3. Timeframe selector
        timeframe_map = {
            "Long-term (4 Years)": "long_term",
            "Swing (3 Months)": "swing",
            "Day (1 Month)": "day"
        }
        timeframe_label = st.radio(
            "Select timeframe for trend analysis:",
            list(timeframe_map.keys()),
            horizontal=True
        )
        timeframe = timeframe_map[timeframe_label]

        tracker = self.initialize_tracker()
        report = get_trend_report(self.config_manager.config, getattr(tracker, "binance_client", None), timeframe)

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
                st.session_state.current_chart_symbol = all_symbols[0]  # Default to first coin
            
            # Dropdown for selecting coin
            selected_symbol = st.selectbox(
                "Select coin to view chart:",
                all_symbols,
                key="unified_chart_coin_select",
                index=all_symbols.index(st.session_state.current_chart_symbol)
            )
            
            # Update session state with selected symbol
            st.session_state.current_chart_symbol = selected_symbol

            # Always render the chart when coins are available
            plot_coin_chart(
                st.session_state.current_chart_symbol,
                timeframe,
                self.config_manager.config,
                getattr(tracker, "binance_client", None)
            )
        else:
            st.info("No coin data available for charting.")

        # 5. Coin-by-Coin Table
        st.markdown("### 🪙 Coin Analysis")
        rows = []
        for symbol, analysis in coin_analyses.items():
            rows.append({
                "Symbol": symbol,
                "Price": f"${analysis.get('current_price', 0):,.2f}",
                "Change (%)": f"{analysis.get('price_change_pct', 0):+.2f}",
                "RSI": f"{analysis.get('rsi', 0):,.2f}",
                "Support": f"${analysis.get('support_level', 0):,.2f}",
                "Resistance": f"${analysis.get('resistance_level', 0):,.2f}",
                "Active Conditions": ", ".join(analysis.get('active_conditions', []))
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No coin analysis data available.")

        # --- Prepare exportable data ---
        df_export = pd.DataFrame([
            {
                "Symbol": symbol,
                "Price": analysis.get('current_price', 0),
                "Change (%)": analysis.get('price_change_pct', 0),
                "RSI": analysis.get('rsi', 0),
                "Support": analysis.get('support_level', 0),
                "Resistance": analysis.get('resistance_level', 0),
                "Active Conditions": ", ".join(analysis.get('active_conditions', []))
            }
            for symbol, analysis in coin_analyses.items()
        ])

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
                        "HTML": "🌐 HTML"
                    }[x],
                    horizontal=True
                )

            with col_action:
                if st.button("🚀 Generate Export", use_container_width=True):
                    with st.spinner(f"Creating {export_format} export..."):
                        try:
                            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"trend_report_{now_str}.{export_format.lower()}"
                            exported_file = export_dir / filename

                            if export_format == "CSV":
                                df_export.to_csv(exported_file, index=False)
                            elif export_format == "JSON":
                                with open(exported_file, "w") as f:
                                    json.dump(report, f, indent=2)
                            elif export_format == "HTML":
                                from crypto_portfolio_tracker.exporters import HtmlExporter
                                exporter = HtmlExporter(self.config_manager.config)
                                exported_file = exporter.export_trend_report(report, df_export)

                            st.success(f"Successfully created {filename}")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")

        # Consolidated exports dropdown
        st.markdown("---")
        st.markdown("#### My Exports")

        # Get all export files sorted by modification time (newest first)
        all_files = sorted(
            list(export_dir.glob("trend_report_*.*")),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not all_files:
            st.info("No exports found. Generate one above!")
            return

        # Dropdown to select export file
        selected_export = st.selectbox(
            "Select an export file:",
            options=all_files,
            format_func=lambda x: f"{x.name} ({x.stat().st_size/1024:.1f} KB, {datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})",
            index=0,
            help="Select an export file to view, download or delete"
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
                    mime=f"text/{selected_export.suffix[1:].lower()}" if selected_export.suffix != ".json" else "application/json",
                    use_container_width=True
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
                    st.error(f"Failed to delete: {str(e)}")

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
                    st.components.v1.html(preview_file.read_text(), height=800, scrolling=True)
            except Exception as e:
                st.error(f"Could not preview file: {str(e)}")

            if st.button("❌ Close Preview"):
                st.session_state.preview_file = None
                st.rerun()

    def render_rebalancing(self):
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
                type="primary"
            )
        
        with col2:
            if st.session_state.rebalance_metrics:
                last_update = datetime.now().strftime("%H:%M:%S")
                st.success(f"✅ Last updated: {last_update}")

        if generate_clicked or st.session_state.rebalance_metrics is None:
            with st.spinner("🔄 Analyzing portfolio and generating rebalancing suggestions..."):
                tracker = self.initialize_tracker()
                
                # 1. Get current metrics
                metrics = asyncio.run(tracker.calculate_portfolio_metrics())
                st.session_state.rebalance_metrics = metrics

                # 2. Get USDT balance (Spot + Earn)
                usdt_balance = 0.0
                try:
                    spot_balance = float(tracker.binance_client.get_asset_balance(asset="USDT").get("free", 0.0))
                    usdt_balance += spot_balance
                    if not self.config_manager.is_testnet_mode:
                        earn_positions = tracker.fetcher.fetch_simple_earn_balances(pd.DataFrame([{"symbol": "USDT"}]))
                        earn_balance = earn_positions.get("USDT", 0.0)
                        usdt_balance += earn_balance
                except Exception:
                    pass
                st.session_state.rebalance_usdt = usdt_balance

                # 3. Get rebalancing suggestions
                suggestions_df = asyncio.run(tracker.get_core_portfolio_rebalance_suggestions_technical())
                st.session_state.rebalance_suggestions = suggestions_df

                # 4. Check for actionable trades
                actionable = False
                if suggestions_df is not None and not suggestions_df.empty:
                    actionable = any(s in ["BUY", "SELL"] for s in suggestions_df["Signal"])
                st.session_state.rebalance_actionable = actionable

                # Clear previous results and accepted trades
                st.session_state.rebalance_results = None
                st.session_state.accepted_trades = set()

        # --- Display Core Portfolio Status ---
        st.markdown("### 🎯 Current Core Portfolio Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Current Allocation")
            metrics = st.session_state.rebalance_metrics
            if metrics and "core_holdings_df" in metrics:
                core_df = metrics["core_holdings_df"]
                # Create a more compact view for status
                status_df = core_df[['symbol', 'core_allocation', 'value_usd']].copy()
                status_df['core_allocation'] = (status_df['core_allocation'] * 100).round(2)
                status_df.columns = ['Asset', 'Current %', 'Value (USD)']
                status_df['Value (USD)'] = status_df['Value (USD)'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(status_df, use_container_width=True, hide_index=True)
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
            actionable_trades = suggestions_df[suggestions_df["Signal"].isin(["BUY", "SELL"])].copy()
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
                st.markdown("*Select the trades you want to execute by checking the boxes below:*")
                
                # Add select all / deselect all buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                with col1:
                    if st.button("✅ Select All", key="select_all_trades", use_container_width=True):
                        for idx, row in actionable_trades.iterrows():
                            trade_key = f"{row['Symbol']}_{row['Signal']}"
                            st.session_state.accepted_trades.add(trade_key)
                        st.rerun()
                
                with col2:
                    if st.button("❌ Deselect All", key="deselect_all_trades", use_container_width=True):
                        st.session_state.accepted_trades.clear()
                        st.rerun()
                
                with col3:
                    if st.button("🔄 Refresh", key="refresh_selection", use_container_width=True):
                        st.rerun()
                
                st.markdown("---")
                
                # Enhanced trade cards
                for idx, row in actionable_trades.iterrows():
                    trade_key = f"{row['Symbol']}_{row['Signal']}"
                    
                    # Create a card-like container for each trade
                    with st.container():
                        # Use colored border based on signal type
                        if row['Signal'] == 'BUY':
                            st.markdown("""
                            <div style="border-left: 4px solid #28a745; padding-left: 15px; margin: 10px 0;">
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="border-left: 4px solid #dc3545; padding-left: 15px; margin: 10px 0;">
                            """, unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
                        
                        with col1:
                            # Enhanced checkbox with styling
                            accepted = st.checkbox(
                                "Execute",
                                key=f"accept_{trade_key}",
                                value=trade_key in st.session_state.accepted_trades,
                                help=f"Check to include this {row['Signal']} order in execution"
                            )
                            if accepted:
                                st.session_state.accepted_trades.add(trade_key)
                            else:
                                st.session_state.accepted_trades.discard(trade_key)
                        
                        with col2:
                            st.markdown(f"**{row['Symbol']}**")
                            # Enhanced trade details
                            drift_color = "🔴" if abs(row['Drift (pts)']) > 5 else "🟡"
                            st.markdown(f"**Drift:** {drift_color} {row['Drift (pts)']:.2f} pts")
                            st.markdown(f"**Allocation:** {row['Current %']:.1f}% → {row['Target %']:.1f}%")
                            
                            # Show the reason for the trade
                            if row['Signal'] == 'BUY':
                                st.markdown(f"*📈 Underweight by {abs(row['Drift (pts)']):,.2f} percentage points*")
                            else:
                                st.markdown(f"*📉 Overweight by {abs(row['Drift (pts)']):,.2f} percentage points*")
                        
                    with col3:
                            st.markdown("**Current Value**")
                            st.markdown(f"${row['Current Value (USD)']:,.0f}")
                            
                            # Show estimated trade size (this would need to be calculated)
                            if 'Estimated Trade Size' in row:
                                st.markdown("**Est. Trade Size**")
                                st.markdown(f"${row['Estimated Trade Size']:,.0f}")
                        
                    with col4:
                            # Enhanced signal display
                            if row['Signal'] == 'BUY':
                                st.markdown("""
                                <div style="background: #d4edda; color: #155724; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;">
                                    📈 BUY
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background: #f8d7da; color: #721c24; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;">
                                    📉 SELL
                                </div>
                                """, unsafe_allow_html=True)
                        
                                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Show HOLD trades in a more compact format
            if not hold_trades.empty:
                with st.expander(f"📊 Hold Positions ({len(hold_trades)} assets) - No Action Required", expanded=False):
                    for idx, row in hold_trades.iterrows():
                        col1, col2, col3 = st.columns([1, 4, 1])
                        with col1:
                            st.markdown("""
                            <div style="background: #e2e3e5; color: #495057; padding: 4px 8px; border-radius: 4px; text-align: center; font-size: 0.8em;">
                                HOLD
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"**{row['Symbol']}** - Within target range")
                            st.markdown(f"Drift: {row['Drift (pts)']:.2f} pts | Current: {row['Current %']:.1f}% (Target: {row['Target %']:.1f}%)")
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
                    st.warning("⚠️ **Large Rebalancing Operation**: You're about to execute multiple trades. Please review carefully.")
            else:
                st.info("ℹ️ No trades selected. Check the boxes above to select trades for execution.")
        
        else:
            st.info("📊 Run portfolio analysis to see rebalancing suggestions here.")

        # --- Enhanced Execute Button Section ---
        st.markdown("### ⚡ Execute Selected Trades")
        
        # Execution controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            execute_enabled = (
                len(st.session_state.accepted_trades) > 0 and 
                not st.session_state.rebalance_executing
            )
            
            execute_clicked = st.button(
                f"🚀 Execute {len(st.session_state.accepted_trades)} Selected Trade(s)",
                key="rebalance_execute_btn",
                disabled=not execute_enabled,
                use_container_width=True,
                type="primary" if execute_enabled else "secondary"
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
                            earn_balances = tracker.fetcher.fetch_simple_earn_balances(spot_balances_df)
                        
                        progress_bar.progress(70)
                        status_text.text("🚀 Executing trades...")
                        
                        # Run execution
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            import io, sys
                            old_stdout = sys.stdout
                            sys.stdout = mystdout = io.StringIO()
                            
                            loop.run_until_complete(
                                tracker._execute_rebalancing_trades(
                                    filtered_df,
                                    earn_balances,
                                    interactive=False,
                                    auto_confirm=True
                                )
                            )
                            
                            sys.stdout = old_stdout
                            execution_output = mystdout.getvalue()
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Execution completed!")
                            
                            # Enhanced results formatting
                            st.session_state.rebalance_results = f"""
    === REBALANCING EXECUTION COMPLETED ===
    Executed {len(accepted_suggestions)} trade(s)
    Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    {execution_output}
    === END EXECUTION LOG ===
                            """
                            
                        except Exception as e:
                            progress_bar.progress(100)
                            status_text.text("❌ Execution failed!")
                            st.session_state.rebalance_results = f"❌ **Execution Error**: {str(e)}\n\nPlease check your settings and try again."
                        finally:
                            loop.close()
                    else:
                        st.session_state.rebalance_results = "ℹ️ No trades were selected for execution."
                    
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
                if st.button("🔄 Clear Selection & Refresh Portfolio", type="secondary"):
                    st.session_state.accepted_trades.clear()
                    st.session_state.rebalance_metrics = None
                    st.session_state.rebalance_suggestions = None
                    st.session_state.rebalance_results = None
                    st.success("✅ Selection cleared. Run analysis again to see updated portfolio.")
                    st.rerun()
                    
            elif "Error" in results:
                st.error("❌ **Execution Failed**")
                st.code(results, language="text")
            else:
                st.info("ℹ️ **Execution Information**")
                st.code(results, language="text")
        else:
            st.info("📝 Execution results and detailed logs will appear here after running trades.")
            
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
        st.markdown("## 💰 Trading")
        
        # Initialize session state
        if "trading_mode" not in st.session_state:
            st.session_state.trading_mode = None
        if "trading_results" not in st.session_state:
            st.session_state.trading_results = None
        if "trading_executing" not in st.session_state:
            st.session_state.trading_executing = False
        if "manual_trade_data" not in st.session_state:
            st.session_state.manual_trade_data = {}
        if "strategy_signals" not in st.session_state:
            st.session_state.strategy_signals = None

        # Check live trading status
        is_live = self.config_manager.config.get("portfolio", {}).get("live_trading_enabled", False)
        is_testnet = self.config_manager.is_testnet_mode
        
        # Trading Status Banner
        col1, col2, col3 = st.columns([1, 2, 1])
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

        # Trading Mode Selection
        st.markdown("### 🎯 Select Trading Mode")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Manual Trade", use_container_width=True, type="primary"):
                st.session_state.trading_mode = "manual"
                st.rerun()
        with col2:
            if st.button("🤖 Live Strategy", use_container_width=True, type="primary"):
                st.session_state.trading_mode = "strategy"
                st.rerun()

        # Render appropriate trading interface
        if st.session_state.trading_mode == "manual":
            self._render_manual_trading()
        elif st.session_state.trading_mode == "strategy":
            self._render_live_strategy_trading()
        else:
            # Show trading overview
            st.markdown("### 📊 Trading Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📝 Manual Trading")
                st.markdown("""
                - Place individual BUY/SELL orders
                - Specify exact amounts and assets
                - Full control over trade parameters
                - Real-time market execution
                """)
                
            with col2:
                st.markdown("#### 🤖 Live Strategy Trading")
                st.markdown("""
                - Automated signal generation
                - Strategy-based trade execution
                - Multi-asset portfolio analysis
                - Account-specific strategies
                """)

        if st.session_state.trading_mode != "manual":
            st.session_state.trading_results = None
            st.session_state.manual_trade_data = {}
        if st.session_state.trading_mode != "strategy":
            st.session_state.trading_results = None
            st.session_state.strategy_signals = None

    def _render_manual_trading(self):
        """Renders the manual trading interface."""
        st.markdown("### 📝 Manual Trading")

        # Fetch and display available USDT balance
        usdt_balance = None
        tracker = self.initialize_tracker()
        try:
            usdt_balance = float(tracker.binance_client.get_asset_balance(asset="USDT").get("free", 0.0))
        except Exception:
            usdt_balance = None

        st.markdown("#### 💰 Available USDT")
        if usdt_balance is not None:
            st.success(f"${usdt_balance:,.2f}")
        else:
            st.info("USDT balance unavailable.")
        
        # Gather core coins
        core_coins = list(self.config_manager.config.get("target_allocation", {}).keys())
        core_coins_upper = [c.upper() for c in core_coins]

        # Get all available symbols from local mappings (JSON)
        symbol_mapper = self.config_manager.symbol_mapper
        all_symbols = list(symbol_mapper.get_all_mappings().keys())

        # If local mappings are empty, fallback to master list from CoinGecko
        if not all_symbols:
            all_coin_dicts = symbol_mapper._fetch_master_coins_list()
            all_symbols = [coin['symbol'].upper() for coin in all_coin_dicts if 'symbol' in coin]

        # Remove duplicates, prioritize core coins
        non_core_symbols = sorted(set(all_symbols) - set(core_coins_upper))
        coin_options = core_coins_upper + non_core_symbols

        col1, col2 = st.columns(2)
        with col1:
            trade_type = st.selectbox(
                "Trade Action",
                ["BUY", "SELL"],
                help="Choose whether to buy or sell the asset"
            )
            symbol = st.selectbox(
                "Asset Symbol",
                coin_options,
                help="Select the asset symbol"
            )
        with col2:
            amount_input = st.text_input(
                f"Amount to {trade_type}",
                placeholder="e.g., 0.1 BTC or 100 USDT",
                help="Enter amount in asset units or USDT"
            )
            is_quote_qty = st.checkbox(
                "Amount is in USDT",
                help="Check if the amount is in USDT, uncheck if in asset units"
            )
        
        # Dynamic button with real-time validation
        st.markdown("---")
        
        # Check if form is valid
        is_valid = bool(symbol and amount_input)
        
        # Dynamic button text
        button_text = f"🚀 {trade_type} {symbol if symbol else 'ASSET'}"
        
        # Execute button
        if st.button(
            button_text,
            type="primary",
            disabled=not is_valid,
            use_container_width=True
        ):
            # Store trade data in session state for confirmation
            st.session_state.manual_trade_data = {
                "trade_type": trade_type,
                "symbol": symbol,
                "amount_input": amount_input,
                "is_quote_qty": is_quote_qty
            }
            st.rerun()
        
        # Show validation feedback
        if not symbol:
            st.warning("⚠️ Please enter an asset symbol")
        if not amount_input:
            st.warning("⚠️ Please enter an amount")
        if symbol and amount_input:
            st.success("✅ Form is ready for submission")
        
        # Show trade confirmation if data is stored
        if st.session_state.manual_trade_data:
            self._show_trade_confirmation()
        
        # Show execution results if available
        if st.session_state.trading_results:
            st.markdown("### 📋 Execution Results")
            st.code(st.session_state.trading_results, language="text")
            
            # Post-execution actions
            if "EXECUTION COMPLETED" in st.session_state.trading_results:
                st.success("✅ **Trade executed successfully!**")
                
                # Suggest portfolio sync
                st.markdown("### 🔄 Portfolio Update")
                st.info("💡 **Recommendation**: After executing a trade, it's a good practice to sync your portfolio to see the updated balances and positions.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Sync Portfolio", type="primary", use_container_width=True):
                        self._sync_portfolio_and_reset()
                with col2:
                    if st.button("🆕 New Trade", type="secondary", use_container_width=True):
                        # Reset to initial state for new trade
                        st.session_state.trading_results = None
                        st.session_state.manual_trade_data = {}
                        st.rerun()
                
                # Show helpful tips
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
                    if st.button("🔄 Try Again", type="primary", use_container_width=True):
                        st.session_state.trading_results = None
                        st.session_state.manual_trade_data = {}
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", type="secondary", use_container_width=True):
                        st.session_state.trading_results = None
                        st.session_state.manual_trade_data = {}
                        st.rerun()

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
                if data['is_quote_qty']:
                    st.markdown(f"**Amount:** ${data['amount_input']} USDT")
                else:
                    st.markdown(f"**Amount:** {data['amount_input']} {data['symbol']}")
        
        # Confirmation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ EXECUTE TRADE", type="primary", use_container_width=True):
                self._confirm_manual_trade(
                    data['trade_type'], 
                    data['symbol'], 
                    data['amount_input'], 
                    data['is_quote_qty']
                )
        with col2:
            if st.button("❌ CANCEL", type="secondary", use_container_width=True):
                st.session_state.manual_trade_data = {}
                st.session_state.trading_results = "Trade cancelled by user."
                st.rerun()

    def _confirm_manual_trade(self, trade_type, symbol, amount_input, is_quote_qty):
        """Confirms and executes the manual trade."""
        st.session_state.trading_executing = True
        
        with st.spinner(f"Executing {trade_type} order for {symbol}..."):
            try:
                tracker = self.initialize_tracker()
                is_live = self.config_manager.config.get("portfolio", {}).get("live_trading_enabled", False)
                
                # Parse amount
                import re
                numeric_part = re.search(r"[\d\.]+", amount_input)
                if not numeric_part:
                    st.error("❌ Invalid amount format. Please enter a valid number.")
                    return
                
                amount = float(numeric_part.group(0))
                trade_ticker = f"{symbol}USDT"
                
                # Execute the trade
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    import io, sys
                    old_stdout = sys.stdout
                    sys.stdout = mystdout = io.StringIO()
                    
                    loop.run_until_complete(
                        tracker._execute_manual_trade(
                            trade_type, symbol, trade_ticker, amount, is_quote_qty, is_live
                        )
                    )
                    
                    sys.stdout = old_stdout
                    execution_output = mystdout.getvalue()
                    
                    st.session_state.trading_results = f"""
=== MANUAL TRADE EXECUTION COMPLETED ===
Trade: {trade_type} {amount} {symbol if not is_quote_qty else 'USDT'}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{execution_output}
=== END EXECUTION LOG ===
                    """
                    
                except Exception as e:
                    st.session_state.trading_results = f"❌ **Execution Error**: {str(e)}"
                finally:
                    loop.close()
                    
            except Exception as e:
                st.session_state.trading_results = f"❌ **Setup Error**: {str(e)}"
            finally:
                st.session_state.trading_executing = False
                # Clear the trade data after execution
                st.session_state.manual_trade_data = {}
                st.rerun()

    def _sync_portfolio_and_reset(self):
        with st.spinner("🔄 Syncing portfolio and updating data..."):
            try:
                tracker = self.initialize_tracker()
                metrics = asyncio.run(tracker.run_full_sync())
                st.session_state.portfolio_metrics = metrics
                st.session_state.last_sync = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Reset trading page state
                st.session_state.trading_results = None
                st.session_state.manual_trade_data = {}
                st.session_state.trading_mode = None
                st.session_state.strategy_signals = None
                st.session_state.strategy_selected_coins = []
                st.session_state.strategy_trade_amount_mode = "% of USDT balance"
                st.session_state.strategy_trade_pct = 20.0
                st.session_state.strategy_trade_amount = 50.0
                for key in list(st.session_state.keys()):
                    if key.startswith("strategy_param_"):
                        del st.session_state[key]
                st.session_state.strategy_reset_flag = True  # <--- ADD THIS LINE
                st.success("✅ Portfolio synced successfully! Trading page reset to initial state.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Portfolio sync failed: {str(e)}")

    def _render_live_strategy_trading(self):
        if st.session_state.get("strategy_reset_flag"):
            st.session_state.strategy_signals = None
            st.session_state.trading_results = None
            st.session_state.strategy_reset_flag = False
            st.stop()  # <--- This halts the UI for this rerun
        st.markdown("### 🤖 Live Strategy Trading")

        # --- 1. Account Selection ---
        tracker = self.initialize_tracker()
        config = self.config_manager.config
        main_account = {"name": "Main Account", "type": "main", **config.get("main_api_keys", {})}
        sub_accounts = config.get("sub_accounts", [])
        accounts = [main_account] + sub_accounts

        # Defensive: No API keys
        if not any(acc.get("api_key") or acc.get("binance_key") for acc in accounts):
            st.error("❌ No API keys found for any account. Cannot run live strategies.")
            return

        account_names = [acc["name"] for acc in accounts]
        selected_account_name = st.selectbox("Select Account", account_names, key="strategy_account")
        selected_account = next(acc for acc in accounts if acc["name"] == selected_account_name)
        account_type = selected_account.get("type", "main")

        # --- 2. Strategy Selection ---
        # Discover all strategies
        all_strategies = {
            name: obj for name, obj in inspect.getmembers(trading_strategies, inspect.isclass)
            if issubclass(obj, trading_strategies.Strategy) and obj is not trading_strategies.Strategy
        }
        # Filter by account type
        if account_type == "main":
            available_strategies = all_strategies
        elif account_type == "swing":
            available_strategies = {k: v for k, v in all_strategies.items() if getattr(v, "strategy_type", None) in ["swing", "general"]}
        elif account_type == "day":
            available_strategies = {k: v for k, v in all_strategies.items() if getattr(v, "strategy_type", None) in ["day", "general"]}
        else:
            available_strategies = all_strategies

        if not available_strategies:
            st.error(f"❌ No suitable strategies found for account type '{account_type}'.")
            return

        strategy_names = list(available_strategies.keys())
        selected_strategy_name = st.selectbox("Select Strategy", strategy_names, key="strategy_name")
        strategy_class = available_strategies[selected_strategy_name]

         # --- Show Available USDT Balance ---
        usdt_balance = None
        try:
            if selected_account["name"] == "Main Account":
                # Use the already-initialized client for main account
                usdt_balance = float(tracker.binance_client.get_asset_balance(asset="USDT").get("free", 0.0))
            else:
                # For sub-accounts, initialize a new client with sub-account keys
                live_client = tracker._init_binance_client(
                    api_key=selected_account.get("api_key") or selected_account.get("binance_key"),
                    api_secret=selected_account.get("api_secret") or selected_account.get("binance_secret")
                )
                usdt_balance = float(live_client.get_asset_balance(asset="USDT").get("free", 0.0))
        except Exception as e:
            st.info(f"USDT balance unavailable. Error: {e}")
            usdt_balance = None

        st.markdown("#### 💰 Available USDT")
        if usdt_balance is not None:
            st.success(f"${usdt_balance:,.2f}")
        else:
            st.info("USDT balance unavailable.")

        # --- 3. Coin and Parameter Selection (all in one form) ---
        symbol_mapper = self.config_manager.symbol_mapper
        core_coins = list(config.get("target_allocation", {}).keys())
        core_coins_upper = [f"{c.upper()}-USD" for c in core_coins]
        all_symbols = list(symbol_mapper.get_all_mappings().keys())
        if not all_symbols:
            all_coin_dicts = symbol_mapper._fetch_master_coins_list()
            all_symbols = [f"{coin['symbol'].upper()}-USD" for coin in all_coin_dicts if 'symbol' in coin]
        non_core_symbols = sorted(set(all_symbols) - set(core_coins_upper))
        coin_options = core_coins_upper + non_core_symbols

        if "strategy_selected_coins" not in st.session_state:
            st.session_state.strategy_selected_coins = core_coins_upper

        param_specs = getattr(strategy_class, "strategy_param_specs", {})
        param_inputs = {}

        def select_all_core_coins():
            st.session_state.strategy_selected_coins = core_coins_upper

        with st.form("strategy_params_form"):
            st.markdown("#### 🪙 Select Coins for Strategy Trading")
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_coins = st.multiselect(
                    "Coins to include",
                    options=coin_options,
                    default=st.session_state.get("strategy_selected_coins", core_coins_upper),
                    key="strategy_selected_coins"
                )
            with col2:
                st.form_submit_button("Select All Core Coins", on_click=select_all_core_coins)

            # Defensive: Require at least one coin
            if not selected_coins:
                st.warning("Please select at least one coin to run the strategy on.")

            # Parameter fields
            for param, spec in param_specs.items():
                label = spec.get("label", param.replace("_", " ").capitalize())
                default = spec.get("default", "")
                key = f"strategy_param_{param}"
                if spec.get("type") == "float":
                    st.number_input(
                        label, value=float(default), min_value=spec.get("min_value", 0.0), max_value=spec.get("max_value", 100.0), key=key
                    )
                elif spec.get("type") == "int":
                    st.number_input(
                        label, value=int(default), min_value=spec.get("min_value", 0), max_value=spec.get("max_value", 100), step=1, key=key
                    )
                elif spec.get("type") == "bool":
                    st.checkbox(label, value=bool(default), key=key)
                else:
                    st.text_input(label, value=str(default), key=key)

            # --- Trade amount mode ---
            trade_amount_mode = st.radio(
                "How do you want to specify trade size?",
                ["% of USDT balance", "Fixed USDT amount"],
                horizontal=True,
                key="strategy_trade_amount_mode"
            )
            if trade_amount_mode == "% of USDT balance":
                trade_pct = st.number_input(
                    "Percent of available USDT to use per trade",
                    min_value=1.0, max_value=100.0, value=20.0, step=1.0,
                    key="strategy_trade_pct"
                )
                trade_amount = None
            else:
                trade_amount = st.number_input(
                    "Fixed USDT amount to use per trade",
                    min_value=1.0, value=50.0, step=1.0,
                    key="strategy_trade_amount"
                )
                trade_pct = None

            submitted = st.form_submit_button("Generate Signals")

        # After the form, collect the user input from session_state:
        for param in param_specs:
            param_inputs[param] = st.session_state.get(f"strategy_param_{param}", param_specs[param].get("default"))
        selected_coins = st.session_state.strategy_selected_coins

        # Defensive: Require at least one coin
        if not selected_coins:
            return

        # --- 4. Signal Generation ---
        if submitted:
            st.session_state.strategy_signals = None  # Reset
            with st.spinner("Generating signals..."):
                try:
                    live_client = tracker._init_binance_client(
                        api_key=selected_account.get("api_key") or selected_account.get("binance_key"),
                        api_secret=selected_account.get("api_secret") or selected_account.get("binance_secret")
                    )
                    analyzer = CryptoTrendAnalyzer(config=config, binance_client=live_client)
                    signals = []
                    for coin in selected_coins:
                        yf_ticker = f"{coin}-USD"
                        analyzer.set_symbol(yf_ticker)
                        # Convert percent params to fraction if needed
                        strategy_kwargs = param_inputs.copy()
                        for k, v in param_specs.items():
                            if v.get("type") == "float" and "pct" in k and strategy_kwargs[k] is not None and strategy_kwargs[k] > 1:
                                strategy_kwargs[k] = strategy_kwargs[k] / 100.0
                        strategy_instance = strategy_class(analyzer=analyzer, **strategy_kwargs)
                        interval = getattr(strategy_instance, "valid_intervals", ["1d"])[0]
                        period = "7d" if "m" in interval or "h" in interval else "1y"
                        data = asyncio.run(analyzer.fetch_crypto_data_async(yf_ticker, period=period, interval=interval))
                        if data is None or data.empty:
                            continue
                        signal, size, reason = asyncio.run(strategy_instance.generate_signal(data))
                        if signal in ["BUY", "SELL"]:
                            signals.append({"Symbol": coin, "Signal": signal, "Size": size, "Reason": reason})
                        st.session_state.strategy_signals = signals
                except Exception as e:
                    st.error(f"Signal generation failed: {e}")
                    st.session_state.strategy_signals = []

        # --- 5. Signal Review ---
        signals = st.session_state.strategy_signals or []
        if not signals:
            st.info("No actionable BUY or SELL signals generated by the strategy.")
            return

        st.markdown("#### 🚨 Proposed Trades")
        st.dataframe(signals, use_container_width=True)

        # --- 6. Execution ---
        if st.button("🚀 Execute All Signals", type="primary", use_container_width=True, disabled=not signals):
            with st.spinner("Executing trades..."):
                try:
                    # For each signal, execute the trade
                    results = []
                    for trade in signals:
                        # Use the same execution logic as manual trade, but for each signal
                        trade_type = trade["Signal"]
                        symbol = trade["Symbol"]
                        amount = trade["Size"]
                        # For simplicity, treat size as fraction of available USDT
                        usdt_balance = float(tracker.binance_client.get_asset_balance(asset="USDT").get("free", 0.0))
                        min_trade_usd = config.get("portfolio", {}).get("minimum_trade_usd", 10.0)
                        if trade_amount_mode == "% of USDT balance":
                            usdt_to_spend = max((trade_pct / 100.0) * usdt_balance, min_trade_usd)
                        else:
                            usdt_to_spend = max(trade_amount, min_trade_usd)
                        trade_ticker = f"{symbol}USDT"
                        is_live = config.get("portfolio", {}).get("live_trading_enabled", False)
                        # Use the same async execution as manual trade
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        import io, sys
                        old_stdout = sys.stdout
                        sys.stdout = mystdout = io.StringIO()
                        try:
                            loop.run_until_complete(
                                tracker._execute_manual_trade(
                                    trade_type, symbol, trade_ticker, usdt_to_spend, True, is_live
                                )
                            )
                            sys.stdout = old_stdout
                            execution_output = mystdout.getvalue()
                            results.append(
                                f"{trade_type} {symbol}: \n"
                                f"Preparing MARKET {trade_type} for {symbol}...\n"
                                f"Amount: {usdt_to_spend:.2f} USDT\n"
                                f"🚀 PLACING LIVE ORDER...\n"
                                f"{execution_output}"
                            )
                        except Exception as e:
                            results.append(f"{trade_type} {symbol}: ❌ Execution Error: {e}")
                        finally:
                            loop.close()
                    st.session_state.trading_results = "\n".join(results)
                    st.success("✅ All signals executed. See results below.")
                except Exception as e:
                    st.session_state.trading_results = f"❌ Execution Error: {e}"
                    st.error(f"Execution failed: {e}")

        # --- 7. Show Execution Results ---
        if st.session_state.trading_results:
            st.markdown("### 📋 Execution Results")
            st.code(st.session_state.trading_results, language="text")
            # Check for any failed trades in the results
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
        from crypto_portfolio_tracker.rebalancing_backtester import RebalancingBacktester
        from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer

        st.markdown("## 🧪 Backtesting")
        st.markdown("Test your trading strategies and rebalancing approaches with historical data")

        tab1, tab2 = st.tabs(["🧪 Strategy Backtest", "⚖️ Rebalancing Backtest"])

        # --- Strategy Backtest Tab ---
        with tab1:
            st.header("🧪 Strategy Backtest")
            st.markdown("Select a trading strategy and customize its parameters to test performance against historical data")

            # Strategy Selection Section
            with st.container():
                st.subheader("📈 Strategy Selection")
                all_strategies = {
                    name: obj for name, obj in inspect.getmembers(trading_strategies, inspect.isclass)
                    if issubclass(obj, trading_strategies.Strategy) and obj is not trading_strategies.Strategy
                }
                strategy_names = list(all_strategies.keys())
                selected_strategy_name = st.selectbox(
                    "Choose Trading Strategy",
                    strategy_names,
                    help="Select the trading strategy you want to backtest"
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
                                    label,
                                    value=float(default),
                                    key=key,
                                    help=help_text
                                )
                            elif spec.get("type") == "int":
                                param_inputs[param] = st.number_input(
                                    label,
                                    value=int(default),
                                    key=key,
                                    help=help_text
                                )
                            elif spec.get("type") == "bool":
                                param_inputs[param] = st.checkbox(
                                    label,
                                    value=bool(default),
                                    key=key,
                                    help=help_text
                                )
                            else:
                                param_inputs[param] = st.text_input(
                                    label,
                                    value=str(default),
                                    key=key,
                                    help=help_text
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
                        help="Time interval for price data (higher frequency = more granular analysis)"
                    )

                with col2:
                    symbol_mapper = self.config_manager.symbol_mapper
                    config = self.config_manager.config
                    core_coins = list(config.get("target_allocation", {}).keys())
                    core_coins_upper = [f"{c.upper()}-USD" for c in core_coins]
                    all_symbols = list(symbol_mapper.get_all_mappings().keys())
                    if not all_symbols:
                        all_coin_dicts = symbol_mapper._fetch_master_coins_list()
                        all_symbols = [f"{coin['symbol'].upper()}-USD" for coin in all_coin_dicts if 'symbol' in coin]
                    non_core_symbols = sorted(set(all_symbols) - set(core_coins_upper))
                    coin_options = core_coins_upper + non_core_symbols

                    selected_coin = st.selectbox(
                        "Select Cryptocurrency",
                        coin_options,
                        help="Choose the cryptocurrency to backtest the strategy on"
                    )

                with col3:
                    period_options = ["1y", "3y", "5y", "60d", "90d", "180d", "Custom"]
                    selected_period_option = st.selectbox(
                        "Backtest Period",
                        period_options,
                        index=1,  # Default to 3y
                        help="Historical period to run the backtest over"
                    )

                    if selected_period_option == "Custom":
                        period = st.text_input(
                            "Custom Period (e.g., 7y, 30d)",
                            value="30d",
                            key="strategy_period_custom",
                            help="Enter custom period: Xy for years, Xd for days"
                        )
                    else:
                        period = selected_period_option

                # --- Warn about yfinance data limits for high-frequency intervals ---
                if ("15m" in interval and "y" in period) or ("1h" in interval and "y" in period and int(period.replace("y", "")) > 2):
                    st.warning("⚠️ For 15m interval, use a period ≤ 60d. For 1h interval, use a period ≤ 729d. Adjust your selection to avoid data errors.")

                # Capital input in its own row for better visibility
                st.markdown("##### 💰 Initial Investment")
                initial_capital = st.number_input(
                    "Initial Capital (USD)",
                    min_value=100.0,
                    value=10000.0,
                    step=500.0,
                    help="Starting capital for the backtest simulation"
                )

            # Run Backtest Button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_backtest = st.button(
                    "🚀 Run Strategy Backtest",
                    type="primary",
                    use_container_width=True
                )

            if run_backtest:
                with st.spinner("🔄 Running backtest analysis..."):
                    try:
                        analyzer = CryptoTrendAnalyzer(config=config, binance_client=None)
                        backtester = StrategyBacktester(config=config, analyzer=analyzer)
                        strategy_instance = strategy_class(analyzer=analyzer, **param_inputs)
                        asyncio.run(backtester.run(
                            strategy=strategy_instance,
                            symbol=selected_coin,
                            initial_capital=initial_capital,
                            period=period,
                            interval=interval
                        ))

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
                                    delta=f"{stats['Strategy Outperformance']:+.1%} vs B&H"
                                )

                            with metric_cols[1]:
                                st.metric(
                                    "Final Value",
                                    f"${stats['Final Portfolio Value']:,.0f}",
                                    delta=f"${stats['Final Portfolio Value'] - stats['Initial Capital']:+,.0f}"
                                )

                            with metric_cols[2]:
                                st.metric(
                                    "Max Drawdown",
                                    f"{stats['Maximum Drawdown']:.1%}"
                                )

                            with metric_cols[3]:
                                st.metric(
                                    "Sharpe Ratio",
                                    f"{stats['Sharpe Ratio']:.2f}"
                                )

                            # Detailed Results Table
                            with st.expander("📋 Detailed Performance Metrics", expanded=True):
                                st.table({
                                    "Metric": [
                                        "Initial Capital",
                                        "Final Portfolio Value",
                                        "Strategy Total Return",
                                        "Buy & Hold Return",
                                        "Strategy Outperformance",
                                        "Maximum Drawdown",
                                        "Annualized Volatility",
                                        "Sharpe Ratio",
                                        "Total Trades Executed"
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
                                        f"{stats['Total Trades Executed']}"
                                    ]
                                })

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
                                    {"Portfolio Value ($)": backtester.portfolio_value_history},
                                    index=backtester.data.index[:len(backtester.portfolio_value_history)]
                                )
                                st.line_chart(df, use_container_width=True)

                        # Trade Log
                        if hasattr(backtester, "trade_log"):
                            with st.expander("📝 Trade Execution Log"):
                                st.code("\n".join(backtester.trade_log), language="text")

                    except Exception as e:
                        st.error(f"❌ Backtest failed: {str(e)}")
                        st.info("💡 Try adjusting the parameters or selecting a different time period")

        # --- Rebalancing Backtest Tab ---
        with tab2:
            st.header("⚖️ Rebalancing Backtest")
            st.markdown("Test portfolio rebalancing strategies with customizable drift thresholds and market conditions")

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
                        help="Starting capital for the rebalancing backtest"
                    )

                with col2:
                    period_options = ["1y", "2y", "3y", "4y", "5y", "Custom"]
                    selected_period_option = st.selectbox(
                        "Backtest Period",
                        period_options,
                        index=2,  # Default to 3y
                        key="rebalance_period_dropdown",
                        help="Historical period for rebalancing analysis"
                    )

                    if selected_period_option == "Custom":
                        period = st.text_input(
                            "Custom Period (e.g., 7y)",
                            value="6y",
                            key="rebalance_period_custom",
                            help="Enter custom period: Xy for years"
                        )
                    else:
                        period = selected_period_option

            # Advanced Parameters Section
            with st.expander("🔧 Advanced Rebalancing Parameters", expanded=False):
                st.markdown("##### 📊 Allocation Drift Thresholds")
                st.info("Set how much deviation from target allocation triggers rebalancing")

                col1, col2 = st.columns(2)
                with col1:
                    majors_drift_threshold = st.slider(
                        "Major Coins Drift Threshold (%)",
                        min_value=1.0, max_value=20.0, value=3.0, step=0.5,
                        help="Drift threshold for BTC/ETH before rebalancing triggers"
                    )
                with col2:
                    alts_drift_threshold = st.slider(
                        "Altcoins Drift Threshold (%)",
                        min_value=1.0, max_value=20.0, value=5.0, step=0.5,
                        help="Drift threshold for altcoins before rebalancing triggers"
                    )

                st.markdown("##### 💱 Trade Aggressiveness")
                st.info("Control how much of the imbalance to correct in each rebalancing event")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Major Coins (BTC/ETH)**")
                    majors_sell_multiplier = st.slider(
                        "Sell Multiplier",
                        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                        help="Portion of overweight amount to sell (0.5 = sell 50%)",
                        key="majors_sell"
                    )
                    majors_buy_multiplier = st.slider(
                        "Buy Multiplier",
                        min_value=0.1, max_value=2.0, value=0.75, step=0.1,
                        help="Portion of underweight amount to buy (0.75 = buy 75%)",
                        key="majors_buy"
                    )
                with col2:
                    st.markdown("**Altcoins**")
                    alts_sell_multiplier = st.slider(
                        "Sell Multiplier",
                        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                        help="Portion of overweight amount to sell for altcoins",
                        key="alts_sell"
                    )
                    alts_buy_multiplier = st.slider(
                        "Buy Multiplier",
                        min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                        help="Portion of underweight amount to buy for altcoins",
                        key="alts_buy"
                    )

                st.markdown("##### 🐻 Market Regime Rules")
                suppress_buys_in_bear = st.checkbox(
                    "Suppress Buys in Bear Market",
                    value=True,
                    help="Avoid buying during bearish market conditions (except oversold scenarios)"
                )

                st.markdown("##### 🎯 Custom Target Allocation")
                with st.container():
                    st.info("💡 Customize allocation or leave as-is to use your default config")

                    # Get current target allocation from config
                    default_allocation = config.get("target_allocation", {})
                    custom_allocation = {}

                    # Create a more organized layout for allocation inputs
                    if default_allocation:
                        allocation_cols = st.columns(min(3, len(default_allocation)))
                        for i, (coin, default_pct) in enumerate(default_allocation.items()):
                            with allocation_cols[i % len(allocation_cols)]:
                                custom_pct = st.number_input(
                                    f"{coin.upper()}",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=default_pct * 100,
                                    step=1.0,
                                    key=f"custom_alloc_{coin}",
                                    help=f"Target allocation percentage for {coin.upper()}"
                                )
                                custom_allocation[coin] = custom_pct / 100.0

                        # Allocation validation
                        total_alloc = sum(custom_allocation.values())
                        if abs(total_alloc - 1.0) > 0.001:
                            if total_alloc > 1.0:
                                st.warning(f"⚠️ Total allocation is {total_alloc:.1%} (over 100%)")
                            else:
                                st.warning(f"⚠️ Total allocation is {total_alloc:.1%} (under 100%)")
                        else:
                            st.success(f"✅ Total allocation: {total_alloc:.1%}")

            # Run Rebalancing Backtest Button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_rebalance_backtest = st.button(
                    "🚀 Run Rebalancing Backtest",
                    type="primary",
                    use_container_width=True
                )

            if run_rebalance_backtest:
                with st.spinner("🔄 Running rebalancing backtest analysis..."):
                    try:
                        # Create custom config with user parameters
                        custom_config = config.copy()

                        # Update rebalancing parameters
                        custom_config["rebalance_technical"] = {
                            "market_regime_rules": {
                                "suppress_buys_in_bear": suppress_buys_in_bear
                            },
                            "majors": {
                                "allocation_drift_threshold_pct": majors_drift_threshold,
                                "sell_percentage_multiplier": majors_sell_multiplier,
                                "buy_amount_multiplier": majors_buy_multiplier
                            },
                            "alts": {
                                "allocation_drift_threshold_pct": alts_drift_threshold,
                                "sell_percentage_multiplier": alts_sell_multiplier,
                                "buy_amount_multiplier": alts_buy_multiplier
                            }
                        }

                        # Update target allocation if user customized it
                        if abs(sum(custom_allocation.values()) - 1.0) < 0.001:
                            custom_config["target_allocation"] = custom_allocation

                        backtester = RebalancingBacktester(config=custom_config)
                        backtester.run(initial_capital=initial_capital, period=period)

                        st.success("✅ Rebalancing backtest completed successfully!")

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
                                    delta=f"{stats['Strategy Outperformance']:+.1%} vs B&H"
                                )

                            with metric_cols[1]:
                                st.metric(
                                    "Final Value",
                                    f"${stats['Final Portfolio Value']:,.0f}",
                                    delta=f"${stats['Final Portfolio Value'] - stats['Initial Capital']:+,.0f}"
                                )

                            with metric_cols[2]:
                                st.metric(
                                    "Max Drawdown",
                                    f"{stats['Maximum Drawdown']:.1%}"
                                )

                            with metric_cols[3]:
                                st.metric(
                                    "Total Trades",
                                    f"{stats['Total Trades Executed']}"
                                )

                            # Detailed Results Table
                            with st.expander("📋 Detailed Performance Metrics", expanded=True):
                                st.table({
                                    "Metric": [
                                        "Initial Capital",
                                        "Final Portfolio Value",
                                        "Strategy Total Return",
                                        "Buy & Hold Return",
                                        "Strategy Outperformance",
                                        "Maximum Drawdown",
                                        "Annualized Volatility",
                                        "Sharpe Ratio",
                                        "Total Trades Executed"
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
                                        f"{stats['Total Trades Executed']}"
                                    ]
                                })

                        # Equity Curve
                        if hasattr(backtester, "portfolio_value_history"):
                            st.markdown("### 📈 Portfolio Equity Curve")
                            if isinstance(backtester.portfolio_value_history, list) and backtester.portfolio_value_history:
                                # Convert to DataFrame with proper date index
                                df = pd.DataFrame(backtester.portfolio_value_history)
                                df['date'] = pd.to_datetime(df['date'])
                                df = df.set_index('date')
                                df = df.rename(columns={'value': 'Portfolio Value ($)'})

                                # Create the chart with proper date axis
                                st.line_chart(df, use_container_width=True)

                        # Trade Log
                        if hasattr(backtester, "trade_log"):
                            with st.expander("📝 Rebalancing Trade Log"):
                                st.code("\n".join(backtester.trade_log), language="text")

                    except Exception as e:
                        st.error(f"❌ Rebalancing backtest failed: {str(e)}")
                        st.info("💡 Try adjusting the parameters or check your allocation settings")

    def render_charts(self):


        st.markdown("## 📊 Portfolio Charts")
        st.info("Visualize your portfolio with interactive charts. Select a chart type below.")

        # tracker = self.initialize_tracker()
        # metrics = st.session_state.get("portfolio_metrics")
        # if not metrics:
        #     st.info("No portfolio metrics available. Please run a full sync first.")
        #     return

        # holdings_df = metrics.get("holdings_df")
        # target_allocation = self.config_manager.config.get("target_allocation", {})
        # snapshots = None
        # try:
        #     snapshots = tracker.db_manager.get_all_snapshots()
        # except Exception:
        #     snapshots = None

        # chart_tabs = st.tabs([
        #     "🟣 Allocation Pie",
        #     "🟡 Allocation Comparison",
        #     "🟢 P/L by Asset",
        #     "🔵 Value Over Time"
        # ])

        # # --- 1. Portfolio Allocation Pie Chart ---
        # with chart_tabs[0]:
        #     st.subheader("Portfolio Allocation (by Value)")
        #     if holdings_df is not None and not holdings_df.empty:
        #         pie_df = holdings_df.copy()
        #         pie_df = pie_df[pie_df["value_usd"] > 0]
        #         # Plotly chart for interactivity
        #         fig_plotly = px.pie(
        #             pie_df,
        #             names="symbol",
        #             values="value_usd",
        #             title="Portfolio Allocation by Value",
        #             hole=0.4
        #         )
        #         st.plotly_chart(fig_plotly, use_container_width=True)
        #         # Matplotlib chart for universal PNG download
        #         fig, ax = plt.subplots()
        #         ax.pie(pie_df["value_usd"], labels=pie_df["symbol"], autopct='%1.1f%%', startangle=90)
        #         ax.set_title("Portfolio Allocation by Value")
        #         buf = io.BytesIO()
        #         fig.savefig(buf, format="png", bbox_inches="tight")
        #         buf.seek(0)
        #         st.download_button(
        #             "Download Pie Chart (PNG, universal)",
        #             buf,
        #             file_name="portfolio_allocation_pie.png",
        #             mime="image/png"
        #         )
        #         plt.close(fig)
        #     else:
        #         st.info("No holdings data available for allocation pie chart.")

        # # --- 2. Current vs. Target Allocation Bar Chart ---
        # with chart_tabs[1]:
        #     st.subheader("Current vs. Target Allocation")
        #     if holdings_df is not None and not holdings_df.empty and target_allocation:
        #         current_alloc = holdings_df.set_index('symbol')['allocation'] * 100
        #         target_alloc = pd.Series(target_allocation) * 100
        #         comparison_df = pd.DataFrame({
        #             'Current (%)': current_alloc,
        #             'Target (%)': target_alloc
        #         }).fillna(0)
        #         # Plotly chart for interactivity
        #         fig_plotly = go.Figure()
        #         fig_plotly.add_trace(go.Bar(
        #             x=comparison_df.index,
        #             y=comparison_df['Current (%)'],
        #             name='Current (%)',
        #             marker_color='indigo'
        #         ))
        #         fig_plotly.add_trace(go.Bar(
        #             x=comparison_df.index,
        #             y=comparison_df['Target (%)'],
        #             name='Target (%)',
        #             marker_color='orange'
        #         ))
        #         fig_plotly.update_layout(
        #             barmode='group',
        #             title="Current vs. Target Portfolio Allocation",
        #             xaxis_title="Asset",
        #             yaxis_title="Allocation (%)"
        #         )
        #         st.plotly_chart(fig_plotly, use_container_width=True)
        #         # Matplotlib chart for universal PNG download
        #         fig, ax = plt.subplots()
        #         width = 0.35
        #         x = range(len(comparison_df.index))
        #         ax.bar([i - width/2 for i in x], comparison_df['Current (%)'], width=width, label='Current (%)', color='indigo')
        #         ax.bar([i + width/2 for i in x], comparison_df['Target (%)'], width=width, label='Target (%)', color='orange')
        #         ax.set_xticks(list(x))
        #         ax.set_xticklabels(comparison_df.index, rotation=45)
        #         ax.set_ylabel("Allocation (%)")
        #         ax.set_title("Current vs. Target Portfolio Allocation")
        #         ax.legend()
        #         fig.tight_layout()
        #         buf = io.BytesIO()
        #         fig.savefig(buf, format="png", bbox_inches="tight")
        #         buf.seek(0)
        #         st.download_button(
        #             "Download Allocation Comparison (PNG, universal)",
        #             buf,
        #             file_name="allocation_comparison_bar.png",
        #             mime="image/png"
        #         )
        #         plt.close(fig)
        #     else:
        #         st.info("No data available for allocation comparison chart.")

        # # --- 3. Unrealized P/L by Asset Bar Chart ---
        # with chart_tabs[2]:
        #     st.subheader("Unrealized Profit/Loss (P/L) by Asset")
        #     if holdings_df is not None and not holdings_df.empty and "unrealized_pl_usd" in holdings_df.columns:
        #         pl_df = holdings_df.set_index('symbol')["unrealized_pl_usd"].sort_values()
        #         colors = ['red' if x < 0 else 'green' for x in pl_df]
        #         # Plotly chart for interactivity
        #         fig_plotly = go.Figure([go.Bar(
        #             x=pl_df.index,
        #             y=pl_df.values,
        #             marker_color=colors
        #         )])
        #         fig_plotly.update_layout(
        #             title="Unrealized P/L by Asset",
        #             xaxis_title="Asset",
        #             yaxis_title="Unrealized P/L (USD)"
        #         )
        #         st.plotly_chart(fig_plotly, use_container_width=True)
        #         # Matplotlib chart for universal PNG download
        #         fig, ax = plt.subplots()
        #         bar_colors = ['green' if v >= 0 else 'red' for v in pl_df.values]
        #         ax.bar(pl_df.index, pl_df.values, color=bar_colors)
        #         ax.set_ylabel("Unrealized P/L (USD)")
        #         ax.set_title("Unrealized P/L by Asset")
        #         ax.axhline(0, color='black', linewidth=0.8)
        #         ax.tick_params(axis='x', rotation=45)  # <-- Use this line
        #         fig.tight_layout()
        #         buf = io.BytesIO()
        #         fig.savefig(buf, format="png", bbox_inches="tight")
        #         buf.seek(0)
        #         st.download_button(
        #             "Download P/L Chart (PNG, universal)",
        #             buf,
        #             file_name="pl_by_asset_bar.png",
        #             mime="image/png"
        #         )
        #         plt.close(fig)
        #     else:
        #         st.info("No P/L data available for this chart.")

        # # --- 4. Portfolio Value Over Time Line Chart ---
        # with chart_tabs[3]:
        #     st.subheader("Portfolio Value Over Time")
        #     if snapshots is not None and not snapshots.empty:
        #         snapshots = snapshots.copy()
        #         snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"])
        #         snapshots = snapshots.dropna(subset=["timestamp"])
        #         snapshots = snapshots[snapshots["timestamp"].dt.year > 2000]  # Filter out weird old dates

        #         # Drop duplicate timestamps, keeping the last entry for each timestamp
        #         snapshots = snapshots.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        #         # Debug output (optional, for development)
        #         st.write("Snapshot Data Preview", snapshots.head(20))
        #         st.write("Unique timestamps:", snapshots["timestamp"].nunique(), "Total rows:", len(snapshots))
        #         duplicates = snapshots[snapshots.duplicated(subset=["timestamp"], keep=False)]
        #         if not duplicates.empty:
        #             st.warning("Duplicate timestamps found in snapshots!")
        #             st.dataframe(duplicates)

        #         # Plotly chart for interactivity
        #         fig_plotly = px.line(
        #             snapshots,
        #             x="timestamp",
        #             y="total_value_usd",
        #             title="Portfolio Value Over Time",
        #             markers=True,
        #             labels={"total_value_usd": "Total Value (USD)", "timestamp": "Date"}
        #         )
        #         fig_plotly.update_xaxes(
        #             tickformat="%Y-%m-%d",  # Format as date
        #             ticklabelmode="period"
        #         )
        #         st.plotly_chart(fig_plotly, use_container_width=True)
        #         # Matplotlib chart for universal PNG download
        #         fig, ax = plt.subplots()
        #         ax.plot(snapshots["timestamp"], snapshots["total_value_usd"], marker='o')
        #         ax.set_xlabel("Date")
        #         ax.set_ylabel("Total Value (USD)")
        #         ax.set_title("Portfolio Value Over Time")
        #         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        #         fig.autofmt_xdate()
        #         fig.tight_layout()
        #         buf = io.BytesIO()
        #         fig.savefig(buf, format="png", bbox_inches="tight")
        #         buf.seek(0)
        #         st.download_button(
        #             "Download Value History (PNG, universal)",
        #             buf,
        #             file_name="portfolio_value_history.png",
        #             mime="image/png"
        #         )
        #         plt.close(fig)
        #     else:
        #         st.info("No snapshot data available for value history chart.")

    def render_data_management(self):
        import pandas as pd
        from pathlib import Path

        st.markdown("## 🗄️ Data Management")

        tracker = self.initialize_tracker()
        db_path = tracker.db_manager.db_path
        backup_dir = tracker.db_manager.backup_dir

        tab1, tab2, tab3, tab4 = st.tabs([
            "💾 Backup & Restore", "⬆️ Import / ⬇️ Export", "📸 Snapshots", "ℹ️ Info"
        ])

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
                        st.download_button("Download", f, file_name=selected_backup, use_container_width=True)
                with col2:
                    if st.button("Delete", key=f"delete_{selected_backup}", use_container_width=True):
                        try:
                            backup_path.unlink()
                            st.success(f"Deleted {selected_backup}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
                with col3:
                    if st.button("Restore", key=f"restore_{selected_backup}", use_container_width=True):
                        st.warning("⚠️ This will overwrite your current database. This action is irreversible!")
                        if st.button("Confirm Restore", key=f"confirm_restore_{selected_backup}"):
                            success = tracker.db_manager.restore_from_backup(backup_path)
                            if success:
                                st.success("Database restored. Please restart the app to use the restored data.")
                            else:
                                st.error("Restore failed. See logs for details.")

        # --- 2. Import/Export ---
        with tab2:
            st.header("⬆️ Import / ⬇️ Export Raw Data")
            export_dir = Path(self.config_manager.config.get("exports", {}).get("path", "data/exports/"))
            data_export_dir = export_dir / "data_management"
            data_export_dir.mkdir(parents=True, exist_ok=True)

            st.markdown("#### Create New Export")
            
            col1, col2 = st.columns(2)
            with col1:
                export_type = st.selectbox("Select data to export", ["Holdings", "Transactions"])
            with col2:
                export_format = st.radio("Select format", ["CSV", "Excel"], horizontal=True)

            if st.button("🚀 Generate Export", use_container_width=True, type="primary"):
                with st.spinner(f"Generating {export_type} export..."):
                    df = None
                    if export_type == "Holdings":
                        df = tracker.db_manager.get_holdings()
                    else: # Transactions
                        df = tracker.db_manager.get_all_transactions()

                    if df is not None and not df.empty:
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_extension = "xlsx" if export_format == "Excel" else "csv"
                        filename = f"{export_type.lower()}_export_{now_str}.{file_extension}"
                        filepath = data_export_dir / filename

                        # Create a copy and handle timezones for Excel compatibility
                        export_df = df.copy()
                        for col in export_df.select_dtypes(['datetimetz']).columns:
                            export_df[col] = export_df[col].dt.tz_localize(None)

                        if export_format == "CSV":
                            export_df.to_csv(filepath, index=False)
                        else: # Excel
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
                reverse=True
            )

            if not all_files:
                st.info("No exports found. Generate one above!")
            else:
                selected_export = st.selectbox(
                    "Select an export file:",
                    options=all_files,
                    format_func=lambda x: f"{x.name} ({x.stat().st_size/1024:.1f} KB, {datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
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
                            mime=f"text/{selected_export.suffix[1:].lower()}" if selected_export.suffix == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
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
            uploaded_holdings = st.file_uploader("Import Holdings (CSV/Excel)", type=["csv", "xlsx"], key="import_holdings")
            if uploaded_holdings:
                try:
                    if uploaded_holdings.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_holdings)
                    else:
                        df = pd.read_excel(uploaded_holdings)
                    st.dataframe(df)
                    if st.button("Import Holdings"):
                        tracker.db_manager.update_holdings(df)
                        st.success("Holdings imported successfully!")
                except Exception as e:
                    st.error(f"Failed to import holdings: {e}")

            uploaded_tx = st.file_uploader("Import Transactions (CSV/Excel)", type=["csv", "xlsx"], key="import_tx")
            if uploaded_tx:
                try:
                    if uploaded_tx.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_tx)
                    else:
                        df = pd.read_excel(uploaded_tx)
                    st.dataframe(df)
                    # You would need to implement a bulk insert for transactions
                    # if st.button("Import Transactions"):
                    #     tracker.db_manager.bulk_insert_transactions(df.to_dict(orient="records"))
                    #     st.success("Transactions imported successfully!")
                except Exception as e:
                    st.error(f"Failed to import transactions: {e}")

        # --- 3. Manage Snapshots ---
        with tab3:
            st.header("📸 Snapshots")
            try:
                snapshots = tracker.db_manager.get_all_snapshots()
                snapshots = snapshots.drop_duplicates(subset=["timestamp"])
            except Exception as e:
                st.error(f"Failed to load snapshots: {e}")
                snapshots = pd.DataFrame()
            if snapshots.empty:
                st.info("No snapshots found. Take a snapshot from the dashboard first.")
            else:
                snapshots = snapshots.sort_values("timestamp")
                # Improved label for NaT/NaN
                snapshot_labels = []
                for _, row in snapshots.iterrows():
                    if pd.isna(row['timestamp']):
                        label = f"⚠️ Invalid Snapshot (No Timestamp) | Value: ${row['total_value_usd']:,.2f}"
                    else:
                        label = f"{row['timestamp']} | Value: ${row['total_value_usd']:,.2f}"
                    snapshot_labels.append(label)
                selected_idx = st.selectbox("Select Snapshot", range(len(snapshots)), format_func=lambda i: snapshot_labels[i])
                selected_row = snapshots.iloc[selected_idx]

                st.write("### Snapshot Details")
                display_dict = {
                    "Timestamp": str(selected_row["timestamp"]),
                    "Total Value (USD)": f"${selected_row['total_value_usd']:,.2f}",
                    "Total Cost Basis (USD)": f"${selected_row['total_cost_basis_usd']:,.2f}",
                    "Unrealized P/L (USD)": f"${selected_row['unrealized_pl_usd']:,.2f}",
                    "Unrealized P/L (%)": f"{selected_row['unrealized_pl_percent']:.2f}%",
                }
                st.table(pd.DataFrame([display_dict]))

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download Snapshot (CSV)", selected_row.to_frame().T.to_csv(index=False), "portfolio_snapshot.csv")
                with col2:
                    if st.button("Delete Selected Snapshot"):
                        timestamp_to_delete = selected_row["timestamp"]
                        try:
                            with tracker.db_manager._get_connection() as conn:
                                if pd.isna(timestamp_to_delete):
                                    sql = "DELETE FROM portfolio_snapshots WHERE timestamp IS NULL"
                                    params = ()
                                else:
                                    naive_timestamp = pd.to_datetime(timestamp_to_delete).to_pydatetime().replace(tzinfo=None)
                                    sql = "DELETE FROM portfolio_snapshots WHERE timestamp = ?"
                                    params = (naive_timestamp,)
                                conn.execute(sql, params)
                                conn.commit()
                            st.success(f"Deleted snapshot {selected_row['timestamp']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete snapshot: {e}")

            st.markdown("---")
            if st.button("Delete all Snapshot"):
                try:
                    tracker.cleanup_old_data()
                    st.success("All Snapshot cleaned up successfully!")
                except Exception as e:
                    st.error(f"Failed to clean old Snapshot: {e}")

        # --- 4. Database Info ---
        with tab4:
            st.header("ℹ️ Database Info")
            st.write(f"**Database Path:** `{db_path}`")
            if os.path.exists(db_path):
                st.write(f"**Size:** {os.path.getsize(db_path)/1024:.1f} KB")
                st.write(f"**Last Modified:** {pd.to_datetime(os.path.getmtime(db_path), unit='s')}")
            else:
                st.warning("Database file not found.")
            st.write(f"**Backup Directory:** `{backup_dir}`")
            if backup_dir.exists():
                st.write(f"**Backups Available:** {len(list(backup_dir.glob('*.bak')))}")
            else:
                st.warning("Backup directory not found.")

    def render_configuration(self):
        st.markdown("## ⚙️ Configuration")
        st.info("Configuration settings would be displayed here")

    def render_system_status(self):
        st.markdown("## 🔧 System Status")
        st.info("System status and API connection info would be displayed here")

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
            "⚙️ Configuration": self.render_configuration,
            "🔧 System Status": self.render_system_status
        }
        if selected_page in page_mapping:
            page_mapping[selected_page]()
        else:
            st.error("Page not found")

# Run the dashboard
if __name__ == "__main__":
    dashboard = PortfolioDashboard()
    dashboard.run()
