import asyncio
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from crypto_portfolio_tracker.dashboard import utils as ui_utils
from crypto_portfolio_tracker.portfolio_tracker import ExecutionMode
from crypto_portfolio_tracker.profit_taking_logic import ProfitTakingAnalyzer
from crypto_portfolio_tracker.dashboard.components import render_transfer_widget, render_trading_status_banner


def render_rebalancing_page(dashboard):
    """
    Render the portfolio rebalancing page.
    
    This page provides tools for analyzing portfolio allocation, generating rebalancing
    suggestions based on target allocations and technical analysis, and executing
    rebalancing trades. It also includes profit-taking functionality when the portfolio
    is balanced.
    
    The page displays current portfolio status, actionable trades, and allows users
    to select and execute specific trades. It handles both online and offline modes,
    with appropriate guards for trading functionality.
    
    Args:
        dashboard (Dashboard): The main dashboard controller instance
    """

    # Clear previous page state if coming from another page
    keys_to_clear = [ # "rebalance_metrics", "rebalance_suggestions" - Excluded to avoid expensive calls everytime.
        "rebalance_usdt",
        "rebalance_actionable", "rebalance_executing", "rebalance_results",
        "accepted_trades", "profit_opportunities", "profit_opportunities_loading",
        "selected_profit_trades", "profit_executing", "profit_results"
    ]
    ui_utils.initialize_page_state("rebalancing", keys_to_clear)

    st.markdown("## ⚖️ Portfolio Rebalancing")

    # Offline guard: disable rebalancing workflows when offline
    if dashboard.offline_mode:
        st.info("⚠️ Offline mode: Rebalancing analysis and execution are unavailable.")
        return

    # Last and Next info
    tracker = dashboard.initialize_tracker()
    last_rb = tracker.db_manager.get_latest_timestamp_for_source("REBALANCE")
    # Fallback to last sync if no rebalancing trades found
    if not last_rb:
        last_sync = tracker.db_manager.get_latest_timestamp_for_source("SYNC")
        last_rb = last_sync
    rb_freq = (
        dashboard.config_manager.config.get("automation", {})
        .get("rebalancing", {})
        .get("frequency")
        or "weekly"
    )
    next_rb = ui_utils.compute_next_time(last_rb, rb_freq)

    col_last, _, col_next = st.columns([2, 1.8, 2])
    with col_last:
        if last_rb:
            st.metric("Last Rebalance", ui_utils.format_datetime_local(last_rb))
        else:
            st.metric("Last Rebalance", "No trades yet")
    with col_next:
        st.metric(
            f"Next Check ({rb_freq.title()})", ui_utils.format_datetime_local(next_rb)
        )

    # Check trading status
    is_live = dashboard.config_manager.is_live
    is_testnet = dashboard.config_manager.is_testnet_mode

    # Trading Status Banner
    render_trading_status_banner(is_live, is_testnet)

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
            tracker = dashboard.initialize_tracker()

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
                if not dashboard.config_manager.is_testnet_mode:
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
                actionable = any(s in ["BUY", "SELL"] for s in suggestions_df["Signal"])
            st.session_state.rebalance_actionable = actionable

            # Clear previous results and accepted trades
            st.session_state.rebalance_results = None
            st.session_state.accepted_trades = set()

    # --- Display Core Portfolio Status ---
    st.markdown("### 🎯 Current Core Portfolio Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "#### 📊 Current Allocation",
            help="Allocations are shown as a percentage of your core portfolio (assets being rebalanced).",
        )
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
        else:
            st.info("Portfolio analysis required")

    with col2:
        st.markdown("#### 💰 Available Liquidity")
        tracker = dashboard.initialize_tracker()
        try:
            usdt_balance = tracker.get_available_usdt_balance()

            total_usdt = float(usdt_balance.get("total", 0.0))
            spot_earn_usdt = float(usdt_balance.get("spot_earn", 0.0))
            funding_usdt = float(usdt_balance.get("funding", 0.0))
        except Exception:
            usdt_balance = None

        if usdt_balance is not None:
            st.metric("USDT Balance (Spot + Earn)", f"${spot_earn_usdt:,.2f}")
            st.metric("USDT Balance (Funding)", f"${funding_usdt:,.2f}")
            st.metric("Total USDT Balance", f"${total_usdt:,.2f}")
            if total_usdt < 100:
                st.warning("⚠️ Low USDT balance may limit rebalancing options")
            else:
                st.success("✅ Sufficient liquidity for rebalancing")

            # Offer contextual transfer if funding has balance
            if funding_usdt > 0:
                # If suggestions exist, estimate required USDT for BUY actions
                required_usdt = None
                suggestions_df = st.session_state.get("rebalance_suggestions")
                if (
                    suggestions_df is not None
                    and not suggestions_df.empty
                    and "action_usd_value" in suggestions_df.columns
                    and "Signal" in suggestions_df.columns
                ):
                    try:
                        required_usdt = float(
                            suggestions_df.loc[
                                suggestions_df["Signal"] == "BUY",
                                "action_usd_value",
                            ]
                            .clip(lower=0)
                            .sum()
                        )
                    except Exception:
                        required_usdt = None

                render_transfer_widget(
                    dashboard, context="rebalancing", required_amount=required_usdt
                )

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

        # Check if portfolio is balanced and profit-taking is enabled
        # Try to get config from tracker first, fallback to config_manager
        if (
            hasattr(dashboard, "tracker")
            and dashboard.tracker
            and hasattr(dashboard.tracker, "config")
        ):
            profit_config = dashboard.tracker.config.get("profit_taking", {})
        elif hasattr(dashboard, "config_manager") and dashboard.config_manager:
            profit_config = dashboard.config_manager.config.get("profit_taking", {})
        else:
            profit_config = {}

        profit_enabled = profit_config.get("enabled", False)

        if actionable_trades.empty and profit_enabled:
            # Portfolio is balanced, check for profit-taking opportunities
            st.markdown("#### ✅ Portfolio Balanced - Checking Profit Opportunities")

            if "profit_opportunities" not in st.session_state:
                st.session_state.profit_opportunities = None
                st.session_state.profit_opportunities_loading = False

            if not st.session_state.profit_opportunities_loading:
                if st.button(
                    "🔍 Analyze Profit-Taking Opportunities",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.profit_opportunities_loading = True
                    st.rerun()

            if st.session_state.profit_opportunities_loading:
                with st.spinner("🔄 Analyzing profit-taking opportunities..."):
                    tracker = dashboard.initialize_tracker()

                    # Get current portfolio metrics for optimized analysis
                    metrics = asyncio.run(tracker.calculate_portfolio_metrics())
                    core_holdings_df = metrics.get("core_holdings_df", pd.DataFrame())

                    # Use optimized profit-taking analysis by passing already computed data
                    opportunities = asyncio.run(
                        tracker.get_profit_taking_opportunities(
                            core_holdings_df=core_holdings_df,
                            rebalance_suggestions_df=suggestions_df,
                        )
                    )
                    st.session_state.profit_opportunities = opportunities
                    st.session_state.profit_opportunities_loading = False
                    st.rerun()

            # Display profit opportunities if available
            if st.session_state.profit_opportunities is not None:
                opportunities = st.session_state.profit_opportunities

                if isinstance(opportunities, list):
                    if len(opportunities) > 0:
                        st.markdown("#### 💰 Profit-Taking Opportunities Found!")
                        st.success(f"Found {len(opportunities)} assets with profit-taking potential")
                        _render_profit_opportunities_section(dashboard, opportunities)
                    else:
                        st.info("✅ No profit-taking opportunities found at current thresholds")
                else:
                    st.warning(f"Unexpected opportunities format: {type(opportunities)}")

        # Show actionable trades with enhanced selection interface
        elif not actionable_trades.empty:
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
                            if (
                                row["Signal"] == "SELL"
                                and "action_coin_quantity" in row
                            ):
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
                        st.markdown(f"${row['Current Value (USD)']:,.2f}")

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

                tracker = dashboard.initialize_tracker()
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
                    if not dashboard.config_manager.is_testnet_mode:
                        spot_balances_df = tracker.fetch_binance_balances()
                        earn_balances = tracker.fetcher.fetch_simple_earn_balances(
                            spot_balances_df
                        )

                    progress_bar.progress(70)
                    status_text.text("🚀 Executing trades...")

                    # Run execution
                    result = asyncio.run(
                        tracker.execute_rebalancing_trades_core(
                            filtered_df,
                            earn_balances,
                            confirmation_callback=None,  # Add this parameter
                            execution_mode=ExecutionMode.AUTO,  # Updated to use new enum
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
                        # store batch id for UI
                        st.session_state.rebalance_batch_id = result.data.get(
                            "batch_id"
                        )
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
    if st.session_state.rebalance_results:
        st.markdown("### 📋 Execution Results & Logs")
        results = st.session_state.rebalance_results
        batch_id = st.session_state.get("rebalance_batch_id")

        if "EXECUTION COMPLETED" in results:
            st.success("✅ **Execution Successful!**")

            # Show execution summary in an expandable section
            with st.expander("📊 View Detailed Execution Log", expanded=True):
                st.code(results, language="text")

            # Clear selection after successful execution
            if st.button("🔄 Clear Results", type="secondary"):
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


def _render_profit_opportunities_section(dashboard, opportunities):
    """
    Renders the profit-taking opportunities section in the Web UI.
    """
    # Initialize session state for profit taking
    if "selected_profit_trades" not in st.session_state:
        st.session_state.selected_profit_trades = set()
    if "profit_executing" not in st.session_state:
        st.session_state.profit_executing = False
    if "profit_results" not in st.session_state:
        st.session_state.profit_results = None

    tracker = dashboard.initialize_tracker()
    from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer

    analyzer = CryptoTrendAnalyzer(
        config=tracker.config, binance_client=tracker.binance_client
    )
    profit_analyzer = ProfitTakingAnalyzer(tracker.config, analyzer)

    # Get default take percentage
    default_take_percentage = tracker.config.get("profit_taking", {}).get(
        "default_take_percentage", 30
    )

    # Calculate profit amounts for each opportunity
    opportunities_with_trades = []
    total_profit_value = 0

    for opportunity in opportunities:
        profit_amount_usd, quantity_to_sell = (
            profit_analyzer.calculate_profit_take_amount(
                opportunity, default_take_percentage
            )
        )
        total_profit_value += profit_amount_usd

        opportunities_with_trades.append(
            {
                "opportunity": opportunity,
                "profit_amount_usd": profit_amount_usd,
                "quantity_to_sell": quantity_to_sell,
            }
        )

    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Opportunities", len(opportunities))
    with col2:
        st.metric("Total Profit Available", f"${total_profit_value:,.2f}")
    with col3:
        selected_profit_count = len(st.session_state.selected_profit_trades)
        st.metric("Selected for Execution", selected_profit_count)

    # Control buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button(
            "✅ Select All Profits", key="select_all_profits", use_container_width=True
        ):
            for i in range(len(opportunities)):
                st.session_state.selected_profit_trades.add(str(i))
            st.rerun()

    with col2:
        if st.button(
            "❌ Deselect All Profits",
            key="deselect_all_profits",
            use_container_width=True,
        ):
            st.session_state.selected_profit_trades.clear()
            st.rerun()

    st.markdown("---")

    # Display each profit opportunity
    for i, opp_data in enumerate(opportunities_with_trades):
        opportunity = opp_data["opportunity"]
        profit_amount_usd = opp_data["profit_amount_usd"]
        quantity_to_sell = opp_data["quantity_to_sell"]

        with st.container():
            # Profit opportunity card with green border
            st.markdown(
                """
                <div style="border-left: 4px solid #28a745; padding-left: 15px; margin: 10px 0;">
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns([1, 4, 2, 1])

            with col1:
                selected = st.checkbox(
                    "Take Profit",
                    key=f"profit_{i}",
                    value=str(i) in st.session_state.selected_profit_trades,
                    help=f"Check to include this profit-taking trade for {opportunity.symbol}",
                )
                if selected:
                    st.session_state.selected_profit_trades.add(str(i))
                else:
                    st.session_state.selected_profit_trades.discard(str(i))

            with col2:
                st.markdown(f"**{opportunity.symbol}**")
                st.markdown(
                    f"**Unrealized Gain:** ${opportunity.unrealized_gain_usd:,.2f} (+{opportunity.unrealized_gain_pct:.1f}%)"
                )
                st.markdown(
                    f"**Opportunity Score:** {opportunity.opportunity_score:.0f}/100"
                )
                st.markdown(f"**Reasons:** {', '.join(opportunity.reasons[:2])}")

            with col3:
                st.markdown("**Profit to Take**")
                st.markdown(f"💰 **${profit_amount_usd:,.2f} USDT**")
                st.markdown(f"📈 **{quantity_to_sell:.8g} {opportunity.symbol}**")
                st.markdown(f"📊 **{default_take_percentage}% of gains**")

            with col4:
                st.markdown(
                    """
                    <div style="background: #d4edda; color: #155724; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;">
                        💰 PROFIT
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Execution summary
    selected_count = len(st.session_state.selected_profit_trades)
    if selected_count > 0:
        selected_opportunities = [
            opportunities_with_trades[int(i)]
            for i in st.session_state.selected_profit_trades
        ]
        selected_profit_value = sum(
            opp["profit_amount_usd"] for opp in selected_opportunities
        )

        st.success(f"✅ **{selected_count} profit-taking trade(s) selected**")
        st.markdown(f"**Total Selected Profit Value: ${selected_profit_value:,.2f}**")

        # Execute button
        if not st.session_state.profit_executing:
            if st.button(
                f"🚀 Execute {selected_count} Profit-Taking Trade(s)",
                key="execute_profit_btn",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.profit_executing = True
                st.rerun()
        else:
            st.info("⏳ Executing profit-taking trades... Please wait.")
    else:
        st.info("ℹ️ Select profit opportunities above to enable execution.")

    # Handle profit execution
    if st.session_state.profit_executing:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔄 Preparing profit-taking execution...")
            progress_bar.progress(30)

            # Prepare selected trades for execution
            selected_opportunities = [
                opportunities_with_trades[int(i)]
                for i in st.session_state.selected_profit_trades
            ]

            selected_trades = []
            for opp_data in selected_opportunities:
                selected_trades.append(
                    {
                        "symbol": opp_data["opportunity"].symbol,
                        "usd_amount": opp_data["profit_amount_usd"],
                        "coin_quantity": opp_data["quantity_to_sell"],
                    }
                )

            progress_bar.progress(70)
            status_text.text("🚀 Executing profit-taking trades...")

            # Execute the trades
            is_live = dashboard.config_manager.is_live
            result = asyncio.run(
                tracker.execute_profit_taking_trades(selected_trades, is_live)
            )

            progress_bar.progress(100)
            status_text.text(
                "✅ Profit-taking completed!"
                if result.success
                else "❌ Profit-taking failed!"
            )

            # Store results
            output = "\n".join(result.messages)
            errors_output = "\n".join(result.errors) if result.errors else ""

            if result.success:
                st.session_state.profit_results = (
                    f"=== PROFIT-TAKING EXECUTION COMPLETED ===\n"
                    f"Executed {result.data.get('trades_executed', 0)} profit-taking trade(s)\n"
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"{output}\n"
                    f"=== END EXECUTION LOG ==="
                )
            else:
                st.session_state.profit_results = (
                    f"=== PROFIT-TAKING EXECUTION FAILED ===\n"
                    f"Executed {result.data.get('trades_executed', 0)} profit-taking trade(s)\n"
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"{output}\n"
                    f"{errors_output if errors_output else ''}\n"
                    f"=== END EXECUTION LOG ==="
                )

        except Exception as e:
            progress_bar.progress(100)
            status_text.text("❌ Execution failed!")
            st.session_state.profit_results = f"❌ **Execution Error**: {str(e)}"

        finally:
            st.session_state.profit_executing = False
            st.rerun()

    # Show execution results
    if st.session_state.profit_results:
        st.markdown("#### 📋 Profit-Taking Execution Results")
        results = st.session_state.profit_results

        if "COMPLETED" in results:
            st.success("✅ **Profit-Taking Execution Successful!**")
        elif "FAILED" in results:
            st.error("❌ **Profit-Taking Execution Failed**")
        else:
            st.warning("⚠️ **Profit-Taking Execution Issues**")

        with st.expander("📊 View Detailed Execution Log", expanded=True):
            st.code(results, language="text")

        # Clear results button
        if st.button(
            "🔄 Clear Profit Results", type="secondary", key="clear_profit_results"
        ):
            st.session_state.selected_profit_trades.clear()
            st.session_state.profit_results = None
            st.session_state.profit_opportunities = None
            st.success("✅ Profit-taking results cleared.")
            st.rerun()
