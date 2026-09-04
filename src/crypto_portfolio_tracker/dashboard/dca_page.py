import asyncio
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from crypto_portfolio_tracker.dashboard import utils as ui_utils
from crypto_portfolio_tracker.dashboard.components import render_transfer_widget, render_trading_status_banner


def render_dca_page(dashboard):
    """Render the Dollar Cost Averaging (DCA) page using a state machine."""

    # Clear previous page state if coming from another page
    keys_to_clear = [
        "dca_step", "dca_trades_for_confirmation",
        "dca_execution_result_data", "dca_batch_id",
        "current_portfolio_value"
    ]
    ui_utils.initialize_page_state("dca", keys_to_clear)

    st.markdown("## 💸 Dollar Cost Averaging (DCA)")

    # Offline guard: disable DCA workflows when offline
    if dashboard.offline_mode:
        st.info("⚠️ Offline mode: DCA is unavailable.")
        return

    # Last and Next info
    last_dca = (
        dashboard.initialize_tracker().db_manager.get_latest_timestamp_for_source("DCA")
    )
    dca_freq = (
        dashboard.config_manager.config.get("automation", {})
        .get("dca", {})
        .get("frequency")
        or "monthly"
    )
    next_dca = ui_utils.compute_next_time(last_dca, dca_freq)

    col_last, _, col_next = st.columns([2, 1.8, 2])
    with col_last:
        if last_dca:
            st.metric("Last DCA", ui_utils.format_datetime_local(last_dca))
        else:
            st.metric("Last DCA", "No trades yet")
    with col_next:
        st.metric(
            f"Next DCA ({dca_freq.title()})", ui_utils.format_datetime_local(next_dca)
        )

    tracker = dashboard.initialize_tracker()
    if not tracker:
        st.error("❌ Tracker not initialized")
        return

    # Initialize state
    if "dca_step" not in st.session_state:
        st.session_state.dca_step = "initial"

    # Check trading status
    is_live = dashboard.config_manager.is_live
    is_testnet = dashboard.config_manager.is_testnet_mode

    # Trading Status Banner
    render_trading_status_banner(is_live, is_testnet)

    # --- Balance Display Section ---
    st.markdown("### 💰 Available USDT")
    usdt_balances = tracker.get_available_usdt_balance()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Spot + Earn Balance", f"${usdt_balances['spot_earn']:,.2f}")
    with col2:
        st.metric("Funding Balance", f"${usdt_balances['funding']:,.2f}")
    with col3:
        st.metric("Total Available", f"${usdt_balances['total']:,.2f}")
    st.markdown("---")

    # Contextual transfer (only when funding has balance)
    if float(usdt_balances.get("funding", 0.0)) > 0:
        # If user typed a DCA amount later, widget will still be helpful here with hint None
        render_transfer_widget(dashboard, context="DCA")

    # --- Current Portfolio Status ---
    st.markdown("### 📊 Current Portfolio Status")

    # Get current portfolio metrics
    metrics = st.session_state.get("portfolio_metrics")
    if not metrics:
        st.info("No portfolio metrics available. Please run a full sync first.")
        if st.button("🔄 Run Full Sync Now"):
            dashboard.run_full_sync()
            st.rerun()
        return

    # Get current portfolio data
    core_portfolio = metrics.get("core_holdings_df", pd.DataFrame())
    target_allocation = tracker.config.get("target_allocation", {})

    if not core_portfolio.empty:
        st.markdown("#### 🎯 Current vs Target Allocation")

        # Create comparison table
        comparison_data = []
        for asset in target_allocation.keys():
            asset_row = core_portfolio[core_portfolio["symbol"] == asset]
            current_value = (
                asset_row["value_usd"].iloc[0] if not asset_row.empty else 0.0
            )
            current_pct = (
                asset_row["core_allocation"].iloc[0] * 100
                if not asset_row.empty
                else 0.0
            )
            target_pct = target_allocation[asset] * 100

            comparison_data.append(
                {
                    "Asset": asset,
                    "Current %": f"{current_pct:.2f}%",
                    "Target %": f"{target_pct:.2f}%",
                    "Current Value": f"${current_value:,.2f}",
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Store current portfolio value for DCA calculations
        total_value = core_portfolio["value_usd"].sum()
        st.session_state.current_portfolio_value = total_value

        # Collapsed by default: this is a reference table, not the DCA workflow.
        with st.expander("📐 Finish to targets (no sells)", expanded=False):
            st.caption("Implied total comes from whichever holding demands the "
                       "biggest finished size, so nothing held ever needs selling.")
            plan_values: dict[str, float] = {}
            plan_unknown: list[str] = []
            for asset in target_allocation.keys():
                asset_row = core_portfolio[core_portfolio["symbol"] == asset]
                if asset_row.empty:
                    plan_values[asset] = 0.0
                    continue
                raw_value = asset_row["value_usd"].iloc[0]
                # NaN is truthy: unknown must render as em dash, never $0.00.
                if raw_value is None or pd.isna(raw_value):
                    plan_unknown.append(asset)
                    plan_values[asset] = 0.0
                else:
                    plan_values[asset] = float(raw_value)

            plan_anchors = {
                s: plan_values[s] / float(w)
                for s, w in target_allocation.items()
                if s not in plan_unknown and plan_values[s] > 0 and float(w) > 0
            }
            if not plan_anchors:
                st.info("No holdings to anchor from yet — nothing held, nothing to finish.")
            else:
                plan_total = max(plan_anchors.values())
                plan_winner = max(plan_anchors, key=lambda s: plan_anchors[s])
                st.metric("Implied finished size",
                          f"${plan_total:,.2f}",
                          help=f"Anchored by {plan_winner}")
                plan_rows = []
                plan_need_total = 0.0
                for asset, weight in target_allocation.items():
                    goal = plan_total * float(weight)
                    need = max(0.0, goal - plan_values[asset])
                    plan_need_total += need
                    plan_rows.append(
                        {
                            "Asset": asset,
                            "Target %": f"{float(weight) * 100:.2f}%",
                            "Target Value": f"${goal:,.2f}",
                            "Current Value": ("—" if asset in plan_unknown
                                              else f"${plan_values[asset]:,.2f}"),
                            "Still to Buy": f"${need:,.2f}",
                        }
                    )
                st.dataframe(pd.DataFrame(plan_rows),
                             use_container_width=True, hide_index=True)
                st.metric("Additional cash needed", f"${plan_need_total:,.2f}")
    else:
        st.info("No core portfolio data available")

    st.markdown("---")

    # --- State Controller ---
    if st.session_state.dca_step == "initial":
        _render_dca_initial_view(tracker)
    elif st.session_state.dca_step == "confirm":
        _render_dca_confirmation_view(tracker)
    elif st.session_state.dca_step == "results":
        _render_dca_results_view(tracker)


def _render_dca_initial_view(tracker):
    """Renders the main DCA calculator and trade selection view."""
    st.markdown("### 🧮 DCA Calculator")

    # Create a single row for the inputs
    col1, col2 = st.columns(2)

    with col1:
        dca_method = st.radio(
            "DCA Method:",
            ["Proportional", "Target-Weight"],
            help="Proportional: Distributes funds to maintain current portfolio proportions. Target-Weight: Distributes funds to reach your target allocation.",
        )

    with col2:
        new_funds = st.number_input(
            "New Fund (USDT)",
            min_value=0.0,
            step=0.01,
            help="Amount of USDT to invest",
        )

    is_valid, message = tracker.validate_dca_amount(new_funds)
    if new_funds > 0:
        if is_valid:
            st.success(message)
        else:
            st.error(message)
            return

    if not (new_funds > 0 and is_valid):
        return

    with st.spinner("Calculating DCA suggestions..."):
        suggestions = asyncio.run(tracker.get_dca_suggestions(new_funds))
        if "error" in suggestions:
            st.error(f"❌ {suggestions['error']}")
            return

    st.markdown("### 📋 DCA Trade Suggestions")
    st.info(f"📊 Using **{dca_method}** DCA method")

    trade_amounts = suggestions.get(dca_method.lower().replace("-", "_"), {})

    all_trades = [
        {"asset": asset, "amount": amount, "method": dca_method}
        for asset, amount in trade_amounts.items()
        if abs(amount) > 0.01
    ]

    if not all_trades:
        st.info("No actionable trades found for the selected method.")
        return

    st.markdown("#### ✅ Select Trades to Execute")
    selected_for_execution = []
    for i, trade in enumerate(all_trades):
        trade_type = "BUY" if trade["amount"] > 0 else "SELL"
        label = f"**{trade['asset']}**: {trade_type} ${abs(trade['amount']):,.2f}"
        if st.checkbox(label, key=f"dca_trade_{i}", value=True):
            selected_for_execution.append(trade)

    st.markdown("---")
    if st.button(
        "Execute Selected Trades",
        type="primary",
        disabled=not selected_for_execution,
    ):
        st.session_state.dca_trades_for_confirmation = selected_for_execution
        st.session_state.dca_step = "confirm"
        st.rerun()


def _render_dca_confirmation_view(tracker):
    """Renders the trade confirmation dialog."""
    st.markdown("### 📋 Confirm DCA Execution")
    selected_trades = st.session_state.get("dca_trades_for_confirmation", [])
    if not selected_trades:
        st.warning("No trades found for confirmation. Returning to calculator.")
        st.session_state.dca_step = "initial"
        st.rerun()
        return

    validation = tracker.validate_dca_execution(selected_trades)
    summary = validation["summary"]

    st.markdown("#### 📊 Execution Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total USDT Needed", f"${summary['total_needed']:,.2f}")
    with col2:
        st.metric("Number of Trades", summary["num_trades"])

    if validation["errors"]:
        st.error("❌ Execution Blocked:")
        for error in validation["errors"]:
            st.write(f"• {error}")
        if st.button("Back to Calculator"):
            st.session_state.dca_step = "initial"
            st.rerun()
        return

    if validation["warnings"]:
        st.warning("⚠️ Please Review:")
        for warning in validation["warnings"]:
            st.write(f"• {warning}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ CONFIRM", type="primary", use_container_width=True):
            _execute_dca_trades_now(tracker, selected_trades, validation)
    with col2:
        if st.button("❌CANCEL", use_container_width=True):
            st.session_state.dca_step = "initial"
            del st.session_state.dca_trades_for_confirmation
            st.rerun()


def _execute_dca_trades_now(tracker, trades, validation):
    """Executes the trades and transitions to the results view."""
    try:
        with st.spinner("Executing DCA trades..."):
            is_live = validation.get("summary", {}).get("is_live", False)
            method = trades[0]["method"] if trades else "proportional"
            result = asyncio.run(
                tracker.execute_dca_trades(
                    selected_trades=trades, method=method, is_live=is_live
                )
            )
        st.session_state.dca_execution_result_data = result
        st.session_state.dca_batch_id = getattr(result, "data", {}).get("batch_id")
    except Exception as e:
        st.session_state.dca_execution_result_data = (
            f"An unexpected error occurred during execution: {e}"
        )

    st.session_state.dca_step = "results"
    if "dca_trades_for_confirmation" in st.session_state:
        del st.session_state.dca_trades_for_confirmation
    st.rerun()


def _render_dca_results_view(tracker):
    """Renders the final results of the DCA execution."""
    st.markdown("### 📊 DCA Execution Results")
    result = st.session_state.get("dca_execution_result_data")

    # If no result data exists, reset to initial state
    if not result:
        st.session_state.dca_step = "initial"
        st.rerun()
        return

    if isinstance(result, str):  # Handle error strings
        st.error(result)
    elif result:  # Handle TradeResult object
        if result.success:
            st.success("✅ DCA trades processed successfully!")
        else:
            st.error("❌ Some DCA trades failed.")

        st.code("\n".join(result.messages), language="text")
        batch_id = st.session_state.get("dca_batch_id")
        if batch_id:
            st.info(f"Batch ID: {batch_id}")
        if result.errors:
            st.error("Errors Reported:")
            st.code("\n".join(result.errors), language="text")

    if st.button("🔄 Clear Results"):
        # Clear all DCA-related session state
        st.session_state.dca_step = "initial"
        if "dca_execution_result_data" in st.session_state:
            del st.session_state.dca_execution_result_data
        if "dca_trades_for_confirmation" in st.session_state:
            del st.session_state.dca_trades_for_confirmation
        st.rerun()
