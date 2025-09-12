import asyncio
import inspect
from datetime import datetime

import streamlit as st

from crypto_portfolio_tracker.dashboard import utils as ui_utils
from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer
from crypto_portfolio_tracker.dashboard.components import render_transfer_widget, render_trading_status_banner


def render_trading_page(dashboard):
    """Render trading page"""

    # Clear previous page state if coming from another page
    keys_to_clear = [
        'trading_mode', 'trading_results', 'trading_executing',
        'manual_trade_data', 'strategy_signals', 'strategy_selected_coins',
        'strategy_param_', 'strategy_trade_amount_mode', 'strategy_trade_pct',
        'strategy_trade_amount', 'strategy_account', 'strategy_name',
        'strategy_reset_flag'
    ]

    ui_utils.initialize_page_state("trading", keys_to_clear)

    st.markdown("## 💰 Trading")

    # Offline guard: disable trading workflows when offline
    if dashboard.offline_mode:
        st.info("⚠️ Offline mode: Trading is unavailable.")
        return

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

    # Check trading status
    is_live = dashboard.config_manager.is_live
    is_testnet = dashboard.config_manager.is_testnet_mode

    # Trading Status Banner
    render_trading_status_banner(is_live, is_testnet)

    _render_manual_trading(dashboard)


def _render_manual_trading(dashboard):
    """Render the manual trading page."""
    st.markdown("### 📝 Manual Trading")
    # Fetch and display available USDT balance
    usdt_balance = None
    tracker = dashboard.initialize_tracker()
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

    # Contextual transfer (only when funding has balance)
    render_transfer_widget(dashboard, context="trading")

    # --- SHOW EXECUTION RESULTS IF AVAILABLE ---
    if st.session_state.trading_results:
        st.markdown("### 📋 Execution Results")
        if "EXECUTION COMPLETED" in st.session_state.trading_results:
            st.success("✅ **Trade execution successfull!**")
            st.code(st.session_state.trading_results, language="text")
        elif "Error" in st.session_state.trading_results:
            st.error("❌ **Trade execution failed**")
            st.code(st.session_state.trading_results, language="text")
        if st.button("🔄 Clear Results", type="secondary", use_container_width=True):
            st.session_state.trading_results = None
            st.session_state.manual_trade_data = {}
            st.rerun()
        return  # <-- Prevents showing confirmation or form

    # --- SHOW TRADE CONFIRMATION IF DATA IS STORED ---
    if st.session_state.manual_trade_data:
        _show_trade_confirmation(dashboard)
        return  # <-- Prevents showing the form

    # --- OTHERWISE, SHOW THE TRADE FORM ---
    core_coins = list(
        dashboard.config_manager.config.get("target_allocation", {}).keys()
    )
    core_coins_upper = [c.upper() for c in core_coins]
    symbol_mapper = dashboard.config_manager.symbol_mapper
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


def _show_trade_confirmation(dashboard):
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
        if st.button("✅ CONFIRM", type="primary", use_container_width=True):
            _confirm_trade(
                dashboard,
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


def _confirm_trade(dashboard, trade_type, symbol, amount_input, is_quote_qty):
    """Makes the actual trade."""
    st.session_state.trading_executing = True
    try:
        with st.spinner(f"Executing {trade_type} order for {symbol}..."):
            try:
                tracker = dashboard.initialize_tracker()
                is_live = dashboard.config_manager.config.get("portfolio", {}).get(
                    "live_trading_enabled", False
                )
                # Parse amount
                import re

                numeric_part = re.search(r"[\d\.]+", amount_input)
                if not numeric_part:
                    st.error("❌ Invalid amount format. Please enter a valid number.")
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
                        f"❌ Trade Execution failed:\n" + "\n".join(result.errors)
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
                st.session_state.trading_results = f"❌ **Execution Error**: {str(e)}"
                # CRITICAL: Clear manual_trade_data even on error
                st.session_state.manual_trade_data = {}
    finally:
        st.session_state.trading_executing = False
        st.rerun()
