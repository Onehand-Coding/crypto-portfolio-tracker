import asyncio
import inspect
from datetime import datetime

import streamlit as st

from crypto_portfolio_tracker import trading_strategies
from crypto_portfolio_tracker.dashboard.components.transfer_widget import (
    render_transfer_widget,
)
from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer


def render_trading_page(dashboard):
    """Render trading page"""
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

    # Check live trading status
    is_live = dashboard.config_manager.config.get("portfolio", {}).get(
        "live_trading_enabled", False
    )
    is_testnet = dashboard.config_manager.is_testnet_mode

    # Trading Status Banner
    col1, col2, col3 = st.columns(3)
    with col1:
        if is_live:
            st.error(" LIVE TRADING ENABLED")
        else:
            st.warning("🟡 LIVE TRADING DISABLED")
    with col2:
        if is_testnet:
            st.info(" TESTNET CONNECTION")
        else:
            st.info(" MAINNET CONNECTION")
    with col3:
        if is_live:
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
        _render_manual_trading(dashboard)
    else:
        if st.session_state.trading_mode != "strategy":
            st.session_state.trading_mode = "strategy"
            st.session_state.manual_trade_data = {}
            if (
                "trading_results" in st.session_state
                and st.session_state.trading_results
            ):
                st.session_state.trading_results = None
        _render_live_strategy_trading(dashboard)

    # --- Always show debug output at the end of the trading page ---
    # (Removed debug output at the end of the trading page)
    pass


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
    try:
        balances = tracker.get_available_usdt_balance()
        if float(balances.get("funding", 0.0)) > 0:
            render_transfer_widget(dashboard, context="trading")
    except Exception:
        pass

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
            _confirm_manual_trade(
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


def _confirm_manual_trade(dashboard, trade_type, symbol, amount_input, is_quote_qty):
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


def _sync_portfolio_and_reset(dashboard):
    with st.spinner("🔄 Syncing portfolio and updating data..."):
        try:
            tracker = dashboard.initialize_tracker()
            metrics = asyncio.run(tracker.run_full_sync())
            st.session_state.portfolio_metrics = metrics
            st.session_state.last_sync = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

                st.success(
                    "✅ Portfolio synced successfully! Strategy trading page reset."
                )

            else:  # manual or any other mode
                # Reset only manual trading state
                st.session_state.trading_results = None
                st.session_state.manual_trade_data = {}

                # Clean up any manual trading related keys if they exist
                manual_keys = [
                    "manual_trade_symbol",
                    "manual_trade_amount",
                    "manual_trade_type",
                ]
                for key in manual_keys:
                    if key in st.session_state:
                        del st.session_state[key]

                st.success(
                    "✅ Portfolio synced successfully! Manual trading page reset."
                )

            st.rerun()

        except Exception as e:
            st.error(f"❌ Portfolio sync failed: {str(e)}")

    # Also update the reset flag handling in the main method


def _render_live_strategy_trading(dashboard):
    if st.session_state.get("strategy_reset_flag"):
        st.session_state.strategy_signals = None
        st.session_state.trading_results = None
        st.session_state.strategy_reset_flag = False

    st.markdown("### 🤖 Live Strategy Trading")

    # --- 1. Account Selection ---
    tracker = dashboard.initialize_tracker()
    config = dashboard.config_manager.config
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
    usdt_balance = _get_usdt_balance(dashboard, tracker, selected_account)

    st.markdown("#### 💰 Available USDT")
    if usdt_balance is not None:
        st.success(f"${usdt_balance:,.2f}")
    else:
        st.info("USDT balance unavailable.")

    # Contextual transfer (only when funding has balance)
    try:
        balances = tracker.get_available_usdt_balance()
        if float(balances.get("funding", 0.0)) > 0:
            render_transfer_widget(dashboard, context="strategy trading")
    except Exception:
        pass

    # --- 2. Strategy Selection ---
    available_strategies = _get_available_strategies(dashboard, account_type)

    if not available_strategies:
        st.error(f"❌ No suitable strategies found for account type '{account_type}'.")
        return

    strategy_names = list(available_strategies.keys())
    selected_strategy_name = st.selectbox(
        "Select Strategy", strategy_names, key="strategy_name"
    )
    strategy_class = available_strategies[selected_strategy_name]

    # --- 3. Coin Selection ---
    coin_options = _get_coin_options(dashboard, config)

    if "strategy_selected_coins" not in st.session_state:
        core_coins = list(config.get("target_allocation", {}).keys())
        core_coins_upper = [f"{c.upper()}-USD" for c in core_coins]
        st.session_state.strategy_selected_coins = core_coins_upper

    st.markdown("#### 🪙 Select Coins for Strategy Trading")
    selected_coins = st.multiselect(
        "Coins to include",
        options=coin_options,
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
            param_inputs[param] = _render_parameter_input(dashboard, param, spec)

        # Trade amount configuration
        trade_config = _render_trade_amount_config(dashboard)

        submitted = st.form_submit_button("Generate Signals")

    # --- 5. Signal Generation ---
    if submitted:
        signals = _generate_signals(
            dashboard,
            tracker,
            selected_account,
            config,
            strategy_class,
            selected_coins,
            param_inputs,
            param_specs,
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
        _execute_strategy_signals(
            dashboard,
            tracker,
            selected_account,
            config,
            signals,
            trade_config,
            usdt_balance,
        )

    # --- 8. Show Execution Results ---
    _display_execution_results(dashboard)


def _get_usdt_balance(dashboard, tracker, selected_account):
    """Get USDT balance for the selected account."""
    try:
        if selected_account["name"] == "Main Account":
            balance = tracker.binance_client.get_asset_balance(asset="USDT")
            return float(balance.get("free", 0.0))
        else:
            live_client = tracker._init_binance_client(
                api_key=selected_account.get("api_key")
                or selected_account.get("binance_key"),
                api_secret=selected_account.get("api_secret")
                or selected_account.get("binance_secret"),
            )
            balance = live_client.get_asset_balance(asset="USDT")
            return float(balance.get("free", 0.0))
    except Exception as e:
        st.info(f"USDT balance unavailable. Error: {e}")
        return None


def _get_available_strategies(dashboard, account_type):
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
            k: v
            for k, v in all_strategies.items()
            if getattr(v, "strategy_type", None) in ["swing", "general"]
        }
    elif account_type == "day":
        return {
            k: v
            for k, v in all_strategies.items()
            if getattr(v, "strategy_type", None) in ["day", "general"]
        }
    else:
        return all_strategies


def _get_coin_options(dashboard, config):
    """Get available coin options."""
    symbol_mapper = dashboard.config_manager.symbol_mapper
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


def _render_parameter_input(dashboard, param, spec):
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


def _render_trade_amount_config(dashboard):
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


def _generate_coin_signal(
    dashboard, analyzer, strategy_class, coin, param_inputs, param_specs
):
    """Generate signal for a single coin."""
    # Extract just the coin name from "BTC-USD" format
    coin_name = coin.split("-")[0] if "-" in coin else coin
    yf_ticker = f"{coin_name}-USD"
    analyzer.set_symbol(yf_ticker)

    # Convert percent params to fraction if needed
    strategy_kwargs = param_inputs.copy()
    for k, v in param_specs.items():
        if (
            v.get("type") == "float"
            and "pct" in k
            and strategy_kwargs[k] is not None
            and strategy_kwargs[k] > 1
        ):
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


def _generate_signals(
    dashboard,
    tracker,
    selected_account,
    config,
    strategy_class,
    selected_coins,
    param_inputs,
    param_specs,
):
    """Generate trading signals for multiple coins."""
    signals = []

    # Create analyzer instance with required config and binance_client
    analyzer = CryptoTrendAnalyzer(config=config, binance_client=tracker.binance_client)

    with st.spinner("Generating signals..."):
        for coin in selected_coins:
            try:
                signal = _generate_coin_signal(
                    dashboard, analyzer, strategy_class, coin, param_inputs, param_specs
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                st.warning(f"Error generating signal for {coin}: {e}")
                continue

    return signals


def _execute_strategy_signals(
    dashboard, tracker, selected_account, config, signals, trade_config, usdt_balance
):
    """Execute all trading signals."""
    with st.spinner("Executing trades..."):
        try:
            results = []
            min_trade_usd = config.get("portfolio", {}).get("minimum_trade_usd", 10.0)
            is_live = config.get("portfolio", {}).get("live_trading_enabled", False)

            for trade in signals:
                try:
                    result = _execute_single_trade(
                        dashboard,
                        tracker,
                        trade,
                        trade_config,
                        usdt_balance,
                        min_trade_usd,
                        is_live,
                    )
                    results.append(result)
                except Exception as e:
                    results.append(
                        f"{trade['Signal']} {trade['Symbol']}: ❌ Execution Error: {e}"
                    )

            st.session_state.trading_results = "\n".join(results)
            st.success("✅ All signals executed. See results below.")

        except Exception as e:
            st.session_state.trading_results = f"❌ Execution Error: {e}"
            st.error(f"Execution failed: {e}")


def _execute_single_trade(
    dashboard, tracker, trade, trade_config, usdt_balance, min_trade_usd, is_live
):
    """Execute a single trade."""
    trade_type = trade["Signal"]
    symbol = trade["Symbol"]

    if trade_config["mode"] == "percentage":
        usdt_to_spend = max(
            (trade_config["value"] / 100.0) * usdt_balance, min_trade_usd
        )
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
            return (
                f"{trade_type} {symbol}: \n"
                f"Preparing MARKET {trade_type} for {symbol}...\n"
                f"Amount: {usdt_to_spend:.2f} USDT\n"
                f"�� PLACING LIVE ORDER...\n"
                f"{execution_output}"
            )
        else:
            # Return error message
            error_msg = (
                "\n".join(result.errors) if result.errors else "Unknown error occurred"
            )
            return f"{trade_type} {symbol}: ❌ FAILED - {error_msg}"

    finally:
        loop.close()


def _display_execution_results(dashboard):
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
    st.info(
        "💡 Recommendation: After executing trades, sync your portfolio to see updated balances and positions."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Sync Portfolio", type="primary", use_container_width=True):
            _sync_portfolio_and_reset(dashboard)
    with col2:
        if st.button("🆕 New Strategy Run", type="secondary", use_container_width=True):
            st.session_state.trading_results = None
            st.session_state.strategy_signals = None
            st.rerun()
