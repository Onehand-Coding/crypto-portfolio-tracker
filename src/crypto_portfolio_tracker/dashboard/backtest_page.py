import asyncio
import inspect

import pandas as pd
import streamlit as st

from crypto_portfolio_tracker import trading_strategies
from crypto_portfolio_tracker.dashboard import utils as ui_utils
from crypto_portfolio_tracker.strategy_backtester import StrategyBacktester
from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer
from crypto_portfolio_tracker.rebalancing_backtester import RebalancingBacktester


def create_custom_config(
    dashboard,
    custom_allocation,
    majors_drift_threshold,
    alts_drift_threshold,
    majors_sell_multiplier,
    majors_buy_multiplier,
    alts_sell_multiplier,
    alts_buy_multiplier,
    selected_frequency,
    suppress_buys_in_bear,
):
    """Create custom config with user parameters"""
    custom_config = dashboard.config_manager.config.copy()

    # Update rebalancing parameters
    custom_config["rebalance_technical"] = {
        "market_regime_rules": {"suppress_buys_in_bear": suppress_buys_in_bear},
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

    # Add frequency setting
    automation_config = custom_config.setdefault("automation", {})
    rebalancing_config = automation_config.setdefault("rebalancing", {})
    rebalancing_config["frequency"] = selected_frequency

    return custom_config


def render_backtest_page(dashboard):
    """Render backtesting page."""

    # Clear previous page state if coming from another page
    keys_to_clear = [
        "show_save_confirmation", "backtest_results", "last_backtest_params",
        "custom_allocation", "add_asset_counter", "rebalance_initial_capital",
        "rebalance_period_dropdown", "rebalance_period_custom", "rebalance_frequency",
        "majors_drift", "alts_drift", "majors_sell", "majors_buy", "alts_sell",
        "alts_buy", "strategy_period_custom"
    ]

    ui_utils.initialize_page_state("backtest", keys_to_clear)

     # Initialize session state (only if not already set)
    if "show_save_confirmation" not in st.session_state:
        st.session_state.show_save_confirmation = False
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None
    if "last_backtest_params" not in st.session_state:
        st.session_state.last_backtest_params = None
    if "custom_allocation" not in st.session_state:
        st.session_state.custom_allocation = dashboard.config_manager.config.get("target_allocation", {}).copy()

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
            col1, col2, col3 = st.columns(3)

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

            with col3:
                # Add frequency selection
                frequency_options = ["weekly", "monthly", "quarterly"]
                selected_frequency = st.selectbox(
                    "Rebalancing Frequency",
                    frequency_options,
                    index=1,  # Default to monthly
                    key="rebalance_frequency",
                    help="How often to check and rebalance the portfolio",
                )

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
                    tracker = dashboard.initialize_tracker()
                    majors_drift_threshold = st.slider(
                        "Drift Threshold (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=tracker.config.get("rebalance_technical", {})
                        .get("majors", {})
                        .get("allocation_drift_threshold_pct", 3.0),
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
                        value=tracker.config.get("rebalance_technical", {})
                        .get("alts", {})
                        .get("allocation_drift_threshold_pct", 3.50),
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
                    dashboard.config_manager.symbol_mapper.get_all_mappings().keys()
                )
                default_allocation = dashboard.config_manager.config.get(
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
                                    new_value = st.session_state[f"alloc_{asset_name}"]
                                    st.session_state.custom_allocation[asset_name] = (
                                        new_value / 100.0
                                    )

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
                                "Select Asset",
                                ["Select an asset..."] + available_to_add,
                                key=f"add_asset_select_{st.session_state.get("add_asset_counter", 0)}",
                                label_visibility="collapsed",  # Hide the label completely
                            )
                        with col2:
                            # Show button only when a valid asset is selected
                            if add_asset and add_asset != "Select an asset...":
                                if st.button("➕ Add", key="add_asset_btn"):
                                    if add_asset and add_asset != "Select an asset...":
                                        custom_allocation[add_asset] = (
                                            0.0  # Default to 0% allocation
                                        )
                                        # Increment counter to force selectbox reset
                                        st.session_state["add_asset_counter"] = (
                                            st.session_state.get("add_asset_counter", 0)
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
                        st.session_state.custom_allocation = default_allocation.copy()
                        st.rerun()

        # Current Settings Display
        with st.expander("📋 Current Default Settings", expanded=False):
            tracker = dashboard.initialize_tracker()
            current_config = tracker.config

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 Target Allocation:**")
                target_alloc = current_config.get("target_allocation", {})
                for asset, pct in target_alloc.items():
                    st.text(f"- {asset}: {pct:.1%}")

            with col2:
                st.markdown("**⚙️ Rebalancing Settings:**")
                rebalance_config = current_config.get("rebalance_technical", {})
                majors_config = rebalance_config.get("majors", {})
                alts_config = rebalance_config.get("alts", {})

                st.text(
                    f" - Majors Drift Threshold: {majors_config['allocation_drift_threshold_pct']:.1f}%"
                )
                st.text(
                    f"- Alts Drift Threshold: {alts_config['allocation_drift_threshold_pct']:.1f}%"
                )
                st.text(
                    f"- Majors Sell Multiplier: {majors_config['sell_percentage_multiplier']:.2f}"
                )
                st.text(
                    f"- Majors Buy Multiplier: {majors_config['buy_amount_multiplier']:.2f}"
                )
                st.text(
                    f"- Alts Sell Multiplier: {alts_config['sell_percentage_multiplier']:.2f}"
                )
                st.text(
                    f"- Alts Buy Multiplier: {alts_config['buy_amount_multiplier']:.2f}"
                )
                st.text(
                    f"- Frequency: {current_config['automation']['rebalancing']['frequency'].title()}"
                )
                st.text(
                    f"- Bear Market Suppression: {rebalance_config['market_regime_rules']['suppress_buys_in_bear']}"
                )

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
            total_alloc = sum(custom_allocation.values()) if custom_allocation else 0
            if total_alloc > 1.0:
                st.error(
                    f"❌ Cannot run backtest: Total allocation is {total_alloc:.2%} which exceeds 100%. Please adjust your allocations."
                )
            else:
                # Create custom config with user parameters
                custom_config = create_custom_config(
                    dashboard,
                    custom_allocation,
                    majors_drift_threshold,
                    alts_drift_threshold,
                    majors_sell_multiplier,
                    majors_buy_multiplier,
                    alts_sell_multiplier,
                    alts_buy_multiplier,
                    selected_frequency,
                    suppress_buys_in_bear,
                )

                # Store parameters for potential saving
                st.session_state.last_backtest_params = {
                    "custom_allocation": custom_allocation.copy(),
                    "majors_drift_threshold": majors_drift_threshold,
                    "alts_drift_threshold": alts_drift_threshold,
                    "majors_sell_multiplier": majors_sell_multiplier,
                    "majors_buy_multiplier": majors_buy_multiplier,
                    "alts_sell_multiplier": alts_sell_multiplier,
                    "alts_buy_multiplier": alts_buy_multiplier,
                    "selected_frequency": selected_frequency,
                    "suppress_buys_in_bear": suppress_buys_in_bear,
                }

                with st.spinner("🔄 Running rebalancing backtest analysis..."):
                    try:
                        backtester = RebalancingBacktester(config=custom_config)
                        backtester.run(
                            initial_capital=initial_capital,
                            period=period,
                            frequency=selected_frequency,
                        )

                        # Store results in session state
                        st.session_state.backtest_results = backtester

                        st.success(
                            f"✅ Rebalancing backtest completed successfully! ({selected_frequency.title()} frequency)"
                        )

                    except Exception as e:
                        st.error(f"❌ Rebalancing backtest failed: {str(e)}")
                        st.info(
                            "💡 Try adjusting the parameters or check your allocation settings"
                        )

        # Display Results Section (from session state)
        if st.session_state.backtest_results is not None:
            backtester = st.session_state.backtest_results

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
                with st.expander("📋 Detailed Performance Metrics", expanded=True):
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
                    df = pd.DataFrame(backtester.portfolio_value_history)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                    df = df.rename(columns={"value": "Portfolio Value ($)"})

                    # Create the chart with proper date axis
                    st.line_chart(df, use_container_width=True)

            # Trade Log
            if hasattr(backtester, "trade_log"):
                with st.expander("📝 Rebalancing Trade Log"):
                    st.code("\n".join(backtester.trade_log), language="text")

            # Save as Default button
            if hasattr(backtester, "summary_stats"):
                st.markdown("---")
                st.markdown(
                    "### 💾 Save Configuration",
                    help="💡 **Satisfied with these results?** Save these proven parameters as your default settings for live trading.",
                )

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(
                        "💾 Save as Default",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.show_save_confirmation = True
                        st.rerun()

        # Show confirmation dialog (outside of results section)
        if (
            st.session_state.show_save_confirmation
            and st.session_state.last_backtest_params
        ):
            params = st.session_state.last_backtest_params

            st.warning("⚠️ **Confirm Settings Overwrite**")

            st.markdown(
                "You are about to overwrite your current default settings with:"
            )
            col_target, col_param = st.columns(2)

            with col_target:
                st.markdown("**🎯 Target Allocation:**")

                # Show current allocation
                for asset, pct in params["custom_allocation"].items():
                    st.text(f"- {asset}: {pct:.1%}")

            with col_param:
                st.markdown("**⚙️ Rebalancing Settings:**")
                st.text(
                    f" - Majors Drift Threshold: {params['majors_drift_threshold']:.1f}%"
                )
                st.text(
                    f"- Alts Drift Threshold: {params['alts_drift_threshold']:.1f}%"
                )
                st.text(
                    f"- Majors Sell Multiplier: {params['majors_sell_multiplier']:.2f}"
                )
                st.text(
                    f"- Majors Buy Multiplier: {params['majors_buy_multiplier']:.2f}"
                )
                st.text(f"- Alts Sell Multiplier: {params['alts_sell_multiplier']:.2f}")
                st.text(f"- Alts Buy Multiplier: {params['alts_buy_multiplier']:.2f}")
                st.text(f"- Frequency: {params['selected_frequency'].title()}")
                st.text(f"- Bear Market Suppression: {params['suppress_buys_in_bear']}")

            st.markdown("---")

            # Confirmation buttons
            confirm_col1, confirm_col2, confirm_col3 = st.columns([1, 1, 1])
            with confirm_col1:
                if st.button(
                    "✅ Yes, Save Settings", type="primary", key="confirm_save"
                ):
                    try:
                        # Create and save the configuration
                        config_to_save = create_custom_config(
                            dashboard,
                            params["custom_allocation"],
                            params["majors_drift_threshold"],
                            params["alts_drift_threshold"],
                            params["majors_sell_multiplier"],
                            params["majors_buy_multiplier"],
                            params["alts_sell_multiplier"],
                            params["alts_buy_multiplier"],
                            params["selected_frequency"],
                            params["suppress_buys_in_bear"],
                        )

                        dashboard.config_manager.config = config_to_save
                        dashboard.config_manager.save_config()
                        dashboard.reload()

                        st.success("✅ Settings saved as default!")
                        st.session_state.show_save_confirmation = False
                        st.session_state.backtest_results = None
                        st.session_state.last_backtest_params = None
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Failed to save settings: {str(e)}")

            with confirm_col3:
                if st.button("❌ Cancel", type="secondary", key="cancel_save"):
                    st.session_state.show_save_confirmation = False
                    st.session_state.backtest_results = None
                    st.session_state.last_backtest_params = None
                    st.rerun()

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
                for name, obj in inspect.getmembers(trading_strategies, inspect.isclass)
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
                symbol_mapper = dashboard.config_manager.symbol_mapper
                config = dashboard.config_manager.config
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
                "1h" in interval and "y" in period and int(period.replace("y", "")) > 2
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
                    analyzer = CryptoTrendAnalyzer(config=config, binance_client=None)
                    backtester = StrategyBacktester(config=config, analyzer=analyzer)
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
                            st.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")

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
                            st.code("\n".join(backtester.trade_log), language="text")

                except Exception as e:
                    st.error(f"❌ Backtest failed: {str(e)}")
                    st.info(
                        "💡 Try adjusting the parameters or selecting a different time period"
                    )
