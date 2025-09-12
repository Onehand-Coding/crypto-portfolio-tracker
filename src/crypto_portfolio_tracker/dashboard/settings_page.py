import re
import os
import json
import psutil
import getpass
import platform
from pathlib import Path
from datetime import datetime

import streamlit as st


def render_settings_page(dashboard):
    st.markdown("## ⚙️ Settings")

    tab_config, tab_status = st.tabs(["⚙️ Configuration", "🔧 System Status"])

    # --- Configuration Tab ---
    with tab_config:
        st.header("⚙️ Configuration")
        config = dashboard.config_manager.config

        # Configuration file locations
        st.markdown("---")
        st.subheader(f" 🗂️ Configuration locations")
        st.success(f"📁 Config file: `{dashboard.config_manager.config_file_path}`\n\n")
        st.success(f"📁 Environment file: `{dashboard.config_manager.env_path}`")
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

                    # Add trade schedules here
                    st.markdown("---")
                    st.text("📆 Trade Schedules")

                    freq_options = [
                        "daily",
                        "weekly",
                        "biweekly",
                        "monthly",
                        "quarterly",
                    ]
                    auto_cfg = config.setdefault("automation", {})
                    dca_cfg = auto_cfg.setdefault("dca", {})
                    rb_cfg = auto_cfg.setdefault("rebalancing", {})

                    col_dca, col_rb = st.columns(2)
                    with col_dca:
                        new_dca_freq = st.selectbox(
                            "DCA Frequency",
                            freq_options,
                            index=freq_options.index(
                                dca_cfg.get("frequency", "monthly")
                            )
                            if dca_cfg.get("frequency", "monthly") in freq_options
                            else 3,
                            help="Frequency for Dollar Cost Averaging trades.",
                        )
                    with col_rb:
                        new_rb_freq = st.selectbox(
                            "Rebalancing Frequency",
                            freq_options,
                            index=freq_options.index(rb_cfg.get("frequency", "weekly"))
                            if rb_cfg.get("frequency", "weekly") in freq_options
                            else 1,
                            help="Frequency for portfolio rebalancing checks.",
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

                    profit_taking_config = config.get("profit_taking", {}).get(
                        "enabled", False
                    )
                    new_profit_taking = st.toggle(
                        "💰 Enable Profit Taking",
                        value=profit_taking_config,
                        help="💰 Enable automated profit-taking when portfolio is balanced.",
                    )

                with st.expander("💰 Profit Taking Configuration", expanded=False):
                    profit_config = config.get("profit_taking", {})
                    
                    # Min opportunity score
                    min_opportunity_score = profit_config.get("min_opportunity_score", 60)
                    new_min_opportunity_score = st.number_input(
                        "⭐ Minimum Opportunity Score",
                        min_value=0,
                        max_value=100,
                        value=int(min_opportunity_score),
                        step=1,
                        help="⭐ Minimum opportunity score (0-100) required to trigger profit-taking.",
                    )
                    
                    # Min unrealized gain percentage
                    min_unrealized_gain_pct = profit_config.get("min_unrealized_gain_pct", 15.0)
                    new_min_unrealized_gain_pct = st.number_input(
                        "📈 Minimum Unrealized Gain (%)",
                        min_value=0.0,
                        max_value=500.0,
                        value=float(min_unrealized_gain_pct),
                        step=0.5,
                        help="📈 Minimum unrealized gain percentage required for profit-taking.",
                    )
                    
                    # Min unrealized gain USD
                    min_unrealized_gain_usd = profit_config.get("min_unrealized_gain_usd", 10.0)
                    new_min_unrealized_gain_usd = st.number_input(
                        "💵 Minimum Unrealized Gain (USD)",
                        min_value=0.0,
                        max_value=10000.0,
                        value=float(min_unrealized_gain_usd),
                        step=1.0,
                        help="💵 Minimum unrealized gain in USD required for profit-taking.",
                    )
                    
                    col_max_take, col_default_take = st.columns(2)
                    with col_max_take:
                        # Max gain take percentage
                        max_gain_take_pct = profit_config.get("max_gain_take_pct", 50)
                        new_max_gain_take_pct = st.number_input(
                            "📊 Max Gain Take (%)",
                            min_value=1,
                            max_value=100,
                            value=int(max_gain_take_pct),
                            step=1,
                            help="📊 Maximum percentage of gains that can be taken in one profit-taking action.",
                        )
                    
                    with col_default_take:
                        # Default take percentage
                        default_take_percentage = profit_config.get("default_take_percentage", 30)
                        new_default_take_percentage = st.number_input(
                            "🎯 Default Take (%)",
                            min_value=1,
                            max_value=100,
                            value=int(default_take_percentage),
                            step=1,
                            help="🎯 Default percentage of gains to take when profit-taking is triggered.",
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

                    crypto_quotes = config.get("portfolio", {}).get("crypto_quotes", [])
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
                        config["automation"]["dca"]["frequency"] = new_dca_freq
                        config["automation"]["rebalancing"]["frequency"] = new_rb_freq
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
                        
                        # Initialize profit_taking config if it doesn't exist
                        if "profit_taking" not in config:
                            config["profit_taking"] = {}
                        
                        # Update profit taking settings
                        config["profit_taking"]["enabled"] = new_profit_taking
                        config["profit_taking"]["min_opportunity_score"] = new_min_opportunity_score
                        config["profit_taking"]["min_unrealized_gain_pct"] = new_min_unrealized_gain_pct
                        config["profit_taking"]["min_unrealized_gain_usd"] = new_min_unrealized_gain_usd
                        config["profit_taking"]["max_gain_take_pct"] = new_max_gain_take_pct
                        config["profit_taking"]["default_take_percentage"] = new_default_take_percentage
                        
                        # Save and Reload
                        dashboard.config_manager.save_config()
                        dashboard.reload()

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
                        config.get("apis", {}).get("coingecko", {}).get("timeout", 30)
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
                    st.text("Data Lookback Periods")
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
                        config["apis"]["binance"]["recv_window"] = new_bi_recv_window
                        config["apis"]["binance"]["request_delay_ms"] = new_bi_delay
                        config["apis"]["coingecko"]["request_delay_ms"] = new_cg_delay
                        config["history_lookback_days"] = new_lookback
                        # Save and Reload
                        dashboard.config_manager.save_config()
                        dashboard.reload()

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

                    console_config = config.get("logging", {}).get("console_config", {})
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
                            dashboard.config_manager.save_config()
                            dashboard.reload()
                            dashboard.setup_logging(level_override=new_level)
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
                        config["trend_analyzer"]["rsi_overbought"] = new_rsi_overbought
                        config["trend_analyzer"]["timeframe_settings"] = (
                            timeframe_settings
                        )
                        # Save and Reload
                        dashboard.config_manager.save_config()
                        dashboard.reload()

                        st.success("✅ Trend Analyzer settings updated and applied!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to save trend analyzer settings: {str(e)}")

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
                        del export_config["main_api_keys"]
                        del export_config["apis"]["coingecko"]["api_key"]

                        export_path_obj = Path(export_path)
                        export_path_obj.parent.mkdir(parents=True, exist_ok=True)

                        with open(export_path_obj, "w") as f:
                            json.dump(export_config, f, indent=2)

                        st.success(f"✅ Configuration exported to `{export_path_obj}`")
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
                        default_config = dashboard.config_manager.config
                        required_keys = {
                            key: type(value)
                            for key, value in default_config.items()
                            if key not in ["main_api_keys"]
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

                                    config.update(new_config)
                                    dashboard.config_manager.save_config()
                                    dashboard.reload()

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
                        st.error(f"❌ Failed to process configuration file: {str(e)}")

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
                tracker = dashboard.initialize_tracker()

                # API Status Expander
                with st.expander("🔗 API Status", expanded=True):
                    if tracker.binance_client:
                        try:
                            tracker.binance_client.ping()
                            st.success("🔶 Binance API: Connected")
                        except Exception as e:
                            st.error(f"❌ Binance API: Connection failed")
                    else:
                        st.warning("⚠️ Binance API: Not connected")

                    # Other API info
                    test_id = "BTC"
                    prices = tracker._get_current_prices([test_id])
                    price = prices.get(test_id)
                    if price:
                        st.success(f"🦎 CoinGecko: Available")
                    else:
                        st.warning(f"❌ CoinGecko: Not Available")
                    st.info("🌍 YFinance: API available")

                # Database & Storage Expander
                with st.expander("💾 Database & Storage", expanded=True):
                    db_path = tracker.db_manager.db_path
                    if os.path.exists(db_path):
                        db_size_kb = os.path.getsize(db_path) / 1024
                        db_modified = datetime.fromtimestamp(os.path.getmtime(db_path))
                        st.success(f"Database: `{os.path.basename(db_path)}`")
                        st.text(f"Size: {db_size_kb:.1f} KB")
                        st.text(
                            f"Last Modified: {db_modified.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    else:
                        st.error("❌ Database file not found")

                    st.markdown("---")
                    export_dir = config.get("exports", {}).get("path", "data/exports/")
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
