#!/usr/bin/env python3
"""
Crypto Portfolio Tracker - Main Entry Point
"""

import re
import json
import copy
import asyncio
import logging
import colorlog
import argparse
import inspect
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

from . import trading_strategies
from .portfolio_tracker import CryptoPortfolioTracker, NetworkUnavailableError
from .config import ConfigManager
from .exceptions import NetworkOperationError
from .rebalancing_backtester import RebalancingBacktester
from .strategy_backtester import StrategyBacktester
from .crypto_trend_analyzer import CryptoTrendAnalyzer


logger = logging.getLogger(__name__)


def setup_logging(level_override: Optional[str] = None):
    """
    Sets up application logging with color-coded console output.
    """
    config_manager = ConfigManager()
    logging_config = config_manager.config.get("logging", {})

    # Priority: 1. CLI override, 2. Config file, 3. Default "INFO"
    log_level_str = level_override or logging_config.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(log_level)

    # Configure console handler with colors
    if logging_config.get("console_config", {}).get("enabled", True):
        log_format = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = colorlog.ColoredFormatter(
            log_format,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Configure file handler (without colors)
    file_config = logging_config.get("file_config", {})
    if file_config.get("enabled", True):
        file_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_path = Path(file_config.get("path", "logs/portfolio_tracker.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(file_log_format))
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured to level: {log_level_str}")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def print_main_menu(offline_mode=False):
    """Prints the main menu options."""
    print("\n" + "=" * 50)
    print("🚀 Crypto Portfolio Tracker v2.1.0")
    if offline_mode:
        print("⚠️  OFFLINE MODE: Network features are disabled.")
    print("=" * 50)
    print("1. 🔄 Full Sync & Analysis")
    print("2. 💰 Quick Portfolio Summary")
    print("3. 📈 View Trends")
    print("4. ⚖️ Rebalance")
    print("5. 🔀 Trade")
    print("6. 🧪 Backtest")
    print("7. 📋 Reports")
    print("8. 📊 Charts")
    print("9. 🗄️  Database")
    print("10. 🧹 Data Cleanup")
    print("11. ⚙️  View Configuration")
    print("12. 🔧 Test Connections")
    print("13. ❌ Exit")
    print("=" * 50)


def _print_wallet_summary(
    title: str,
    balances: List[Dict[str, Any]],
    balance_key: str,
    asset_key: str = "asset",
):
    """
    Helper function to print a formatted summary for a given wallet,
    handling empty lists and lists with only zero balances gracefully.
    """
    LINE_WIDTH = 115
    print("\n" + f"--- {title} ---".center(LINE_WIDTH))

    # First, create a new list containing only assets with a non-zero balance.
    non_zero_balances = []
    if balances:  # Ensure balances is not None
        for item in balances:
            balance = float(item.get(balance_key, 0.0))
            if balance > 1e-8:
                non_zero_balances.append(item)

    # Now, check if our new list is empty.
    if not non_zero_balances:
        print("No balances found.".center(LINE_WIDTH))
        return

    # If we have non-zero balances, print the header and the rows.
    header = f"{'Asset':<15} {'Balance':<20}"
    print(header)
    print("-" * len(header))

    for item in non_zero_balances:
        balance = float(item.get(balance_key, 0.0))
        asset = item.get(asset_key, "N/A")
        print(f"{asset:<15} {balance:<20,.8g}")


def print_portfolio_summary(tracker: CryptoPortfolioTracker, metrics: Dict[str, Any]):
    """Prints a consolidated summary of the portfolio, including a breakdown of all wallet values."""
    LINE_WIDTH = 115
    print("\n" + "=" * LINE_WIDTH)
    print("📊 CONSOLIDATED PORTFOLIO SUMMARY")
    print("=" * LINE_WIDTH)

    if "error" in metrics:
        print(f"❌ Could not generate summary: {metrics['error']}")
        print("=" * LINE_WIDTH)
        return

    timestamp = metrics.get("timestamp", pd.Timestamp.now())
    print(f"Timestamp:                   {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    db_path = tracker.config.get("database", {}).get("path", "N/A")
    db_name = Path(db_path).name
    if tracker.config_manager.is_testnet_mode:
        print(f"Database:                    {db_name} (TESTNET MODE)")
    else:
        print(f"Database:                    {db_name}")

    print("-" * LINE_WIDTH)
    print(f"TOTAL PORTFOLIO VALUE:       ${metrics.get('total_value_usd', 0):,.2f}")
    print("-" * LINE_WIDTH)

    # --- Breakdown of Total Portfolio Value ---
    print("Wallet Value Breakdown:")
    print(f"  Spot & Earn Value:         ${metrics.get('spot_earn_value_usd', 0):,.2f}")
    print(f"  Futures Wallet Value:      ${metrics.get('futures_value_usd', 0):,.2f}")
    print(f"  Funding Wallet Value:      ${metrics.get('funding_value_usd', 0):,.2f}")
    print("-" * LINE_WIDTH)

    total_invested = metrics.get("total_invested_capital", 0)
    overall_pl_usd = metrics.get("overall_pl_usd", 0)
    overall_pl_pct = metrics.get("overall_pl_percent", 0)
    color_overall = "\033[92m" if overall_pl_usd >= 0 else "\033[91m"
    color_end = "\033[0m"

    print("Performance vs. Invested capital:")
    print(f"Total Invested Capital:      ${total_invested:,.2f}")
    print(
        f"Overall P/L:                 {color_overall}${overall_pl_usd:,.2f} ({overall_pl_pct:.2f}%){color_end}"
    )

    print("-" * LINE_WIDTH)

    print("Performance vs. Rolling cost basis (Spot/Earn only):")
    print(
        f"Total Cost Basis (FIFO):     ${metrics.get('total_cost_basis_usd', 0):,.2f}"
    )

    pl_usd = metrics.get("unrealized_pl_usd", 0)
    pl_pct = metrics.get("unrealized_pl_percent", 0)
    color_unrealized = "\033[92m" if pl_usd >= 0 else "\033[91m"

    print(
        f"Unrealized P/L (FIFO):     {color_unrealized}${pl_usd:,.2f} ({pl_pct:.2f}%){color_end}"
    )

    # Print Core Holdings Table
    core_holdings_df = metrics.get("core_holdings_df")
    if core_holdings_df is not None and not core_holdings_df.empty:
        print(
            "\n"
            + "--- 🎯 Core Portfolio Holdings (Used for Rebalancing) ---".center(
                LINE_WIDTH
            )
        )
        header = f"{'Asset':<11} {'Total Qty':<15} {'Spot Qty':<15} {'Earn Qty':<15} {'Value (USD)':<15} {'Cost Basis':<15} {'P/L (USD)':<15} {'Core Alloc.':<15} {'Total Alloc.':<10}"
        print(header)
        print("-" * len(header))
        for _, row in core_holdings_df.sort_values(
            by="value_usd", ascending=False
        ).iterrows():
            row_pl_usd = row.get("unrealized_pl_usd", 0)
            row_color_start = "\033[92m" if row_pl_usd >= 0 else "\033[91m"
            core_alloc_str = f"{row.get('core_allocation', 0) * 100:.2f}%"
            total_alloc_str = f"{row.get('allocation', 0) * 100:.2f}%"
            print(
                f"{row.get('symbol', 'N/A'):<11} "
                f"{row.get('total_quantity', 0):<15,.8g} "
                f"{row.get('spot_quantity', 0):<15,.8g} "
                f"{row.get('earn_quantity', 0):<15,.8g} "
                f"${row.get('value_usd', 0):<14,.2f} "
                f"${row.get('cost_basis_total', 0):<14,.2f} "
                f"{row_color_start}${row_pl_usd:<14,.2f}{color_end} "
                f"{core_alloc_str:<15} "
                f"{total_alloc_str:<10}"
            )

    # Print Other Holdings Table
    other_holdings_df = metrics.get("other_holdings_df")
    if other_holdings_df is not None and not other_holdings_df.empty:
        print("\n" + "--- 📈 Other Holdings ---".center(LINE_WIDTH))
        header = f"{'Asset':<11} {'Total Qty':<15} {'Spot Qty':<15} {'Earn Qty':<15} {'Value (USD)':<15} {'Cost Basis':<15} {'P/L (USD)':<15} {'Total Alloc.':<10}"
        print(header)
        print("-" * len(header))
        for _, row in other_holdings_df.sort_values(
            by="value_usd", ascending=False
        ).iterrows():
            row_pl_usd = row.get("unrealized_pl_usd", 0)
            row_color_start = "\033[92m" if row_pl_usd >= 0 else "\033[91m"
            total_alloc_str = f"{row.get('allocation', 0) * 100:.2f}%"
            print(
                f"{row.get('symbol', 'N/A'):<11} "
                f"{row.get('total_quantity', 0):<15,.8g} "
                f"{row.get('spot_quantity', 0):<15,.8g} "
                f"{row.get('earn_quantity', 0):<15,.8g} "
                f"${row.get('value_usd', 0):<14,.2f} "
                f"${row.get('cost_basis_total', 0):<14,.2f} "
                f"{row_color_start}${row_pl_usd:<14,.2f}{color_end} "
                f"{total_alloc_str:<10}"
            )
        print("=" * len(header))

    # --- Print Individual Wallet Summaries ---
    futures_balances = metrics.get("futures_balances")
    _print_wallet_summary("Futures Wallet Summary", futures_balances, "balance")

    funding_balances = metrics.get("funding_balances")
    _print_wallet_summary("Funding Wallet Summary", funding_balances, "free")


def print_trend_report(report: Dict[str, Any]):
    """Prints a formatted trend analysis report to the console."""
    if not report:
        print("\n--- No Trend Report Data ---")
        return

    print("\n" + "=" * 80)
    print(f"📈 TREND ANALYSIS REPORT ({report.get('timeframe', 'N/A').upper()})")
    print(f"Timestamp: {report.get('timestamp')}")
    print("=" * 80)

    summary = report.get("market_summary", {})
    print("\n--- 🌍 Market Summary ---")
    print(f"Coins Analyzed: {summary.get('total_coins', 'N/A')}")
    print(f"Most Common Condition: {summary.get('most_common_condition', 'N/A')}")
    print(
        f"Bullish Coins: {summary.get('bull_count', 0)} | Bearish Coins: {summary.get('bear_count', 0)}"
    )

    btc = report.get("benchmark_analysis", {})
    if btc:
        print(f"\n--- 🎯 Benchmark Analysis: {btc.get('symbol', 'N/A')} ---")
        print(
            f"  Price: ${btc.get('current_price', 0):,.2f} ({btc.get('price_change_pct', 0):+.2f}%) | RSI: {btc.get('rsi', 0):.2f}"
        )
        print(
            f"  Support: ${btc.get('support_level', 0):,.2f} | Resistance: ${btc.get('resistance_level', 0):,.2f}"
        )
        print(
            f"  Active Conditions: {', '.join(btc.get('active_conditions', ['None']))}"
        )

    print("\n--- 🪙 Coin-by-Coin Analysis ---")
    for symbol, analysis in report.get("coin_analyses", {}).items():
        if symbol == btc.get("symbol"):
            continue  # Skip printing benchmark again

        print(f"\n➡️ {symbol}")
        print(
            f"  Price: ${analysis.get('current_price', 0):,.2f} ({analysis.get('price_change_pct', 0):+.2f}%) | RSI: {analysis.get('rsi', 0):.2f}"
        )
        print(
            f"  Support: ${analysis.get('support_level', 0):,.2f} | Resistance: ${analysis.get('resistance_level', 0):,.2f}"
        )
        print(
            f"  Active Conditions: {', '.join(analysis.get('active_conditions', ['None']))}"
        )

    print("\n" + "=" * 80)


def print_rebalance_suggestions(
    tracker: CryptoPortfolioTracker,
    suggestions_df: Optional[pd.DataFrame],
    available_usdt: Optional[float] = None,
):
    """Prints the rebalancing suggestions in a formatted way."""
    if suggestions_df is None or suggestions_df.empty:
        print("No rebalancing suggestions available.")
        return

    # Sort the DataFrame before printing for a logical display
    signal_order = {"SELL": 0, "BUY": 1, "HOLD": 2}
    suggestions_df["signal_order"] = suggestions_df["Signal"].map(signal_order)
    suggestions_df = suggestions_df.sort_values(
        by=["signal_order", "Drift (pts)"], ascending=[True, True]
    )
    suggestions_df = suggestions_df.drop(columns=["signal_order"])

    print("\n" + "=" * 88)
    print("⚖️  REBALANCING SUGGESTIONS (Multi-Timeframe Analysis)")
    print("=" * 88)

    # Calculate and display the total value of the core portfolio being rebalanced
    total_core_value = suggestions_df["Current Value (USD)"].sum()
    print("-" * 88)
    print(f"💰 Core Portfolio Value: ${total_core_value:,.2f}")

    # Display the USDT balance inside the header if it was provided
    if available_usdt is not None:
        print(f"💰 Available USDT (Spot + Earn): ${available_usdt:,.2f}")
        print("-" * 88)

    for _, row in suggestions_df.iterrows():
        signal = row["Signal"]
        if signal == "SELL":
            color_code = "\033[91m"
        elif signal == "BUY":
            color_code = "\033[92m"
        else:
            color_code = "\033[93m"
        reset_code = "\033[0m"

        signal_icon = {"SELL": "🔴", "BUY": "🟢", "HOLD": "🟡"}.get(signal, "⚪️")

        print(
            f"{color_code}{signal_icon} {row['Symbol']:<7}| Signal: {signal:<4}{reset_code}"
        )
        print(
            f"   Allocation: {row['Current %']:.2f}% (Target: {row['Target %']:.1f}%) | Drift: {row['Drift (pts)']:.2f} pts | Value: ${row['Current Value (USD)']:,.2f}"
        )

        price_str = f"${row['TA_Price']:,.2f}".ljust(12)
        support_str = f"${row.get('Support', 0.0):,.2f}"
        resistance_str = f"${row.get('Resistance', 0.0):,.2f}"

        print(
            f"   Price: {price_str} | Support: {support_str} | Resistance: {resistance_str}"
        )

        print(f"   Long-Term Trend: {row['TA_Conditions']}")

        print(f"   Action: {row['Suggested Action Detail']}")

        if row.name != suggestions_df.index[-1]:
            print("-" * 54)
    print("=" * 88)


async def view_trends(tracker: CryptoPortfolioTracker):
    """
    Provides an interactive menu to view cryptocurrency trend analysis reports.
    """
    try:
        analyzer = CryptoTrendAnalyzer(
            config=tracker.config, binance_client=tracker.binance_client
        )
        print("✅ Trend Analyzer initialized with centralized config.")
    except Exception as e:
        tracker.logger.error(
            f"Failed to initialize CryptoTrendAnalyzer: {e}", exc_info=True
        )
        print(
            "❌ Error: Could not initialize the Trend Analyzer. Please check the logs."
        )
        return

    while True:
        print("\n--- 📈 Crypto Trend Analysis ---\n")
        print("Select the timeframe for the analysis:")
        print("1. Long-term (4 Years)")
        print("2. Swing (3 Months)")
        print("3. Day (1 Month)")

        try:
            choice_str = input("Select option (1-3): ").strip()
            if not choice_str:
                return
            if not choice_str.isdigit() or int(choice_str) not in range(1, 4):
                print("❌ Invalid input. Please enter a number between 1 and 3.")
                continue
            choice = int(choice_str)
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
            continue

        timeframe = None
        if choice == 1:
            timeframe = "long_term"
        elif choice == 2:
            timeframe = "swing"
        elif choice == 3:
            timeframe = "day"
        elif choice == 4:
            break

        if timeframe:
            print(f"\n🔄 Generating {timeframe.replace('_', ' ')} trend report...")
            try:
                report = await analyzer.generate_report(timeframe)

                if report:
                    print_trend_report(report)
                    export_choice = input(
                        "\nDo you want to export this report? (y/n): "
                    ).lower()
                    if export_choice == "y":
                        success = export_trend_report_interactive(tracker, report, timeframe)
                        if not success:
                            print("💡 Try using the web interface for more export options")
                else:
                    print(
                        f"❌ Could not generate the {timeframe} trend report. See logs for details."
                    )

            except Exception as e:
                tracker.logger.error(
                    f"An error occurred during report generation for timeframe '{timeframe}': {e}",
                    exc_info=True,
                )
                print(f"❌ An unexpected error occurred. Please check the logs.")

            input("\n✅ Press Enter to continue...")


async def run_rebalancing_backtest(tracker: CryptoPortfolioTracker):
    """
    Handles the user flow for running the rebalancing backtest.
    """
    print("\n--- ⚖️ Rebalancing Strategy Backtesting Mode ---")
    try:
        backtester = RebalancingBacktester(config=tracker.config)
        loop = asyncio.get_event_loop()

        initial_capital_str = await loop.run_in_executor(
            None, input, "Enter initial capital for simulation (default: 10000): "
        )
        initial_capital = float(initial_capital_str) if initial_capital_str else 10000.0

        period = ""
        while True:
            period_str = await loop.run_in_executor(
                None, input, "Enter backtest period (e.g., 2y, 3y, 5y - default: 3y): "
            )
            period_str = period_str.strip().lower()
            if not period_str:
                period = "3y"  # Default value
                break
            if re.match(r"^\d+[dmy]$", period_str):
                period = period_str
                break
            else:
                print(
                    "❌ Invalid format. Please use a number followed by 'd', 'm', or 'y' (e.g., '90d', '6m', '3y')."
                )

        backtester.run(initial_capital=initial_capital, period=period)

    except Exception as e:
        logger.error(
            f"An error occurred during rebalancing backtest: {e}", exc_info=True
        )
        print(f"❌ An unexpected error occurred: {e}")


async def run_manual_trade_session(tracker: CryptoPortfolioTracker):
    """Runs an interactive session for placing a manual trade."""
    print("\n--- TRADE Manual Trading ---")
    is_live = tracker.config.get("portfolio", {}).get("live_trading_enabled", False)
    if not is_live:
        print(
            "🟡 NOTE: Live Trading is DISABLED. All trades will be simulated (Dry Run)."
        )
    else:
        print("🔴 WARNING: Live Trading is ENABLED. Real orders will be placed.")

    loop = asyncio.get_event_loop()

    # 1. Get Trade Type
    trade_type = ""
    while trade_type not in ["BUY", "SELL"]:
        trade_type = await loop.run_in_executor(
            None, input, "Choose action [BUY / SELL]: "
        )
        trade_type = trade_type.upper().strip()
        if not trade_type:
            print("Returning to main menu...")
            return

    # 2. Get Asset
    symbol = await loop.run_in_executor(None, input, "Enter asset symbol (e.g., BTC): ")
    symbol = symbol.upper().strip()
    if not symbol:
        print("Returning to main menu...")
        return
    trade_ticker = f"{symbol}USDT"

    # 3. Get Amount
    amount_str_input = await loop.run_in_executor(
        None, input, f"Enter amount to {trade_type} (e.g., '0.1 {symbol}'): "
    )
    amount_str_input = amount_str_input.upper().strip()
    if not amount_str_input:
        print("Returning to main menu...")
        return

    is_quote_qty = "USDT" in amount_str_input
    try:
        # Extract numbers from the string
        numeric_part = re.search(r"[\d\.]+", amount_str_input).group(0)
        amount = float(numeric_part)
    except (AttributeError, ValueError):
        print("❌ Invalid amount format. Could not parse number.")
        return

    # 4. Confirmation
    print("\n" + "=" * 50)
    print("🚨 PLEASE CONFIRM THE FOLLOWING MARKET ORDER 🚨")
    print(f"   Action: {trade_type}")
    print(f"   Asset:  {symbol}")
    if is_quote_qty:
        print(f"   Amount: {amount:,.2f} USDT")
    else:
        print(f"   Amount: {amount:,.8g} {symbol}")
    print("=" * 50)

    confirm = await loop.run_in_executor(None, input, "Type 'EXECUTE' to confirm: ")
    confirm = confirm.strip()

    if confirm == "EXECUTE":
        result = await tracker.execute_manual_trade_core(
            trade_type, symbol, trade_ticker, amount, is_quote_qty, is_live
        )
        for msg in result.messages:
            print(msg)
        if result.success:
            print("✅ Trade executed successfully!")
        else:
            print("❌ Trade failed:")
            for err in result.errors:
                print("   -", err)
    else:
        print("🛑 Trade cancelled by user.")


async def run_rebalance_and_execute(tracker: CryptoPortfolioTracker):
    """
    Orchestrates the process of getting rebalancing suggestions and allows users
    to execute them all at once or one-by-one.
    """
    logger.info("Starting automated rebalancing process...")
    print("\n--- ⚖️ Automated Rebalancing ---")
    loop = asyncio.get_event_loop()

    # 1. Get Suggestions
    print("🔄 Generating rebalancing suggestions...")
    suggestions_df = await tracker.get_core_portfolio_rebalance_suggestions_technical()

    if suggestions_df is None or suggestions_df.empty:
        print("\n✅ No rebalancing suggestions available at this time.")
        logger.info("No rebalancing suggestions generated. Exiting process.")
        return

    # 2. Calculate TOTAL available USDT (Spot + Earn)
    total_usdt_balance = 0
    try:
        spot_balance = float(
            tracker.binance_client.get_asset_balance(asset="USDT").get("free", 0.0)
        )
        total_usdt_balance += spot_balance
        if not tracker.config_manager.is_testnet_mode:
            earn_positions = tracker.fetcher.fetch_simple_earn_balances(
                pd.DataFrame([{"symbol": "USDT"}])
            )
            earn_balance = earn_positions.get("USDT", 0.0)
            total_usdt_balance += earn_balance
    except Exception as e:
        logger.error(f"Could not fetch total USDT balance for display: {e}")

    # 3. Print Suggestions for Review
    print_rebalance_suggestions(
        tracker, suggestions_df, available_usdt=total_usdt_balance
    )

    # 4. Check for actionable signals
    actionable_trades = suggestions_df[suggestions_df["Signal"].isin(["BUY", "SELL"])]
    if actionable_trades.empty:
        print("\n✅ Your portfolio is balanced. No rebalancing needed at this time.")
        return

    # 5. Add interactive execution choice
    while True:
        action = await loop.run_in_executor(
            None, input, "Type EXECUTE ALL or EXECUTE for one-by-one confirmation: "
        )
        action = action.upper().strip()

        if not action:
            print("Returning to main menu...")
            return

        elif action in ["EXECUTE ALL", "EXECUTE"]:
            logger.info("Preparing for rebalance execution...")
            tracker._sync_binance_client_time(
                tracker.binance_client, context="rebalancing"
            )
            interactive_mode = action == "EXECUTE"
            earn_balances = {}
            if not tracker.config_manager.is_testnet_mode:
                print("Verifying balances in Spot and Earn wallets...")
                spot_balances_df = tracker.fetch_binance_balances()
                earn_balances = tracker.fetcher.fetch_simple_earn_balances(
                    spot_balances_df
                )
            else:
                print("🟡 TESTNET MODE: Skipping Earn wallet check.")
            result = await tracker.execute_rebalancing_trades_core(
                suggestions_df,
                earn_balances,
                interactive=interactive_mode,
                auto_confirm=True,
            )
            for msg in result.messages:
                print(msg)
            if result.success:
                print(
                    f"✅ {result.data.get('trades_executed', 0)} trade(s) executed successfully!"
                )
            else:
                print("❌ No trades executed.")
                for err in result.errors:
                    print("   -", err)
            return
        else:
            print("Invalid command. Please choose from EXECUTE ALL, EXECUTE.")


async def run_trading_strategy_backtest(tracker: CryptoPortfolioTracker):
    """
    Handles the user flow for running a directional strategy backtest.
    """
    print("\n--- 🧪 Directional Strategy Backtesting Mode ---")
    loop = asyncio.get_event_loop()

    try:
        analyzer = CryptoTrendAnalyzer(
            config=tracker.config, binance_client=tracker.binance_client
        )
        backtester = StrategyBacktester(config=tracker.config, analyzer=analyzer)
    except Exception as e:
        logger.error(f"Failed to initialize backtesting components: {e}", exc_info=True)
        return

    # Strategy Selection
    available_strategies = {
        name: obj
        for name, obj in inspect.getmembers(trading_strategies, inspect.isclass)
        if issubclass(obj, trading_strategies.Strategy)
        and obj is not trading_strategies.Strategy
    }
    if not available_strategies:
        print("❌ No strategy classes found in 'src/trading_strategies.py'.")
        return

    print("\nAvailable Strategies:")
    strategy_list = list(available_strategies.keys())
    for i, name in enumerate(strategy_list):
        print(f"  {i + 1}. {name}")

    strategy_to_run = None
    while strategy_to_run is None:
        try:
            choice_str = await loop.run_in_executor(
                None,
                input,
                f"Select the strategy to backtest (1-{len(strategy_list)}): ",
            )
            if not choice_str:
                return
            choice = int(choice_str) - 1
            if 0 <= choice < len(strategy_list):
                strategy_name = strategy_list[choice]
                strategy_class = available_strategies[strategy_name]
                print(f"\nConfiguring strategy: {strategy_name}")
                user_params = (
                    strategy_class.get_user_params()
                    if hasattr(strategy_class, "get_user_params")
                    else {}
                )
                strategy_to_run = strategy_class(analyzer=analyzer, **user_params)
            else:
                print("❌ Invalid selection.")
        except (ValueError, IndexError):
            print("❌ Invalid input.")

    if not strategy_to_run:
        return
    print(f"\nSelected Strategy: {strategy_to_run.name}")

    # Interval Selection
    interval = "1d"
    # ... [interval selection logic can be added here if needed]

    # Coin Selection
    # ... [coin selection logic can be added here, similar to the original]
    symbol_to_test = await loop.run_in_executor(
        None, input, "Enter the symbol to backtest (e.g., BTC-USD): "
    )

    try:
        initial_capital_str = await loop.run_in_executor(
            None, input, "Enter initial capital (default: 10000): "
        )
        initial_capital = float(initial_capital_str) if initial_capital_str else 10000.0
        period_str = await loop.run_in_executor(
            None, input, "Enter backtest period (e.g., '3y', '60d', default: '3y'): "
        )
        period = period_str if period_str else "3y"
    except ValueError:
        initial_capital = 10000.0
        period = "3y"

    await backtester.run(
        strategy=strategy_to_run,
        symbol=symbol_to_test,
        initial_capital=initial_capital,
        period=period,
        interval=interval,
    )
    backtester.generate_report()


async def run_live_strategy(tracker: CryptoPortfolioTracker):
    """
    Master workflow for running a directional strategy for live signal generation and execution.
    """
    print("\n--- 🤖 Live Trading Strategy Runner ---")

    # 1. Select Account
    accounts = [
        {
            "name": "Main Account",
            "type": "main",
            **tracker.config.get("main_api_keys", {}),
        }
    ]
    sub_accounts = tracker.config.get("sub_accounts", [])
    for sub in sub_accounts:
        # Infer account type from name
        if "swing" in sub["name"].lower():
            sub["type"] = "swing"
        elif "day" in sub["name"].lower():
            sub["type"] = "day"
        else:
            sub["type"] = "main"  # Default if not specified
        accounts.append(sub)

    if not any(acc.get("binance_key") for acc in accounts):
        print("❌ No API keys found for any account. Cannot run live strategies.")
        return

    print("\nSelect account to trade on:")
    for i, acc in enumerate(accounts):
        print(f"  {i + 1}. {acc['name']} (Type: {acc.get('type', 'N/A')})")

    selected_account = None
    while selected_account is None:
        choice_str = input(f"Select account (1-{len(accounts)}): ").strip()
        if not choice_str:
            print("Returning to main menu...")
            return

        try:
            choice = int(choice_str) - 1
            if 0 <= choice < len(accounts):
                selected_account = accounts[choice]
            else:
                print("❌ Invalid selection.")
        except (ValueError, IndexError):
            print("❌ Invalid input.")

    print(f"\nTrading on account: {selected_account['name']}")
    live_client = tracker._init_binance_client(
        api_key=selected_account.get("binance_key"),
        api_secret=selected_account.get("binance_secret"),
    )
    if not live_client:
        print(
            f"❌ Failed to initialize Binance client for account: {selected_account['name']}. Check API keys."
        )
        return

    # 2. Select and Configure Strategy based on account type
    all_strategies = {
        name: obj
        for name, obj in inspect.getmembers(trading_strategies, inspect.isclass)
        if issubclass(obj, trading_strategies.Strategy)
        and obj is not trading_strategies.Strategy
    }

    account_type = selected_account.get("type")
    available_strategies = {}
    if account_type == "main":
        available_strategies = all_strategies
    elif account_type == "swing":
        available_strategies = {
            k: v
            for k, v in all_strategies.items()
            if v.strategy_type in ["swing", "general"]
        }
    elif account_type == "day":
        available_strategies = {
            k: v
            for k, v in all_strategies.items()
            if v.strategy_type in ["day", "general"]
        }

    if not available_strategies:
        print(
            f"❌ No suitable strategies found for an account of type '{account_type}'."
        )
        return

    print("\nAvailable Strategies:")
    strategy_list = list(available_strategies.keys())
    for i, name in enumerate(strategy_list):
        print(f"  {i + 1}. {name}")

    strategy_class = None
    user_params = {}
    strategy_name = ""
    while strategy_class is None:
        choice_str = input(f"Select strategy to run (1-{len(strategy_list)}): ").strip()
        if not choice_str:
            print("Returning to main menu...")
            return

        try:
            choice = int(choice_str) - 1
            if 0 <= choice < len(strategy_list):
                strategy_name = strategy_list[choice]
                strategy_class = available_strategies[strategy_name]
                print(f"\nConfiguring strategy: {strategy_name}")
                if hasattr(strategy_class, "get_user_params"):
                    user_params = strategy_class.get_user_params()
            else:
                print("❌ Invalid selection.")
        except (ValueError, IndexError):
            print("❌ Invalid input.")

    # Create a temporary instance just to get the name for the print message
    temp_strategy_for_name = strategy_class(analyzer=None, **user_params)
    print(f"\n🔄 Running '{temp_strategy_for_name.name}' to generate live signals...")

    # 3. Generate Signals for all portfolio assets
    target_coins = list(tracker.config.get("target_allocation", {}).keys())
    signals_to_execute = []
    analyzer = CryptoTrendAnalyzer(config=tracker.config, binance_client=live_client)

    for coin in target_coins:
        yf_ticker = f"{coin}-USD"
        analyzer.set_symbol(yf_ticker)
        state_key = f"{selected_account['name']}_{strategy_name}_{coin}"
        previous_state = tracker.strategy_states.get(state_key)

        strategy_instance = strategy_class(
            analyzer=analyzer, state=previous_state, **user_params
        )

        interval = strategy_instance.valid_intervals[0]
        # Use a shorter period for live trading to be faster
        period = "7d" if "m" in interval or "h" in interval else "1y"
        data = await analyzer.fetch_crypto_data_async(
            yf_ticker, period=period, interval=interval
        )

        if data is None or data.empty:
            tracker.logger.warning(
                f"Could not fetch data for {yf_ticker}, cannot generate signal."
            )
            continue

        signal, size, reason = await strategy_instance.generate_signal(data)
        tracker.strategy_states[state_key] = strategy_instance.get_state()

        if signal in ["BUY", "SELL"]:
            signals_to_execute.append(
                {"Symbol": coin, "Signal": signal, "Size": size, "Reason": reason}
            )

    tracker._save_strategy_state()

    # 4. Present Trades and Ask for Execution
    if not signals_to_execute:
        print(
            "\n✅ Analysis complete. No new BUY or SELL signals generated by the strategy."
        )
        return

    is_live = tracker.config.get("portfolio", {}).get("live_trading_enabled", False)
    is_testnet = tracker.config.get("apis", {}).get("binance", {}).get("testnet", False)

    print("\n" + "=" * 80)
    print("🚨 PROPOSED TRADES - PLEASE REVIEW CAREFULLY 🚨")
    print("=" * 80)

    if is_testnet:
        print("🟡🟡🟡 NOTE: Connected to TESTNET. No real funds will be used. 🟡🟡🟡")

    if is_live:
        print(
            "🔴🔴🔴 WARNING: Live Trading is ENABLED. Real orders will be placed. 🔴🔴🔴"
        )
    else:
        print("🟡🟡🟡 NOTE: Live Trading is DISABLED. This is a DRY RUN. 🟡🟡🟡")

    print("=" * 80)
    for trade in signals_to_execute:
        print(f"-> {trade['Signal']} {trade['Symbol']} (Reason: {trade['Reason']})")
    print("=" * 80)

    try:
        confirm = input("Type 'EXECUTE' to proceed with the trades listed above: ")
        if confirm != "EXECUTE":
            print("🛑 Trade execution cancelled by user.")
            return
    except KeyboardInterrupt:
        print("\n🛑 Trade execution cancelled by user.")
        return

    # 5. Execute Trades
    for trade in signals_to_execute:
        tracker._execute_directional_trade(trade, live_client)


def _restore_database_interactive(tracker: CryptoPortfolioTracker):
    """Handles the interactive process of restoring a database."""
    print("\n--- Restoring Database from Backup ---")
    backups = tracker.db_manager.list_backups()

    if not backups:
        print("❌ No backup files found in the 'data/db_backups/' directory.")
        return

    print("Available backups (newest first):")
    for i, backup_file in enumerate(backups):
        try:
            relative_path = backup_file.relative_to(tracker.config_manager.project_root)
        except ValueError:
            relative_path = backup_file  # Fallback
        print(f"  {i + 1}. {relative_path}")

    try:
        selection_str = input(
            f"\nEnter the number of the backup to restore (or press Enter to cancel): "
        ).strip()
        if not selection_str:
            print("Restore cancelled.")
            return

        selection_idx = int(selection_str) - 1
        if not 0 <= selection_idx < len(backups):
            print("❌ Invalid selection.")
            return

        selected_backup = backups[selection_idx]
        print("\n" + "=" * 50)
        print("🚨 WARNING: THIS IS A DESTRUCTIVE ACTION 🚨")
        print(f"You are about to restore from:\n  -> {selected_backup.name}")
        print("=" * 50)

        confirm = input("Type 'RESTORE' to proceed: ")
        if confirm == "RESTORE":
            print("Restoring database...")
            if tracker.db_manager.restore_from_backup(selected_backup):
                print("\n✅ Restore successful. PLEASE RESTART THE APPLICATION.")
            else:
                print("❌ Restore failed. Check logs.")
        else:
            print("🛑 Restore cancelled.")

    except (ValueError, IndexError):
        print("❌ Invalid input.")


def export_trend_report_interactive(tracker: CryptoPortfolioTracker, report: Dict[str, Any], timeframe: str) -> bool:
    """
    Interactive trend report export for CLI with format selection.
    
    Args:
        report: The trend analysis report dictionary
        timeframe: The timeframe of the analysis
    
    Returns:
        True if export was successful, False otherwise
    """
    try:
        print("\n--- �� Export Options ---")
        print("1. Export as HTML (recommended)")
        print("2. Export as CSV")
        print("3. Export as JSON")
        print("4. Export ALL formats")
        
        choice = input("Select option (1-): ").strip()

        if not choice:
            return
        
        if choice == "1":
            exported_file = tracker.export_trend_report(report, timeframe, "HTML")
            if exported_file:
                print(f"✅ HTML report exported to: {exported_file}")
                return True
                
        elif choice == "2":
            exported_file = tracker.export_trend_report(report, timeframe, "CSV")
            if exported_file:
                print(f"✅ CSV report exported to: {exported_file}")
                return True
                
        elif choice == "3":
            exported_file = tracker.export_trend_report(report, timeframe, "JSON")
            if exported_file:
                print(f"✅ JSON report exported to: {exported_file}")
                return True
                
        elif choice == "4":
            results = tracker.export_trend_report_all_formats(report, timeframe)
            success_count = sum(1 for path in results.values() if path is not None)
            print(f"✅ Exported {success_count}/3 formats successfully")
            for format_type, file_path in results.items():
                if file_path:
                    print(f"   {format_type}: {file_path}")
            return success_count > 0
            
        else:
            print("❌ Invalid option")
            return False
            
    except Exception as e:
        tracker.logger.error(f"Interactive export failed: {e}", exc_info=True)
        print(f"❌ Export failed: {e}")
        return False


def run_backup_and_restore_session(tracker: CryptoPortfolioTracker):
    """Orchestrates creating a backup or restoring from one."""
    print("\n--- 🗄️ Database Backup & Restore ---\n")
    print("1. Create a new database backup")
    print("2. Restore from an existing backup")
    print("Press Enter to return to the main menu.")

    choice = input("Select an option: ").strip()

    if choice == "1":
        print("\n💾 Creating database backup...")
        if tracker.db_manager.backup_database():
            print("✅ Backup successful.")
        else:
            print("❌ Backup failed. Please check the logs.")
    elif choice == "2":
        _restore_database_interactive(tracker)
    else:
        print("Returning to main menu...")


def print_configuration(tracker: CryptoPortfolioTracker):
    """Prints a security-redacted version of the current configuration."""
    print("\n" + "=" * 50 + "\n⚙️ Current Configuration\n" + "=" * 50)
    safe_config = copy.deepcopy(tracker.config)
    if "main_api_keys" in safe_config:
        safe_config["main_api_keys"] = {
            k: "********" for k in safe_config["main_api_keys"]
        }
    if "sub_accounts" in safe_config:
        for account in safe_config["sub_accounts"]:
            if "binance_key" in account:
                account["binance_key"] = "********"
            if "binance_secret" in account:
                account["binance_secret"] = "********"
    print(json.dumps(safe_config, indent=2))
    print("=" * 50)


async def test_connections(tracker: CryptoPortfolioTracker):
    """Test connections to Binance and CoinGecko."""
    if tracker.binance_client:
        try:
            tracker.binance_client.ping()
            print("✅ Binance Connection: SUCCESS")
        except Exception as e:
            print(f"❌ Binance Connection: FAILED ({e})")
    else:
        print("⚠️ Binance Connection: SKIPPED (No API keys or client init failed)")

    try:
        prices = await tracker.enricher.get_current_prices(["BTC"])
        if prices and prices.get("BTC"):
            print(f"✅ CoinGecko Connection: SUCCESS (BTC price: ${prices['BTC']})")
        else:
            print("❌ CoinGecko Connection: FAILED (No price data returned)")
    except Exception as e:
        print(f"❌ CoinGecko Connection: FAILED ({str(e)})")
    print("-" * 30)


async def run_interactive_mode(tracker: CryptoPortfolioTracker):
    """Runs the main interactive menu loop, now fully asynchronous."""
    loop = asyncio.get_event_loop()
    offline_mode = getattr(tracker, "offline_mode", False)
    unavailable_offline = {1, 2, 3, 4, 5, 6, 7, 8, 12}
    while True:
        print_main_menu(offline_mode)
        try:
            choice_str = await loop.run_in_executor(
                None, input, "Select option (1-13): "
            )
            choice = int(choice_str) if choice_str.isdigit() else -1

            if offline_mode and choice in unavailable_offline:
                print("❌ This feature is unavailable in offline mode.")
                continue

            match choice:
                case 1:
                    print("\n🔄 Running full sync and analysis...")
                    metrics = await tracker.run_full_sync()
                    print_portfolio_summary(tracker, metrics)
                    # tracker.save_snapshot(metrics)
                case 2:
                    print("\n📊 Generating quick portfolio summary...")
                    metrics = await tracker.calculate_portfolio_metrics()
                    print_portfolio_summary(tracker, metrics)
                case 3:
                    await view_trends(tracker)
                case 4:
                    await run_rebalance_and_execute(tracker)
                case 5:
                    print("\n--- 🔀 Trading ---")
                    print("1. Manual Trade (Buy/Sell)")
                    print("2. Live Trading Strategy")
                    sub_choice = await loop.run_in_executor(
                        None, input, "Select option (1-2) or Enter to return: "
                    )
                    if sub_choice == "1":
                        await run_manual_trade_session(tracker)
                    elif sub_choice == "2":
                        await run_live_strategy(tracker)
                case 6:
                    print("\n--- 🧪 Backtesting ---")
                    print("1. Rebalancing Backtest")
                    print("2. Strategy Backtest")
                    sub_choice = await loop.run_in_executor(
                        None, input, "Select option (1-2) or Enter to return: "
                    )
                    if sub_choice == "1":
                        await run_rebalancing_backtest(tracker)
                    elif sub_choice == "2":
                        await run_trading_strategy_backtest(tracker)
                case 7:
                    print("\n--- 📋 Export Reports / Data ---")
                    print("1. Export as Excel")
                    print("1. Export as HTML")
                    print("1. Export as CSV")
                    print("1. Export ALL (Excel, HTML, CSV)")
                    sub_choice = await loop.run_in_executor(
                        None, input, "Select option (1-4) or Enter to return: "
                    )
                    metrics = await tracker.calculate_portfolio_metrics()
                    if sub_choice == "1":
                        format = "Excel"
                        print(f"Exporting to {format}...")
                        metrics = await tracker.calculate_portfolio_metrics()
                        tracker.export_to_excel(metrics)
                        print(f"Portfolio Metrics Exported to {format}.")
                    elif sub_choice == "2":
                        format = "HTML"
                        print(f"Exporting to {format}...")
                        metrics = await tracker.calculate_portfolio_metrics()
                        tracker.export_to_html(metrics)
                        print(f"Portfolio Metrics Exported to {format}")
                    elif sub_choice == "3":
                        format = "CSV"
                        print(f"Exporting to {format}...")
                        metrics = await tracker.calculate_portfolio_metrics()
                        tracker.export_csv_backup()
                        print(f"Portfolio Metrics Exported to {format}")
                    elif sub_choice == "4":
                        print(f"Exporting to (Excel, HTML, CSV)...")
                        metrics = await tracker.calculate_portfolio_metrics()
                        tracker.export_to_excel(metrics)
                        tracker.export_to_html(metrics)
                        tracker.export_csv_backup()
                        print(f"Portfolio Metrics Exported to (Excel, HTML, CSV).")
                case 8:
                    print("\n📈 Generating charts...")
                    metrics = await tracker.calculate_portfolio_metrics()
                    tracker.create_portfolio_charts(metrics)
                case 9:
                    run_backup_and_restore_session(tracker)
                case 10:
                    print("\n🧹 Data Cleanup")
                    print("=" * 50)

                    # Get cleanup configuration
                    cleanup_days = tracker.config.get("database", {}).get(
                        "cleanup_days", 90
                    )
                    print(f"📊 Current Retention Period: {cleanup_days} days")

                    if cleanup_days <= 0:
                        print(
                            "⚠️  Data cleanup is currently disabled (cleanup_days = 0)"
                        )
                        continue

                    # Calculate what would be deleted
                    from datetime import datetime, timedelta

                    cutoff_date = datetime.now() - timedelta(days=cleanup_days)
                    print(
                        f"📅 Cutoff Date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    # Get cleanup statistics
                    stats = tracker.db_manager.get_cleanup_statistics()

                    if not stats["cleanup_enabled"]:
                        print(
                            "⚠️  Data cleanup is currently disabled (cleanup_days = 0)"
                        )
                        continue

                    if "error" in stats:
                        print(f"❌ Could not analyze database: {stats['error']}")
                        continue

                    old_transactions = stats["old_transactions"]
                    old_snapshots = stats["old_snapshots"]
                    total_transactions = stats["total_transactions"]
                    total_snapshots = stats["total_snapshots"]
                    cutoff_date = stats["cutoff_date"]

                    # Display what will be deleted
                    print(
                        f"\n📊 Old Transactions: {old_transactions:,} of {total_transactions:,} total"
                    )
                    print(
                        f"📸 Old Snapshots: {old_snapshots:,} of {total_snapshots:,} total"
                    )

                    if old_transactions > 0 or old_snapshots > 0:
                        print("\n⚠️  WARNING: This will permanently delete:")
                        print(
                            "   - Historical transaction data older than the retention period"
                        )
                        print(
                            "   - Portfolio snapshots older than the retention period"
                        )
                        print(
                            "   - Impact: This may affect tax reporting, historical analysis, and portfolio tracking"
                        )

                        print("\n🔐 Confirmation Required")
                        confirm = input(
                            "Type 'DELETE' to confirm, or press Enter to cancel: "
                        ).strip()

                        if confirm == "DELETE":
                            # Create backup before deletion
                            backup_path = tracker.db_manager.backup_database()
                            if backup_path:
                                print(f"✅ Backup created: {backup_path}")

                            # Perform cleanup
                            tracker.cleanup_old_data()
                            print("✅ Data cleanup completed successfully!")
                        else:
                            print("🛑 Cleanup cancelled.")
                    else:
                        print("\n✅ No old data to clean up!")
                        print("ℹ️  All your data is within the retention period.")
                case 11:
                    print_configuration(tracker)
                case 12:
                    await test_connections(tracker)
                case 13:
                    print("👋 Exiting. Goodbye!")
                    break
                case _:
                    print("❌ Invalid option. Please try again.")

        except NetworkOperationError as e:
            print(f"\n❌ Network error: {e}\nOperation aborted.")
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            print(f"\n❌ An unexpected error occurred: {e}")

        await loop.run_in_executor(None, input, "\n✅ Press Enter to continue...")


async def amain():
    """The main asynchronous entry point for the application."""
    parser = argparse.ArgumentParser(description="Crypto Portfolio Tracker")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const="DEBUG",
        dest="loglevel",
        help="Enable verbose DEBUG logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const="WARNING",
        dest="loglevel",
        help="Enable quiet logging.",
    )
    args = parser.parse_args()

    setup_logging(level_override=args.loglevel)

    logger.info("Starting Crypto Portfolio Tracker")
    config_manager = ConfigManager()

    tracker = None
    try:
        tracker = CryptoPortfolioTracker(config_manager)
    except NetworkUnavailableError:
        resp = (
            input("⚠️  Network appears unavailable. Enter offline mode? [Y/n]: ")
            .strip()
            .lower()
        )
        if resp not in ("", "y", "yes"):
            print("Exiting. Please check your network and try again.")
            return
        tracker = CryptoPortfolioTracker(config_manager, force_offline=True)

    if getattr(tracker, "offline_mode", False):
        print("⚠️  OFFLINE MODE: Network features are disabled.")
    await run_interactive_mode(tracker)


def main():
    """Synchronous entry point that starts the asyncio event loop."""
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n👋 Exiting due to user interruption.")


if __name__ == "__main__":
    main()
