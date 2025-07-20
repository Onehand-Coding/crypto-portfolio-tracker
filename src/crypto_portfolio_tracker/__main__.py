#!/usr/bin/env python3
"""
Crypto Portfolio Tracker - Main Entry Point
"""
import asyncio
import logging
import colorlog
import argparse
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

from .portfolio_tracker import CryptoPortfolioTracker, NetworkUnavailableError
from .config import ConfigManager
from .exceptions import NetworkOperationError

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
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Configure file handler (without colors)
    file_config = logging_config.get("file_config", {})
    if file_config.get("enabled", True):
        file_log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_path = Path(file_config.get("path", "logs/portfolio_tracker.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(file_log_format))
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured to level: {log_level_str}")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def print_main_menu(offline_mode=False):
    """Prints the main menu options."""
    print("\n" + "="*50)
    print("🚀 Crypto Portfolio Tracker v2.1.0")
    if offline_mode:
        print("⚠️  OFFLINE MODE: Network features are disabled.")
    print("="*50)
    print("1. 🔄 Full Sync & Analysis")
    print("2. 📊 Quick Portfolio Summary")
    print("3. 📈 View Crypto Trends")
    print("4. 🤖 Execute Rebalancing Trades")
    print("5. 🔀 Trading")
    print("6. 🧪 Backtesting")
    print("7. 📋 Export Reports / Data")
    print("8. 📈 Generate Charts")
    print("9. 🗄️  Backup / Restore Database")
    print("10. 🧹 Clean Old Data")
    print("11. ⚙️  View Configuration")
    print("12. 🔧 Test API Connections")
    print("13. ❌ Exit")
    print("="*50)


async def run_interactive_mode(tracker: CryptoPortfolioTracker):
    """Runs the main interactive menu loop, now fully asynchronous."""
    loop = asyncio.get_event_loop()
    offline_mode = getattr(tracker, "offline_mode", False)
    unavailable_offline = {1,2,3,4,5,6,7,8}
    while True:
        print_main_menu(offline_mode)
        try:
            choice_str = await loop.run_in_executor(None, input, "Select option (1-13): ")
            choice = int(choice_str) if choice_str.isdigit() else -1

            if offline_mode and choice in unavailable_offline:
                print("❌ This feature is unavailable in offline mode.")
                continue

            match choice:
                case 1:
                    print("\n🔄 Running full sync and analysis...")
                    metrics = await tracker.run_full_sync()
                    tracker.print_portfolio_summary(metrics)
                    tracker.save_snapshot(metrics)
                case 2:
                    print("\n📊 Generating quick portfolio summary...")
                    metrics = await tracker.calculate_portfolio_metrics()
                    tracker.print_portfolio_summary(metrics)
                case 3:
                    await tracker.view_trends()
                case 4:
                    await tracker.run_rebalance_and_execute()
                case 5:
                    # Trading submenu: Manual or Live
                    print("\n--- 🔀 Trading ---")
                    print("1. Manual Trade (Buy/Sell)")
                    print("2. Live Trading Strategy")
                    print("Press Enter to return to main menu.")
                    sub_choice = await loop.run_in_executor(None, input, "Select option (1-2): ")
                    if sub_choice == "1":
                        await tracker.run_manual_trade_session()
                    elif sub_choice == "2":
                        await tracker.run_live_strategy()
                    else:
                        print("Returning to main menu...")
                case 6:
                    # Backtesting submenu: Strategy or Rebalancing
                    print("\n--- 🧪 Backtesting ---")
                    print("1. Strategy Backtest")
                    print("2. Rebalancing Backtest")
                    print("Press Enter to return to main menu.")
                    sub_choice = await loop.run_in_executor(None, input, "Select option (1-2): ")
                    if sub_choice == "1":
                        await tracker.run_trading_strategy_backtest()
                    elif sub_choice == "2":
                        await tracker.run_rebalancing_backtest()
                    else:
                        print("Returning to main menu...")
                case 7:
                    # Export submenu: Excel, HTML, CSV, or All
                    print("\n--- 📋 Export Reports / Data ---")
                    print("1. Export Excel Report")
                    print("2. Export HTML Report")
                    print("3. Export CSV Data Backup")
                    print("4. Export ALL (Excel, HTML, CSV)")
                    print("Press Enter to return to main menu.")
                    sub_choice = await loop.run_in_executor(None, input, "Select option (1-4): ")
                    metrics = await tracker.calculate_portfolio_metrics()
                    if sub_choice == "1":
                        tracker.export_to_excel(metrics)
                    elif sub_choice == "2":
                        tracker.export_to_html(metrics)
                    elif sub_choice == "3":
                        tracker.export_csv_backup()
                    elif sub_choice == "4":
                        tracker.export_to_excel(metrics)
                        tracker.export_to_html(metrics)
                        tracker.export_csv_backup()
                    else:
                        print("Returning to main menu...")
                case 8:
                    print("\n📈 Generating charts...")
                    metrics = await tracker.calculate_portfolio_metrics()
                    tracker.create_portfolio_charts(metrics)
                case 9:
                    tracker.run_backup_and_restore_session()
                case 10:
                    print("\n🧹 Cleaning old data...")
                    tracker.cleanup_old_data()
                case 11:
                    tracker.print_configuration()
                case 12:
                    tracker.test_connections()
                case 13:
                    print("👋 Exiting. Goodbye!")
                    break
                case _:
                    print("❌ Invalid option. Please try again.")

        except NetworkOperationError as e:
            print(f"\n❌ Network error: {e}\nOperation aborted. Please check your connection and try again.")
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            print(f"\n❌ An unexpected error occurred: {e}\nPlease check logs for more details.")

        await loop.run_in_executor(None, input, "\n✅ Press Enter to continue...")


async def amain():
    """The main asynchronous entry point for the application."""
    parser = argparse.ArgumentParser(description="Crypto Portfolio Tracker")
    parser.add_argument('-v', '--verbose', action='store_const', const='DEBUG', dest='loglevel', help="Enable verbose DEBUG logging.")
    parser.add_argument('-q', '--quiet', action='store_const', const='WARNING', dest='loglevel', help="Enable quiet logging.")
    args = parser.parse_args()

    setup_logging(level_override=args.loglevel)

    logger.info("Starting Crypto Portfolio Tracker")
    config_manager = ConfigManager()

    tracker = None
    try:
        tracker = CryptoPortfolioTracker(config_manager)
    except NetworkUnavailableError:
        resp = input("⚠️  Network appears unavailable. Enter offline mode? [Y/n]: ").strip().lower()
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
