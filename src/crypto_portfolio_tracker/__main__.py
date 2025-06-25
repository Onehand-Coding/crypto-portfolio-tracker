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

from .portfolio_tracker import CryptoPortfolioTracker
from .config import ConfigManager

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


def print_main_menu():
    """Prints the main menu options."""
    print("\n" + "="*50)
    print("🚀 Crypto Portfolio Tracker v2.1.0")
    print("="*50)
    print("1. 🔄 Full Sync & Analysis (Recommended)")
    print("2. 📊 Quick Portfolio Summary")
    print("3. 📈 View Crypto Trends")
    print("4. ⚖️  View Rebalance Suggestions")
    print("5. 🤖 Execute Rebalancing Trades")
    print("6. 🔀 TRADE Manual Trade (Buy/Sell)")
    print("7. 💰 Live Trading (Directional Strategy)")
    print("8. 🧪 Run Strategy Backtest")
    print("9. ⚖️  Run Rebalancing Backtest")
    print("10. 📋 Export Reports Only")
    print("11. 📈 Generate Charts Only")
    print("12. 💾 Export Data Backup")
    print("13. 🧹 Clean Old Data")
    print("14. ⚙️  View Configuration")
    print("15. 🔧 Test API Connections")
    print("16. ❌ Exit")
    print("="*50)


async def run_interactive_mode(tracker: CryptoPortfolioTracker):
    """Runs the main interactive menu loop, now fully asynchronous."""
    loop = asyncio.get_event_loop()
    while True:
        print_main_menu()
        try:
            choice_str = await loop.run_in_executor(None, input, "Select option (1-16): ")
            choice = int(choice_str) if choice_str.isdigit() else -1

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
                    print("\n⚖️  Generating rebalance suggestions...")
                    suggestions = await tracker.get_core_portfolio_rebalance_suggestions_technical()
                    tracker.print_rebalance_suggestions(suggestions)
                case 5:
                    await tracker.run_rebalance_and_execute()
                case 6:
                    await tracker.run_manual_trade_session()
                case 7:
                    await tracker.run_live_strategy()
                case 8:
                    await tracker.run_trading_strategy_backtest()
                case 9:
                    await tracker.run_rebalancing_backtest()
                case 10:
                    print("\n📋 Exporting reports...")
                    metrics = await tracker.calculate_portfolio_metrics()
                    tracker.export_to_excel(metrics)
                    tracker.export_to_html(metrics)
                case 11:
                    print("\n📈 Generating charts...")
                    metrics = await tracker.calculate_portfolio_metrics()
                    tracker.create_portfolio_charts(metrics)
                case 12:
                    print("\n💾 Exporting data backup...")
                    tracker.export_csv_backup()
                case 13:
                    print("\n🧹 Cleaning old data...")
                    tracker.cleanup_old_data()
                case 14:
                    tracker.print_configuration()
                case 15:
                    tracker.test_connections()
                case 16:
                    print("👋 Exiting. Goodbye!")
                    break
                case _:
                    print("❌ Invalid option. Please try again.")

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

    try:
        tracker = CryptoPortfolioTracker(config_manager)
        await run_interactive_mode(tracker)
    except Exception as e:
        logger.critical(f"A critical error occurred at the top level: {e}", exc_info=True)
    finally:
        logger.info("Application finished.")


def main():
    """Synchronous entry point that starts the asyncio event loop."""
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n👋 Exiting due to user interruption.")


if __name__ == "__main__":
    main()
