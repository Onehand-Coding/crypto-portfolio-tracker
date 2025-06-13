#!/usr/bin/env python3
"""
Crypto Portfolio Tracker - Main Entry Point
Entry point for the cryptocurrency portfolio tracking application.
"""
import os
import sys
import asyncio
import logging
import platform
import argparse
import warnings
from pathlib import Path

# Suppress the UserWarning from the pandas_ta library right away.
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from portfolio_tracker import CryptoPortfolioTracker
from config import setup_logging, ConfigManager


def clear_screen() -> None:
    """Clears the terminal screen."""
    os.system('cls' if platform.system() == "Windows" else 'clear')


def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Crypto Portfolio Tracker - Analyze your cryptocurrency investments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive mode
  python main.py --sync-only               # Sync data only
  python main.py --export-only             # Export existing data
  python main.py --format excel            # Export to Excel only
  python main.py --verbose                 # Enable debug logging
  python main.py --config my_config.json   # Use custom config
        """
    )

    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Run data synchronization only (no analysis or exports)"
    )

    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export existing data without syncing"
    )

    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Generate charts only from existing data"
    )

    parser.add_argument(
        "--format",
        choices=["excel", "html", "csv", "all"],
        default="all",
        help="Export format (default: all)"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output except errors"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Crypto Portfolio Tracker v2.1.0" # Ensure this matches actual version if it changes
    )

    return parser


def show_interactive_menu():
    """Display interactive menu and get user choice"""
    print("\n" + "="*50)
    print("🚀 Crypto Portfolio Tracker v2.1.0")
    print("="*50)
    print("1. 🔄 Full Sync & Analysis (Recommended)")
    print("2. 📊 Quick Portfolio Summary")
    print("3. 📈 View Crypto Trends")
    print("4. ⚖️  View Rebalance Suggestions")
    print("5. 🤖 Execute Rebalancing Trades")
    print("6. 💰 Live Trading (Directional Strategy)")
    print("7. 🧪 Run Strategy Backtest")
    print("8. ⚖️  Run Rebalancing Backtest")
    print("9. 📋 Export Reports Only")
    print("10. 📈 Generate Charts Only")
    print("11. 💾 Export Data Backup")
    print("12. ⚙️  View Configuration")
    print("13. 🧹 Clean Old Data")
    print("14. 🔧 Test API Connections")
    print("15. ❌ Exit")
    print("="*50)

    while True:
        try:
            choice = input("\nSelect option (1-15): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= 15:
                return int(choice)
            else:
                print("❌ Invalid choice. Please select a number from 1 to 15.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except EOFError: # Handle Ctrl+D
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def run_interactive_mode(tracker: CryptoPortfolioTracker):
    """Run the application in interactive mode"""
    while True:
        choice = show_interactive_menu()

        try:
            if choice == 1:
                print("\n🔄 Running full sync and analysis...")
                metrics = asyncio.run(tracker.run_full_sync())
                if "error" not in metrics:
                    tracker.print_portfolio_summary(metrics)
                    tracker.save_snapshot(metrics)
                input("\n✅ Full sync & analysis complete. Press Enter to continue...")

            elif choice == 2:
                print("\n📊 Generating quick portfolio summary...")
                metrics = tracker.calculate_portfolio_metrics()
                if "error" in metrics:
                    print(f"\n❌ Error: {metrics['error']}")
                else:
                    tracker.print_portfolio_summary(metrics)
                input("\n✅ Press Enter to continue...")

            elif choice == 3:
                print("\n📈 Viewing Crypto Trends...")
                asyncio.run(tracker.view_trends())

            elif choice == 4:
                suggestions = asyncio.run(tracker.get_core_portfolio_rebalance_suggestions_technical())
                tracker.print_rebalance_suggestions(suggestions)
                input("\n✅ Press Enter to continue...")

            elif choice == 5:
                asyncio.run(tracker.run_rebalance_and_execute())
                input("\n✅ Press Enter to continue...")

            elif choice == 6:
                print("\n💰 Entering Live Trading Mode...")
                asyncio.run(tracker.run_live_strategy())
                input("\n✅ Live trading session finished. Press Enter to continue...")

            elif choice == 7:
                asyncio.run(tracker.run_backtest())
                input("\n✅ Press Enter to continue...")

            elif choice == 8:
                asyncio.run(tracker.run_rebalancing_backtest())
                input("\n✅ Press Enter to continue...")

            elif choice == 9:
                print("\n📋 Exporting reports...")
                metrics = tracker.calculate_portfolio_metrics()
                if "error" not in metrics:
                    if tracker.config.get("exports",{}).get("formats",{}).get("excel",{}).get("enabled", False):
                        tracker.export_to_excel(metrics)
                    if tracker.config.get("exports",{}).get("formats",{}).get("html",{}).get("enabled", False):
                        tracker.export_to_html(metrics)
                    print("\n✅ Reports exported (if enabled in config).")
                else:
                    print(f"❌ Could not generate metrics for export: {metrics['error']}")
                input("Press Enter to continue...")

            elif choice == 10:
                print("\n📈 Generating charts...")
                metrics = tracker.calculate_portfolio_metrics()
                if "error" not in metrics:
                    tracker.create_portfolio_charts(metrics)
                    print("\n✅ Charts generated (if enabled and data available).")
                else:
                    print(f"❌ Could not generate metrics for charts: {metrics['error']}")
                input("Press Enter to continue...")

            elif choice == 11:
                print("\n💾 Exporting data backup (CSV)...")
                tracker.export_csv_backup()
                input("\n✅ CSV Backup completed. Press Enter to continue...")

            elif choice == 12:
                print("\n⚙  Current Configuration:")
                tracker.print_configuration()
                input("\nPress Enter to continue...")

            elif choice == 13:
                print("\n🧹 Cleaning old data...")
                tracker.cleanup_old_data()
                input("\n✅ Cleanup completed. Press Enter to continue...")

            elif choice == 14:
                print("\n🔧 Testing API connections...")
                tracker.test_connections()
                input("\n✅ Test completed. Press Enter to continue...")

            elif choice == 15:
                print("\n👋 Goodbye!")
                break

        except KeyboardInterrupt:
            print("\n\n⚠ Operation cancelled by user. Returning to menu.")
            continue
        except Exception as e:
            logging.exception(f"Error in interactive mode choice {choice}: {e}")
            print(f"\n❌ An unexpected error occurred: {e}")
            print("Please check logs for more details.")
            input("Press Enter to continue...")


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Step 1: Instantiate the new ConfigManager. It handles all config loading internally.
    manager = ConfigManager(args.config)
    config_data = manager.config # Get the non-sensitive config dict for logging setup

    # Determine the final log level
    log_level = config_data.get("logging", {}).get("level", "INFO")
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"

    # Setup logging
    setup_logging(config=config_data.get("logging"), level=log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting Crypto Portfolio Tracker")

    try:
        # Step 2: Pass the entire ConfigManager instance to the tracker.
        tracker = CryptoPortfolioTracker(config_manager=manager)

        if args.sync_only:
            logger.info("Running sync-only mode")
            asyncio.run(tracker.sync_data())
            print("✅ Data synchronization completed")
        elif args.export_only:
            logger.info("Running export-only mode")
            metrics = tracker.calculate_portfolio_metrics()
            if "error" in metrics:
                print(f"❌ Error calculating metrics: {metrics['error']}")
            else:
                export_performed = False
                if args.format == "excel" or args.format == "all":
                    if tracker.config.get("exports",{}).get("formats",{}).get("excel",{}).get("enabled", False):
                        tracker.export_to_excel(metrics)
                        export_performed = True
                if args.format == "html" or args.format == "all":
                    if tracker.config.get("exports",{}).get("formats",{}).get("html",{}).get("enabled", False):
                        tracker.export_to_html(metrics)
                        export_performed = True
                if args.format == "csv" or args.format == "all":
                    if tracker.config.get("exports",{}).get("formats",{}).get("csv",{}).get("enabled", False):
                        tracker.export_csv_backup()
                        export_performed = True

                if export_performed:
                    print("✅ Export completed (for enabled formats).")
                else:
                    print("⚠️ No export formats enabled or specified for export-only mode.")
        elif args.charts_only:
            logger.info("Running charts-only mode")
            metrics = tracker.calculate_portfolio_metrics()
            if "error" in metrics:
                print(f"❌ Error calculating metrics: {metrics['error']}")
            else:
                tracker.create_portfolio_charts(metrics)
                print("✅ Charts generated (if data available).")
        else:
            run_interactive_mode(tracker)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        logging.info("Application interrupted by user")
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found or path error: {e}", exc_info=True)
        print(f"\n💥 Configuration Error: {e}")
        print("Please ensure your config/config.json or config/default_config.json file exists.")
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}", exc_info=True)
        print(f"\n💥 A fatal error occurred: {e}")
        print("Please check logs for detailed error information.")
        sys.exit(1)
    finally:
        logging.info("Application finished")


if __name__ == "__main__":
    main()
