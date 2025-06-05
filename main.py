#!/usr/bin/env python3
"""
Crypto Portfolio Tracker - Main Entry Point
Entry point for the cryptocurrency portfolio tracking application.
"""
import os
import sys
import logging
import platform
import argparse
from pathlib import Path
import asyncio # Added for running async methods

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from portfolio_tracker import CryptoPortfolioTracker
from config import setup_logging, load_config

# load_config is called within ConfigManager now, which is good.

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
        version="Crypto Portfolio Tracker v2.0.0" # Ensure this matches your actual version if it changes
    )

    return parser


def show_interactive_menu():
    """Display interactive menu and get user choice"""
    print("\n" + "="*50)
    print("🚀 Crypto Portfolio Tracker v2.0")
    print("="*50)
    print("1. 🔄 Full Sync & Analysis (Recommended)")
    print("2. 📊 Quick Portfolio Summary")
    print("3. 📋 Export Reports Only")
    print("4. 📈 Generate Charts Only")
    print("5. 💾 Export Data Backup")
    print("6. ⚙️  View Configuration")
    print("7. 🧹 Clean Old Data")
    print("8. ⚖️  View Rebalance Suggestions (Technical Strategy)") # Updated text
    print("9. 🔧 Test API Connections")
    print("10. ❌ Exit")
    print("="*50)

    while True:
        try:
            choice = input("\nSelect option (1-10): ").strip()
            if choice in [str(i) for i in range(1, 11)]:
                return int(choice)
            else:
                print("❌ Invalid choice. Please select 1-10.")
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
                    tracker.save_snapshot(metrics) # <-- ADD THIS LINE
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

            elif choice == 4:
                print("\n📈 Generating charts...")
                metrics = tracker.calculate_portfolio_metrics()
                if "error" not in metrics:
                    tracker.create_portfolio_charts(metrics)
                    print("\n✅ Charts generated (if enabled and data available).")
                else:
                    print(f"❌ Could not generate metrics for charts: {metrics['error']}")
                input("Press Enter to continue...")

            elif choice == 5:
                print("\n💾 Exporting data backup (CSV)...")
                tracker.export_csv_backup()
                input("\n✅ CSV Backup completed. Press Enter to continue...")

            elif choice == 6:
                print("\n⚙  Current Configuration:")
                tracker.print_configuration()
                input("\nPress Enter to continue...")

            elif choice == 7:
                print("\n🧹 Cleaning old data...")
                tracker.cleanup_old_data()
                input("\n✅ Cleanup completed. Press Enter to continue...")

            elif choice == 8:
                print("\n⚖  Rebalance Suggestions (Core Portfolio - Technical Analysis)")
                suggestions_df = asyncio.run(tracker.get_core_portfolio_rebalance_suggestions_technical())

                if suggestions_df is not None and not suggestions_df.empty:
                    tracker.print_rebalance_suggestions(suggestions_df)
                elif suggestions_df is not None and suggestions_df.empty:
                    print("No rebalancing suggestions based on the current technical criteria, or no core assets found.")
                else:
                    print("Could not generate rebalance suggestions (function returned None, check logs).")
                input("\n✅ Press Enter to continue...")

            elif choice == 9:
                print("\n🔧 Testing API connections...")
                tracker.test_connections()
                input("\n✅ Test completed. Press Enter to continue...")

            elif choice == 10:
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

    config_data_for_logging = load_config(args.config)
    log_level_console = "DEBUG" if args.verbose else ("ERROR" if args.quiet else "INFO")
    setup_logging(config=config_data_for_logging, level=log_level_console)

    logger = logging.getLogger(__name__)
    logger.info("Starting Crypto Portfolio Tracker")
    if args.verbose:
        logger.debug("Verbose logging enabled.")
    if args.quiet:
        logger.info("Quiet mode enabled (console output suppressed for INFO/DEBUG).")

    try:
        config_path = args.config if args.config else None
        tracker = CryptoPortfolioTracker(config_path=config_path)

        if args.sync_only:
            logger.info("Running sync-only mode")
            # Make the call async
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
                    print("No export formats enabled or specified for export-only mode.")

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
        logger.info("Application interrupted by user")
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found or path error: {e}", exc_info=True)
        print(f"\n💥 Configuration Error: {e}")
        print("Please ensure your config file path is correct or a default_config.json exists.")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        print(f"\n💥 A fatal error occurred: {e}")
        print("Please check logs for detailed error information.")
        sys.exit(1)
    finally:
        logger.info("Application finished")


if __name__ == "__main__":
    main()
