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

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from portfolio_tracker import CryptoPortfolioTracker
from config import setup_logging, load_config


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
  python main.py                          # Interactive mode
  python main.py --sync-only              # Sync data only
  python main.py --export-only            # Export existing data
  python main.py --format excel           # Export to Excel only
  python main.py --verbose                # Enable debug logging
  python main.py --config my_config.json  # Use custom config
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
        version="Crypto Portfolio Tracker v2.0.0"
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
    print("8. ⚖️  View Rebalance Suggestions (Cost Basis)")
    print("9. 🔧 Test API Connections")
    print("10. ❌ Exit")
    print("="*50)

    while True:
        try:
            choice = input("\nSelect option (1-10): ").strip()
            if choice in [str(i) for i in range(1, 11)]:
                return int(choice)
            else:
                print("❌ Invalid choice. Please select 1-9.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def run_interactive_mode(tracker):
    """Run the application in interactive mode"""
    while True:
        # clear_screen()
        choice = show_interactive_menu()

        try:
            if choice == 1:
                print("\n🔄 Running full sync and analysis...")
                metrics = tracker.run_full_sync()
                input("\n✅ Press Enter to continue...")

            elif choice == 2:
                print("\n📊 Generating quick portfolio summary...")
                metrics = tracker.calculate_portfolio_metrics()
                tracker.print_portfolio_summary(metrics)
                input("\n✅ Press Enter to continue...")

            elif choice == 3:
                print("\n📋 Exporting reports...")
                metrics = tracker.calculate_portfolio_metrics()
                tracker.export_to_excel(metrics)
                tracker.export_to_html(metrics)
                input("\n✅ Reports exported. Press Enter to continue...")

            elif choice == 4:
                print("\n📈 Generating charts...")
                metrics = tracker.calculate_portfolio_metrics()
                tracker.create_portfolio_charts(metrics)
                input("\n✅ Charts created. Press Enter to continue...")

            elif choice == 5:
                print("\n💾 Exporting data backup...")
                tracker.export_csv_backup()
                input("\n✅ Backup completed. Press Enter to continue...")

            elif choice == 6:
                print("\n⚙️ Current Configuration:")
                tracker.print_configuration()
                input("\n✅ Press Enter to continue...")

            elif choice == 7:
                print("\n🧹 Cleaning old data...")
                tracker.cleanup_old_data()
                input("\n✅ Cleanup completed. Press Enter to continue...")

            elif choice == 8:
                print("\n⚖️  Generating rebalance suggestions (Cost Basis)...")
                suggestions_df = tracker.get_rebalance_suggestions_by_cost()
                if suggestions_df is not None:
                    print(suggestions_df.to_string(index=False))
                else:
                    print("Could not generate suggestions (Check logs - cost basis likely missing).")
                input("\n✅ Press Enter to continue...")

            elif choice == 9:
                print("\n🔧 Testing API connections...")
                tracker.test_connections()
                input("\n✅ Test completed. Press Enter to continue...")

            elif choice == 10:
                print("\n👋 Thank you for using Crypto Portfolio Tracker!")
                break

        except KeyboardInterrupt:
            print("\n\n⚠️ Operation interrupted by user.")
            continue
        except Exception as e:
            logging.error(f"Error in interactive mode: {e}")
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")
            clear_screen()


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging based on arguments
    log_level = "DEBUG" if args.verbose else ("ERROR" if args.quiet else "INFO")
    setup_logging(level=log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting Crypto Portfolio Tracker")

    try:
        # Initialize tracker
        config_path = args.config if args.config else None
        tracker = CryptoPortfolioTracker(config_path=config_path)

        # Handle command line arguments
        if args.sync_only:
            logger.info("Running sync-only mode")
            tracker.sync_data()
            print("✅ Data synchronization completed")

        elif args.export_only:
            logger.info("Running export-only mode")
            metrics = tracker.calculate_portfolio_metrics()

            if args.format == "excel" or args.format == "all":
                tracker.export_to_excel(metrics)
            if args.format == "html" or args.format == "all":
                tracker.export_to_html(metrics)
            if args.format == "csv" or args.format == "all":
                tracker.export_csv_backup()

            print("✅ Export completed")

        elif args.charts_only:
            logger.info("Running charts-only mode")
            metrics = tracker.calculate_portfolio_metrics()
            tracker.create_portfolio_charts(metrics)
            print("✅ Charts generated")

        else:
            # Interactive mode
            logger.info("Starting interactive mode")
            run_interactive_mode(tracker)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        logger.info("Application interrupted by user")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n💥 Fatal error: {e}")
        print("Check logs for detailed error information.")
        sys.exit(1)

    finally:
        logger.info("Application finished")


if __name__ == "__main__":
    main()
