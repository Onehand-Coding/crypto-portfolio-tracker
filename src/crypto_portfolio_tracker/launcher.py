#!/usr/bin/env python3
"""
Unified launcher for Crypto Portfolio Tracker.

Usage:
    cpt cli [-q|-v]  - Run CLI interface (default)
    cpt web [-q|-v]  - Run web UI
    cpt              - Same as 'cpt cli'
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Portfolio Tracker - Unified Launcher"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["cli", "web"],
        default="cli",
        help="Run mode: 'cli' for terminal interface, 'web' for browser UI (default: cli)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_const",
        const="DEBUG",
        dest="loglevel",
        help="Enable verbose DEBUG logging"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_const",
        const="WARNING",
        dest="loglevel",
        help="Enable quiet logging (warnings only)"
    )
    
    args = parser.parse_args()
    
    # Build the command to run
    if args.mode == "web":
        from crypto_portfolio_tracker.dashboard_launcher import main as web_main
        web_main()
    else:
        # CLI mode - import and run the main function with args
        from crypto_portfolio_tracker.__main__ import amain
        import asyncio

        # Set sys.argv for the CLI to pick up the log level args
        if args.loglevel:
            if args.loglevel == "DEBUG":
                sys.argv = [sys.argv[0], "-v"]
            elif args.loglevel == "WARNING":
                sys.argv = [sys.argv[0], "-q"]
        else:
            sys.argv = [sys.argv[0]]

        try:
            asyncio.run(amain())
        except KeyboardInterrupt:
            print("\n👋 Exiting due to user interruption.")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
