#!/usr/bin/env python3
"""
Development entry point for the Crypto Portfolio Tracker.

This script is a convenience wrapper for running the application from the project
root during development. It requires the package to be installed in editable
mode (`uv pip sync`).

The primary entry point for the installed application is defined in
`src/crypto_portfolio_tracker/__main__.py`.
"""
from crypto_portfolio_tracker.__main__ import main

if __name__ == "__main__":
    main()
