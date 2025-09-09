# Crypto Portfolio Tracker & Rebalancing Advisor

A comprehensive, personal cryptocurrency portfolio tracking application built on a modern, high-performance Python toolkit. It connects to Binance to provide detailed analysis of your holdings, accurate P/L calculations using dual accounting perspectives, intelligent rebalancing suggestions, and advanced backtesting capabilities.

## 🚀 Key Features

### 📊 **Complete Portfolio Intelligence**
- **Comprehensive Transaction Syncing**: Automatically fetches all transaction types from your Binance account:
  - Spot Trades & P2P Trades (Fiat to crypto)
  - Deposits & Withdrawals
  - Simple Earn (Subscriptions, Redemptions, and Rewards)
  - Staking History (Subscriptions, Redemptions, and Interest)
  - Dividends, Asset Conversions, Internal Transfers, and Copy Trading
- **Dual Accounting Perspectives**:
  - **FIFO Cost Basis**: First-In, First-Out calculation for precise unrealized P/L on a per-asset basis, essential for tax-lot accounting
  - **Net Invested Capital**: Tracks true cash-in vs. cash-out to provide an absolute "Overall P/L" on your entire portfolio
- **Context-Aware Portfolio Summary**: The main summary view is split into two clear sections:
  - **Core Portfolio Holdings**: Lists only the assets you are actively rebalancing (defined in your `target_allocation`). It includes a dedicated "Core Alloc. %" column to show you the exact numbers used by the rebalancing engine
  - **Other Holdings**: Lists all other assets in your wallet, giving you a complete financial overview without cluttering your strategic view
- **Intelligent Data Processing**: Automatically de-duplicates P2P transaction records from the Binance API to ensure data integrity
- **Persistent Local Database**: All transactions stored in local SQLite database (`data/portfolio.db` or `data/testnet_portfolio.db` for testnet) for fast queries and complete historical record

### ⚡ **Advanced Trading & Analysis**
- **Intelligent Rebalancing Engine**:
  - **Accurate Calculations**: Rebalancing logic is calculated based only on the value of your core portfolio assets, not the entire wallet, leading to much more accurate and logical trade suggestions
- **Live Trading & Rebalancing Execution**: Execute trades based on analysis (optional, disabled by default):
  - **Granular Control**: Use one-by-one confirmation of each trade, or  approve all suggested trades at once.
  - **Manual Trading**: A dedicated menu option allows you to place ad-hoc BUY or SELL market orders for any asset, independent of rebalancing suggestions. Perfect for acting on news or opportunities
  - **Directional Strategy Trading**: Run technical strategies live with safety checks
- **Automated Profit-Taking**: Intelligently identifies opportunities to lock in gains when your portfolio is balanced. It uses a multi-factor scoring system to suggest selling a small portion of assets with significant unrealized profits, without touching your core position.
- **Fund Transfers**: Easily transfer assets between your Funding and Spot wallets directly from the interface.
- **Dollar Cost Averaging (DCA)**: Automated investment strategy for consistent portfolio growth:
  - **Proportional DCA**: Distribute new funds to maintain current portfolio proportions, ideal for maintaining existing allocation ratios
  - **Target-Weight DCA**: Allocate new funds to reach your target allocation percentages, perfect for gradually achieving desired portfolio balance
  - **Smart Validation**: Automatic validation of DCA amounts against available USDT balance and minimum trade requirements
  - **Flexible Execution**: Choose between manual confirmation or automated execution with comprehensive trade preview
  - **Balance Integration**: Seamlessly combines Spot and Earn wallet balances for maximum buying power
- **Dual Backtesting Engines**: Test and validate strategies with two specialized backtesters:
  - **Directional Strategy Backtester**: Evaluate entry/exit signals from technical trading strategies
  - **Rebalancing Strategy Backtester**: Simulate long-term performance of allocation-based rebalancing
- **Multi-Timeframe Technical Analysis**: Advanced `CryptoTrendAnalyzer` generates comprehensive reports:
  - Long-term (4-year), Swing (3-month), and Day (60-day) analysis
  - RSI, MACD, SMA crossovers, and momentum indicators
  - Confidence-weighted recommendations
- **Pluggable Strategy Architecture**: Easily develop custom trading strategies by extending the base `Strategy` class

### 🔧 **Robust & Performance Optimized**
- **Modern Python Toolchain**: Uses `uv` for lightning-fast dependency management and `ruff` for high-performance formatting and linting
- **Intelligent API Management**:
  - **Resilient Error Handling**: Automatically retries failed API calls with exponential backoff using `tenacity`
  - **Timestamp Synchronization**: Eliminates `recvWindow` errors by actively synchronizing client clock with Binance server
  - **Testnet Support**: Full testnet integration for safe testing without real funds
  - **CoinGecko API Integration**: Supports both free and premium CoinGecko API tiers
- **Performance Optimized**:
  - Asynchronous concurrent API calls for significantly faster data syncing
  - Persistent disk caching reduces API calls by 90% after initial sync
  - Incremental updates process only new transactions
- **Professional Reporting**: Generate reports in multiple formats:
  - Excel (`.xlsx`)
  - Mobile-optimized HTML (`.html`) with interactive elements
  - Complete CSV backups with separate files for different data types

## 📁 Project Structure

The project follows a standard `src` layout for clean, maintainable, and distributable packaging:

```
crypto-portfolio-tracker/
├── config/
│   └── .gitkeep
├── data/
│   ├── cache/
│   │   ├── backtest_data/
│   │   ├── coingecko_historical/
│   │   │   └── cache.db
│   │   ├── fiat_exchange_rates/
│   │   │   └── cache.db
│   │   ├── rebalancing_presets/
│   │   └── yfinance_historical/
│   │       └── cache.db
│   ├── db_backups/
│   │   ├── portfolio.db.20250628_120358.bak
│   │   └── ... (automated backups)
│   ├── exports/
│   │   ├── charts/
│   │   ├── portfolio_report_*.xlsx
│   │   ├── tax_report_*.xlsx
│   │   ├── holdings_backup_*.csv
│   │   ├── transactions_backup_*.csv
│   │   └── trend_report_*.html
│   ├── coingecko_mappings.json
│   ├── connection_state.json
│   ├── portfolio.db
│   ├── strategy_state.json
│   └── testnet_portfolio.db
├── logs/
│   └── portfolio_tracker.log
├── src/
│   └── crypto_portfolio_tracker/
│       ├── dashboard/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── backtest_page.py
│       │   ├── components/
│       │   │   ├── __init__.py
│       │   │   ├── trading_status_banner.py
│       │   │   └── transfer_widget.py
│       │   ├── database_page.py
│       │   ├── dca_page.py
│       │   ├── home_page.py
│       │   ├── main_dashboard.py
│       │   ├── market_page.py
│       │   ├── rebalancing_page.py
│       │   ├── settings_page.py
│       │   ├── trading_page.py
│       │   ├── ui_controller.py
│       │   └── utils.py
│       ├── templates/
│       │   ├── report_template.html
│       │   ├── tax_report_template.html
│       │   └── trend_report_template.html
│       ├── __init__.py
│       ├── __main__.py
│       ├── binance_fetcher.py
│       ├── check_deps.py
│       ├── config.py
│       ├── crypto_trend_analyzer.py
│       ├── dashboard_launcher.py
│       ├── data_manager.py
│       ├── data_synchronizer.py
│       ├── database.py
│       ├── dca_manager.py
│       ├── exceptions.py
│       ├── exporters.py
│       ├── models.py
│       ├── portfolio_analyzer.py
│       ├── portfolio_tracker.py
│       ├── price_enricher.py
│       ├── profit_taking_logic.py
│       ├── rebalancing_backtester.py
│       ├── rebalancing_logic.py
│       ├── report_generator.py
│       ├── strategy_backtester.py
│       ├── symbol_mapper.py
│       ├── trade_executor.py
│       ├── trading_strategies.py
│       ├── utils.py
│       └── visualizations.py
├── tests/
│   ├── conftest.py
│   ├── test_api_integration.py
│   ├── test_binance_fetcher.py
│   ├── test_database.py
│   ├── test_portfolio_tracker.py
│   ├── test_portfolio_tracker_comprehensive.py
│   └── test_price_enricher.py
├── default_config.json.example
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) installed (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Binance account with API access
- Git (optional but recommended)

### Quick Start

1. **Clone the Repository:**
```bash
git clone https://github.com/Onehand-Coding/crypto-portfolio-tracker.git
cd crypto-portfolio-tracker
```

2. **Create and Activate Virtual Environment (with `uv`):**
```bash
uv sync
```

3. **Install Dependencies:**
```bash
uv sync --extra dev
```

4. **Set Up Environment Variables:**
```bash
cp .env.example .env
```

Edit the `.env` file with your API credentials:
```env
# Main Production API Keys
MAIN_API_KEY=your_binance_api_key_here
MAIN_API_SECRET=your_binance_api_secret_here

# Optional: Testnet API Keys (recommended for testing)
TESTNET_API_KEY=testnet_api_key_here
TESTNET_API_SECRET=testnet_api_secret_here
# CoinGecko API (recommended for higher rate limits)
COINGECKO_API_KEY=your_coingecko_api_key_here
```

5. **Configure Your Portfolio and Testnet Mode:**

```bash
cp default_config.json.example config/default_config.json
```

Edit `config/default_config.json` to set your target allocation and preferences. To enable testnet mode, set:
```json
{
  "testnet_mode": true,
  "target_allocation": {
    "BTC": 0.35,
    "ETH": 0.20,
    "SOL": 0.12,
    "RENDER": 0.08,
    "TAO": 0.08,
    "AVAX": 0.06,
    "LINK": 0.06,
    "ONDO": 0.05
  },
  "asset_classes": {
    "majors": ["BTC", "ETH"]
  },
  "rebalance_technical": {
    "majors": {
      "allocation_drift_threshold_pct": 3.0,
      "sell_percentage_multiplier": 0.25,
      "buy_amount_multiplier": 0.75
    },
    "alts": {
      "allocation_drift_threshold_pct": 7.0,
      "sell_percentage_multiplier": 0.5,
      "buy_amount_multiplier": 1.0
    }
  },
  "trend_analyzer": {
    "cryptocurrencies": [],
    "timeframe_settings": {
      "long_term": {
        "period": "4y",
        "sma_short_window": 50,
        "sma_long_window": 200
      },
      "swing": {
        "period": "3mo",
        "sma_short_window": 10,
        "sma_long_window": 30
      },
      "day": {
        "period": "60d",
        "sma_short_window": 5,
        "sma_long_window": 15
      }
    },
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70
  }
}
```

### Advanced Rebalancing Configuration

Configure different thresholds for major coins vs altcoins:

```json
{
  "asset_classes": {
    "majors": ["BTC", "ETH"]
  },
  "rebalance_technical": {
    "majors": {
      "allocation_drift_threshold_pct": 3.0,
      "sell_percentage_multiplier": 0.25,
      "buy_amount_multiplier": 0.75
    },
    "alts": {
      "allocation_drift_threshold_pct": 7.0,
      "sell_percentage_multiplier": 0.5,
      "buy_amount_multiplier": 1.0
    }
  }
}
```

### Profit-Taking Configuration

The automated profit-taking system is triggered only when your portfolio is balanced (i.e., no rebalancing actions are suggested). It uses a multi-factor score to find the best opportunities to lock in gains without selling your core position.

```json
{
  "profit_taking": {
    "enabled": true,
    "min_opportunity_score": 60,
    "min_unrealized_gain_pct": 15.0,
    "min_unrealized_gain_usd": 10.0,
    "max_gain_take_pct": 50,
    "default_take_percentage": 30
  }
}
```
- `enabled`: Turns the entire feature on or off.
- `min_opportunity_score`: The minimum score (0-100) an asset needs to be considered for profit-taking.
- `min_unrealized_gain_pct`: The minimum unrealized gain percentage required.
- `min_unrealized_gain_usd`: The minimum unrealized gain in USD required.
- `max_gain_take_pct`: The maximum percentage of the *gains* that can be sold in a single action.
- `default_take_percentage`: The default percentage of gains to sell when an opportunity is executed.

### Technical Analysis Settings

Configure multi-timeframe analysis parameters:

```json
{
  "trend_analyzer": {
    "cryptocurrencies": [],
    "timeframe_settings": {
      "long_term": {
        "period": "4y",
        "sma_short_window": 50,
        "sma_long_window": 200
      },
      "swing": {
        "period": "3mo",
        "sma_short_window": 10,
        "sma_long_window": 30
      },
      "day": {
        "period": "60d",
        "sma_short_window": 5,
        "sma_long_window": 15
      }
    },
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70
  }
}
```

## 🎯 Usage

The application offers both a modern web interface and a command-line interface for different use cases.

### 🌐 Web UI (Recommended)

The **Streamlit web interface** provides an intuitive, feature-rich experience for portfolio management:

```bash
uv run track-portfolio-web
```

**Features:**
- 🏠 Real-time portfolio dashboard with interactive charts
- 📈 Market trends and technical analysis with visual indicators
- ⚖️ Rebalancing suggestions with interactive controls
- 🔀 Manual and automated trading interfaces
- 🧪 Backtesting with interactive parameter controls
- 🗄️ Data management tools
- ⚙️ Settings and configuration management
- 🎨 Professional, responsive design

The web UI runs on `http://localhost:8502` and is the **recommended interface** for daily portfolio management.

### 💻 CLI Interface (For Debugging & Automation)

The command-line interface is useful for debugging, automation, and quick checks:

```bash
# Interactive CLI
uv run track-portfolio

# Command-line mode
track-portfolio [options]
# -v, --verbose   Enable DEBUG logging
# -q, --quiet     Suppress output except errors

# Developer entry point
python main.py
```

**CLI Menu Options:**
```
==================================================
🪙 Crypto Portfolio Tracker v2.1.0
==================================================
1. 🔄 Full Sync & Analysis
2. 💰 Quick Portfolio Summary
3. 📈 View Trends
4. ⚖️  Rebalance
5. 💸 Dollar Cost Averaging (DCA)
6. 🔀 Trade
7. 💵 Transfer Funds
8. 🧪 Backtest
9. 📋 Reports
10. 📊 Charts
11. 🗄️  Database
12. 🧹 Data Cleanup
13. ⚙️  View Configuration
14. 🔧 Test Connections
15. ❌ Exit
==================================================
Select option (1-15):  
```

**Use Cases for CLI:**
- 🔧 Debugging API connections and data sync issues
- 🤖 Automation scripts and scheduled tasks
- ⚡ Quick portfolio summaries and status checks
- 🧪 Testing new features and configurations

### Recommended Workflow

1. **First Setup**: Set `testnet_mode=true` and use the web UI for initial testing
2. **Production Setup**: Set `testnet_mode=false` for real portfolio tracking
3. **Daily Monitoring**: Use the web UI for portfolio overview and management
4. **Technical Analysis**: Use web UI's trend analysis for multi-timeframe market insights
5. **Strategic Planning**: Use web UI's rebalancing interface for allocation management
6. **Strategy Testing**: Use web UI's backtesting tools before live trading
7. **Live Execution**: Use web UI's trading interface for manual and automated trades
8. **Debugging**: Use CLI when troubleshooting API or configuration issues

## 📊 Understanding the Output

### Context-Aware Portfolio Summary

The portfolio summary provides two distinct views of your performance with enhanced clarity:

```
==================================================
🪙 Crypto Portfolio Tracker v2.1.0
==================================================
1. 🔄 Full Sync & Analysis
2. 💰 Quick Portfolio Summary
3. 📈 View Trends
4. ⚖️  Rebalance
5. 💸 Dollar Cost Averaging (DCA)
6. 🔀 Trade
7. 💵 Transfer Funds
8. 🧪 Backtest
9. 📋 Reports
10. 📊 Charts
11. 🗄️  Database
12. 🧹 Data Cleanup
13. ⚙️  View Configuration
14. 🔧 Test Connections
15. ❌ Exit
==================================================
Select option (1-15): 1

🔄 Running full sync and analysis...

===================================================================================================================
                                                📊 PORTFOLIO SUMMARY
===================================================================================================================
Timestamp:                   2025-08-06 12:21:09
Database:                    portfolio.db
-------------------------------------------------------------------------------------------------------------------
TOTAL PORTFOLIO VALUE:       $213.52
-------------------------------------------------------------------------------------------------------------------
Wallet Value Breakdown:
  Spot & Earn Value:         $145.16
  Futures Wallet Value:      $0.00
  Funding Wallet Value:      $68.36
-------------------------------------------------------------------------------------------------------------------
Performance vs. Invested capital:
Total Invested Capital:      $190.54
Overall P/L:                 $22.98 (12.06%)
-------------------------------------------------------------------------------------------------------------------
Performance vs. Rolling cost basis (Spot/Earn only):
Total Cost Basis (FIFO):     $131.30
Unrealized P/L (FIFO):     $13.86 (10.55%)

                          --- ️ All Holdings (Alloc. = % of spot/earn portfolio value) ---
Asset       Total Qty       Spot Qty        Earn Qty        Value (USD)     Cost Basis      P/L (USD)       Alloc.
----------------------------------------------------------------------------------------------------------------------
BTC         0.00046946      0               0.00046946      $53.25          $42.77          $10.48          36.68%
ETH         0.00992765      0               0.00992765      $35.50          $26.37          $9.13           24.46%
SOL         0.08647399      0               0.08647399      $13.97          $14.98          $-1.01          9.62%
RENDER      3.64635         3.64635         0               $12.54          $16.66          $-4.11          8.64%
LINK        0.49951961      0               0.49951961      $8.03           $6.99           $1.04           5.53%
TAO         0.02380511      0               0.02380511      $7.99           $10.68          $-2.69          5.50%
AVAX        0.33997039      0               0.33997039      $7.34           $6.91           $0.44           5.06%
ONDO        7.1967374       0               7.1967374       $6.49           $5.95           $0.54           4.47%
BNB         4.59e-05        0               4.59e-05        $0.03           $0.00           $0.03           0.02%
W           0.04832963      0               0.04832963      $0.00           $0.00           $0.00           0.00%
PIXEL       0.07425932      0               0.07425932      $0.00           $0.00           $0.00           0.00%

                            ---  Core Holdings (Alloc. = % of core portfolio value) ---
Asset       Total Qty       Spot Qty        Earn Qty        Value (USD)     Cost Basis      P/L (USD)       Alloc.
----------------------------------------------------------------------------------------------------------------------
BTC         0.00046946      0               0.00046946      $53.25          $42.77          $10.48          36.69%
ETH         0.00992765      0               0.00992765      $35.50          $26.37          $9.13           24.47%
SOL         0.08647399      0               0.08647399      $13.97          $14.98          $-1.01          9.63%
RENDER      3.64635         3.64635         0               $12.54          $16.66          $-4.11          8.64%
LINK        0.49951961      0               0.49951961      $8.03           $6.99           $1.04           5.53%
TAO         0.02380511      0               0.02380511      $7.99           $10.68          $-2.69          5.50%
AVAX        0.33997039      0               0.33997039      $7.34           $6.91           $0.44           5.06%
ONDO        7.1967374       0               7.1967374       $6.49           $5.95           $0.54           4.47%

                         --- 📈 Other Holdings (Alloc. = % of spot/earn portfolio value) ---
Asset       Total Qty       Spot Qty        Earn Qty        Value (USD)     Cost Basis      P/L (USD)       Alloc.
----------------------------------------------------------------------------------------------------------------------
BNB         4.59e-05        0               4.59e-05        $0.03           $0.00           $0.03           0.02%
W           0.04832963      0               0.04832963      $0.00           $0.00           $0.00           0.00%
PIXEL       0.07425932      0               0.07425932      $0.00           $0.00           $0.00           0.00%
======================================================================================================================

                                           --- Futures Wallet Summary ---
                                                 No balances found.

                                           --- Funding Wallet Summary ---
Asset           Balance
------------------------------------
USDT            68.37
📸 Save snapshot? (y/n):
```

**Key Metrics Explained:**
- **Core Alloc. %**: An asset's percentage of the core portfolio's value. This is the number used for rebalancing calculations
- **Total Alloc. %**: An asset's percentage of your total wallet value
- **Invested Capital Performance**: Your true performance against actual cash invested
- **FIFO Cost Basis Performance**: Tax-relevant unrealized P/L based on First-In, First-Out accounting
- **Database Indicator**: Shows whether you're using testnet or production data

### Enhanced Rebalancing Suggestions

The rebalancing view now provides full context for your decisions:

```
========================================================================================
⚖️ REBALANCING SUGGESTIONS (Multi-Timeframe Analysis)
========================================================================================
----------------------------------------------------------------------------------------
💰 Core Portfolio Value: $177,000.40
💰 Available USDT (Spot + Earn): $85,611.26
----------------------------------------------------------------------------------------
🔴 ETH    | Signal: SELL
   Allocation: 23.70% (Target: 20.0%) | Drift: 3.70 pts | Value: $29.49
   Price: $2,526.74    | Support: $2,387.61 | Resistance: $2,877.63
   Long-Term Trend: Golden Cross, Price > SMA200, Neutral RSI (40-60)
   Action: Sell ~$2.30 worth, which is 0.00091149451 ETH
------------------------------------------------------
🔴 BTC    | Signal: SELL
   Allocation: 39.53% (Target: 35.0%) | Drift: 4.53 pts | Value: $49.18
   Price: $105,022.07  | Support: $100,436.88 | Resistance: $111,970.17
   Long-Term Trend: Golden Cross, Price > SMA200
   Action: Sell ~$4.22 worth, which is 4.0213452e-05 BTC
------------------------------------------------------
🟡 RENDER | Signal: HOLD
   Allocation: 5.35% (Target: 8.0%) | Drift: -2.65 pts | Value: $6.65
   Price: $3.28        | Support: $3.16 | Resistance: $5.34
   Long-Term Trend: Golden Cross, Price < SMA200, Neutral RSI (40-60)
   Action: Hold: Allocation is within tolerance.
------------------------------------------------------
========================================================================================
Verifying balances in Spot and Earn wallets...

================================================================================
🔴🔴🔴 WARNING: Live Trading is ENABLED. 🔴🔴🔴
================================================================================
🚨 PROPOSED TRADES - PLEASE REVIEW CAREFULLY 🚨
================================================================================
Symbol Signal                       Suggested Action Detail
   ETH   SELL Sell ~$2.30 worth, which is 0.00091149451 ETH
   BTC   SELL Sell ~$4.22 worth, which is 4.0213452e-05 BTC
================================================================================
Type 'EXECUTE ALL' or 'EXECUTE' for one-by-one confirmation:
```

**Enhanced Features:**
- **Core Portfolio Value**: The total value of only the assets being considered for rebalancing
- **Available USDT**: Your total buying power from both Spot and Earn wallets
- **Granular Control**: Choose `EXECUTE ALL` for batch confirmation or `EXECUTE` for one-by-one approval
- **Smart Prompts**: Execution prompt is hidden if all signals are HOLD

### Multi-Timeframe Technical Analysis
```
================================================================================
📈 TREND ANALYSIS REPORT (LONG_TERM)
Timestamp: 2025-06-19T13:28:29.097339
================================================================================

--- 🌍 Market Summary ---
Coins Analyzed: 8
Most Common Condition: Neutral RSI (40-60)
Bullish Coins: 5 | Bearish Coins: 3

--- 🎯 Benchmark Analysis: BTC-USD ---
  Price: $105,150.34 (+194.55%) | RSI: 62.64
  Support: $74,436.68 | Resistance: $111,970.17
  Active Conditions: Golden Cross, Price > SMA200

--- 🪙 Coin-by-Coin Analysis ---

➡️ ETH-USD
  Price: $2,530.84 (+12.66%) | RSI: 51.93
  Support: $1,386.80 | Resistance: $4,106.96
  Active Conditions: Golden Cross, Price > SMA200, Neutral RSI (40-60)

➡️ SOL-USD
  Price: $146.99 (+316.14%) | RSI: 45.84
  Support: $96.59 | Resistance: $294.33
  Active Conditions: Golden Cross, Price > SMA200, Neutral RSI (40-60)

================================================================================
```

## 🔧 Development & Tooling

This project uses modern, high-performance Python tools for development:

### Using `uv` for Dependencies

```bash
# Install all dependencies including dev tools
uv sync --extra dev

# Add a new package
# Edit pyproject.toml to add the package under [project.dependencies] or [project.optional-dependencies.dev], then:
uv sync --extra dev

# Update all packages
uv sync --extra dev --upgrade
```

### Using `ruff` for Code Quality

```bash
# Format all files
ruff format .

# Check for linting issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest tests/ --cov=src/crypto_portfolio_tracker

# Run specific test file
uv run pytest tests/test_portfolio_tracker.py
```

## 🐛 Troubleshooting

### Common Issues

#### Testnet vs Production Confusion
- **Check Database**: Look for "TESTNET MODE" indicator in portfolio summary
- **Verify Settings**: Ensure `testnet_mode` is set correctly in `default_config.json`
- **Separate Databases**: Testnet uses `testnet_portfolio.db`, production uses `portfolio.db`

#### API Connection Failed
```bash
# Test API connections through menu
track-portfolio  # Choose option 12 (Test API Connections)

# Check configuration
track-portfolio  # Choose option 11 (View Configuration)

# Verify environment variables
cat .env
```

#### Rate Limiting Issues
- **Use CoinGecko API Key**: Add `COINGECKO_API_KEY` to `.env` for higher rate limits
- **Check Rate Limits**: Monitor API usage in your exchange account
- **Increase Delays**: Adjust request timing in configuration

#### Empty Portfolio or No Data
- **Run Full Sync**: Choose option 1 from the interactive menu
- **Check API Permissions**: Ensure "Enable Reading" is enabled on your API key
- **Verify Holdings**: Confirm you have assets in Spot or Earn wallets
- **Review Logs**: Check `logs/portfolio_tracker.log` for detailed error information

### Debug Mode

Enable comprehensive logging:
```bash
# Enable verbose logging
track-portfolio --verbose

# Set environment variable
export LOG_LEVEL=DEBUG
track-portfolio

# View live logs
tail -f logs/portfolio_tracker.log
```

### Database Issues

```bash
# Check database integrity
sqlite3 data/portfolio.db ".schema"

# For testnet database
sqlite3 data/testnet_portfolio.db ".schema"

# Manual cleanup (use with caution)
track-portfolio  # Choose option 10 (Data Cleanup)

# Complete reset (will require full re-sync)
rm data/portfolio.db data/testnet_portfolio.db
```

## 🔒 Security Best Practices

### API Security
1. **Start with Testnet**: Always test with `testnet_mode=true` first
2. **Use Read-Only Keys**: Enable only "Enable Reading" permission unless live trading is necessary
3. **Enable IP Whitelisting**: Restrict API access to your IP address
4. **Rotate API Keys Regularly**: Monthly rotation recommended
5. **Monitor API Usage**: Check usage in your exchange account settings

### Local Security
1. **Never Commit `.env`**: File is in `.gitignore` but double-check
2. **Restrict File Permissions**: `chmod 600 .env` on Linux/macOS
3. **Use Encrypted Storage**: Encrypt sensitive configuration files
4. **Regular Backups**: Backup database and configuration regularly
5. **Keep Software Updated**: Update dependencies regularly with `uv sync --upgrade`

### Trading Security
1. **Always Start with Testnet**: Test all strategies on testnet first
2. **Use Dry-Run Mode**: Keep `live_trading_enabled: false` initially
3. **Set Position Limits**: Configure maximum trade amounts
4. **Monitor Actively**: Watch automated trades closely
5. **Have Kill Switch**: Know how to stop automated trading immediately

## 📈 Performance Metrics

Optimized performance characteristics:
- **Initial Sync**: ~5-10 minutes (depending on transaction history)
- **Daily Updates**: ~10-15 seconds with caching
- **Report Generation**: ~5-10 seconds including charts
- **API Efficiency**: 90% fewer requests after initial sync
- **Database Performance**: Sub-second queries with proper indexing
- **Memory Usage**: ~50-100MB during typical operation

## ⚠️ Important Disclaimers

### Financial Risk Warning
This software is for **educational and informational purposes only** and does **NOT** constitute financial advice. Cryptocurrency investments carry significant risk and can result in substantial losses.

### Testnet vs Production
- **Always test on testnet first** before using real funds
- **Testnet data is separate** from production data
- **Set `testnet_mode=false`** only when ready for real trading
- **Double-check database indicator** in portfolio summaries

### Strategy Performance Warning
- **Default strategies are educational**: Not optimized for profit
- **Backtesting may show underperformance**: Many strategies underperform buy-and-hold
- **Past performance ≠ future results**: Historical data doesn't guarantee future success
- **Do your own research**: Optimize strategies for your risk tolerance

### Live Trading Risks
- **Disabled by default**: Live trading requires explicit enablement
- **Potential for losses**: Bugs or market conditions can cause financial loss
- **Start small**: Test with minimal amounts first
- **Active monitoring required**: Don't leave automated systems unattended

## ⚠️ Data Limitations & Important Notes

### Transaction Data Limitations
**⚠️ CRITICAL: Transaction History Limited to 90 Days**
- **Binance API Restriction**: The Binance API only provides transaction data for the last 90 days
- **Impact on Calculations**: This limitation means your cost basis calculations, P/L analysis, and portfolio performance metrics may be **incomplete or inaccurate** if you have transactions older than 90 days
- **Historical Data Gap**: Transactions older than 90 days are not fetched and cannot be included in calculations
- **Recommendation**: For accurate long-term analysis, consider manually importing historical transaction data or using the export/import features to supplement missing data

### Import/Export Limitations
**⚠️ IMPORTANT: Import Functionality Limitations**
- **Binance Export Format**: Direct import of transaction exports from Binance is **not yet supported**
- **Manual Import Required**: You may need to manually format transaction data for import
- **Data Validation**: The import feature validates required columns but may not handle all Binance export formats
- **Testing Recommended**: Always test import functionality with small datasets first

### Data Completeness Impact
**⚠️ COMPUTATIONAL ACCURACY WARNING**
- **Incomplete Cost Basis**: Missing historical transactions can lead to incorrect FIFO cost basis calculations
- **Inaccurate P/L**: Portfolio performance metrics may be skewed without complete transaction history
- **Rebalancing Impact**: Incomplete data may affect rebalancing suggestions and backtesting results
- **Tax Implications**: Incomplete transaction history could lead to inaccurate tax reporting
- **Futures Trading Limitations**: Detailed futures trading history and profit/loss calculations are not fully tracked, which may affect cost basis accuracy when funds are transferred between spot and futures wallets

### Recommendations for Data Completeness
1. **Export Current Data**: Use the export feature to backup your current transaction data
2. **Manual Supplementation**: Consider manually adding historical transactions older than 90 days
3. **Regular Backups**: Export your data regularly to maintain complete records
4. **Verify Calculations**: Cross-reference portfolio calculations with your exchange records
5. **Start Fresh**: If possible, begin tracking from a point where you have complete 90-day history

### Alternative Data Sources
- **Manual Entry**: Consider manually entering critical historical transactions
- **External Tools**: Use external portfolio tracking tools for historical data analysis
- **Exchange Records**: Download and maintain separate records of transactions older than 90 days
- **Tax Software**: Consider using dedicated crypto tax software for complete historical analysis

## 📁 Output Files

All files are organized in the `data/` directory:

### Database Files
- **Production**: `data/portfolio.db` (SQLite with complete transaction history)
- **Testnet**: `data/testnet_portfolio.db` (Separate testnet database)

### Cache and Performance
- **API Cache**: `data/cache/coingecko_historical/` (Price data caching)
- **Backtest Cache**: `data/cache/backtest_data/` (Strategy backtesting cache)
- **Symbol Mappings**: `data/coingecko_mappings.json` (CoinGecko ID mappings)

### Reports and Exports
- **Excel Reports**: `data/exports/portfolio_report_YYYYMMDD_HHMMSS.xlsx`
- **HTML Reports**: `data/exports/portfolio_report_YYYYMMDD_HHMMSS.html`
- **Data Backups**: `data/exports/transactions_backup_YYYYMMDD_HHMMSS.csv`
- **Holdings Backups**: `data/exports/holdings_backup_YYYYMMDD_HHMMSS.csv`
- **Charts**: `data/exports/charts*.png`

### Analysis and Logs
- **Technical Analysis**: `data/trend_reports/` (Detailed market analysis JSON files)
- **Strategy State**: `data/strategy_state.json` (Trading strategy persistence)
- **Application Logs**: `logs/portfolio_tracker.log` (Rotating logs with 5 backup files)

## 🚀 Roadmap

### Next Major Features
- [ ] Real-time price alerts and notifications
- [ ] Manual transaction import from CSV files
- [ ] Additional exchange integrations
- [ ] Multi-portfolio management for different strategies
- [ ] Advanced options and futures tracking
- [ ] Integrate a gem sniper tool for new token detection and analysis (BNB Chain, honeypot check, liquidity, etc.)

### Web UI Future
- The current web UI is built with Streamlit for rapid development and ease of use.
- **Planned:** Upgrade to a React-based frontend for a more modern, responsive, and customizable user experience.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- Binance API: Subject to Binance Terms of Service
- CoinGecko API: Subject to CoinGecko Terms of Service
- All Python dependencies: See individual package licenses in `pyproject.toml`

## 🙏 Acknowledgments

- **Binance** for providing comprehensive API access and testnet environment
- **CoinGecko** for reliable cryptocurrency data and generous free tier
- **uv team** for revolutionary Python dependency management
- **Ruff team** for high-performance Python tooling
- **Python community** for excellent libraries and tools
- **Contributors** who help improve the project
- **Users** who provide feedback and bug reports

---

**Built with ❤️ for the crypto community**

*Remember: This tool is designed to help you make informed decisions, but always do your own research and never invest more than you can afford to lose. Start with testnet mode to safely explore all features.*
