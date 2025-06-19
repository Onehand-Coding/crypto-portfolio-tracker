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
- **Intelligent Data Processing**: Automatically de-duplicates P2P transaction records from the Binance API to ensure data integrity
- **Persistent Local Database**: All transactions stored in local SQLite database (`data/portfolio.db` or `data/testnet_portfolio.db` for testnet) for fast queries and complete historical record

### ⚡ **Advanced Trading & Analysis**
- **Dual Backtesting Engines**: Test and validate strategies with two specialized backtesters:
  - **Directional Strategy Backtester**: Evaluate entry/exit signals from technical trading strategies
  - **Rebalancing Strategy Backtester**: Simulate long-term performance of allocation-based rebalancing
- **Live Trading & Rebalancing Execution**: Execute trades based on analysis (optional, disabled by default):
  - **Automated Rebalancing**: Execute trades to align portfolio with target allocations
  - **Directional Strategy Trading**: Run technical strategies live with safety checks
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
  - Excel (`.xlsx`) with embedded charts and password protection options
  - Mobile-optimized HTML (`.html`) with interactive elements
  - Complete CSV backups with separate files for different data types

## 📁 Project Structure

The project follows a standard `src` layout for clean, maintainable, and distributable packaging:

```
crypto-portfolio-tracker/
├── config/
│   └── default_config.json        # Default configuration settings
├── data/
│   ├── portfolio.db               # SQLite database (production)
│   ├── testnet_portfolio.db       # SQLite database (testnet)
│   ├── cache/                     # API response cache
│   │   ├── backtest_data/         # Cached backtesting data
│   │   └── coingecko_historical/  # CoinGecko price cache
│   ├── exports/                   # Generated reports and backups
│   └── trend_reports/             # Technical analysis reports
├── logs/
│   └── portfolio_tracker.log      # Application logs with rotation
├── src/
│   └── crypto_portfolio_tracker/
│       ├── templates/
│       │   └── report_template.html
│       ├── __init__.py
│       ├── __main__.py            # Main application entry point
│       ├── portfolio_tracker.py   # Core orchestration logic
│       ├── binance_fetcher.py     # Binance API data fetching
│       ├── price_enricher.py      # Price data enrichment
│       ├── database.py            # SQLite database operations
│       ├── config.py              # Configuration management
│       ├── exporters.py           # Export functionality (Excel/HTML/CSV)
│       ├── visualizations.py      # Chart and graph generation
│       ├── crypto_trend_analyzer.py # Multi-timeframe technical analysis
│       ├── trading_strategies.py  # Pluggable trading strategies
│       ├── strategy_backtester.py # Backtester for directional strategies
│       ├── rebalancing_backtester.py # Backtester for rebalancing strategies
│       ├── rebalancing_logic.py   # Rebalancing calculation logic
│       └── symbol_mapper.py       # Symbol mapping utilities
├── tests/                         # Unit tests
├── main.py                        # Developer convenience entry point
├── pyproject.toml                 # Project definition and dependencies
├── default_config.json.example    # Configuration template
└── README.md                      # This documentation
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
uv venv
source .venv/bin/activate  # Linux/macOS

.venv\Scripts\activate     # Windows
```

3. **Install Dependencies:**
```bash
uv sync --dev
```

4. **Set Up Environment Variables:**
```bash
cp config/.env.example .env
```

Edit the `.env` file with your API credentials:
```env
# Main Production API Keys
MAIN_API_KEY=your_binance_api_key_here
MAIN_API_SECRET=your_binance_api_secret_here

# Optional: Testnet API Keys (recommended for testing)
TESTNET_API_KEY=testnet_api_key_here
TESTNET_API_SECRET=testnet_api_secret_here

# Master Testnet Switch - Set to "true" for safe testing
BINANCE_TESTNET=true

# CoinGecko API (recommended for higher rate limits)
COINGECKO_API_KEY=your_coingecko_api_key_here

# Logging Level
LOG_LEVEL=INFO
```

5. **Configure Your Portfolio Targets:**
```bash
cp default_config.json.example config/default_config.json
```

Edit `config/default_config.json` with your target allocation and preferences.

## 🔐 Binance API Setup

### Production API Setup

**CRITICAL SECURITY STEPS:**

1. Visit [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create a new API key with descriptive name
3. **For Portfolio Tracking Only**: Enable ONLY "Enable Reading" permission
4. **For Live Trading**: Enable "Enable Reading" + "Enable Spot & Margin Trading" (use with extreme caution)
5. **Add your IP address to whitelist** for enhanced security
6. Copy API Key and Secret to your `.env` file as `MAIN_API_KEY` and `MAIN_API_SECRET`

### Testnet API Setup (Recommended for Testing)

1. Visit [Binance Testnet](https://testnet.binance.vision/)
2. Create testnet account and generate API keys
3. Add testnet keys to `.env` file as `TESTNET_API_KEY` and `TESTNET_API_SECRET`
4. Set `BINANCE_TESTNET=true` in `.env` to use testnet
5. **Note**: Testnet uses a separate database (`testnet_portfolio.db`) to avoid mixing test and real data

### CoinGecko API Setup (Recommended)

1. Visit [CoinGecko API](https://www.coingecko.com/en/api) (optional but recommended)
2. Sign up for free or premium API access
3. Add your API key to `.env` file as `COINGECKO_API_KEY`
4. Higher rate limits prevent API throttling issues

## ⚙️ Configuration

### Target Portfolio Allocation

Define your desired allocation in `config/default_config.json` (percentages must sum to 1.0):

```json
{
  "target_allocation": {
    "BTC": 0.35,
    "ETH": 0.20,
    "SOL": 0.12,
    "RENDER": 0.08,
    "TAO": 0.08,
    "AVAX": 0.06,
    "LINK": 0.06,
    "ONDO": 0.05
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

The application offers a comprehensive interactive menu and command-line options for automation.

### Interactive Mode (Recommended)

For day-to-day portfolio management:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run the application
track-portfolio
# OR for development
python main.py
# OR using uv
uv run track-portfolio
```

**Interactive Menu Options:**
```
==================================================
🚀 Crypto Portfolio Tracker v2.1.0
==================================================
1. 🔄 Full Sync & Analysis (Recommended for first run)
2. 📊 Quick Portfolio Summary (Fast daily updates)
3. 📈 View Crypto Trends (Multi-timeframe analysis)
4. ⚖️  View Rebalance Suggestions (Technical analysis)
5. 🤖 Execute Rebalancing Trades (Live trading)
6. 💰 Live Trading (Directional strategies)
7. 🧪 Run Strategy Backtest (Test strategies)
8. ⚖️  Run Rebalancing Backtest (Test allocations)
9. 📋 Export Reports Only
10. 📈 Generate Charts Only
11. 💾 Export Data Backup
12. ⚙️  View Configuration
13. 🧹 Clean Old Data
14. 🔧 Test API Connections
15. ❌ Exit
==================================================
```

### Command-Line Mode

For automation, scripting, and advanced usage:

```bash
# General format
track-portfolio [options]

# Available options
-v, --verbose            # Enable detailed DEBUG logging
-q, --quiet              # Suppress console output except errors
```

**Command-Line Examples:**
```bash
# Silent background sync for automation
track-portfolio --quiet

# Verbose logging for troubleshooting
track-portfolio --verbose

# View live logs
tail -f logs/portfolio_tracker.log
```

### Recommended Workflow

1. **First Setup**: Set `BINANCE_TESTNET=true` and run option 1 (Full Sync & Analysis) to test safely
2. **Production Setup**: Set `BINANCE_TESTNET=false` for real portfolio tracking
3. **Daily Monitoring**: Option 2 (Quick Portfolio Summary) for fast portfolio updates
4. **Technical Analysis**: Option 3 (View Crypto Trends) for multi-timeframe market analysis
5. **Strategic Planning**: Option 4 (Rebalance Suggestions) for allocation-based recommendations
6. **Strategy Testing**: Options 7-8 for backtesting before live trading
7. **Live Execution**: Options 5-6 for automated rebalancing or strategy trading (use with extreme caution)

## 📊 Understanding the Output

### Dual Accounting Perspective

The portfolio summary provides two distinct views of your performance:

```
===================================================================================================================
📊 CONSOLIDATED PORTFOLIO SUMMARY (Spot + Earn)
===================================================================================================================
Timestamp:                   2025-06-19 12:09:35
Database:                    testnet_portfolio.db (TESTNET MODE)
-------------------------------------------------------------------------------------------------------------------
PERFORMANCE VS. INVESTED CAPITAL:
Total Invested Capital:      $121.22
Overall P/L:                 $4.28 (3.53%)
-------------------------------------------------------------------------------------------------------------------
PERFORMANCE VS. ROLLING COST BASIS:
Total Portfolio Value:       $125.50
Total Cost Basis (FIFO):     $129.57
Unrealized P/L (FIFO):       $-4.07 (-3.14%)
-------------------------------------------------------------------------------------------------------------------
Asset    Total Qty          Spot Qty        Earn Qty        Value (USD)     Cost Basis      P/L (USD)       Allocation
-------------------------------------------------------------------------------------------------------------------
BTC      0.00046944         0               0.00046944      $49.28          $42.73          $6.55           39.26%
ETH      0.01190199         0               0.01190199      $32.83          $31.49          $1.34           24.38%
SOL      0.08608167         0               0.08608167      $13.74          $14.91          $-1.17          10.20%
...
===================================================================================================================
```

**Key Metrics Explained:**
- **Invested Capital Performance**: Your true performance against actual cash invested
- **FIFO Cost Basis Performance**: Tax-relevant unrealized P/L based on First-In, First-Out accounting
- **Database Indicator**: Shows whether you're using testnet or production data

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

➡️ RENDER-USD
  Price: $3.30 (+471.01%) | RSI: 40.46
  Support: $2.53 | Resistance: $11.62
  Active Conditions: Golden Cross, Price < SMA200, Neutral RSI (40-60)

➡️ TAO-USD
  Price: $0.00 (+0.00%) | RSI: 50.00
  Support: $0.00 | Resistance: $0.00
  Active Conditions: Insufficient Historical Data

➡️ AVAX-USD
  Price: $18.04 (+29.12%) | RSI: 40.43
  Support: $14.70 | Resistance: $55.70
  Active Conditions: Death Cross, Price < SMA200, Neutral RSI (40-60)

➡️ LINK-USD
  Price: $13.26 (-38.98%) | RSI: 44.30
  Support: $10.20 | Resistance: $30.81
  Active Conditions: Golden Cross, Price < SMA200, Neutral RSI (40-60)

➡️ ONDO-USD
  Price: $0.78 (+178.48%) | RSI: 43.05
  Support: $0.67 | Resistance: $2.14
  Active Conditions: Neutral RSI (40-60)

================================================================================

```

### Rebalancing Suggestion
```
========================================================================================
⚖️ REBALANCING SUGGESTIONS (Multi-Timeframe Analysis)
========================================================================================
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
🟡 SOL    | Signal: HOLD
   Allocation: 9.89% (Target: 12.0%) | Drift: -2.11 pts | Value: $12.31
   Price: $146.70      | Support: $141.44 | Resistance: $187.28
   Long-Term Trend: Golden Cross, Price > SMA200, Neutral RSI (40-60)
   Action: Hold: Allocation is within tolerance.
------------------------------------------------------
🟡 AVAX   | Signal: HOLD
   Allocation: 4.83% (Target: 6.0%) | Drift: -1.17 pts | Value: $6.01
   Price: $17.98       | Support: $17.84 | Resistance: $25.95
   Long-Term Trend: Death Cross, Price < SMA200, Neutral RSI (40-60)
   Action: Hold: Allocation is within tolerance.
------------------------------------------------------
🟡 TAO    | Signal: HOLD
   Allocation: 6.85% (Target: 8.0%) | Drift: -1.15 pts | Value: $8.52
   Price: $360.20      | Support: $334.30 | Resistance: $500.00
   Long-Term Trend: Insufficient Historical Data
   Action: Hold: Allocation is within tolerance.
------------------------------------------------------
🟡 LINK   | Signal: HOLD
   Allocation: 5.29% (Target: 6.0%) | Drift: -0.71 pts | Value: $6.58
   Price: $13.26       | Support: $12.68 | Resistance: $17.14
   Long-Term Trend: Golden Cross, Price < SMA200, Neutral RSI (40-60)
   Action: Hold: Allocation is within tolerance.
------------------------------------------------------
🟡 ONDO   | Signal: HOLD
   Allocation: 4.53% (Target: 5.0%) | Drift: -0.47 pts | Value: $5.63
   Price: $0.78        | Support: $0.73 | Resistance: $1.05
   Long-Term Trend: Neutral RSI (40-60)
   Action: Hold: Allocation is within tolerance.
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
Type 'EXECUTE' to proceed with the trades listed above:
```

## 🔧 Development & Tooling

This project uses modern, high-performance Python tools for development:

### Using `uv` for Dependencies

```bash
# Install all dependencies including dev tools
uv sync --dev

# Add a new package
# Edit pyproject.toml to add the package, then:
uv sync --extra dev pyproject.toml

# Update all packages
uv sync --upgrade --dev
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
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src/crypto_portfolio_tracker

# Run specific test file
uv run pytest tests/test_portfolio_tracker.py
```

## 🐛 Troubleshooting

### Common Issues

#### Testnet vs Production Confusion
- **Check Database**: Look for "TESTNET MODE" indicator in portfolio summary
- **Verify Settings**: Ensure `BINANCE_TESTNET` is set correctly in `.env`
- **Separate Databases**: Testnet uses `testnet_portfolio.db`, production uses `portfolio.db`

#### API Connection Failed
```bash
# Test API connections through menu
track-portfolio  # Choose option 14 (Test API Connections)

# Check configuration
track-portfolio  # Choose option 12 (View Configuration)

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
track-portfolio  # Choose option 13 (Clean Old Data)

# Complete reset (will require full re-sync)
rm data/portfolio.db data/testnet_portfolio.db
```

## 🔒 Security Best Practices

### API Security
1. **Start with Testnet**: Always test with `BINANCE_TESTNET=true` first
2. **Use Read-Only Keys**: Enable only "Enable Reading" permission unless live trading is necessary
3. **Enable IP Whitelisting**: Restrict API access to your IP address
4. **Rotate API Keys Regularly**: Monthly rotation recommended
5. **Monitor API Usage**: Check usage in your exchange account settings

### Local Security
1. **Never Commit `.env`**: File is in `.gitignore` but double-check
2. **Restrict File Permissions**: `chmod 600 .env` on Linux/macOS
3. **Use Encrypted Storage**: Encrypt sensitive configuration files
4. **Regular Backups**: Backup database and configuration regularly
5. **Keep Software Updated**: Update dependencies regularly with `uv pip sync --upgrade`

### Trading Security
1. **Always Start with Testnet**: Test all strategies on testnet first
2. **Use Dry-Run Mode**: Keep `live_trading_enabled: false` initially
3. **Set Position Limits**: Configure maximum trade amounts
4. **Monitor Actively**: Watch automated trades closely
5. **Have Kill Switch**: Know how to stop automated trading immediately

## 📈 Performance Metrics

Optimized performance characteristics:
- **Initial Sync**: ~2-3 minutes (depending on transaction history)
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
- **Set `BINANCE_TESTNET=false`** only when ready for real trading
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
- **Charts**: `data/exports/*.png` and `data/exports/*.svg`

### Analysis and Logs
- **Technical Analysis**: `data/trend_reports/` (Detailed market analysis JSON files)
- **Strategy State**: `data/strategy_state.json` (Trading strategy persistence)
- **Application Logs**: `logs/portfolio_tracker.log` (Rotating logs with 5 backup files)

## 🚀 Roadmap

### Immediate Improvements (v2.2)
- [ ] Enhanced error recovery for network interruptions
- [ ] Manual transaction import from CSV files
- [ ] Additional exchange integrations (Coinbase Pro, Kraken)
- [ ] Mobile app companion for portfolio monitoring
- [ ] Real-time price alerts and notifications

### Medium-term Goals (v2.5)
- [ ] Tax reporting features (Form 8949, capital gains calculation)
- [ ] DeFi protocol integration (Uniswap, Compound, Aave)
- [ ] Advanced options and futures tracking
- [ ] Multi-portfolio management for different strategies
- [ ] Social trading features and strategy sharing

### Long-term Vision (v3.0)
- [ ] Web-based dashboard with real-time updates
- [ ] Machine learning-powered strategy optimization
- [ ] Cross-chain asset tracking (Ethereum, BSC, Polygon)
- [ ] Professional-grade risk management tools
- [ ] Institutional features for larger portfolios

## 🤝 Contributing

We welcome contributions from the community! Here's how to get started:

1. **Fork the Repository** on GitHub
2. **Create a Feature Branch**: `git checkout -b feature-amazing-feature`
3. **Set Up Development Environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync --dev
   ```
4. **Make Your Changes** with clear, commented code
5. **Run Tests**: `pytest tests/`
6. **Format Code**: `ruff format .` and `ruff check --fix .`
7. **Add Tests** for new functionality
8. **Update Documentation** as needed
9. **Submit a Pull Request** with detailed description of changes

### Development Guidelines
- Follow PEP 8 style guidelines (enforced by `ruff`)
- Write comprehensive docstrings for new functions
- Add unit tests for new features
- Update configuration examples when adding new options
- Test both testnet and production modes when applicable

## 🆘 Support & Community

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/Onehand-Coding/crypto-portfolio-tracker/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/Onehand-Coding/crypto-portfolio-tracker/discussions)
- **Documentation**: [Project Wiki](https://github.com/Onehand-Coding/crypto-portfolio-tracker/wiki)

### Getting Help
1. **Check the troubleshooting section** above
2. **Search existing issues** on GitHub
3. **Provide detailed information** when reporting bugs:
   - Operating system and Python version
   - Whether using testnet or production mode
   - Complete error messages and stack traces
   - Steps to reproduce the issue
   - Relevant configuration (without API keys)
   - Log files (`logs/portfolio_tracker.log`)

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
