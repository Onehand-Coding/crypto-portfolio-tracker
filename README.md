# Crypto Portfolio Tracker & Rebalancing Advisor

A comprehensive, personal cryptocurrency portfolio tracking application that connects to Binance to provide detailed analysis of your holdings, accurate P/L calculations using FIFO cost basis, intelligent rebalancing suggestions based on technical analysis, and advanced backtesting capabilities with optional live trading execution.

## 🚀 Key Features

### 📊 **Complete Portfolio Intelligence**
- **Comprehensive Transaction Syncing**: Automatically fetches all transaction types from your Binance account:
  - Spot Trades & P2P Trades (Fiat to crypto)
  - Deposits & Withdrawals
  - Simple Earn (Subscriptions, Redemptions, and Rewards)
  - Staking History (Subscriptions, Redemptions, and Interest)
  - Dividends, Asset Conversions, Internal Transfers, and Copy Trading
- **Consolidated Wallet View**: Aggregates balances from Spot and Simple Earn wallets with automatic LD (Locked/Staked) asset normalization
- **Accurate P/L Calculation**: Implements First-In, First-Out (FIFO) accounting method for precise cost basis and unrealized profit/loss calculations
- **Persistent Local Database**: All transactions stored in local SQLite database (`data/portfolio.db`) for fast queries and complete historical record

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
- **Intelligent API Management**:
  - Resilient error handling with automatic retries
  - Fallback mechanism from yfinance to Binance API for reliable data
  - Smart rate limiting and request batching
- **Performance Optimized**:
  - Asynchronous concurrent API calls for 5x faster syncing
  - Persistent disk caching reduces API calls by 90% after initial sync
  - Incremental updates process only new transactions
- **Professional Reporting**: Generate reports in multiple formats:
  - Excel (`.xlsx`) with embedded charts and password protection options
  - Mobile-optimized HTML (`.html`) with interactive elements
  - Complete CSV backups with separate files for different data types
- **Advanced Visualization**: Create professional charts showing portfolio allocation, performance trends, and technical analysis

## 📁 Project Structure

```
crypto-portfolio-tracker/
├── src/
│   ├── templates/                  # HTML report templates
│   │   └── report_template.html
│   ├── __init__.py
│   ├── portfolio_tracker.py       # Main tracker & orchestration
│   ├── config.py                  # Configuration management
│   ├── database.py                # SQLite database operations
│   ├── exporters.py               # Export functionality (Excel/HTML/CSV)
│   ├── visualizations.py          # Chart and graph generation
│   ├── crypto_trend_analyzer.py   # Multi-timeframe technical analysis
│   ├── trading_strategies.py      # Pluggable trading strategies
│   ├── strategy_backtester.py     # Backtester for directional strategies
│   └── rebalancing_backtester.py  # Backtester for rebalancing strategies
├── config/
│   ├── .env.example               # Environment variables template
│   └── default_config.json        # Default configuration settings
├── data/                          # Local data storage
│   ├── portfolio.db               # SQLite database
│   ├── cache/                     # API response cache
│   ├── exports/                   # Generated reports
│   └── trend_reports/             # Technical analysis reports
├── logs/                          # Application logs with rotation
├── requirements.txt               # Python dependencies
├── main.py                        # Application entry point
├── README.md                      # This documentation
└── setup.py                       # Installation script
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Binance account with API access
- Git (optional but recommended)

### Quick Start

1. **Clone the Repository:**
```bash
git clone https://github.com/Onehand-Coding/crypto-portfolio-tracker.git
cd crypto-portfolio-tracker
```

2. **Create and Activate Virtual Environment:**
```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables (Recommended):**
```bash
cp config/.env.example .env
```

Edit `.env` with your API credentials:
```env
# .env - Most secure way to handle credentials
BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET="YOUR_BINANCE_API_SECRET"
COINGECKO_API_KEY="your_coingecko_api_key_optional"
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

5. **Configure Your Portfolio Targets:**
```bash
cp config/default_config.json config/config.json
```

Edit `config/config.json` with your target allocation and preferences.

**Note**: Settings in `.env` file will override settings in `config.json`.

## 🔐 Binance API Setup

**CRITICAL SECURITY STEPS:**

1. Visit [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create a new API key with descriptive name
3. **For Portfolio Tracking Only**: Enable ONLY "Enable Reading" permission
4. **For Live Trading**: Enable "Enable Reading" + "Enable Spot & Margin Trading" (use with extreme caution)
5. **Add your IP address to whitelist** for enhanced security
6. Copy API Key and Secret to your `.env` file
7. **NEVER share your API secret**

## ⚙️ Configuration

### Target Portfolio Allocation

Define your desired allocation in `config/config.json` (percentages must sum to 1.0):

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

### Portfolio & Trading Settings

```json
{
  "portfolio": {
    "minimum_trade_usd": 5.0,
    "live_trading_enabled": false,
    "p2p_fiat_currency": "PHP",
    "stablecoin_symbols": ["USDT"]
  }
}
```

## 🎯 Usage

The application offers a comprehensive interactive menu and command-line options for automation.

### Interactive Mode (Recommended)

For day-to-day portfolio management:

```bash
python main.py
```

**Interactive Menu Options:**
```
==================================================
🚀 Crypto Portfolio Tracker v2.1
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
python main.py [command] [options]

# Available options
--sync-only              # Run data synchronization only
--export-only            # Export existing data without syncing
--charts-only            # Generate charts from existing data
--format [excel|html|csv|all]  # Specify export format
--config <path>          # Use custom configuration file
-v, --verbose            # Enable detailed debug logging
-q, --quiet              # Suppress console output except errors
--version                # Show application version
```

**Command-Line Examples:**
```bash
# Silent background sync for automation
python main.py --sync-only --quiet

# Export only Excel report
python main.py --export-only --format excel

# Use separate config for different portfolio
python main.py --config /path/to/my_other_config.json

# Generate charts with verbose logging
python main.py --charts-only --verbose
```

### Recommended Workflow

1. **First Run**: Choose option 1 (Full Sync & Analysis) for complete transaction history sync
2. **Daily Monitoring**: Option 2 (Quick Portfolio Summary) for fast portfolio updates
3. **Technical Analysis**: Option 3 (View Crypto Trends) for multi-timeframe market analysis
4. **Strategic Planning**: Option 4 (Rebalance Suggestions) for allocation-based recommendations
5. **Strategy Testing**: Options 7-8 for backtesting before live trading
6. **Live Execution**: Options 5-6 for automated rebalancing or strategy trading (use with caution)

## 📊 Understanding the Output

### Consolidated Portfolio Summary
```
===================================================================================================================
📊 CONSOLIDATED PORTFOLIO SUMMARY (Spot + Earn)
===================================================================================================================
Timestamp:             2025-06-12 17:42:28
Total Portfolio Value: $134.65
Total Cost Basis:      $129.70
Unrealized P/L:        $4.95 (3.81%)
-------------------------------------------------------------------------------------------------------------------
Asset    Total Qty          Spot Qty        Earn Qty        Value (USD)     Cost Basis      P/L (USD)       Allocation
-------------------------------------------------------------------------------------------------------------------
BTC      0.00046944         0               0.00046944      $50.54          $42.77          $7.77           37.53%
ETH      0.01190199         0               0.01190199      $32.83          $31.49          $1.34           24.38%
SOL      0.08608167         0               0.08608167      $13.74          $14.91          $-1.17          10.20%
TAO      0.02378306         0               0.02378306      $9.43           $10.67          $-1.24          7.00%
RENDER   2.02797            2.02797         0               $7.60           $9.99           $-2.38          5.65%
LINK     0.49950165         0               0.49950165      $7.22           $6.99           $0.23           5.36%
AVAX     0.33968686         0               0.33968686      $7.16           $6.90           $0.26           5.32%
ONDO     7.1930891          0               7.1930891       $6.08           $5.95           $0.13           4.51%
===================================================================================================================
```

### Multi-Timeframe Technical Analysis
```
========================================================================================================================
📈 CRYPTO TREND ANALYSIS REPORT
========================================================================================================================
Generated: 2025-06-12 18:30:45
Analyzing: BTC, ETH, SOL, RENDER, TAO, AVAX, LINK, ONDO
========================================================================================================================

🟢 BTC (Bitcoin) - BULLISH CONFIDENCE: 82%
------------------------------------------------------------------------
Long-Term (4y):  ✅ Golden Cross | Price: $107,632 (+18.2% vs SMA200)
Swing (3mo):     ✅ Above MA30   | RSI: 58.3 (Neutral)
Day (60d):       ⚠️  Near MA15   | MACD: Bullish Crossover
Recommendation:  STRONG BUY - Multiple timeframes confirm bullish trend

🔴 RENDER (Render Token) - BEARISH CONFIDENCE: 71%
------------------------------------------------------------------------
Long-Term (4y):  ❌ Death Cross  | Price: $3.75 (-35.8% vs SMA200)
Swing (3mo):     ❌ Below MA30   | RSI: 28.1 (Oversold)
Day (60d):       ⚠️  Testing MA15| MACD: Bearish Divergence
Recommendation:  HOLD/ACCUMULATE - Oversold conditions, potential reversal
========================================================================================================================
```

### Technical Rebalancing Suggestions
```
========================================================================================================================
⚖️ REBALANCING SUGGESTIONS (Multi-Timeframe Analysis)
========================================================================================================================
🔴 BTC     | Signal: SELL
   Allocation: 37.53% (Target: 35.0%) | Drift: +2.53 pts | Value: $50.54
   Technical: Golden Cross, RSI: 58.3, Price > SMA200 (+18.2%)
   Action: Sell ~25% of excess (~$0.85), which is 0.0000079 BTC
   Reason: Above target allocation, strong technical position allows partial profit-taking
----------------------------------------------------------------------
🟢 AVAX    | Signal: BUY
   Allocation: 5.32% (Target: 6.0%) | Drift: -0.68 pts | Value: $7.16
   Technical: Death Cross, RSI: 42.1, Price vs SMA200 (-12.3%)
   Action: Buy ~$0.92 worth, which is 0.044 AVAX
   Reason: Below target allocation, oversold conditions present buying opportunity
----------------------------------------------------------------------
🟡 ETH     | Signal: HOLD
   Allocation: 24.38% (Target: 20.0%) | Drift: +4.38 pts | Value: $32.83
   Technical: Golden Cross, RSI: 64.2, Price > SMA200 (+8.1%)
   Action: Hold: Drift within major asset tolerance (3.0%), strong technical setup
================================================================================
🟡🟡🟡 NOTE: Live Trading is DISABLED. 🟡🟡🟡
================================================================================
🚨 PROPOSED TRADES - PLEASE REVIEW CAREFULLY 🚨
================================================================================

Symbol  Signal   Suggested Action Detail
  BTC    SELL    Sell ~$0.85 worth, which is 0.0000079 BTC
 AVAX     BUY    Buy ~$0.92 worth, which is 0.044 AVAX
================================================================================
Type 'EXECUTE' to proceed with dry-run simulation:
```

### Strategy Backtesting Results
```
================================================================================
📈 BACKTEST PERFORMANCE REPORT: BTC-USD (2022 - 2025)
Strategy: Golden Cross (50/200)
================================================================================
Initial Capital:         $10,000.00
Final Portfolio Value:   $21,205.03
----------------------------------------
Strategy Total Return:   112.05%
Buy & Hold Return:       300.79%
Strategy Outperformance: -188.74%
----------------------------------------
Total Trades Executed:   7
================================================================================

--- Trade Log (First 15) ---
2023-02-07 00:00: BUY 0.4294 of BTC-USD @ $23,264.29 (Golden Cross: 50d > 200d)
2023-09-12 00:00: SELL 0.4294 of BTC-USD @ $25,833.34 (Death Cross: 50d < 200d)
2023-10-30 00:00: BUY 0.3209 of BTC-USD @ $34,502.36 (Golden Cross: 50d > 200d)
2024-08-10 00:00: SELL 0.3209 of BTC-USD @ $60,945.81 (Death Cross: 50d < 200d)
2024-10-28 00:00: BUY 0.2792 of BTC-USD @ $69,907.76 (Golden Cross: 50d > 200d)
2025-04-07 00:00: SELL 0.2792 of BTC-USD @ $79,235.34 (Death Cross: 50d < 200d)
2025-05-22 00:00: BUY 0.1977 of BTC-USD @ $111,673.28 (Golden Cross: 50d > 200d)
================================================================================

✅ Press Enter to continue...

```

## 📁 Output Files

All files are organized in the `data/` directory:

- **Database**: `data/portfolio.db` (SQLite with complete transaction history)
- **Cache**: `data/cache/` (API response caching for performance)
- **Reports**:
  - `data/exports/portfolio_report_YYYYMMDD_HHMMSS.xlsx` (Excel with charts)
  - `data/exports/portfolio_report_YYYYMMDD_HHMMSS.html` (Mobile-optimized)
- **Backups**: `data/exports/transactions_backup_YYYYMMDD_HHMMSS.csv`
- **Charts**: `data/exports/portfolio_allocation_pie_YYYYMMDD_HHMMSS.png`
- **Technical Analysis**: `data/trend_reports/` (Detailed market analysis)
- **Logs**: `logs/portfolio_tracker.log` (Rotating logs with 5 backup files)

## 🔧 Advanced Features

### FIFO Cost Basis Calculation
- Accurately tracks cost basis using First-In, First-Out methodology
- Handles complex scenarios: staking rewards, conversions, internal transfers
- Properly accounts for fees in cost basis calculations
- Supports different fiat currencies for P2P trades

### Multi-Timeframe Technical Analysis
- **Long-term (4-year)**: 50/200 SMA for major trend identification
- **Swing (3-month)**: 10/30 SMA for medium-term momentum
- **Day (60-day)**: 5/15 SMA for short-term entry/exit timing
- **Confidence Scoring**: Weighted recommendations based on timeframe alignment

### Advanced Rebalancing Logic
- **Asset Classification**: Different thresholds for majors (BTC/ETH) vs altcoins
- **Technical Integration**: RSI and moving average filters for timing
- **Safety Mechanisms**: Minimum trade amounts, never-sell lists, drift thresholds
- **Smart Execution**: Batch trades, dry-run mode, confirmation steps

### Performance Optimizations
- **Concurrent Processing**: Async API calls for 5x faster data fetching
- **Intelligent Caching**: 90% reduction in API calls after initial sync
- **Incremental Updates**: Only process new transactions after first run
- **Database Optimization**: Indexed queries, automatic cleanup, backup rotation

## 🐛 Troubleshooting

### Common Issues

**"API Connection Failed"**
```bash
# Test API connections
python main.py  # Choose option 14 (Test API Connections)

# Check configuration
python main.py  # Choose option 12 (View Configuration)

# Verify environment variables
cat .env
```

**"Permission Denied" or "Invalid API Key"**
- Ensure API key has correct permissions (Read-only for tracking, Spot Trading for live trades)
- Verify IP whitelist settings in Binance
- Check if API key is expired or suspended
- Confirm recv_window setting (default: 60000ms)

**"No Data Found" or "Empty Portfolio"**
- Run full sync first: Choose option 1
- Verify you have holdings in your Binance account (Spot or Earn wallets)
- Check minimum value threshold in config (`minimum_value_usd`)
- Review logs: `tail -f logs/portfolio_tracker.log`

**"Symbol Not Found" in CoinGecko**
- Update `symbol_mappings.coingecko_ids` in config
- Check if token is listed on CoinGecko API
- Add custom mapping for new tokens
- Use symbol normalization for ticker variations (RNDR → RENDER)

**"Rate Limit Exceeded"**
- Increase `request_delay_ms` in API configuration
- Reduce `batch_days` for smaller batches
- Check rate limits in Binance account
- Consider upgrading CoinGecko API plan

### Debug Mode

Enable comprehensive logging:
```bash
# Via command line
python main.py --verbose

# Via environment variable
export LOG_LEVEL=DEBUG
python main.py

# View live logs
tail -f logs/portfolio_tracker.log
```

### Database Issues

```bash
# Check database integrity
sqlite3 data/portfolio.db ".schema"

# Manual cleanup (use with caution)
python main.py  # Choose option 13 (Clean Old Data)

# Complete reset (will require full re-sync)
rm data/portfolio.db
```

## 🔒 Security Best Practices

### API Security
1. **Use read-only API keys** unless live trading is absolutely necessary
2. **Enable IP whitelisting** on all API keys
3. **Rotate API keys regularly** (monthly recommended)
4. **Monitor API usage** in Binance account settings
5. **Use testnet** for strategy development when available

### Local Security
1. **Never commit `.env` file** to version control (added to .gitignore)
2. **Set restrictive file permissions** on configuration files (chmod 600)
3. **Use encrypted storage** for sensitive data when possible
4. **Regular backups** of database and configuration
5. **Keep software updated** with latest security patches

### Trading Security
1. **Start with dry-run mode** always (`live_trading_enabled: false`)
2. **Use minimum trade amounts** to limit exposure
3. **Set up stop-losses** and position limits
4. **Monitor trades actively** when live trading is enabled
5. **Have emergency procedures** for stopping automated trading

## 🚀 Performance Metrics

After optimization improvements:
- **Initial sync**: ~2-3 minutes (depending on transaction history)
- **Daily updates**: ~10-15 seconds with caching
- **Report generation**: ~5-10 seconds including charts
- **API calls reduced**: 90% fewer requests after first run
- **Database queries**: Sub-second response times with indexing
- **Memory usage**: ~50-100MB typical operation

## ⚠️ Important Disclaimers

### Financial Risk Warning
This software is for **educational and informational purposes only** and does **NOT** constitute financial advice. Cryptocurrency investments carry significant risk and can result in substantial losses.

### Strategy Performance Warning
- **Default strategies are for demonstration** and educational purposes only
- **Backtesting results show strategies may underperform** simple buy-and-hold approaches
- **Past performance does not guarantee future results**
- **You must perform your own research and strategy optimization**

### Live Trading Risks
- **Live trading is disabled by default** for safety
- **Bugs, API issues, or flawed logic can lead to financial loss**
- **Use extensive backtesting and dry-run mode** before enabling live trading
- **Start with small amounts** to test functionality
- **Monitor automated trades actively**

### Data Accuracy
- **Tool relies on third-party APIs** (Binance, CoinGecko, yfinance)
- **Data is not guaranteed to be 100% accurate or available**
- **Always verify critical data** independently
- **Network issues can cause incomplete synchronization**

## 📈 Roadmap

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

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature-amazing-feature`
3. **Make your changes** with clear, commented code
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request** with detailed description of changes

## 🆘 Support & Community

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/Onehand-Coding/crypto-portfolio-tracker/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/Onehand-Coding/crypto-portfolio-tracker/discussions)
- **Documentation**: [Project Wiki](https://github.com/Onehand-Coding/crypto-portfolio-tracker/wiki)
- **Community Chat**: [Discord Server](https://discord.gg/crypto-portfolio-tracker)

### Getting Help
1. **Check the troubleshooting section** above
2. **Search existing issues** on GitHub
3. **Provide detailed information** when reporting bugs:
   - Operating system and Python version
   - Complete error messages and stack traces
   - Steps to reproduce the issue
   - Relevant configuration (without API keys)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- Binance API: Subject to Binance Terms of Service
- CoinGecko API: Subject to CoinGecko Terms of Service
- All Python dependencies: See individual package licenses

## 🙏 Acknowledgments

- **Binance** for providing comprehensive API access
- **CoinGecko** for reliable cryptocurrency data
- **Python community** for excellent libraries and tools
- **Contributors** who help improve the project
- **Users** who provide feedback and bug reports

---

**Built with ❤️ for the crypto community**

*Remember: This tool is designed to help you make informed decisions, but always do your own research and never invest more than you can afford to lose.*
