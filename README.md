# Crypto Portfolio Tracker & Rebalancing Advisor

A comprehensive, personal cryptocurrency portfolio tracking application that connects to Binance to provide detailed analysis of your holdings, accurate P/L calculations using FIFO cost basis, and intelligent rebalancing suggestions based on technical analysis.

## 🚀 Key Features

### 📊 **Complete Transaction Syncing**
- **Comprehensive Data Coverage**: Automatically fetches all transaction types from your Binance account:
  - Spot Trades & P2P Trades (Fiat to USDT)
  - Deposits & Withdrawals
  - Simple Earn Subscriptions, Redemptions, and Rewards
  - Staking History (Subscriptions, Redemptions, and Interest)
  - Dividends, Asset Conversions, and More
- **Accurate P/L Calculation**: Implements First-In, First-Out (FIFO) accounting method for precise cost basis and unrealized profit/loss calculations
- **Persistent Local Database**: All transactions stored in local SQLite database (`data/portfolio.db`) for fast queries and complete historical record

### ⚡ **Performance Optimized**
- **Asynchronous Syncing**: Concurrent API calls for significantly faster data fetching
- **Smart Selective Sync**: After initial setup, only fetches new transactions for lightning-fast daily updates
- **Persistent Caching**: API calls and historical price data cached locally to minimize redundant requests
- **Local SQLite Database**: All data stored in `data/portfolio.db` for fast queries and offline access

### 🎯 **Strategic Rebalancing Advisor**
- **Live Portfolio Analysis**: Analyzes your current portfolio value against defined target allocations
- **Technical Analysis Integration**: Incorporates RSI and 200-week Moving Average indicators for context-aware decisions
- **Smart Recommendations**: Provides intelligent `BUY`, `SELL`, or `HOLD` suggestions with clear reasoning
- **Actionable Trade Details**: Exact USD values and coin amounts for each recommended trade
- **Configurable Strategy**: Protect specific assets from selling, customize rebalancing preferences

### 📋 **Data Export & Visualization**
- **Professional Reports**: Export portfolio summaries to Excel (`.xlsx`) or HTML (`.html`) formats
- **Visual Analytics**: Generate charts showing portfolio allocation and performance
- **Complete Data Backup**: Full CSV export of all transactions and holdings data
- **Mobile-Optimized**: HTML reports designed for easy mobile viewing and sharing

### 🔧 **Advanced Usage Features**
- **Environment Variable Support**: Secure API key management using `.env` files (recommended)
- **Command-Line Interface**: Non-interactive modes for automation and scripting
- **Symbol Normalization**: Handles ticker discrepancies (RENDER vs RNDR) automatically
- **Configurable Logging**: Structured logging with rotation and adjustable verbosity levels
- **Custom Configuration**: Override settings via environment variables or custom config files

## 📁 Project Structure

```
crypto-portfolio-tracker/
├── src/
│   ├── templates/                  # HTML report templates
│   │   └── report_template.html
│   ├── __init__.py
│   ├── portfolio_tracker.py       # Main tracker implementation
│   ├── config.py                  # Configuration management
│   ├── database.py                # SQLite database operations
│   ├── exporters.py               # Export functionality (Excel/HTML/CSV)
│   └── visualizations.py          # Chart and graph generation
├── config/
│   ├── .env.example               # Environment variables template
│   └── default_config.json        # Default configuration settings
├── data/                          # Local data storage
│   ├── portfolio.db               # SQLite database
│   └── exports/                   # Generated reports
├── logs/                          # Application logs
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
3. **Enable ONLY "Enable Reading" permission** (disable trading for security)
4. For complete history, enable "Enable Spot & Margin Trading" read permissions
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

### Symbol Mappings & Normalization

Handle ticker discrepancies between exchanges:

```json
{
  "symbol_mappings": {
    "coingecko_ids": {
      "USDT": "tether",
      "BTC": "bitcoin",
      "RENDER": "render-token",
      "RNDR": "render-token"
    }
  },
  "symbol_normalization_map": {
    "RNDR": "RENDER"
  }
}
```

### Rebalancing Strategy

Configure rebalancing behavior:

```json
{
  "rebalancing_strategy": {
    "base_on_cost": true,
    "allow_selling": true,
    "never_sell_symbols": ["BTC", "ETH"]
  }
}
```

### P2P Fiat Currency

Set your local fiat currency for P2P trades:

```json
{
  "p2p_fiat_currency": "USD"
}
```

## 🎯 Usage

The application offers both interactive and command-line modes for maximum flexibility.

### Interactive Mode (Recommended)

For day-to-day portfolio management, simply run:

```bash
python main.py
```

**Interactive Menu Options:**
```
Crypto Portfolio Tracker & Rebalancing Advisor
==============================================
1. 🔄 Full Sync & Analysis (Recommended for first run)
2. 📊 Quick Portfolio Summary (Fast daily updates)
3. 📋 Export Reports Only
4. 📈 Generate Charts Only
5. 💾 Export Data Backup
6. ⚙️  View Configuration
7. 🧹 Clean Old Data
8. ⚖️  Rebalance Suggestions (Technical Analysis)
9. 🔧 Test API Connections
10. ❌ Exit

Select option (1-10):
```

### Command-Line Mode

For automation, scripting, and advanced usage:

```bash
# General format
python main.py [command] [options]
```

**Available Commands & Options:**
- `--sync-only`: Run data synchronization only
- `--export-only`: Export existing data without syncing
- `--charts-only`: Generate charts from existing data
- `--format [excel|html|csv|all]`: Specify export format (default: all)
- `--config <path>`: Use custom configuration file
- `-v, --verbose`: Enable detailed debug logging
- `-q, --quiet`: Suppress console output except errors
- `--version`: Show application version

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
3. **Strategic Analysis**: Option 8 (Rebalance Suggestions) for technical analysis-based recommendations
4. **Professional Reporting**: Option 3 (Export Reports) to generate comprehensive portfolio reports

## 📊 Understanding the Output

### Portfolio Summary
```
📊 CRYPTO PORTFOLIO SUMMARY
================================================================================
Timestamp:             2024-01-15 14:30:22
Total Portfolio Value: $15,234.56
Total Cost Basis:      $12,000.00
Unrealized P/L:        $3,234.56 (+26.95%)
--------------------------------------------------------------------------------
Asset    Quantity       Price (USD)    Value (USD)      P/L (USD)     Allocation
--------------------------------------------------------------------------------
BTC      0.2500         $43,000.00     $10,750.00       +$3,750.00    45.2%
ETH      5.0000         $2,200.00      $11,000.00       +$2,000.00    25.8%
SOL      25.0000        $65.00         $1,625.00        +$125.00      10.7%
================================================================================
```

### Technical Analysis Rebalancing
```
================================================================================
⚖️ REBALANCING SUGGESTIONS (Core Portfolio - Technical Analysis)
================================================================================
🔴 BTC       | Signal: SELL
   Allocation: 56.68% (Target: 35.0%) | Current Value: $62.58
   TA: RSI: 51.1, Price vs 200w MA: +116.1%
   Action: Sell ~7.5% of position (~$4.70), which is 0.000045 BTC
--------------------------------------------------------------------------------
🟡 ETH       | Signal: HOLD
   Allocation: 13.16% (Target: 20.0%) | Current Value: $14.52
   TA: RSI: 61.6, Price vs 200w MA: +6.4%
   Action: Hold: Allocation is within tolerance.
--------------------------------------------------------------------------------
🟢 AVAX      | Signal: BUY
   Allocation: 0.00% (Target: 6.0%) | Current Value: $0.00
   TA: RSI: 40.3, Price vs 200w MA: -40.8%
   Action: Buy ~$3.31 worth (0.165 AVAX)
--------------------------------------------------------------------------------
```

## 📁 Output Files

All files saved in `data/` directory:

- **Database**: `data/portfolio.db` (SQLite with complete transaction history)
- **Excel Reports**: `data/exports/portfolio_report_YYYYMMDD_HHMMSS.xlsx`
- **HTML Reports**: `data/exports/portfolio_report_YYYYMMDD_HHMMSS.html`
- **CSV Backups**: `data/exports/transactions_backup_YYYYMMDD_HHMMSS.csv`
- **Charts**: `data/exports/portfolio_allocation_pie_YYYYMMDD_HHMMSS.png`
- **Logs**: `logs/portfolio_tracker.log`

## 🔧 Advanced Features

### FIFO Cost Basis Calculation
- Accurately tracks cost basis using First-In, First-Out methodology
- Handles complex scenarios like staking rewards, conversions, and internal transfers
- Properly accounts for fees in cost basis calculations

### Technical Analysis Integration
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **200-week Moving Average**: Long-term trend analysis for strategic decisions
- **Context-Aware Recommendations**: Combines allocation targets with technical signals

### Performance Optimizations
- **Concurrent API Calls**: Async processing for 5x faster data fetching
- **Intelligent Caching**: Reduces API calls by 90% after initial sync
- **Incremental Updates**: Only processes new transactions after first run

## 🐛 Troubleshooting

### Common Issues

**"API Connection Failed"**
```bash
# Verify credentials
python main.py  # Choose option 9 (Test API Connections)

# Check .env file
cat .env
```

**"Permission Denied"**
- Ensure API key has correct read permissions
- Verify IP whitelist settings
- Check if API key is expired

**"No Data Found"**
- Run full sync first: Choose option 1
- Verify you have holdings in Binance account
- Check logs: `tail -f logs/portfolio_tracker.log`

**"Symbol Not Found"**
- Update `symbol_mappings.coingecko_ids` in config
- Check if token is listed on CoinGecko
- Add custom mapping if needed

### Debug Mode

Enable verbose logging:
```bash
python main.py --verbose
```

Or set environment:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

## 📱 Mobile-Friendly Reports

HTML exports are optimized for mobile viewing:

1. Generate HTML report: Choose option 3, select HTML format
2. Open file on mobile device
3. Bookmark for quick portfolio checks
4. Share via email or messaging

## 🔒 Security Best Practices

1. **Use read-only API keys exclusively**
2. **Never commit `.env` file to version control**
3. **Enable IP whitelisting on Binance**
4. **Regularly rotate API keys**
5. **Keep software updated**
6. **Backup data securely**

## 🚀 Performance Metrics

After optimization improvements:
- **Initial sync**: ~2-3 minutes (depending on transaction history)
- **Daily updates**: ~10-15 seconds
- **Report generation**: ~5 seconds
- **API calls reduced**: 90% fewer requests after first run

## 📈 Roadmap

### Immediate Improvements
- [ ] Enhanced fee calculation for non-stablecoin fees
- [ ] LD (Earn/Staked) asset aggregation with spot holdings
- [ ] Complete USD pricing for all trading pairs
- [ ] Manual transaction import interface

### Long-term Goals
- [ ] Multi-exchange support (Coinbase, Kraken, etc.)
- [ ] Real-time price alerts and notifications
- [ ] Tax reporting features
- [ ] Web dashboard interface
- [ ] DeFi protocol integration
- [ ] Advanced backtesting capabilities

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Submit pull request with detailed description

## ⚠️ Disclaimer

This software is for educational and informational purposes only and does not constitute financial advice. Cryptocurrency investments carry significant risk and can result in substantial losses. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/Onehand-Coding/crypto-portfolio-tracker/issues)
- **Documentation**: [Project Wiki](https://github.com/Onehand-Coding/crypto-portfolio-tracker/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/Onehand-Coding/crypto-portfolio-tracker/discussions)

---

**Built with ❤️ for the crypto community**
