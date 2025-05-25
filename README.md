# Crypto Portfolio Tracker

A comprehensive cryptocurrency portfolio tracking tool that connects to Binance and CoinGecko APIs to analyze your holdings, calculate P/L using FIFO cost basis, and provide detailed portfolio insights including intelligent rebalancing suggestions.

## 🚀 Key Features

- **🔐 Secure API Integration**: Connect to Binance with read-only permissions for trades, deposits, and withdrawals
- **📊 FIFO Cost Basis Calculation**: Accurately calculates average cost basis and P/L for all assets
- **📈 Historical Price Fetching**: Retrieves historical prices for deposits to determine accurate cost basis
- **⚖️ Smart Rebalancing**: Compare actual vs target allocations with cost-basis rebalancing suggestions
- **🎯 Configurable Strategy**: Define rebalancing preferences, protect assets from selling, and set target allocations
- **🔄 Symbol Normalization**: Handles ticker discrepancies (e.g., RENDER vs RNDR) via configuration
- **📦 Batch API Requests**: Efficiently fetches current prices from CoinGecko
- **📋 Multiple Export Formats**: Excel, HTML (mobile-friendly), and CSV exports
- **📊 Visual Analytics**: Charts and graphs for portfolio visualization
- **💾 Local Data Storage**: SQLite database for transaction history and calculated holdings
- **📝 Professional Logging**: Structured logging with rotation and configurable levels

## 📁 Project Structure

```
crypto-portfolio-tracker/
├── src/
│   ├── templates/                  # HTML templates for reports
│   │   └── report_template.html
│   ├── __init__.py
│   ├── portfolio_tracker.py       # Main tracker class
│   ├── config.py                  # Configuration management
│   ├── database.py                # Database operations
│   ├── exporters.py               # Export functionality
│   └── visualizations.py          # Chart generation
├── config/
│   ├── .env.example               # Environment variables template
│   └── default_config.json        # Default configuration
├── data/                          # Database and exports
│   └── exports/                   # Reports output directory
├── logs/                          # Application logs
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── main.py                        # Entry point
└── setup.py                       # Installation script
```

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Binance account with API access
- Git (optional)

### Quick Setup

1. **Clone or download the project:**
```bash
git clone https://github.com/Onehand-Coding/crypto-portfolio-tracker.git
cd crypto-portfolio-tracker
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp config/.env.example .env
```
Edit `.env` with your API credentials:
```env
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
COINGECKO_API_KEY=your_coingecko_api_key_optional
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

4. **Review and customize configuration:**
Edit `config/default_config.json` to set your preferences (see Configuration section below).

5. **Run the tracker:**
```bash
python main.py
```

## 🔐 Binance API Setup

**IMPORTANT SECURITY STEPS:**

1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create a new API key with a descriptive name
3. **Enable ONLY "Enable Reading" permission** (disable all others for security)
4. For comprehensive trade history, you may need "Enable Spot & Margin Trading" read permissions
5. Add your IP address to the whitelist for enhanced security
6. Copy the API Key and Secret to your `.env` file
7. **Never share your API secret with anyone**

## ⚙️ Configuration

The tracker uses a layered configuration system in `config/default_config.json`:

### Target Portfolio Allocation

Define your desired portfolio allocation percentages (should sum to 1.0):

```json
{
  "target_allocation": {
    "BTC": 0.40,
    "ETH": 0.25,
    "SOL": 0.15,
    "RENDER": 0.10,
    "TAO": 0.10
  }
}
```

### Symbol Mappings

Map exchange tickers to CoinGecko API IDs for price fetching:

```json
{
  "symbol_mappings": {
    "coingecko_ids": {
      "USDT": "tether",
      "LDUSDT": "tether",
      "BTC": "bitcoin",
      "LDBTC": "bitcoin",
      "RENDER": "render-token",
      "RNDR": "render-token",
      "HMSTR": "hamster-kombat"
    }
  }
}
```

### Symbol Normalization

Handle ticker discrepancies between exchanges and your preferred naming:

```json
{
  "symbol_normalization_map": {
    "RNDR": "RENDER"
  }
}
```

### Rebalancing Strategy

Configure how rebalancing suggestions are made:

```json
{
  "rebalancing_strategy": {
    "base_on_cost": true,
    "allow_selling": true,
    "never_sell_symbols": ["BTC", "ETH"]
  }
}
```

- `base_on_cost`: Use cost basis for rebalancing calculations
- `allow_selling`: Include sell suggestions for overweight assets
- `never_sell_symbols`: Assets to never suggest selling

### Gift/Airdrop Handling

Identify specific airdrops to assign $0 cost basis:

```json
{
  "pepe_gift_details": {
    "symbol": "PEPE",
    "amount": "322.6452382"
  }
}
```

## 🎯 Usage

### Command Line Interface

```bash
python main.py [options]

Options:
  --sync-only          Run sync without analysis
  --export-only        Export existing data only
  --charts-only        Generate charts only
  --format FORMAT      Export format (excel|html|csv|all)
  --verbose / -v       Enable verbose logging (DEBUG level)
  --quiet / -q         Suppress console output except errors
  --config CONFIG      Custom config file path
  --version            Show version information
```

### Interactive Menu

Run without arguments for the interactive menu:

```bash
python main.py
```

```
Crypto Portfolio Tracker v2.0
==============================
1. 🔄 Full Sync & Analysis (Recommended)
2. 📊 Quick Portfolio Summary
3. 📋 Export Reports Only
4. 📈 Generate Charts Only
5. 💾 Export Data Backup
6. ⚙️  View Configuration
7. 🧹 Clean Old Data
8. ⚖️  View Rebalance Suggestions (Cost Basis)
9. 🔧 Test API Connections
10. ❌ Exit

Select option (1-10):
```

### Programmatic Usage

```python
from src.portfolio_tracker import CryptoPortfolioTracker

# Initialize tracker
tracker = CryptoPortfolioTracker()

# Run full analysis
metrics = tracker.run_full_sync()

# Access data
print(f"Total Portfolio Value: ${metrics['total_value']:,.2f}")

# Export specific format
tracker.export_to_excel(metrics, "my_portfolio.xlsx")
```

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

### Rebalancing Suggestions

```
⚖️ REBALANCING SUGGESTIONS (Based on Cost Basis)
================================================================================
Total Portfolio Cost Basis (for target assets only): $10,500.00

Symbol  Target %  Current Cost  Target Cost   Action         Amount
--------------------------------------------------------------------------------
BTC     40.00%    $6,500.00     $4,200.00    SELL           $2,300.00 (0.0535 BTC)
ETH     25.00%    $3,000.00     $2,625.00    SELL           $375.00 (0.170 ETH)
SOL     15.00%    $900.00       $1,575.00    BUY            $675.00 (10.38 SOL)
RENDER  10.00%    $0.00         $1,050.00    BUY            $1,050.00 (35.0 RENDER)
TAO     10.00%    $100.00       $1,050.00    BUY            $950.00 (1.9 TAO)
```

## 📁 Output Files

All files are saved in the `data/` directory (configurable):

- **Excel Reports**: `data/exports/portfolio_report_YYYYMMDD_HHMMSS.xlsx`
- **HTML Reports**: `data/exports/portfolio_report_YYYYMMDD_HHMMSS.html`
- **CSV Backups**: `data/exports/transactions_backup_YYYYMMDD_HHMMSS.csv`
- **Charts**: `data/exports/portfolio_allocation_pie_YYYYMMDD_HHMMSS.png`
- **Database**: `data/portfolio.db` (SQLite)
- **Logs**: `logs/portfolio_tracker.log`

## 🔧 Advanced Features & Known Limitations

### Current Capabilities

- **FIFO Cost Basis**: Accurate calculation using First-In-First-Out methodology
- **Fee Handling**: Calculates USD fees for stablecoin and base asset fees
- **Historical Pricing**: Fetches historical prices for deposits and trades
- **Multi-format Export**: Excel, HTML, and CSV export options
- **Symbol Mapping**: Handles ticker discrepancies between exchanges

### Known Limitations & Roadmap

**Fee Calculation Improvements Needed:**
- Fees paid in BNB or other non-stablecoin currencies need historical price lookup
- Non-USD trading pairs (e.g., SOL/BTC) need quote currency historical pricing

**LD (Earn/Staked) Assets:**
- Strategy needed to aggregate LD assets with base spot assets
- Clear display of staked vs spot quantities

**Trade History:**
- Enhanced fetching for all available trading pairs
- Manual transaction import for off-exchange acquisitions

## 🐛 Troubleshooting

### Common Issues

**"API Connection Failed"**
```bash
# Check your .env file
cat .env

# Test API connectivity
python main.py --option 9
```

**"Permission Denied"**
- Ensure Binance API has correct read permissions
- Check IP whitelist settings
- Verify API key is not expired

**"No Data Found"**
- Run full sync first: `python main.py --sync-only`
- Check if you have crypto holdings in your Binance account
- Review logs: `tail -f logs/portfolio_tracker.log`

**"Price Data Missing"**
- Some tokens may not be available on CoinGecko
- Check `symbol_mappings.coingecko_ids` in config
- Review API rate limits

### Debug Mode

Enable verbose logging:
```bash
python main.py --verbose
```

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

## 📱 Mobile Usage

The HTML export is optimized for mobile viewing:

1. Generate HTML report: `python main.py --export-only --format html`
2. Open the HTML file on your phone
3. Bookmark for quick access
4. Share via email/messaging apps

## 🔒 Security Best Practices

1. **Never commit `.env` file to version control**
2. **Use read-only API keys only**
3. **Enable IP whitelisting on Binance**
4. **Regularly rotate API keys**
5. **Keep the software updated**
6. **Store backups securely**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and informational purposes only. It does not constitute financial advice. Cryptocurrency investments are risky and you should consult with a financial advisor before making investment decisions.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/Onehand-Coding/crypto-portfolio-tracker/issues)
- **Documentation**: [Wiki](https://github.com/Onehand-Coding/crypto-portfolio-tracker/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/Onehand-Coding/crypto-portfolio-tracker/discussions)

## 📈 Roadmap

### Near-term Improvements
- [ ] Enhanced Binance trade fetching (all relevant pairs)
- [ ] Full USD pricing for non-USD trades (historical quote currency lookup)
- [ ] Complete fee USD pricing (historical fee currency lookup)
- [ ] LD (Earn/Staked) asset aggregation strategy
- [ ] Manual transaction import/editing interface

### Long-term Goals
- [ ] Support for additional exchanges (Coinbase, Kraken, etc.)
- [ ] Real-time price alerts and notifications
- [ ] Comprehensive tax reporting features
- [ ] Portfolio backtesting capabilities
- [ ] Web dashboard interface
- [ ] Mobile application
- [ ] DeFi protocol integration
- [ ] Advanced analytics and insights

---

**Made with ❤️ for the crypto community**
