# Crypto Portfolio Tracker

A comprehensive cryptocurrency portfolio tracking tool that connects to Binance and CoinGecko APIs to analyze your holdings, calculate P/L, and provide detailed portfolio insights.

## 🚀 Features

- **Secure API Integration**: Connect to Binance with read-only permissions
- **Comprehensive Tracking**: Transaction history, current balances, and cost basis calculation
- **Portfolio Analysis**: Compare actual vs target allocations with rebalancing suggestions
- **Multiple Export Formats**: Excel, HTML (mobile-friendly), and CSV exports
- **Visual Analytics**: Charts and graphs for portfolio visualization
- **Local Data Storage**: SQLite database for offline analysis
- **Professional Logging**: Structured logging with rotation

## 📁 Project Structure

```
crypto-portfolio-tracker/
├── src/
│   ├── __init__.py
│   ├── portfolio_tracker.py       # Main tracker class
│   ├── config.py                  # Configuration management
│   ├── database.py                # Database operations
│   ├── exporters.py               # Export functionality
│   └── visualizations.py          # Chart generation
├── config/
│   ├── .env.example              # Environment variables template
│   └── default_config.json       # Default configuration
├── data/                         # Database and exports
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── main.py                       # Entry point
└── setup.py                      # Installation script
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
```

4. **Run the tracker:**
```bash
python main.py
```

## 🔐 Binance API Setup

**IMPORTANT SECURITY STEPS:**

1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create a new API key with a descriptive name
3. **Enable ONLY "Enable Reading" permission** (disable all others for security)
4. Add your IP address to the whitelist
5. Copy the API Key and Secret to your `.env` file
6. **Never share your API secret with anyone**

## ⚙️ Configuration

The tracker uses a layered configuration system:

1. **Environment Variables** (`.env` file) - for sensitive data
2. **Configuration File** (`config/config.json`) - for settings
3. **Command Line Arguments** - for runtime options

### Target Portfolio Allocation

Edit `config/config.json` to set your desired allocation:

```json
{
  "target_allocation": {
    "BTC": 0.40,
    "ETH": 0.25,
    "SOL": 0.15,
    "RNDR": 0.10,
    "TAO": 0.10
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
  --verbose           Enable verbose logging
  --config CONFIG     Custom config file path
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

Select option (1-7):
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
CRYPTO PORTFOLIO SUMMARY
========================

Total Portfolio Value: $15,234.56
Total Cost Basis:      $12,000.00
Unrealized P/L:        $3,234.56 (+26.95%)

Asset    Quantity       Current Price  Value        P/L          Allocation  Target   Action
BTC      0.2500         $43,000.00     $10,750.00   +$3,750.00   45.2%      40.0%    SELL $789
ETH      5.0000         $2,200.00      $11,000.00   +$2,000.00   25.8%      25.0%    HOLD
SOL      25.0000        $65.00         $1,625.00    +$125.00     10.7%      15.0%    BUY $654
```

### Key Metrics
- **Total Value**: Current USD value of all holdings
- **Cost Basis**: Total amount invested
- **P/L**: Profit/Loss (unrealized)
- **Allocation**: Current vs target percentage
- **Action**: Buy/Sell/Hold recommendations

## 📁 Output Files

All output files are saved in the `data/` directory:

- **Excel Reports**: `portfolio_YYYYMMDD_HHMMSS.xlsx`
- **HTML Reports**: `portfolio_YYYYMMDD_HHMMSS.html` (mobile-friendly)
- **Charts**: `charts_YYYYMMDD_HHMMSS.png`
- **Database**: `portfolio.db` (SQLite)
- **Logs**: `logs/portfolio_tracker.log`

## 🔧 Advanced Configuration

### Logging Configuration

Modify logging in `src/config.py`:

```python
LOGGING_CONFIG = {
    'level': 'INFO',           # DEBUG, INFO, WARNING, ERROR
    'max_file_size': 10,       # MB
    'backup_count': 5,         # Number of log files to keep
    'console_output': True     # Show logs in console
}
```

### Database Settings

```python
DATABASE_CONFIG = {
    'path': 'data/portfolio.db',
    'backup_enabled': True,
    'backup_interval': 24,     # hours
    'cleanup_days': 90         # days to keep old data
}
```

### API Rate Limiting

```python
API_CONFIG = {
    'binance_rate_limit': 1200,    # requests per minute
    'coingecko_rate_limit': 100,   # requests per minute
    'retry_attempts': 3,
    'retry_delay': 5               # seconds
}
```

## 🐛 Troubleshooting

### Common Issues

**"API Connection Failed"**
```bash
# Check your .env file
cat .env

# Test API connectivity
python -c "from src.portfolio_tracker import CryptoPortfolioTracker; CryptoPortfolioTracker().test_connection()"
```

**"Permission Denied"**
- Ensure Binance API has "Enable Reading" permission only
- Check IP whitelist settings

**"No Data Found"**
- Run full sync first: `python main.py --sync-only`
- Check if you have any crypto in your Binance account

**"Price Data Missing"**
- Some tokens may not be available on CoinGecko
- Check logs for specific API errors: `tail -f logs/portfolio_tracker.log`

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
6. **Use strong passwords for any exports**

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

- [ ] Support for additional exchanges (Coinbase, Kraken, etc.)
- [ ] Real-time price alerts
- [ ] Tax reporting features
- [ ] Portfolio backtesting
- [ ] Web dashboard
- [ ] Mobile app
- [ ] DeFi protocol integration

---

**Made with ❤️ for the crypto community**
