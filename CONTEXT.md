# Crypto Portfolio Tracker - Project Context & State

**Last Updated:** 2026-05-04
**Version:** 2.1.2
**Project Root:** `/home/kenneth/Coding/projects/crypto-portfolio-tracker`

---

## 🎯 CRITICAL: Recent Changes (2026-05-04)

### Binance API Time Range Fix - 30-Day Chunking

**Problem:** "Query time range too large" errors (APIError code -6021):
```
ERROR - Error fetching Simple Earn subscriptions (Page 1): APIError(code=-6021): Query time range too large
ERROR - Error fetching Simple Earn redemptions (Page 1): APIError(code=-6021): Query time range too large
```
Binance Simple Earn endpoints have a **30-day max time range**, but code was fetching 90 days.

**Solution Implemented:**

1. **Added `_get_30day_chunks()`** - Helper method to split time range into 30-day chunks

2. **Added `_fetch_in_30day_chunks()`** - Generic method for chunked fetching with logging:
   - Logs each chunk: `[Binance Simple Earn Subscription] Fetching chunk 1/3: 2026-02-03 to 2026-03-05`
   - Handles exceptions per chunk (one failed chunk doesn't stop others)

3. **Updated three methods to use 30-day chunks:**
   - `fetch_simple_earn_subscriptions()` - Splits 90-day window into 3 chunks
   - `fetch_simple_earn_redemptions()` - Splits 90-day window into 3 chunks
   - `fetch_staking_history()` - Splits 90-day window into 3 chunks

**Files Modified:**
```
src/crypto_portfolio_tracker/binance_fetcher.py     (+204 lines, -128 lines)
```

**Testing Status:** ✅ All 57 tests passing

**Notes for Future:**
- Logs show chunk progress: "Fetching chunk 1/3", "Fetching chunk 2/3", etc.
- If Binance further reduces limits, chunk size is centralized in `_get_30day_chunks()`
- Simple Earn rewards endpoint still shows WARNING (deprecated by Binance)

---

## 🎯 CRITICAL: Recent Changes (2026-04-02)

### Connection Pool & Rate Limiting Improvements

**Problem:** urllib3 connection pool full warnings during sync:
```
WARNING - Connection pool is full, discarding connection: api.binance.com. Connection pool size: 10
```

**Solution Implemented:**

1. **Connection Pool Expansion** - Increased from 10 to 50 connections:
   - `pool_connections=50` (default)
   - `pool_maxsize=50` (default)
   - `pool_block=True` (requests wait instead of discarding)
   - Added logging to confirm configuration on init

2. **Request Delays** - Added configurable delays between API calls:
   - Added `_get_request_delay()` helper to BinanceFetcher (reads from config, default 500ms)
   - Applied to: Simple Earn pagination (rewards/subscriptions/redemptions)
   - Applied to: Staking history calls (between ETH/SOL/general endpoints)
   - Applied to: Trade fetch chunks (between each pair/time window)

3. **Concurrency Semaphore** - Limited concurrent fetcher tasks:
   - Added `asyncio.Semaphore(7)` to data_synchronizer.py
   - Wrapped all fetcher tasks in `wrapped_fetch()` function
   - Max 7 simultaneous Binance API tasks (previously 10-12)

**Files Modified:**
```
src/crypto_portfolio_tracker/data_synchronizer.py     (+20 lines, -5 lines)
src/crypto_portfolio_tracker/binance_fetcher.py       (+130 lines, -5 lines)
default_config.json.example                           (+3 lines)
```

**Testing Status:** ✅ All 57 tests passing

**Notes for Future:**
- Simple Earn rewards warning preserved (helpful deprecation notice)
- Request delay configurable via `apis.binance.request_delay_ms` in config
- Pool size configurable via `apis.binance.pool_connections/pool_maxsize`
- Semaphore limit hardcoded to 7 (can be made configurable if needed)

---

## 🎯 CRITICAL: Recent Changes (2026-03-07)

### Binance API Endpoint Updates - DEPRECATED ENDPOINTS FIXED ✅

**Problem:** Binance deprecated several API endpoints, causing ERROR messages in logs:
- `/sapi/v1/staking/history` - Completely deprecated
- `simple-earn/flexible/history/rewardRecord` - Deprecated/changed

**User Impact:**
- **Before Fix:** Multiple ERROR messages on every sync
  ```
  ERROR - Error fetching paginated data from staking/history: APIError(code=None): None
  ERROR - Error fetching Simple Earn rewards (Page 1): APIError(code=None): None
  ```
- **After Fix:** Clean logs with appropriate WARNING for deprecated endpoints
  ```
  WARNING - Simple Earn rewards endpoint unavailable... (helpful explanation)
  ```

**Solution Implemented:**

1. **Staking History** - Complete rewrite to use new 2026+ endpoints:
   - Added `_fetch_eth_staking_history()` - Uses `margin_v1_get_eth_staking_eth_history_*()` methods
   - Added `_fetch_sol_staking_history()` - Uses `margin_v1_get_sol_staking_sol_history_*()` methods
   - Added `_fetch_general_staking_rewards()` - Uses `get_staking_rewards_history_us()`
   - All exceptions logged as WARNING (not ERROR) to reduce noise

2. **Simple Earn Rewards** - Updated error handling:
   - Changed from ERROR to WARNING level logging
   - Added helpful explanation: auto-earn rewards compound automatically
   - Users informed that rewards aren't separate transactions with auto-earn

**Files Modified:**
```
src/crypto_portfolio_tracker/binance_fetcher.py     (+265 lines, -56 lines)
```

**Testing Status:** ✅ Verified working
- Sync completes successfully with no ERROR messages
- Staking endpoints gracefully skip (no staking positions)
- Simple Earn subscriptions/redemptions still work correctly
- Simple Earn rewards show informative WARNING (endpoint deprecated)

**Notes for Future:**
- When you enable ETH/SOL staking, transactions will auto-sync
- Simple Earn rewards are compounded with auto-earn (no separate transactions)
- For detailed reward history, export from Binance web interface

---

## 🎯 CRITICAL: Recent Changes (2026-02-22)

### P2P Sell Transaction Support - IMPLEMENTED ✅

**Problem Fixed:** Portfolio tracker showed false losses (~99%) because P2P sell transactions (cashing out crypto to fiat) were not tracked.

**User Impact (Kenneth's Portfolio):**
- **Before Fix:** Invested Capital $211.71, P/L -$210.26 (-99.32%) ← FALSE
- **After Fix:** Invested Capital $26.00, P/L -$24.55 (-94.45%) ← ACCURATE
- **Root Cause:** $185.71 USDT P2P sell (cash-out on 2026-01-18) was never subtracted from invested capital

**Solution Implemented:**
Added complete P2P sell tracking throughout the entire pipeline:

1. **Transaction Type** - Added `P2P_SELL = "P2P_SELL"` to `TransactionType` enum
2. **Fetching** - Created `fetch_p2p_usdt_sells()` in `BinanceFetcher` (mirrors P2P buy logic)
3. **Sync Pipeline** - Added "Binance P2P Sell" to `data_synchronizer.py` sources
4. **Enrichment** - Added P2P_SELL handling in `PriceEnricher` (records USDT at $1.0, type=SELL)
5. **Database** - Updated query to include P2P sells in invested capital calculation
6. **Analytics** - Updated `calculate_total_invested_capital()` to subtract P2P sells as outflows
7. **Schema** - Database CHECK constraint updated to include P2P_SELL

**Files Modified (7 code + 2 docs):**
```
src/crypto_portfolio_tracker/models.py              (+1 line)
src/crypto_portfolio_tracker/binance_fetcher.py     (+145 lines)
src/crypto_portfolio_tracker/data_synchronizer.py   (+4 lines)
src/crypto_portfolio_tracker/database.py            (+4 lines, -16 lines)
src/crypto_portfolio_tracker/portfolio_analyzer.py  (+31 lines, -16 lines)
src/crypto_portfolio_tracker/price_enricher.py      (+21 lines, -1 line)
default_config.json.example                         (+1 line)
docs/P2P_SELL_TRACKING_ISSUE.md                     (new - comprehensive issue doc)
docs/migrate_p2p_sell_schema.py                     (new - migration script)
```

**Migration for Existing Users:**
- Schema auto-updates on first run via `_check_and_update_schema()`
- If manual migration needed: `python docs/migrate_p2p_sell_schema.py`
- Then run full sync to fetch historical P2P sells

**Testing Status:** ✅ Verified working
- P2P sell fetched from Binance API
- Transaction saved to database with correct type/price
- Invested capital calculation includes P2P sell outflow
- Portfolio summary shows accurate P/L

---

## 📦 Project Overview

**Purpose:** Comprehensive personal cryptocurrency portfolio tracking and rebalancing advisor built on modern Python stack.

**Core Value Propositions:**
1. **Dual Accounting Perspectives:**
   - **FIFO Cost Basis:** First-In-First-Out for per-asset unrealized P/L (tax-lot accounting)
   - **Net Invested Capital:** True cash-in vs cash-out for overall portfolio performance

2. **Context-Aware Portfolio View:**
   - **Core Holdings:** Assets in target allocation (used for rebalancing)
   - **Other Holdings:** All other assets (complete financial overview)

3. **Complete Transaction Syncing:**
   - Spot Trades & P2P Trades (fiat on/off ramps)
   - Deposits & Withdrawals
   - Simple Earn (subscriptions, redemptions, rewards)
   - Staking (subscriptions, redemptions, interest)
   - Dividends, Internal Transfers, Convert history

4. **Intelligent Rebalancing:**
   - Multi-timeframe technical analysis integration
   - Different thresholds for majors (BTC/ETH) vs alts
   - Bear market detection (BTC < SMA200)
   - Signal strength amplification near support/resistance

5. **Advanced Features:**
   - Dollar Cost Averaging (proportional & target-weight)
   - Profit-taking opportunities (multi-factor scoring)
   - Backtesting engine (1:1 simulation of rebalancing strategy)
   - Manual trading interface
   - Fund transfers between wallets

**Tech Stack:**
- **Language:** Python 3.10+
- **Package Manager:** `uv` (lightning-fast)
- **Web Framework:** Streamlit
- **Data Processing:** pandas, numpy, pandas_ta
- **APIs:** python-binance, CoinGecko, yFinance
- **Database:** SQLite with diskcache caching
- **Visualization:** matplotlib, seaborn, plotly

---

## 🏗️ Architecture Deep Dive

### Design Patterns

1. **Facade Pattern** - `CryptoPortfolioTracker` orchestrates specialized components
2. **Separation of Concerns** - Clear boundaries between fetching, processing, storage, execution
3. **Repository Pattern** - `DatabaseManager` abstracts all persistence logic
4. **Strategy Pattern** - Rebalancing logic pluggable for different strategies

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CryptoPortfolioTracker                            │
│                           (Facade / Orchestrator)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ DataSynchronizer │  │ PortfolioAnalyzer│  │  TradeExecutor   │      │
│  │                  │  │                  │  │                  │      │
│  │ - Sync pipeline  │  │ - Metrics calc   │  │ - Execute trades │      │
│  │ - Binance client │  │ - Rebalancing    │  │ - Lot size adj   │      │
│  │ - Time sync      │  │ - Profit-taking  │  │ - Earn redemption│      │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────────┘      │
│           │                     │                                       │
│  ┌────────▼─────────┐  ┌────────▼─────────┐  ┌──────────────────┐      │
│  │  BinanceFetcher  │  │  PriceEnricher   │  │   DCAManager     │      │
│  │                  │  │                  │  │                  │      │
│  │ - Balances       │  │ - CoinGecko API  │  │ - DCA calc       │      │
│  │ - Transactions   │  │ - yFinance       │  │ - Validation     │      │
│  │ - P2P (buy/sell) │  │ - Fiat rates     │  │ - Execution      │      │
│  │ - Earn/Staking   │  │ - Price caching  │  │                  │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ DatabaseManager  │  │ CryptoTrend      │  │ ReportGenerator  │      │
│  │                  │  │ Analyzer         │  │                  │      │
│  │ - SQLite DB      │  │                  │  │ - Excel export   │      │
│  │ - Schema migr.   │  │ - Multi-timeframe│  │ - HTML export    │      │
│  │ - Backups        │  │ - RSI, MACD, SMA │  │ - CSV backup     │      │
│  │ - Transactions   │  │ - Support/Resist │  │ - Charts         │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼───────┐ ┌────▼────┐ ┌───────▼───────┐
            │   Binance     │ │CoinGecko│ │   yFinance    │
            │     API       │ │   API   │ │     API       │
            └───────────────┘ └─────────┘ └───────────────┘
```

### Core Components Detail

#### 1. DataSynchronizer (`data_synchronizer.py`)
**Responsibility:** Orchestrates data synchronization pipeline

**Key Methods:**
- `_init_binance_client()` - Initialize with retry logic, time sync
- `_sync_binance_client_time()` - Prevent recvWindow errors
- `_get_current_prices()` - Batch fetch from CoinGecko
- `_get_coingecko_historical_price()` - With fallback to nearest date
- `sync_data()` - Main pipeline: Gather → Enrich → Process

**Data Sources Synced:**
```python
{
    "Binance Trade": fetch_binance_transactions,
    "Binance Transfer": fetch_transfer_history,
    "Binance Deposit": fetch_deposit_history,
    "Binance Withdrawal": fetch_withdrawal_history,
    "Binance P2P Buy": fetch_p2p_usdt_buys,
    "Binance P2P Sell": fetch_p2p_usdt_sells,  # NEW
    "Binance Convert": fetch_spot_convert_history,
    "Binance Dividend": fetch_dividend_history,
    "Binance Simple Earn Reward": fetch_simple_earn_rewards,
    "Binance Simple Earn Subscription": fetch_simple_earn_subscriptions,
    "Binance Simple Earn Redemption": fetch_simple_earn_redemptions,
    "Binance Staking": fetch_staking_history,  # Subscription/Redemption/Interest
}
```

**Sync Strategy:**
- Incremental: Only fetches new transactions since last sync
- Stores latest timestamp per source in database
- Concurrent fetching with asyncio.gather()
- Testnet mode: Only fetches spot trades (limited API)

#### 2. BinanceFetcher (`binance_fetcher.py`)
**Responsibility:** All Binance API interactions

**Key Features:**
- Centralized error handling (`_handle_binance_api_exception`)
- Pagination support for large datasets
- Rate limiting awareness
- Transaction hash generation for deduplication

**P2P Sell Implementation:**
```python
def fetch_p2p_usdt_sells(self, source_name, days_back=90, latest_known_ts=None):
    """Fetch P2P SELL transactions (user cashing out to fiat)"""
    # Uses tradeType="SELL" (vs "BUY" for P2P buys)
    # Filters for orderStatus == "COMPLETED"
    # De-duplicates by orderNumber
    # Returns transactions with:
    #   - tx_type: "P2P_SELL"
    #   - symbol: "USDT"
    #   - quantity: USDT amount sold
    #   - fiat_amount: Fiat received (PHP, USD, etc.)
```

#### 3. PriceEnricher (`price_enricher.py`)
**Responsibility:** Fetch historical prices, enrich transactions with USD values

**Key Features:**
- Disk caching (CoinGecko, yFinance, fiat rates)
- Batch price fetching for efficiency
- Fallback logic for missing prices

**P2P Sell Enrichment:**
```python
elif tx["tx_type"] == "P2P_SELL":
    # USDT is pegged to USD, so price = $1.0
    enriched_transactions.append({
        "symbol": raw["asset"],  # USDT
        "timestamp": ts,
        "type": "SELL",
        "quantity": raw["quantity"],
        "price_usd": 1.0,  # Stablecoin peg
        "source": "Binance P2P Sell",
        "transaction_hash": tx["transaction_hash"],
        "notes": f"P2P Sell: {raw['fiat_amount']} {raw['fiat_currency']}",
    })
```

#### 4. DatabaseManager (`database.py`)
**Responsibility:** SQLite persistence, schema migrations, backups

**Schema Version:** 2 (includes P2P_SELL in CHECK constraint)

**Tables:**
```sql
-- Assets registry
assets (id, symbol, name, coingecko_id)

-- All transactions (CHECK constraint on type includes P2P_SELL)
transactions (
    id, asset_id, timestamp, type, quantity, price_usd,
    fee_quantity, fee_currency, fee_usd, source, notes,
    transaction_hash, created_at
)

-- Current holdings (quantity, avg cost basis)
holdings (asset_id, quantity, average_cost_basis, last_updated)

-- Historical prices for backtesting
historical_prices (asset_id, date, price_usd)

-- Portfolio snapshots for tracking over time
portfolio_snapshots (timestamp, total_value_usd, total_cost_basis_usd,
                     unrealized_pl_usd, unrealized_pl_percent)

-- Schema versioning
schema_version (version, updated_at)
```

**Migration System:**
- `_check_and_update_schema()` runs on initialization
- Target version: 2
- Migration v2: Updates CHECK constraint to include P2P_SELL
- Auto-backup before schema changes

#### 5. PortfolioAnalyzer (`portfolio_analyzer.py`)
**Responsibility:** Calculate portfolio metrics, generate rebalancing suggestions

**Key Metrics:**
```python
def calculate_total_invested_capital(self) -> float:
    """
    Calculate NET invested capital:
    + P2P Buys (buying crypto with fiat)
    - Withdrawals (crypto leaving Binance)
    - P2P Sells (cashing out crypto to fiat)  # NEW
    """
    p2p_buys = sum(qty * price for P2P Buy transactions)
    withdrawals = sum(qty * price for Withdrawal transactions)
    p2p_sells = sum(qty * price for P2P Sell transactions)  # NEW
    
    return p2p_buys - withdrawals - p2p_sells
```

**Rebalancing Logic:**
- Fetches live balances (Spot + Earn)
- Filters to core portfolio (target_allocation)
- Gets technical analysis (swing + long-term timeframes)
- Detects market regime (bear if BTC < SMA200)
- Calculates drift vs target for each asset
- Applies thresholds (different for majors/alts)
- Generates BUY/SELL/HOLD signals with action amounts

#### 6. CryptoTrendAnalyzer (`crypto_trend_analyzer.py`)
**Responsibility:** Multi-timeframe technical analysis

**Timeframes:**
- **Long-term:** 4 years, SMA50/SMA200, for market regime detection
- **Swing:** 3 months, SMA10/SMA30, for rebalancing signals
- **Day:** 60 days, SMA5/SMA15, for short-term opportunities

**Indicators:**
- RSI (14-period, oversold <30, overbought >70)
- MACD (bullish/bearish crossovers)
- SMA crossovers (Golden Cross, Death Cross)
- Support/Resistance (30-day min/max)

**Market Regime Detection:**
```python
def get_market_regime(long_term_report) -> bool:
    """Returns True if bear market (BTC < SMA200)"""
    btc_analysis = long_term_report.get("benchmark_analysis")
    return TrendCondition.PRICE_BELOW_SMA200.value in btc_analysis.get("active_conditions", [])
```

#### 7. TradeExecutor (`trade_executor.py`)
**Responsibility:** Execute trades with proper validation

**Features:**
- Lot size adjustment (Binance filters)
- Earn redemption (auto-redeem before selling)
- Profit-taking execution
- Manual trade execution
- Rebalancing trade execution (bulk or interactive)

#### 8. DCAManager (`dca_manager.py`)
**Responsibility:** Dollar-cost averaging calculations and execution

**Strategies:**
- **Proportional DCA:** Distribute to maintain current proportions
- **Target-Weight DCA:** Allocate to reach target allocation

**Validation:**
- Check available USDT balance (Spot + Earn)
- Validate against minimum trade size ($5 USD)
- Preview trades before execution

---

## 🔄 Data Flow Deep Dive

### Transaction Sync Flow

```
1. User triggers "Full Sync & Analysis"
         │
2. DataSynchronizer.sync_data()
         │
3. For each data source:
   ├─ Check latest timestamp in DB
   ├─ Calculate fetch window (incremental or full)
   └─ Launch async fetch task
         │
4. BinanceFetcher fetches raw data:
   ├─ P2P Sells: get_c2c_trade_history(tradeType="SELL")
   ├─ Trades: get_my_trades() for all pairs
   ├─ Deposits: get_deposit_history()
   ├─ Withdrawals: get_withdraw_history()
   └─ etc...
         │
5. Raw transactions returned with structure:
   {
       "tx_type": "P2P_SELL",
       "timestamp": datetime,
       "source": "Binance P2P Sell",
       "transaction_hash": orderNumber,
       "raw_data": {
           "asset": "USDT",
           "quantity": 185.71,
           "fiat_currency": "PHP",
           "fiat_amount": 10999.6
       }
   }
         │
6. PriceEnricher.enrich_transactions():
   ├─ For P2P_SELL: Set price_usd = 1.0 (USDT peg)
   ├─ For trades: Fetch historical quote asset price
   ├─ For deposits: Fetch historical asset price
   └─ Output enriched transactions
         │
7. DatabaseManager.bulk_insert_transactions():
   ├─ INSERT ... ON CONFLICT DO UPDATE
   ├─ Deduplicates by transaction_hash
   └─ Returns count of inserted/updated rows
         │
8. PortfolioAnalyzer.calculate_portfolio_metrics():
   ├─ Fetch live balances (Spot, Earn, Futures, Funding)
   ├─ Fetch cost basis from DB holdings
   ├─ Get current prices from CoinGecko
   ├─ Calculate allocations, P/L, invested capital
   └─ Return comprehensive metrics dict
```

### Invested Capital Calculation Flow

```
1. PortfolioAnalyzer.calculate_total_invested_capital()
         │
2. DatabaseManager.get_invested_capital_transactions()
   Query:
   SELECT source, type, quantity, price_usd
   FROM transactions
   WHERE source IN ('Binance P2P Buy', 'Binance P2P Sell')
      OR type = 'WITHDRAWAL'
         │
3. Calculate:
   inflows  = Σ(P2P Buy: quantity × price_usd)
   outflows = Σ(Withdrawals: quantity × price_usd)
            + Σ(P2P Sell: quantity × price_usd)  # NEW
         │
4. Result:
   net_invested = inflows - outflows
   
   Example (Kenneth's portfolio):
   - P2P Buys:      $211.71 (capital in)
   - P2P Sells:     $185.71 (capital out - cashed out)  # NEW
   - Withdrawals:   $0.00
   ─────────────────────────────
   Net Invested:    $26.00 (actual remaining capital)
```

### Rebalancing Suggestion Flow

```
1. User navigates to Rebalance page
         │
2. PortfolioAnalyzer.get_core_portfolio_rebalance_suggestions_technical()
         │
3. Fetch live data:
   ├─ Balances: Spot + Earn wallets
   ├─ Prices: CoinGecko current prices
   └─ Filter to core assets (target_allocation)
         │
4. Generate technical reports (async):
   ├─ Swing report (3-month timeframe)
   └─ Long-term report (4-year timeframe)
         │
5. Detect market regime:
   └─ Bear market if BTC has "Price < SMA200" condition
         │
6. For each core asset:
   ├─ Calculate current allocation %
   ├─ Calculate drift from target %
   ├─ Apply threshold (3% for majors, 5-7% for alts)
   ├─ Check technical conditions (oversold, near support)
   ├─ Apply bear market rules (suppress buys if enabled)
   └─ Generate signal: BUY / SELL / HOLD
         │
7. Display suggestions with:
   ├─ Allocation drift
   ├─ Current value
   ├─ Technical indicators (RSI, MACD, SMA)
   ├─ Support/Resistance levels
   └─ Suggested action with quantity
```

---

## 🗄️ Database Schema (Complete)

### transactions Table (CHECK Constraint)
```sql
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    type TEXT NOT NULL CHECK (type IN (
        'BUY', 'SELL', 'DEPOSIT', 'WITHDRAWAL',
        'P2P_BUY', 'P2P_SELL',  -- P2P_SELL added in schema v2
        'CONVERT', 'EARN_REWARD', 'EARN_SUBSCRIPTION', 'EARN_REDEMPTION',
        'DIVIDEND', 'STAKING_SUBSCRIBE', 'STAKING_REDEMPTION', 'STAKING_INTEREST',
        'TRANSFER', 'TRANSFER_IN', 'TRANSFER_OUT'
    )),
    quantity REAL NOT NULL,
    price_usd REAL,
    fee_quantity REAL,
    fee_currency TEXT,
    fee_usd REAL,
    source TEXT,
    notes TEXT,
    transaction_hash TEXT UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets (id)
);
```

### Indexes
```sql
CREATE INDEX idx_transactions_asset_timestamp 
    ON transactions (asset_id, timestamp);

CREATE INDEX idx_historical_prices_asset_date 
    ON historical_prices (asset_id, date);
```

---

## ⚙️ Configuration System

### Configuration Hierarchy

1. **Environment Variables** (`.env`)
   - `MAIN_API_KEY`, `MAIN_API_SECRET` - Production Binance API
   - `TESTNET_API_KEY`, `TESTNET_API_SECRET` - Testnet Binance API
   - `COINGECKO_API_KEY` - CoinGecko API (optional, for higher rate limits)

2. **Runtime Config** (`config/default_config.json`)
   - Target allocation
   - Asset classes (majors vs alts)
   - Rebalancing thresholds
   - History lookback periods
   - Technical analysis settings
   - Profit-taking rules

3. **Code Defaults** (fallback if config missing)

### Key Configuration Sections

```json
{
  "portfolio": {
    "testnet_mode": false,
    "live_trading_enabled": true,
    "minimum_trade_usd": 5.0,
    "p2p_fiat_currency": "PHP",
    "stablecoin_symbols": ["USDT"]
  },
  
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
    "market_regime_rules": {
      "suppress_buys_in_bear": true
    },
    "majors": {
      "allocation_drift_threshold_pct": 3.0,
      "sell_percentage_multiplier": 0.5,
      "buy_amount_multiplier": 0.75
    },
    "alts": {
      "allocation_drift_threshold_pct": 5.0,
      "sell_percentage_multiplier": 0.5,
      "buy_amount_multiplier": 1.0
    }
  },
  
  "history_lookback_days": {
    "trades": 90,
    "deposits": 90,
    "withdrawals": 90,
    "p2p_buys": 90,
    "p2p_sells": 90,
    "internal_transfers": 90,
    "spot_futures_transfers": 90,
    "spot_convert_history": 90,
    "simple_earn_rewards": 90,
    "simple_earn_subscriptions": 90,
    "simple_earn_redemptions": 90,
    "dividend_history": 90,
    "staking_history": 90
  },
  
  "trend_analyzer": {
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
  },
  
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

---

## 🚀 Usage Patterns

### Web UI (Recommended for Daily Use)

```bash
uv run track-portfolio-web
```

**Pages:**
- 🏠 **Home** - Portfolio overview, charts, quick stats
- 💳 **Wallets** - Detailed wallet breakdowns (Spot, Earn, Futures, Funding)
- 📈 **Market** - Technical analysis, trend reports, multi-timeframe view
- ⚖️ **Rebalance** - Rebalancing suggestions with interactive controls
- 💸 **DCA** - Dollar-cost averaging wizard
- 💰 **Trade** - Manual trading interface
- 🧪 **Backtest** - Strategy backtesting with parameter controls
- 🗄️ **Database** - Data management, backups, cleanup
- ⚙️ **Settings** - Configuration management

### CLI (For Debugging & Automation)

```bash
uv run track-portfolio-cli
```

**Menu Options:**
```
1. 🔄 Full Sync & Analysis      - Complete data sync + metrics calculation
2. 💰 Quick Portfolio Summary   - Fast metrics from cached data
3. 📈 View Trends               - Technical analysis reports
4. ⚖️  Rebalance                - Generate rebalancing suggestions
5. 💸 Dollar Cost Averaging     - DCA calculation and execution
6. 🔀 Trade                     - Manual trade execution
7. 💵 Transfer Funds            - Wallet-to-wallet transfers
8. 🧪 Backtest                  - Strategy backtesting
9. 📋 Reports                   - Generate Excel/HTML/CSV reports
10. 📊 Charts                   - Create portfolio visualizations
11. 🗄️  Database                - Data management tools
12. 🧹 Data Cleanup             - Remove old data per retention policy
13. ⚙️  View Configuration      - Display current config
14. 🔧 Test Connections         - Verify API connectivity
15. ❌ Exit
```

---

## 🧪 Testing Infrastructure

### Test Files
- `tests/conftest.py` - Pytest fixtures (mock config, database, Binance client)
- `tests/test_api_integration.py` - Live API integration tests
- `tests/test_binance_fetcher.py` - Fetcher unit tests
- `tests/test_database.py` - Database operations tests
- `tests/test_portfolio_tracker.py` - Main tracker tests
- `tests/test_portfolio_tracker_comprehensive.py` - End-to-end tests
- `tests/test_price_enricher.py` - Price enrichment tests
- `tests/test_offline_mode.py` - Offline functionality tests
- `tests/test_testnet_mode.py` - Testnet mode tests
- `tests/test_production_confidence.py` - Production readiness tests

### Running Tests
```bash
uv run pytest                    # Run all tests
uv run pytest -v                 # Verbose output
uv run pytest --cov=src          # With coverage
uv run pytest tests/test_database.py -v  # Specific test file
```

---

## 📊 Current Portfolio State (Kenneth's)

**As of 2026-02-22 16:59:**

### Summary
- **Total Portfolio Value:** $1.44
- **Total Invested Capital:** $26.00
- **Overall P/L:** -$24.55 (-94.45%)
- **Total Cost Basis (FIFO):** $199.75
- **Unrealized P/L (FIFO):** -$198.81 (-99.53%)

### Wallet Breakdown
- Spot & Earn Value: $0.94
- Futures Wallet Value: $0.50
- Funding Wallet Value: $0.00

### Holdings (Dust Amounts)
| Asset | Quantity | Value (USD) | Cost Basis | P/L |
|-------|----------|-------------|------------|-----|
| BTC | 9.33e-06 | $0.63 | $0.91 | -$0.28 |
| BNB | 0.00020355 | $0.13 | $0.00 | +$0.13 |
| LINK | 0.00961133 | $0.08 | $0.13 | -$0.05 |
| ETH | 2.162e-05 | $0.04 | $0.06 | -$0.02 |
| SOL | 0.00028173 | $0.02 | $0.05 | -$0.02 |
| AVAX | 0.00122818 | $0.01 | $0.03 | -$0.01 |
| RENDER | 0.0046801 | $0.01 | $0.02 | -$0.01 |
| TAO | 3.029e-05 | $0.01 | $0.01 | -$0.01 |
| ONDO | 0.01294673 | $0.00 | $0.01 | -$0.01 |

### P2P Transaction History
| Date | Type | Quantity (USDT) | Fiat Amount | Notes |
|------|------|-----------------|-------------|-------|
| 2026-01-18 | P2P Sell | 185.71 | 10,999.60 PHP | Cash out |
| 2025-09-08 | P2P Buy | 20.96 | 1,200.00 PHP | |
| 2025-08-04 | P2P Buy | 68.37 | 4,000.00 PHP | |
| 2025-06-07 | P2P Buy | 17.83 | 1,000.00 PHP | |
| ... | ... | ... | ... | (6 more P2P buys) |

**Total P2P Buys:** $211.71  
**Total P2P Sells:** $185.71  
**Net Invested:** $26.00 ✓

---

## 🔐 Security Considerations

### API Key Security
- Keys stored in `.env` (gitignored)
- Recommended: **Read-only** API permissions
- Testnet mode for safe testing
- Never commit API keys to git

### Database Security
- Local SQLite database (not exposed to network)
- Automatic backups before schema changes
- Configurable backup retention

### Trading Security
- `live_trading_enabled` flag to disable live trades
- Confirmation prompts before executing trades
- Testnet mode for testing without real funds

---

## 📚 Documentation Files

- `README.md` - Main user documentation
- `docs/P2P_SELL_TRACKING_ISSUE.md` - P2P sell issue analysis and resolution
- `docs/migrate_p2p_sell_schema.py` - Manual schema migration script
- `.qwen/QWEN.md` - This file (AI assistant context)

---

## 🐛 Known Issues & Resolutions

### ✅ Resolved Issues

1. **P2P Sell Tracking (2026-02-22)**
   - **Symptom:** Portfolio shows ~99% loss after cashing out via P2P
   - **Cause:** P2P sell transactions not tracked
   - **Fix:** Added complete P2P sell support throughout pipeline
   - **Migration:** Schema auto-updates; run full sync to fetch history

### ⚠️ Current Limitations

1. **Testnet Mode Limitations**
   - Only syncs spot trade history (not full transaction set)
   - Some endpoints not available on testnet

2. **API Rate Limits**
   - CoinGecko free tier has rate limits
   - Mitigated by disk caching and batch requests

3. **Historical Data Gaps**
   - Some assets may not have historical data for full backtest period
   - Backtester excludes assets with missing data

---

## 🔧 Development Quick Reference

### Project Structure
```
crypto-portfolio-tracker/
├── src/crypto_portfolio_tracker/
│   ├── dashboard/           # Streamlit web UI
│   ├── templates/           # HTML report templates
│   ├── __main__.py          # CLI entry point
│   ├── binance_fetcher.py   # Binance API fetching
│   ├── config.py            # Configuration management
│   ├── data_synchronizer.py # Data sync orchestration
│   ├── database.py          # SQLite database manager
│   ├── portfolio_analyzer.py# Metrics & rebalancing
│   ├── price_enricher.py    # Price fetching & enrichment
│   ├── trade_executor.py    # Trade execution
│   └── ... (other modules)
├── tests/                   # Test suite
├── config/                  # Runtime configuration
├── data/                    # Database, cache, exports
├── logs/                    # Application logs
└── docs/                    # Documentation
```

### Common Commands
```bash
# Development
uv sync                     # Install dependencies
uv run python main.py       # Run CLI
uv run track-portfolio-web  # Run web UI
uv run pytest               # Run tests
uv run ruff check .         # Lint
uv run ruff format .        # Format

# Database
python docs/migrate_p2p_sell_schema.py  # Manual schema migration

# Configuration
cp .env.example .env                    # Create .env
cp default_config.json.example config/default_config.json  # Create config
```

### Debugging Tips
1. **Check logs:** `logs/portfolio_tracker.log`
2. **Verify DB:** `sqlite3 data/portfolio.db "SELECT * FROM transactions LIMIT 5;"`
3. **Test connection:** CLI option 14 (Test Connections)
4. **Force full sync:** Delete `data/portfolio.db` and resync

---

**End of Context File**  
*This file should be updated whenever significant changes are made to the project architecture or functionality.*
