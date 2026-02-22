# P2P Sell Transaction Tracking Issue

**Created:** 2026-02-22  
**Status:** Identified - Fix Pending  
**Severity:** High - Causes incorrect P/L calculations when users cash out via P2P

---

## Problem Summary

When users sell their crypto assets via **P2P (Peer-to-Peer) trading** on Binance and cash out to fiat, the portfolio tracker does not record these transactions. This results in:

1. **Incorrectly high cost basis** - The system still thinks you hold the assets you sold
2. **Massive false losses** - P/L shows as if you sold at a loss, when you actually cashed out
3. **Inaccurate invested capital calculation** - Capital returned via P2P sales is not subtracted

### Example from Production

After user sold all assets and cashed out via P2P:

```
TOTAL PORTFOLIO VALUE:       $1.44
Total Invested Capital:      $211.71
Overall P/L:                 $-210.26 (-99.32%)  ← FALSE LOSS

Total Cost Basis (FIFO):     $199.75
Unrealized P/L (FIFO):       $-198.81 (-99.53%)  ← Still shows holdings at loss
```

**Reality:** User cashed out ~$200+ via P2P sales, not at a loss.

---

## Root Cause Analysis

### 1. Missing P2P Sell Fetching

**File:** `src/crypto_portfolio_tracker/binance_fetcher.py`

The `fetch_p2p_usdt_buys()` method only fetches **BUY** transactions:

```python
history = self.binance_client.get_c2c_trade_history(
    tradeType="BUY",  # ← Only fetches P2P BUYS
    page=current_page,
    rows=PAGE_SIZE,
    ...
)
```

**Problem:** When you sell USDT for fiat (cash out), Binance records this as a **SELL** trade, which is never fetched.

### 2. Missing Transaction Type

**File:** `src/crypto_portfolio_tracker/models.py`

The `TransactionType` enum has `P2P_BUY` but no `P2P_SELL`:

```python
class TransactionType(Enum):
    P2P_BUY = "P2P_BUY"
    # P2P_SELL is missing!
```

### 3. Incomplete Invested Capital Calculation

**File:** `src/crypto_portfolio_tracker/portfolio_analyzer.py`

The `calculate_total_invested_capital()` method only accounts for:
- **Inflows:** P2P Buys (buying crypto with fiat)
- **Outflows:** Withdrawals (crypto leaving Binance)

```python
def calculate_total_invested_capital(self) -> float:
    # Inflows: P2P Buys
    p2p_transactions = transactions_df[transactions_df['source'] == 'Binance P2P Buy']
    net_invested += (p2p_transactions['quantity'] * p2p_transactions['price_usd']).sum()
    
    # Outflows: Withdrawals only
    withdrawal_transactions = transactions_df[transactions_df['type'] == 'WITHDRAWAL']
    net_invested -= (withdrawal_transactions['quantity'] * withdrawal_transactions['price_usd']).sum()
    
    # P2P Sells (capital returned) are NOT subtracted!
```

**Problem:** P2P sales represent **capital returned to the user** but are not subtracted from invested capital.

### 4. Database Query Excludes P2P Sells

**File:** `src/crypto_portfolio_tracker/database.py`

```python
def get_invested_capital_transactions(self) -> pd.DataFrame:
    query = """
        SELECT source, type, quantity, price_usd
        FROM transactions
        WHERE source = 'Binance P2P Buy' OR type = 'WITHDRAWAL';
        # P2P Sell transactions are not queried!
    """
```

---

## How P2P Trading Works on Binance

| Action | What Happens | Binance API `tradeType` |
|--------|--------------|------------------------|
| **P2P Buy** | You buy USDT with fiat (PHP, USD, etc.) | `"BUY"` |
| **P2P Sell** | You sell USDT for fiat (cash out) | `"SELL"` |

**Current State:** Only P2P Buy is tracked → System knows when you add capital, but not when you withdraw capital via P2P.

---

## Impact

### Affected Users
- Users who cash out via P2P sales (common in many countries)
- Users who partially exit positions via P2P
- Anyone using P2P for fiat on/off ramps

### Symptoms
- Portfolio shows 90-99% loss after cashing out
- Cost basis remains high even with zero holdings
- "Overall P/L" shows massive negative values
- Rebalancing suggestions may be incorrect

### Not Affected
- Users who only withdraw via crypto withdrawal (already tracked)
- Users who haven't cashed out via P2P

---

## Planned Fix

### 1. Add P2P_SELL Transaction Type

**File:** `src/crypto_portfolio_tracker/models.py`

```python
class TransactionType(Enum):
    P2P_BUY = "P2P_BUY"
    P2P_SELL = "P2P_SELL"  # ← Add this
```

### 2. Create P2P Sell Fetching Method

**File:** `src/crypto_portfolio_tracker/binance_fetcher.py`

Option A: Rename and refactor existing method:
```python
def fetch_p2p_transactions(self, source_name: str, days_back: int = 90, 
                           latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict]:
    """Fetch both P2P BUY and P2P SELL transactions."""
    
    # Fetch P2P Buys (existing logic)
    buy_transactions = self._fetch_p2p_trades(tradeType="BUY", ...)
    
    # Fetch P2P Sells (new logic)
    sell_transactions = self._fetch_p2p_trades(tradeType="SELL", ...)
    
    return buy_transactions + sell_transactions
```

Option B: Add separate method:
```python
def fetch_p2p_usdt_sells(self, source_name: str, days_back: int = 90,
                         latest_known_ts: Optional[datetime.datetime] = None) -> List[Dict]:
    """Fetch P2P SELL transactions (when user cashes out)."""
    
    # Similar to fetch_p2p_usdt_buys but with tradeType="SELL"
    # Records as P2P_SELL transaction type
```

### 3. Update Data Synchronizer

**File:** `src/crypto_portfolio_tracker/data_synchronizer.py`

Add P2P Sell to the sync pipeline:

```python
all_data_sources = {
    "Binance P2P Buy": {...},
    "Binance P2P Sell": {  # ← Add this
        "fetcher": self.fetcher.fetch_p2p_usdt_sells,
        "days": lookback_config.get("p2p_sells", 90),
    },
    ...
}
```

### 4. Update Invested Capital Calculation

**File:** `src/crypto_portfolio_tracker/portfolio_analyzer.py`

```python
def calculate_total_invested_capital(self) -> float:
    transactions_df = self.db_manager.get_invested_capital_transactions()
    
    # Inflows: P2P Buys (buying crypto with fiat)
    p2p_buys = transactions_df[transactions_df['source'] == 'Binance P2P Buy']
    net_invested += (p2p_buys['quantity'] * p2p_buys['price_usd']).sum()
    
    # Outflows: Withdrawals (crypto leaving Binance)
    withdrawals = transactions_df[transactions_df['type'] == 'WITHDRAWAL']
    net_invested -= (withdrawals['quantity'] * withdrawals['price_usd']).sum()
    
    # Outflows: P2P Sells (cashing out to fiat) ← Add this
    p2p_sells = transactions_df[transactions_df['source'] == 'Binance P2P Sell']
    net_invested -= (p2p_sells['quantity'] * p2p_sells['price_usd']).sum()
    
    return float(net_invested)
```

### 5. Update Database Query

**File:** `src/crypto_portfolio_tracker/database.py`

```python
def get_invested_capital_transactions(self) -> pd.DataFrame:
    query = """
        SELECT source, type, quantity, price_usd
        FROM transactions
        WHERE source IN ('Binance P2P Buy', 'Binance P2P Sell') 
           OR type = 'WITHDRAWAL';
    """
```

### 6. Schema Migration (if needed)

The database uses a CHECK constraint on the `type` column that's dynamically generated from the `TransactionType` enum. When the app restarts, the schema migration system (`_apply_migration_v2`) will automatically update the constraint to include `P2P_SELL`.

---

## Configuration Changes

**File:** `default_config.json.example`

Add P2P Sells to history lookback:

```json
{
  "history_lookback_days": {
    "p2p_buys": 90,
    "p2p_sells": 90  // ← Add this
  }
}
```

---

## Testing Plan

### 1. Unit Tests
- Test P2P sell fetching with mock data
- Test invested capital calculation with P2P sells
- Test database queries include P2P sells

### 2. Integration Tests
- Sync with testnet account that has P2P sell history
- Verify transactions are recorded correctly
- Verify P/L calculation is accurate

### 3. Manual Testing
- User with existing P2P sell history should:
  1. Run full sync
  2. Verify P2P sell transactions appear in database
  3. Verify invested capital is reduced correctly
  4. Verify P/L shows realistic values

---

## Migration Path for Existing Users

Users who have already cashed out will need to:

1. **Full Resync:** Run a full sync to fetch historical P2P sell transactions
2. **Database Migration:** The schema will auto-update on first run
3. **Verify P/L:** Check that invested capital and P/L are now accurate

**Note:** The sync is incremental by default. Users may need to:
- Delete their existing database and resync from scratch, OR
- Manually set the sync timestamp for P2P Sell to trigger a full historical fetch

---

## Related Files

| File | Changes Needed |
|------|----------------|
| `src/crypto_portfolio_tracker/models.py` | Add `P2P_SELL` enum value |
| `src/crypto_portfolio_tracker/binance_fetcher.py` | Add P2P sell fetching logic |
| `src/crypto_portfolio_tracker/data_synchronizer.py` | Add P2P sell to sync pipeline |
| `src/crypto_portfolio_tracker/database.py` | Update invested capital query |
| `src/crypto_portfolio_tracker/portfolio_analyzer.py` | Update invested capital calculation |
| `default_config.json.example` | Add `p2p_sells` lookback config |

---

## Additional Considerations

### 1. P2P Sell Price Recording

When recording P2P sells, we need to capture:
- USDT quantity sold
- Fiat amount received (PHP, USD, etc.)
- Effective exchange rate

This is important for accurate cost basis calculation.

### 2. FIFO Cost Basis Impact

P2P sells should also affect FIFO cost basis:
- When you sell USDT via P2P, you're reducing your USDT holdings
- This should be recorded as a SELL transaction for FIFO purposes

### 3. Tax Implications

P2P sells may have tax implications in some jurisdictions. Recording these transactions accurately is important for:
- Capital gains calculations
- Tax reporting
- Audit trails

---

## Status Updates

### 2026-02-22 - Issue Identified and Fixed ✅

**Status:** RESOLVED

**Summary:**
- Problem discovered after user cashed out via P2P
- Root cause analysis completed
- Fix implemented and tested successfully
- All code changes merged

**Changes Made:**
1. ✅ Added `P2P_SELL` transaction type to `models.py`
2. ✅ Created `fetch_p2p_usdt_sells()` method in `BinanceFetcher`
3. ✅ Added P2P Sell to data synchronizer pipeline
4. ✅ Updated database query for invested capital
5. ✅ Updated portfolio analyzer calculation
6. ✅ Added P2P_SELL handling to PriceEnricher
7. ✅ Added `p2p_sells` to history lookback config
8. ✅ Updated database schema (CHECK constraint)

**Results:**
- **Before Fix:**
  - Total Invested Capital: $211.71 (incorrect - didn't subtract P2P sells)
  - Overall P/L: -$210.26 (-99.32%) ← FALSE LOSS

- **After Fix:**
  - Total Invested Capital: $26.00 (correct - subtracts $185.71 P2P sell)
  - Overall P/L: -$24.55 (-94.45%) ← Accurate loss from remaining dust

**Migration Required:**
Existing users need to:
1. Run the application (schema will auto-update)
2. Run a full sync to fetch historical P2P sell transactions
3. Verify invested capital is now correct

If the schema doesn't auto-update, run the manual migration:

```bash
cd crypto-portfolio-tracker
python docs/migrate_p2p_sell_schema.py
```

## Manual Schema Migration

If your database schema doesn't automatically update to include `P2P_SELL` in the CHECK constraint, run this script:

```python
# docs/migrate_p2p_sell_schema.py
import sqlite3
from pathlib import Path
from crypto_portfolio_tracker.models import TransactionType

db_path = Path('data/portfolio.db')
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Generate valid_tx_types from the enum
valid_tx_types = ', '.join([f"'{t.value}'" for t in TransactionType])

# Apply migration
cursor.execute('ALTER TABLE transactions RENAME TO transactions_old')

create_sql = f'''
    CREATE TABLE transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL, type TEXT NOT NULL CHECK (type IN ({valid_tx_types})),
        quantity REAL NOT NULL, price_usd REAL, fee_quantity REAL, fee_currency TEXT, fee_usd REAL, 
        source TEXT, notes TEXT, transaction_hash TEXT UNIQUE, 
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (asset_id) REFERENCES assets (id)
    )
'''
cursor.execute(create_sql)
cursor.execute('''
    INSERT INTO transactions SELECT * FROM transactions_old
''')
cursor.execute('DROP TABLE transactions_old')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_asset_timestamp ON transactions (asset_id, timestamp)')
conn.commit()
conn.close()

print('Schema migration complete!')
```

Then run a full sync to fetch your P2P sell history:
```bash
# Run full sync
track-portfolio-cli
# Select option 1: Full Sync & Analysis
```

### [Pending] - Fix Implementation
- [ ] Add P2P_SELL transaction type
- [ ] Implement P2P sell fetching
- [ ] Update data synchronizer
- [ ] Update invested capital calculation
- [ ] Update database queries
- [ ] Add tests
- [ ] Update documentation

---

## References

- Binance P2P API: `get_c2c_trade_history(tradeType="SELL")`
- Related issue: None (first report)
- Affected versions: v2.1.0 and earlier
