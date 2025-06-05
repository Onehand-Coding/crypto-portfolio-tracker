import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import datetime # Ensure this is imported

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize database manager"""
        self.db_path = Path(config.get("database", {}).get("path", "data/portfolio.db"))
        self.db_config = config.get("database", {}) # Store the 'database' sub-dictionary
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection_timeout = self.db_config.get("connection_timeout", 30)

        # Define table names as instance attributes
        self.ASSETS_TABLE_NAME = "assets"
        self.TRANSACTIONS_TABLE_NAME = "transactions"
        self.HOLDINGS_TABLE_NAME = "holdings"
        self.HISTORICAL_PRICES_TABLE_NAME = "historical_prices"
        self.PORTFOLIO_SNAPSHOTS_TABLE_NAME = "portfolio_snapshots"

        self._create_tables()
        logger.info(f"Database initialized at: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=self.connection_timeout)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        commands = [
            """
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL, name TEXT, coingecko_id TEXT UNIQUE
            );""", """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL, type TEXT NOT NULL CHECK (type IN ('BUY', 'SELL', 'DEPOSIT', 'WITHDRAWAL', 'TRANSFER')),
                quantity REAL NOT NULL, price_usd REAL, fee_quantity REAL, fee_currency TEXT, fee_usd REAL, source TEXT, notes TEXT,
                transaction_hash TEXT UNIQUE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (asset_id) REFERENCES assets (id)
            );""", """
            CREATE TABLE IF NOT EXISTS holdings (
                asset_id INTEGER PRIMARY KEY, quantity REAL NOT NULL, average_cost_basis REAL NOT NULL,
                last_updated DATETIME NOT NULL, FOREIGN KEY (asset_id) REFERENCES assets (id)
            );""", """
            CREATE TABLE IF NOT EXISTS historical_prices (
                asset_id INTEGER NOT NULL, date DATE NOT NULL, price_usd REAL NOT NULL,
                PRIMARY KEY (asset_id, date), FOREIGN KEY (asset_id) REFERENCES assets (id)
            );""", """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                timestamp DATETIME PRIMARY KEY, total_value_usd REAL NOT NULL,
                total_cost_basis_usd REAL NOT NULL, unrealized_pl_usd REAL NOT NULL,
                unrealized_pl_percent REAL NOT NULL
            );""",
            "CREATE INDEX IF NOT EXISTS idx_transactions_asset_timestamp ON transactions (asset_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_historical_prices_asset_date ON historical_prices (asset_id, date);"
        ]
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for command in commands:
                    cursor.execute(command)
                conn.commit()
                logger.info("Database tables checked/created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def get_asset_id(self, symbol: str, name: Optional[str] = None, coingecko_id: Optional[str] = None, create_if_missing: bool = True) -> Optional[int]:
        """Get asset ID by symbol, creating it if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM assets WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                if row: return row['id']
                elif create_if_missing:
                    cursor.execute("INSERT INTO assets (symbol, name, coingecko_id) VALUES (?, ?, ?)", (symbol, name, coingecko_id))
                    conn.commit()
                    logger.info(f"Created new asset: {symbol}")
                    return cursor.lastrowid
                else: return None
        except sqlite3.Error as e:
            logger.error(f"Error getting/creating asset ID for {symbol}: {e}")
            return None

    def bulk_insert_transactions(self, transactions: List[Dict[str, Any]]) -> Optional[int]:
        """Bulk insert or update transactions using ON CONFLICT DO UPDATE."""
        if not transactions:
            logger.info("No transactions provided for bulk insert.")
            return 0

        sql = f"""
        INSERT INTO {self.TRANSACTIONS_TABLE_NAME}
        (asset_id, timestamp, type, quantity, price_usd,
         fee_quantity, fee_currency, fee_usd,
         source, notes, transaction_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(transaction_hash) DO UPDATE SET
            asset_id = excluded.asset_id,
            timestamp = excluded.timestamp,
            type = excluded.type,
            quantity = excluded.quantity,
            price_usd = excluded.price_usd,
            fee_quantity = excluded.fee_quantity,
            fee_currency = excluded.fee_currency,
            fee_usd = excluded.fee_usd,
            source = excluded.source,
            notes = excluded.notes;
        """

        data_to_insert = []
        for tx_dict in transactions:
            symbol = tx_dict.get('symbol')
            asset_id = self.get_asset_id(symbol, create_if_missing=True)
            if asset_id is None:
                logger.warning(f"Could not get or create asset_id for symbol: {symbol}. Skipping transaction: {tx_dict.get('transaction_hash')}")
                continue

            timestamp_val = tx_dict.get('timestamp')
            timestamp_for_db = None
            if isinstance(timestamp_val, datetime.datetime):
                if timestamp_val.tzinfo is None:
                    timestamp_val = timestamp_val.replace(tzinfo=datetime.timezone.utc) # Ensure UTC
                else:
                    timestamp_val = timestamp_val.astimezone(datetime.timezone.utc) # Convert to UTC
                timestamp_for_db = timestamp_val.isoformat(sep=' ', timespec='milliseconds')
            elif isinstance(timestamp_val, pd.Timestamp): # Handle pandas Timestamps
                if timestamp_val.tzinfo is None:
                     timestamp_for_db = timestamp_val.tz_localize('utc').to_pydatetime().isoformat(sep=' ', timespec='milliseconds')
                else:
                     timestamp_for_db = timestamp_val.astimezone('utc').to_pydatetime().isoformat(sep=' ', timespec='milliseconds')
            elif timestamp_val is not None: # Fallback for other types, store as string
                logger.warning(f"Timestamp for tx {tx_dict.get('transaction_hash')} is not standard datetime/pandas ({type(timestamp_val)}). Storing as string: {timestamp_val}")
                timestamp_for_db = str(timestamp_val)
            else: # Skip if timestamp is None
                logger.warning(f"Transaction {tx_dict.get('transaction_hash')} has a None timestamp. Skipping.")
                continue


            data_to_insert.append((
                asset_id, timestamp_for_db, tx_dict.get('type'),
                float(tx_dict.get('quantity', 0.0)),
                float(tx_dict.get('price_usd', 0.0) if tx_dict.get('price_usd') is not None else 0.0),
                float(tx_dict.get('fee_quantity', 0.0) if tx_dict.get('fee_quantity') is not None else 0.0),
                tx_dict.get('fee_currency'),
                float(tx_dict.get('fee_usd', 0.0) if tx_dict.get('fee_usd') is not None else 0.0),
                tx_dict.get('source'), tx_dict.get('notes'), tx_dict.get('transaction_hash')
            ))

        if not data_to_insert:
            logger.info("No valid transactions prepared for DB insert/update after type checks.")
            return 0

        logger.info(f"Attempting to insert/update {len(data_to_insert)} transactions using ON CONFLICT DO UPDATE...")

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(sql, data_to_insert)
                conn.commit()

                rows_affected = cursor.rowcount
                # For SQLite, rowcount after executemany with "ON CONFLICT DO UPDATE"
                # might be -1 or the number of rows *inserted*. It's not consistently
                # the total number of rows affected (inserted + updated).
                # A more reliable way to get changes in SQLite is `SELECT changes()`.
                if rows_affected == -1: # Check if SQLite reported -1 (common for complex executemany)
                    changes_cursor = conn.cursor()
                    changes_cursor.execute("SELECT changes()")
                    changes_result = changes_cursor.fetchone()
                    rows_affected_fallback = changes_result[0] if changes_result else 0
                    logger.info(f"DB: `executemany` rowcount was -1. SELECT changes() reported {rows_affected_fallback} changes.")
                    return rows_affected_fallback
                else:
                    logger.info(f"DB: `executemany` reported {rows_affected} rows affected (inserted or updated).")
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error during bulk insert/update: {e}", exc_info=True)
            if data_to_insert: # Log the first problematic item for easier debugging
                logger.error(f"First data item in problematic batch: {data_to_insert[0]}")
            return 0 # Indicate 0 rows affected on error
        except Exception as e_generic: # Catch any other non-SQLite errors
            logger.error(f"Generic error during bulk insert/update: {e_generic}", exc_info=True)
            return 0

    def get_all_transactions(self) -> pd.DataFrame:
        """Fetch all transactions from the database."""
        query = "SELECT t.*, a.symbol FROM transactions t JOIN assets a ON t.asset_id = a.id ORDER BY t.timestamp;"
        try:
            with self._get_connection() as conn: return pd.read_sql_query(query, conn, parse_dates=['timestamp', 'created_at'])
        except sqlite3.Error as e: logger.error(f"Error fetching all transactions: {e}"); return pd.DataFrame()

    def update_holdings(self, holdings_data: pd.DataFrame):
        """Update or Insert holdings data."""
        sql = "INSERT OR REPLACE INTO holdings (asset_id, quantity, average_cost_basis, last_updated) VALUES (?, ?, ?, ?)"
        try:
            with self._get_connection() as conn:
                 cursor = conn.cursor(); now = datetime.datetime.now(datetime.timezone.utc).isoformat(sep=' ', timespec='milliseconds') # Store as UTC
                 data_to_update = []
                 for _, row in holdings_data.iterrows():
                     asset_id = self.get_asset_id(row['symbol'], create_if_missing=False) # Don't create if missing for holdings
                     if asset_id:
                         logger.info(f"DB: Preparing to update holding: Symbol={row['symbol']}, Qty={row['quantity']:.8f}, AvgCost={row['average_cost_basis']:.8f}")
                         data_to_update.append((
                             asset_id,
                             row['quantity'],
                             row['average_cost_basis'],
                             now
                         ))
                     else:
                         logger.warning(f"DB: Could not find asset_id for {row['symbol']} during holdings update. Skipping.")

                 if data_to_update:
                     cursor.executemany(sql, data_to_update)
                     conn.commit()
                     logger.info(f"DB: Holdings update executed. Cursor rowcount: {cursor.rowcount}. Attempted: {len(data_to_update)} records.")
                 else:
                     logger.info("DB: No holdings data prepared for update.")
        except sqlite3.Error as e:
            logger.error(f"Error updating holdings: {e}")

    def get_holdings(self) -> pd.DataFrame:
        """Fetch current holdings."""
        query = "SELECT h.*, a.symbol, a.name, a.coingecko_id FROM holdings h JOIN assets a ON h.asset_id = a.id WHERE h.quantity > 0.000000001;" # More robust zero check
        try:
            with self._get_connection() as conn: return pd.read_sql_query(query, conn, parse_dates=['last_updated'])
        except sqlite3.Error as e: logger.error(f"Error fetching holdings: {e}"); return pd.DataFrame()

    def insert_historical_prices(self, prices_df: pd.DataFrame):
        """Insert historical prices, ignoring duplicates."""
        sql = "INSERT OR IGNORE INTO historical_prices (asset_id, date, price_usd) VALUES (?, ?, ?)"
        data_to_insert = []
        for _, row in prices_df.iterrows():
            asset_id = self.get_asset_id(row['symbol'], create_if_missing=False)
            if asset_id:
                try:
                    # Ensure date is correctly formatted as YYYY-MM-DD string
                    date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                    data_to_insert.append((asset_id, date_str, row['price_usd']))
                except Exception as e:
                    logger.warning(f"Could not process date for historical price for {row['symbol']} (Date: {row['date']}): {e}")
            # else: Asset not in DB, skip.

        if not data_to_insert: return
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor(); cursor.executemany(sql, data_to_insert); conn.commit()
                logger.debug(f"Inserted/Ignored {len(data_to_insert)} historical price records.")
        except sqlite3.Error as e: logger.error(f"Error inserting historical prices: {e}")

    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical prices for a given asset and date range."""
        asset_id = self.get_asset_id(symbol, create_if_missing=False)
        if not asset_id: return pd.DataFrame()
        query = "SELECT date, price_usd FROM historical_prices WHERE asset_id = ? AND date BETWEEN ? AND ? ORDER BY date;"
        try:
            with self._get_connection() as conn: df = pd.read_sql_query(query, conn, params=(asset_id, start_date, end_date), parse_dates=['date']); df.set_index('date', inplace=True); return df
        except sqlite3.Error as e: logger.error(f"Error fetching historical prices for {symbol}: {e}"); return pd.DataFrame()

    def backup_database(self):
        """Create a backup of the database file."""
        # Assuming self.config is available from __init__; if not, it should be self.db_config
        if not self.db_config.get("backup_enabled", False): logger.info("Database backup is disabled."); return
        backup_dir = self.db_path.parent / "backups"; backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{self.db_path.stem}_backup_{timestamp}.db"
        try: shutil.copy2(self.db_path, backup_path); logger.info(f"Database backup created successfully at: {backup_path}")
        except Exception as e: logger.error(f"Failed to create database backup: {e}")

    def cleanup_old_data(self):
        """Clean up old data based on configuration."""
        cleanup_days = self.db_config.get("cleanup_days", 90) # Use self.db_config
        if cleanup_days <= 0: logger.info("Data cleanup is disabled."); return
        cutoff_date_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=cleanup_days)
        cutoff_date_str = cutoff_date_dt.isoformat(sep=' ', timespec='milliseconds') # Match inserted format

        commands = [
            f"DELETE FROM {self.TRANSACTIONS_TABLE_NAME} WHERE timestamp < '{cutoff_date_str}';",
            f"DELETE FROM {self.PORTFOLIO_SNAPSHOTS_TABLE_NAME} WHERE timestamp < '{cutoff_date_str}';"
            # Add historical_prices cleanup if desired:
            # f"DELETE FROM {self.HISTORICAL_PRICES_TABLE_NAME} WHERE date < '{cutoff_date_dt.strftime('%Y-%m-%d')}';"
        ]
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor(); total_deleted = 0
                for command in commands:
                    logger.debug(f"Executing cleanup command: {command}")
                    cursor.execute(command);
                    # SELECT changes() is more reliable for DELETE statements
                    changes_cursor = conn.cursor()
                    changes_cursor.execute("SELECT changes()")
                    changes_result = changes_cursor.fetchone()
                    deleted_this_command = changes_result[0] if changes_result else 0
                    total_deleted += deleted_this_command
                    logger.debug(f"Command affected {deleted_this_command} rows.")
                conn.commit()
                if total_deleted > 0: logger.info(f"Cleaned up {total_deleted} old records.")
                else: logger.info("No old data found to clean up.")
        except sqlite3.Error as e: logger.error(f"Error cleaning up old data: {e}")

    def get_latest_timestamp_for_source(self, source_name: str) -> Optional[datetime.datetime]:
        """Fetch the latest transaction timestamp for a given source."""
        query = f"""
            SELECT MAX(timestamp) as latest_timestamp
            FROM {self.TRANSACTIONS_TABLE_NAME}
            WHERE source = ?;
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (source_name,))
                row = cursor.fetchone()
                if row and row['latest_timestamp']:
                    latest_ts_str = row['latest_timestamp']
                    # Assuming timestamps are stored as ISO format strings UTC (e.g., 'YYYY-MM-DD HH:MM:SS.sss')
                    # The pd.to_datetime will parse it, then convert to python datetime
                    dt_obj = pd.to_datetime(latest_ts_str).to_pydatetime()

                    # Ensure the datetime object is timezone-aware (UTC)
                    if dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
                    else:
                        dt_obj = dt_obj.astimezone(datetime.timezone.utc)

                    logger.info(f"Latest timestamp found for source '{source_name}': {dt_obj}")
                    return dt_obj
                else:
                    logger.info(f"No previous transactions found for source '{source_name}'.")
                    return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching latest timestamp for source {source_name}: {e}")
            return None
        except Exception as ex: # Catch other potential errors like parsing
            logger.error(f"Unexpected error processing latest timestamp for source {source_name}: {ex}", exc_info=True)
            return None

    def save_portfolio_snapshot(self, timestamp: datetime, total_value: float, total_cost_basis: float, unrealized_pl: float, unrealized_pl_percent: float):
        """Saves a snapshot of the total portfolio value and performance at a given time."""
        # Use the correct connection method name as defined in your file
        conn = self._get_connection()
        if not conn:
            logging.error("DB Error: Could not create connection for saving snapshot.")
            return

        try:
            cursor = conn.cursor()

            # The schema from your file has more columns, let's use them all.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp DATETIME PRIMARY KEY,
                    total_value_usd REAL NOT NULL,
                    total_cost_basis_usd REAL NOT NULL,
                    unrealized_pl_usd REAL NOT NULL,
                    unrealized_pl_percent REAL NOT NULL
                )
            """)

            # Insert all the new snapshot data
            cursor.execute(
                """INSERT INTO portfolio_snapshots (timestamp, total_value_usd, total_cost_basis_usd, unrealized_pl_usd, unrealized_pl_percent)
                   VALUES (?, ?, ?, ?, ?)""",
                (timestamp, total_value, total_cost_basis, unrealized_pl, unrealized_pl_percent)
            )
            conn.commit()
            logging.info(f"Saved portfolio snapshot: {timestamp} - Value: ${total_value:,.2f}, P/L: ${unrealized_pl:,.2f}")

        except sqlite3.Error as e:
            logging.error(f"DB Error saving portfolio snapshot: {e}", exc_info=True)

        finally:
            if conn:
                conn.close()
