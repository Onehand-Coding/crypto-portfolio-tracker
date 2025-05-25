"""
Database Management Module
Handles all interactions with the SQLite database.
"""
import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize database manager"""
        self.db_path = Path(config.get("database", {}).get("path", "data/portfolio.db"))
        self.config = config.get("database", {})
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection_timeout = self.config.get("connection_timeout", 30)
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
                quantity REAL NOT NULL, price_usd REAL, fee_usd REAL, source TEXT, notes TEXT,
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

    def bulk_insert_transactions(self, transactions: List[Dict[str, Any]]):
        """Bulk insert transactions into the database."""
        sql = """
        INSERT OR IGNORE INTO transactions
        (asset_id, timestamp, type, quantity, price_usd, fee_usd, source, notes, transaction_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data_to_insert = []
        for tx_dict in transactions:
            asset_id = self.get_asset_id(tx_dict.get('symbol'), create_if_missing=True)
            if asset_id:
                timestamp_val = tx_dict.get('timestamp')
                timestamp_for_db = None

                # <<< MODIFICATION START >>>
                if isinstance(timestamp_val, datetime.datetime):
                    timestamp_for_db = timestamp_val.isoformat(sep=' ', timespec='milliseconds') # Standard ISO format, space separator
                elif isinstance(timestamp_val, pd.Timestamp):
                    timestamp_for_db = timestamp_val.to_pydatetime().isoformat(sep=' ', timespec='milliseconds')
                elif timestamp_val is not None:
                    logger.warning(f"Timestamp for tx {tx_dict.get('transaction_hash')} is not a standard datetime object: {timestamp_val} (type: {type(timestamp_val)}). Storing as string: {str(timestamp_val)}")
                    timestamp_for_db = str(timestamp_val)
                # <<< MODIFICATION END >>>

                if tx_dict.get('type') == 'DEPOSIT':
                    logger.debug(
                        f"Preparing DEPOSIT tx for DB insert: "
                        f"asset_id={asset_id}, "
                        f"symbol={tx_dict.get('symbol')}, "
                        f"timestamp_for_db='{timestamp_for_db}' (type: {type(timestamp_for_db)}), " # Log the string
                        f"price_usd={tx_dict.get('price_usd')} (type: {type(tx_dict.get('price_usd'))}), "
                        f"quantity={tx_dict.get('quantity')}"
                    )

                data_to_insert.append((
                    asset_id,
                    timestamp_for_db, # This will now be a standard ISO string or None
                    tx_dict.get('type'),
                    tx_dict.get('quantity'),
                    tx_dict.get('price_usd'),
                    tx_dict.get('fee_usd'),
                    tx_dict.get('source'),
                    tx_dict.get('notes'),
                    tx_dict.get('transaction_hash')
                ))
            else:
                logger.warning(f"Could not get or create asset_id for symbol: {tx_dict.get('symbol')}. Skipping transaction: {tx_dict}")

        if not data_to_insert:
            logger.info("No new transactions prepared for DB insert.")
            return

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(sql, data_to_insert)
                conn.commit()
                logger.info(f"Attempted to insert/ignore {len(data_to_insert)} transactions. Rows newly inserted: {cursor.rowcount}")
        except sqlite3.Error as e:
            logger.error(f"Error bulk inserting transactions: {e}")
            if data_to_insert:
                 logger.error(f"First data item in batch (potential cause): {data_to_insert[0]}")

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
                 cursor = conn.cursor(); now = datetime.datetime.now()
                 data_to_update = [(self.get_asset_id(row['symbol'], create_if_missing=False), row['quantity'], row['average_cost_basis'], now) for _, row in holdings_data.iterrows() if self.get_asset_id(row['symbol'], create_if_missing=False)]
                 if data_to_update: cursor.executemany(sql, data_to_update); conn.commit(); logger.info(f"Updated {len(data_to_update)} holdings records.")
        except sqlite3.Error as e: logger.error(f"Error updating holdings: {e}")

    def get_holdings(self) -> pd.DataFrame:
        """Fetch current holdings."""
        query = "SELECT h.*, a.symbol, a.name, a.coingecko_id FROM holdings h JOIN assets a ON h.asset_id = a.id WHERE h.quantity > 0;"
        try:
            with self._get_connection() as conn: return pd.read_sql_query(query, conn, parse_dates=['last_updated'])
        except sqlite3.Error as e: logger.error(f"Error fetching holdings: {e}"); return pd.DataFrame()

    def insert_historical_prices(self, prices_df: pd.DataFrame):
        """Insert historical prices, ignoring duplicates."""
        sql = "INSERT OR IGNORE INTO historical_prices (asset_id, date, price_usd) VALUES (?, ?, ?)"
        data_to_insert = [(self.get_asset_id(row['symbol'], create_if_missing=False), pd.to_datetime(row['date']).strftime('%Y-%m-%d'), row['price_usd']) for _, row in prices_df.iterrows() if self.get_asset_id(row['symbol'], create_if_missing=False)]
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
        if not self.config.get("backup_enabled", False): logger.info("Database backup is disabled."); return
        backup_dir = self.db_path.parent / "backups"; backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{self.db_path.stem}_backup_{timestamp}.db"
        try: shutil.copy2(self.db_path, backup_path); logger.info(f"Database backup created successfully at: {backup_path}")
        except Exception as e: logger.error(f"Failed to create database backup: {e}")

    def cleanup_old_data(self):
        """Clean up old data based on configuration."""
        cleanup_days = self.config.get("cleanup_days", 90)
        if cleanup_days <= 0: logger.info("Data cleanup is disabled."); return
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=cleanup_days)).strftime('%Y-%m-%d %H:%M:%S')
        commands = [f"DELETE FROM transactions WHERE timestamp < '{cutoff_date}';", f"DELETE FROM portfolio_snapshots WHERE timestamp < '{cutoff_date}';"]
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor(); total_deleted = 0
                for command in commands: cursor.execute(command); total_deleted += cursor.rowcount
                conn.commit()
                if total_deleted > 0: logger.info(f"Cleaned up {total_deleted} old records.")
                else: logger.info("No old data found to clean up.")
        except sqlite3.Error as e: logger.error(f"Error cleaning up old data: {e}")
