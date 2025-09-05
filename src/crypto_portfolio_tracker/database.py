import shutil
import sqlite3
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from .exceptions import DatabaseOperationError


class DatabaseManager:
    """
    Manages SQLite database operations for the crypto portfolio tracker.
    
    This class handles all database operations including:
    - Transaction storage and retrieval
    - Asset and holdings management
    - Historical price storage
    - Portfolio snapshots
    - Data backup and restoration
    - Data cleanup based on retention policies
    
    The database schema includes tables for assets, transactions, holdings, historical prices,
    and portfolio snapshots, with appropriate indexing for performance.
    """

    def __init__(
        self,
        db_path: Path,
        backup_dir: Optional[Path] = None,
        connection_timeout: int = 30,
        cleanup_days: int = 90,
        auto_delete_backups: bool = True,
        auto_backup_enabled: bool = True,
        max_backups: int = 10,
    ):
        """
        Initialize database manager with explicit database path.

        Args:
            db_path: Path to the database file
            backup_dir: Directory for backups (defaults to db_path.parent / "db_backups")
            connection_timeout: Database connection timeout in seconds
            cleanup_days: Number of days to keep data (0 to disable cleanup)
            auto_delete_backups: Whether to automatically delete old backups
            auto_backup_enabled: Whether to automatically create backups
            max_backups: Maximum number of backups to keep when auto deletion is enabled
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup backup directory - separate for testnet and production
        if backup_dir is None:
            # Check if this is a testnet database by checking the path
            if "testnet" in str(self.db_path).lower():
                self.backup_dir = self.db_path.parent / "testnet_db_backups"
            else:
                self.backup_dir = self.db_path.parent / "db_backups"
        else:
            self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

        self.connection_timeout = connection_timeout
        self.cleanup_days = cleanup_days
        self.auto_delete_backups = auto_delete_backups
        self.auto_backup_enabled = auto_backup_enabled
        self.max_backups = max_backups

        # Define table names as instance attributes
        self.ASSETS_TABLE_NAME = "assets"
        self.TRANSACTIONS_TABLE_NAME = "transactions"
        self.HOLDINGS_TABLE_NAME = "holdings"
        self.HISTORICAL_PRICES_TABLE_NAME = "historical_prices"
        self.PORTFOLIO_SNAPSHOTS_TABLE_NAME = "portfolio_snapshots"
        self.SCHEMA_VERSION_TABLE_NAME = "schema_version"

        self._create_tables()
        self._check_and_update_schema()
        self.logger.info(f"Database initialized at: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=self.connection_timeout)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise DatabaseOperationError(f"Failed to connect to database: {e}")

    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        commands = [
            """
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL, name TEXT, coingecko_id TEXT UNIQUE
            );""",
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL, type TEXT NOT NULL CHECK (type IN ('BUY', 'SELL', 'DEPOSIT', 'WITHDRAWAL', 'TRANSFER')),
                quantity REAL NOT NULL, price_usd REAL, fee_quantity REAL, fee_currency TEXT, fee_usd REAL, source TEXT, notes TEXT,
                transaction_hash TEXT UNIQUE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (asset_id) REFERENCES assets (id)
            );""",
            """
            CREATE TABLE IF NOT EXISTS holdings (
                asset_id INTEGER PRIMARY KEY, quantity REAL NOT NULL, average_cost_basis REAL NOT NULL,
                last_updated DATETIME NOT NULL, FOREIGN KEY (asset_id) REFERENCES assets (id)
            );""",
            """
            CREATE TABLE IF NOT EXISTS historical_prices (
                asset_id INTEGER NOT NULL, date DATE NOT NULL, price_usd REAL NOT NULL,
                PRIMARY KEY (asset_id, date), FOREIGN KEY (asset_id) REFERENCES assets (id)
            );""",
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                timestamp DATETIME PRIMARY KEY, total_value_usd REAL NOT NULL,
                total_cost_basis_usd REAL NOT NULL, unrealized_pl_usd REAL NOT NULL,
                unrealized_pl_percent REAL NOT NULL
            );""",
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );""",
            "CREATE INDEX IF NOT EXISTS idx_transactions_asset_timestamp ON transactions (asset_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_historical_prices_asset_date ON historical_prices (asset_id, date);",
        ]
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for command in commands:
                    cursor.execute(command)
                conn.commit()
                self.logger.info("Database tables checked/created successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def _check_and_update_schema(self):
        """Check the current schema version and apply updates if necessary."""
        current_version = self._get_schema_version()
        target_version = 1  # Current target schema version
        
        if current_version < target_version:
            self.logger.info(f"Upgrading database schema from version {current_version} to {target_version}")
            self._apply_schema_updates(current_version, target_version)
        elif current_version > target_version:
            self.logger.warning(
                f"Database schema version ({current_version}) is newer than expected ({target_version}). "
                f"This may cause compatibility issues."
            )
        else:
            self.logger.debug(f"Database schema is up to date (version {current_version})")

    def _get_schema_version(self) -> int:
        """Get the current schema version from the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
                row = cursor.fetchone()
                return row[0] if row else 0
        except sqlite3.Error as e:
            self.logger.debug(f"Could not get schema version (assuming 0): {e}")
            return 0

    def _apply_schema_updates(self, from_version: int, to_version: int):
        """Apply schema updates from one version to another."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Apply updates incrementally
                for version in range(from_version + 1, to_version + 1):
                    if version == 1:
                        # Version 1: Initial schema with all current tables
                        # No actual changes needed since _create_tables already created them
                        # Just record that we're at version 1
                        cursor.execute(
                            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                            (version,)
                        )
                        self.logger.info(f"Applied schema update to version {version}")
                
                conn.commit()
                self.logger.info(f"Schema successfully updated from version {from_version} to {to_version}")
                
        except sqlite3.Error as e:
            self.logger.error(f"Error applying schema updates: {e}")
            raise DatabaseOperationError(f"Failed to update database schema: {e}")

    def _apply_migration_v1(self, cursor):
        """Apply changes needed for schema version 1."""
        # For version 1, all tables are already created by _create_tables
        # This method exists for future migrations
        pass

    def get_asset_id(
        self,
        symbol: str,
        name: Optional[str] = None,
        coingecko_id: Optional[str] = None,
        create_if_missing: bool = True,
    ) -> Optional[int]:
        """Get asset ID by symbol, creating it if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM assets WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                if row:
                    return row["id"]
                elif create_if_missing:
                    cursor.execute(
                        "INSERT INTO assets (symbol, name, coingecko_id) VALUES (?, ?, ?)",
                        (symbol, name, coingecko_id),
                    )
                    conn.commit()
                    self.logger.info(f"Created new asset: {symbol}")
                    return cursor.lastrowid
                else:
                    return None
        except sqlite3.Error as e:
            self.logger.error(f"Error getting/creating asset ID for {symbol}: {e}")
            return None

    def bulk_insert_transactions(
        self, transactions: List[Dict[str, Any]]
    ) -> Optional[int]:
        """Bulk insert or update transactions using ON CONFLICT DO UPDATE."""
        # Create a backup before bulk insert operations
        self.backup_database("bulk_insert")
        
        if not transactions:
            self.logger.info("No transactions provided for bulk insert.")
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
            symbol = tx_dict.get("symbol")
            asset_id = self.get_asset_id(symbol, create_if_missing=True)
            if asset_id is None:
                self.logger.warning(
                    f"Could not get or create asset_id for symbol: {symbol}. Skipping transaction: {tx_dict.get('transaction_hash')}"
                )
                continue

            timestamp_val = tx_dict.get("timestamp")
            timestamp_for_db = None
            if isinstance(timestamp_val, datetime.datetime):
                if timestamp_val.tzinfo is None:
                    timestamp_val = timestamp_val.replace(
                        tzinfo=datetime.timezone.utc
                    )  # Ensure UTC
                else:
                    timestamp_val = timestamp_val.astimezone(
                        datetime.timezone.utc
                    )  # Convert to UTC
                timestamp_for_db = timestamp_val.isoformat(
                    sep=" ", timespec="milliseconds"
                )
            elif isinstance(timestamp_val, pd.Timestamp):  # Handle pandas Timestamps
                if timestamp_val.tzinfo is None:
                    timestamp_for_db = (
                        timestamp_val.tz_localize("utc")
                        .to_pydatetime()
                        .isoformat(sep=" ", timespec="milliseconds")
                    )
                else:
                    timestamp_for_db = (
                        timestamp_val.astimezone("utc")
                        .to_pydatetime()
                        .isoformat(sep=" ", timespec="milliseconds")
                    )
            elif timestamp_val is not None:  # Fallback for other types, store as string
                self.logger.warning(
                    f"Timestamp for tx {tx_dict.get('transaction_hash')} is not standard datetime/pandas ({type(timestamp_val)}). Storing as string: {timestamp_val}"
                )
                timestamp_for_db = str(timestamp_val)
            else:  # Skip if timestamp is None
                self.logger.warning(
                    f"Transaction {tx_dict.get('transaction_hash')} has a None timestamp. Skipping."
                )
                continue

            data_to_insert.append(
                (
                    asset_id,
                    timestamp_for_db,
                    tx_dict.get("type"),
                    float(tx_dict.get("quantity", 0.0)),
                    float(
                        tx_dict.get("price_usd", 0.0)
                        if tx_dict.get("price_usd") is not None
                        else 0.0
                    ),
                    float(
                        tx_dict.get("fee_quantity", 0.0)
                        if tx_dict.get("fee_quantity") is not None
                        else 0.0
                    ),
                    tx_dict.get("fee_currency"),
                    float(
                        tx_dict.get("fee_usd", 0.0)
                        if tx_dict.get("fee_usd") is not None
                        else 0.0
                    ),
                    tx_dict.get("source"),
                    tx_dict.get("notes"),
                    tx_dict.get("transaction_hash"),
                )
            )

        if not data_to_insert:
            self.logger.info(
                "No valid transactions prepared for DB insert/update after type checks."
            )
            return 0

        self.logger.debug(
            f"Attempting to insert/update {len(data_to_insert)} transactions using ON CONFLICT DO UPDATE..."
        )

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
                if (
                    rows_affected == -1
                ):  # Check if SQLite reported -1 (common for complex executemany)
                    changes_cursor = conn.cursor()
                    changes_cursor.execute("SELECT changes()")
                    changes_result = changes_cursor.fetchone()
                    rows_affected_fallback = changes_result[0] if changes_result else 0
                    self.logger.debug(
                        f"`executemany` rowcount was -1. SELECT changes() reported {rows_affected_fallback} changes."
                    )
                    return rows_affected_fallback
                else:
                    self.logger.debug(
                        f"`executemany` reported {rows_affected} rows affected (inserted or updated)."
                    )
                return rows_affected
        except sqlite3.Error as e:
            self.logger.error(
                f"Database error during bulk insert/update: {e}", exc_info=True
            )
            if data_to_insert:
                self.logger.error(
                    f"First data item in problematic batch: {data_to_insert[0]}"
                )
            raise DatabaseOperationError(
                f"Database error during bulk insert/update: {e}"
            )
        except Exception as e_generic:
            self.logger.error(
                f"Generic error during bulk insert/update: {e_generic}", exc_info=True
            )
            raise DatabaseOperationError(
                f"Generic error during bulk insert/update: {e_generic}"
            )

    def get_all_transactions(self) -> pd.DataFrame:
        """Fetch all transactions from the database."""
        query = "SELECT t.*, a.symbol FROM transactions t JOIN assets a ON t.asset_id = a.id ORDER BY t.timestamp;"
        try:
            with self._get_connection() as conn:
                return pd.read_sql_query(
                    query, conn, parse_dates=["timestamp", "created_at"]
                )
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching all transactions: {e}")
            return pd.DataFrame()

    def update_holdings(self, holdings_data: pd.DataFrame):
        """Update or Insert holdings data."""
        # Create a backup before updating holdings
        self.backup_database("update_holdings")
        
        sql = "INSERT OR REPLACE INTO holdings (asset_id, quantity, average_cost_basis, last_updated) VALUES (?, ?, ?, ?)"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.datetime.now(datetime.timezone.utc).isoformat(
                    sep=" ", timespec="milliseconds"
                )  # Store as UTC
                data_to_update = []
                for _, row in holdings_data.iterrows():
                    asset_id = self.get_asset_id(
                        row["symbol"], create_if_missing=False
                    )  # Don't create if missing for holdings
                    if asset_id:
                        self.logger.debug(
                            f"Preparing to update holding: Symbol={row['symbol']}, Qty={row['quantity']:.8f}, AvgCost={row['average_cost_basis']:.8f}"
                        )
                        data_to_update.append(
                            (asset_id, row["quantity"], row["average_cost_basis"], now)
                        )
                    else:
                        self.logger.warning(
                            f"Could not find asset_id for {row['symbol']} during holdings update. Skipping."
                        )

                if data_to_update:
                    cursor.executemany(sql, data_to_update)
                    conn.commit()
                    self.logger.info(
                        f"Holdings update executed. Cursor rowcount: {cursor.rowcount}. Attempted: {len(data_to_update)} records."
                    )
                else:
                    self.logger.info("No holdings data prepared for update.")
        except sqlite3.Error as e:
            self.logger.error(f"Error updating holdings: {e}")

    def get_holdings(self) -> pd.DataFrame:
        """Fetch current holdings."""
        query = "SELECT h.*, a.symbol, a.name, a.coingecko_id FROM holdings h JOIN assets a ON h.asset_id = a.id WHERE h.quantity > 0.000000001;"  # More robust zero check
        try:
            with self._get_connection() as conn:
                return pd.read_sql_query(query, conn, parse_dates=["last_updated"])
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching holdings: {e}")
            return pd.DataFrame()

    def insert_historical_prices(self, prices_df: pd.DataFrame):
        """Insert historical prices, ignoring duplicates."""
        sql = "INSERT OR IGNORE INTO historical_prices (asset_id, date, price_usd) VALUES (?, ?, ?)"
        data_to_insert = []
        for _, row in prices_df.iterrows():
            asset_id = self.get_asset_id(row["symbol"], create_if_missing=False)
            if asset_id:
                try:
                    # Ensure date is correctly formatted as YYYY-MM-DD string
                    date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                    data_to_insert.append((asset_id, date_str, row["price_usd"]))
                except Exception as e:
                    self.logger.warning(
                        f"Could not process date for historical price for {row['symbol']} (Date: {row['date']}): {e}"
                    )
            # else: Asset not in DB, skip.

        if not data_to_insert:
            return
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(sql, data_to_insert)
                conn.commit()
                self.logger.debug(
                    f"Inserted/Ignored {len(data_to_insert)} historical price records."
                )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting historical prices: {e}")

    def get_historical_prices(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch historical prices for a given asset and date range."""
        asset_id = self.get_asset_id(symbol, create_if_missing=False)
        if not asset_id:
            return pd.DataFrame()
        query = "SELECT date, price_usd FROM historical_prices WHERE asset_id = ? AND date BETWEEN ? AND ? ORDER BY date;"
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(asset_id, start_date, end_date),
                    parse_dates=["date"],
                )
                df.set_index("date", inplace=True)
                return df
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching historical prices for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_timestamp_for_source(
        self, source_name: str
    ) -> Optional[datetime.datetime]:
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
                if row and row["latest_timestamp"]:
                    latest_ts_str = row["latest_timestamp"]
                    # Assuming timestamps are stored as ISO format strings UTC (e.g., 'YYYY-MM-DD HH:MM:SS.sss')
                    # The pd.to_datetime will parse it, then convert to python datetime
                    dt_obj = pd.to_datetime(latest_ts_str).to_pydatetime()

                    # Ensure the datetime object is timezone-aware (UTC)
                    if dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
                    else:
                        dt_obj = dt_obj.astimezone(datetime.timezone.utc)

                    self.logger.info(
                        f"Latest timestamp found for source '{source_name}': {dt_obj}"
                    )
                    return dt_obj
                else:
                    self.logger.info(
                        f"No previous transactions found for source '{source_name}'."
                    )
                    return None
        except sqlite3.Error as e:
            self.logger.error(
                f"Error fetching latest timestamp for source {source_name}: {e}"
            )
            return None
        except Exception as ex:  # Catch other potential errors like parsing
            self.logger.error(
                f"Unexpected error processing latest timestamp for source {source_name}: {ex}",
                exc_info=True,
            )
            return None

    def save_portfolio_snapshot(
        self,
        timestamp: datetime,
        total_value: float,
        total_cost_basis: float,
        unrealized_pl: float,
        unrealized_pl_percent: float,
    ):
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
                (
                    timestamp,
                    total_value,
                    total_cost_basis,
                    unrealized_pl,
                    unrealized_pl_percent,
                ),
            )
            conn.commit()
            logging.info(
                f"Saved portfolio snapshot: {timestamp} - Value: ${total_value:,.2f}, P/L: ${unrealized_pl:,.2f}"
            )

        except sqlite3.Error as e:
            logging.error(f"DB Error saving portfolio snapshot: {e}", exc_info=True)

        finally:
            if conn:
                conn.close()

    def calculate_total_invested_capital(self) -> float:
        """
        Calculates the NET invested capital by summing capital inflows (P2P Buys)
        and subtracting capital outflows (Withdrawals).
        """
        # FIX: Use conditional aggregation to calculate Inflows - Outflows.
        query = """
            SELECT SUM(
                CASE
                    WHEN source = 'Binance P2P Buy' THEN quantity * price_usd
                    WHEN type = 'WITHDRAWAL' THEN -1 * quantity * price_usd
                    ELSE 0
                END
            ) as net_invested
            FROM transactions
            WHERE source = 'Binance P2P Buy' OR type = 'WITHDRAWAL';
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                result = cursor.fetchone()
                net_invested = result[0] if result and result[0] is not None else 0.0
                self.logger.info(
                    f"Calculated NET invested capital: ${net_invested:,.2f}"
                )
                return float(net_invested)
        except Exception as e:
            self.logger.error(
                f"Error calculating net invested capital: {e}", exc_info=True
            )
            return 0.0

    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """Get statistics about what would be cleaned up."""
        if self.cleanup_days <= 0:
            return {
                "cleanup_enabled": False,
                "cleanup_days": self.cleanup_days,
                "old_transactions": 0,
                "old_snapshots": 0,
                "total_transactions": 0,
                "total_snapshots": 0,
                "cutoff_date": None,
            }

        cutoff_date_dt = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(days=self.cleanup_days)
        cutoff_date_str = cutoff_date_dt.isoformat(sep=" ", timespec="milliseconds")

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Count old transactions
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.TRANSACTIONS_TABLE_NAME} WHERE timestamp < ?",
                    (cutoff_date_str,),
                )
                old_transactions = cursor.fetchone()[0]

                # Count old snapshots
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.PORTFOLIO_SNAPSHOTS_TABLE_NAME} WHERE timestamp < ?",
                    (cutoff_date_str,),
                )
                old_snapshots = cursor.fetchone()[0]

                # Count total records for context
                cursor.execute(f"SELECT COUNT(*) FROM {self.TRANSACTIONS_TABLE_NAME}")
                total_transactions = cursor.fetchone()[0]

                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.PORTFOLIO_SNAPSHOTS_TABLE_NAME}"
                )
                total_snapshots = cursor.fetchone()[0]

                return {
                    "cleanup_enabled": True,
                    "cleanup_days": self.cleanup_days,
                    "old_transactions": old_transactions,
                    "old_snapshots": old_snapshots,
                    "total_transactions": total_transactions,
                    "total_snapshots": total_snapshots,
                    "cutoff_date": cutoff_date_dt,
                }

        except sqlite3.Error as e:
            self.logger.error(f"Error getting cleanup statistics: {e}")
            return {
                "cleanup_enabled": True,
                "cleanup_days": self.cleanup_days,
                "old_transactions": 0,
                "old_snapshots": 0,
                "total_transactions": 0,
                "total_snapshots": 0,
                "cutoff_date": cutoff_date_dt,
                "error": str(e),
            }

    def cleanup_old_data(self):
        """Clean up old data based on configuration."""
        # Create a backup before cleanup operations
        self.backup_database("cleanup")
        
        if self.cleanup_days <= 0:
            self.logger.info("Data cleanup is disabled.")
            return

        cutoff_date_dt = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(days=self.cleanup_days)
        cutoff_date_str = cutoff_date_dt.isoformat(
            sep=" ", timespec="milliseconds"
        )  # Match inserted format

        commands = [
            f"DELETE FROM {self.TRANSACTIONS_TABLE_NAME} WHERE timestamp < '{cutoff_date_str}';",
            f"DELETE FROM {self.PORTFOLIO_SNAPSHOTS_TABLE_NAME} WHERE timestamp < '{cutoff_date_str}';",
            # Add historical_prices cleanup if desired:
            # f"DELETE FROM {self.HISTORICAL_PRICES_TABLE_NAME} WHERE date < '{cutoff_date_dt.strftime('%Y-%m-%d')}';"
        ]
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                total_deleted = 0
                for command in commands:
                    self.logger.debug(f"Executing cleanup command: {command}")
                    cursor.execute(command)
                    # SELECT changes() is more reliable for DELETE statements
                    changes_cursor = conn.cursor()
                    changes_cursor.execute("SELECT changes()")
                    changes_result = changes_cursor.fetchone()
                    deleted_this_command = changes_result[0] if changes_result else 0
                    total_deleted += deleted_this_command
                    self.logger.debug(f"Command affected {deleted_this_command} rows.")
                conn.commit()
                if total_deleted > 0:
                    self.logger.info(f"Cleaned up {total_deleted} old records.")
                else:
                    self.logger.info("No old data found to clean up.")
        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    def backup_database(self, reason: str = "manual", force: bool = False) -> Optional[str]:
        """
        Creates a timestamped backup of the current database file in the backup directory.
        Returns the path of the backup file on success, None on failure.

        Args:
            reason: Reason for the backup (used in filename)
            force: If True, create backup even if auto_backup_enabled is False
        """
        # Check if auto backup is enabled (unless forced)
        if not self.auto_backup_enabled and not force:
            self.logger.debug("Auto backup is disabled. Skipping backup creation.")
            return None

        if not self.db_path.exists():
            self.logger.error(
                f"Database file not found at {self.db_path}. Cannot create backup."
            )
            return None
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            reason_tag = reason.replace(" ", "_").lower()
            backup_filename = f"{self.db_path.name}.{timestamp}.{reason_tag}.bak"
            backup_filepath = self.backup_dir / backup_filename
            shutil.copy2(self.db_path, backup_filepath)
            self.logger.info(f"Successfully created database backup: {backup_filepath} (reason: {reason})")
            
            # Clean up old backups if auto deletion is enabled
            if self.auto_delete_backups:
                self._cleanup_old_backups()
            
            return str(backup_filepath)
        except Exception as e:
            self.logger.error(f"Failed to create database backup: {e}", exc_info=True)
            return None

    def _cleanup_old_backups(self):
        """Keep only the most recent backups"""
        # Check if auto deletion is enabled
        if not self.auto_delete_backups:
            self.logger.debug("Auto deletion of backups is disabled. Skipping cleanup.")
            return

        try:
            backup_files = list(self.backup_dir.glob("*.bak"))
            if len(backup_files) > self.max_backups:
                # Sort by modification time, oldest first
                backup_files.sort(key=lambda p: p.stat().st_mtime)
                # Remove oldest backups
                for old_backup in backup_files[:-self.max_backups]:
                    old_backup.unlink()
                    self.logger.info(f"Removed old backup: {old_backup}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")

    def list_backups(self) -> List[Path]:
        """
        Lists available database backups, sorted from newest to oldest.
        """
        try:
            backup_files = list(self.backup_dir.glob("*.bak"))
            # Sort by modification time, newest first
            if backup_files:
                backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return backup_files
        except Exception as e:
            self.logger.error(f"Failed to list database backups: {e}")
            return []

    def restore_from_backup(self, backup_path: Path) -> bool:
        """
        Restores the database from a selected backup file. This is a destructive operation.
        """
        if not backup_path.exists():
            self.logger.error(f"Backup file not found: {backup_path}")
            return False
        try:
            shutil.copy2(backup_path, self.db_path)
            self.logger.info(f"Successfully restored database from: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to restore database from backup: {e}", exc_info=True
            )
            return False

    def fetch_transactions(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Fetch transactions from the database (example method)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # ... your fetch logic ...
                return []  # Replace with actual fetch result
        except sqlite3.Error as e:
            self.logger.error(f"Database error during fetch: {e}", exc_info=True)
            raise DatabaseOperationError(f"Database error during fetch: {e}")
        except Exception as e_generic:
            self.logger.error(f"Generic error during fetch: {e_generic}", exc_info=True)
            raise DatabaseOperationError(f"Generic error during fetch: {e_generic}")

    def update_holding(self, *args, **kwargs) -> None:
        """Update a holding in the database (example method)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # ... your update logic ...
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error during update: {e}", exc_info=True)
            raise DatabaseOperationError(f"Database error during update: {e}")
        except Exception as e_generic:
            self.logger.error(
                f"Generic error during update: {e_generic}", exc_info=True
            )
            raise DatabaseOperationError(f"Generic error during update: {e_generic}")

    def delete_transaction(self, *args, **kwargs) -> None:
        """Delete a transaction from the database (example method)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # ... your delete logic ...
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error during delete: {e}", exc_info=True)
            raise DatabaseOperationError(f"Database error during delete: {e}")
        except Exception as e_generic:
            self.logger.error(
                f"Generic error during delete: {e_generic}", exc_info=True
            )
            raise DatabaseOperationError(f"Generic error during delete: {e_generic}")

    def get_all_snapshots(self) -> pd.DataFrame:
        """Fetch all portfolio snapshots from the database with robust timestamp handling."""
        query = "SELECT * FROM portfolio_snapshots ORDER BY timestamp;"
        try:
            with self._get_connection() as conn:
                # Get raw data without parsing dates
                df = pd.read_sql_query(query, conn)

                # Keep timestamps as strings to avoid parsing issues
                # This ensures all timestamps are in the same format for sorting
                if not df.empty and "timestamp" in df.columns:
                    # Convert all timestamps to strings for consistent handling
                    df["timestamp"] = df["timestamp"].astype(str)

                    # Replace 'None', 'nan', 'NaT' with actual None for proper handling
                    df["timestamp"] = df["timestamp"].replace(
                        ["None", "nan", "NaT", "NULL"], None
                    )

                return df
        except Exception as e:
            self.logger.error(f"Error fetching all snapshots: {e}")
            return pd.DataFrame()

    def delete_snapshot(
        self,
        timestamp,
        total_value_usd,
        total_cost_basis_usd,
        unrealized_pl_usd,
        unrealized_pl_percent,
    ):
        """
        Delete a specific snapshot. Handles NULL timestamps, NaN values, and exact value matching.
        Returns the number of rows deleted.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Check if timestamp is null/NaN/NaT in various ways
                is_null_timestamp = (
                    pd.isna(timestamp)
                    or timestamp is None
                    or (hasattr(timestamp, "value") and pd.isna(timestamp.value))
                    or str(timestamp).lower() in ["nan", "nat", "none", ""]
                )

                if is_null_timestamp:
                    # For NULL timestamps, try multiple approaches
                    sql = """
                        DELETE FROM portfolio_snapshots
                        WHERE (timestamp IS NULL OR timestamp = '' OR timestamp = 'NaN' OR timestamp = 'NaT')
                        AND ROUND(total_value_usd, 2) = ROUND(?, 2)
                        AND ROUND(total_cost_basis_usd, 2) = ROUND(?, 2)
                        AND ROUND(unrealized_pl_usd, 2) = ROUND(?, 2)
                        AND ROUND(unrealized_pl_percent, 2) = ROUND(?, 2)
                    """
                    params = (
                        total_value_usd,
                        total_cost_basis_usd,
                        unrealized_pl_usd,
                        unrealized_pl_percent,
                    )
                else:
                    # For valid timestamps, handle both datetime objects and strings
                    ts = timestamp
                    if hasattr(ts, "isoformat"):
                        ts = ts.isoformat(sep=" ", timespec="microseconds")
                    elif isinstance(ts, str):
                        # If it's already a string, use it as-is
                        ts = ts
                    else:
                        # Convert to string
                        ts = str(ts)

                    sql = """
                        DELETE FROM portfolio_snapshots
                        WHERE timestamp = ?
                        AND ROUND(total_value_usd, 2) = ROUND(?, 2)
                        AND ROUND(total_cost_basis_usd, 2) = ROUND(?, 2)
                        AND ROUND(unrealized_pl_usd, 2) = ROUND(?, 2)
                        AND ROUND(unrealized_pl_percent, 2) = ROUND(?, 2)
                    """
                    params = (
                        ts,
                        total_value_usd,
                        total_cost_basis_usd,
                        unrealized_pl_usd,
                        unrealized_pl_percent,
                    )

                cursor.execute(sql, params)
                rows_deleted = cursor.rowcount

                # Log the deletion attempt for debugging
                self.logger.info(
                    f"Delete attempt: timestamp={timestamp}, "
                    f"total_value_usd={total_value_usd}, "
                    f"is_null_timestamp={is_null_timestamp}, "
                    f"rows_deleted={rows_deleted}"
                )

                return rows_deleted

        except Exception as e:
            self.logger.error(f"Error deleting snapshot: {e}")
            raise

    def debug_snapshots(self):
        """Debug method to see what snapshots exist in the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM portfolio_snapshots ORDER BY timestamp")
                rows = cursor.fetchall()

                self.logger.info(f"Found {len(rows)} snapshots in database:")
                for i, row in enumerate(rows):
                    self.logger.info(
                        f"  {i + 1}. timestamp={row[0]}, "
                        f"total_value_usd={row[1]}, "
                        f"total_cost_basis_usd={row[2]}, "
                        f"unrealized_pl_usd={row[3]}, "
                        f"unrealized_pl_percent={row[4]}"
                    )
                return rows
        except Exception as e:
            self.logger.error(f"Failed to debug snapshots: {e}")
            return []
