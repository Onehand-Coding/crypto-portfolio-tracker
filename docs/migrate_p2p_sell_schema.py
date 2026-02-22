#!/usr/bin/env python3
"""
Manual schema migration to add P2P_SELL to the transactions CHECK constraint.

Run this script if your database schema doesn't automatically update when you
start the application after pulling the P2P sell tracking fix.

Usage:
    python docs/migrate_p2p_sell_schema.py
"""

import sqlite3
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crypto_portfolio_tracker.models import TransactionType


def migrate_schema():
    """Migrate the database schema to include P2P_SELL in the CHECK constraint."""
    
    # Determine database path
    db_path = Path(__file__).parent.parent / 'data' / 'portfolio.db'
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Make sure you're running this from the project root directory.")
        return False
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Check current schema version
        cursor.execute('SELECT version FROM schema_version ORDER BY version DESC LIMIT 1')
        result = cursor.fetchone()
        current_version = result[0] if result else 0
        print(f"Current schema version: {current_version}")
        
        # Check if P2P_SELL is already in the constraint
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='transactions'")
        result = cursor.fetchone()
        if result and 'P2P_SELL' in result[0]:
            print("✓ P2P_SELL is already in the CHECK constraint. No migration needed.")
            return True
        
        # Generate valid_tx_types from the enum
        valid_tx_types = ', '.join([f"'{t.value}'" for t in TransactionType])
        print(f"Valid transaction types: {valid_tx_types}")
        
        # Apply migration
        print("\nApplying schema migration...")
        
        # 1. Rename old table
        cursor.execute('ALTER TABLE transactions RENAME TO transactions_old')
        print('  - Renamed transactions to transactions_old')
        
        # 2. Create new table with updated CHECK constraint
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
        print('  - Created new transactions table with P2P_SELL')
        
        # 3. Copy data
        cursor.execute('INSERT INTO transactions SELECT * FROM transactions_old')
        print('  - Copied data to new table')
        
        # 4. Drop old table
        cursor.execute('DROP TABLE transactions_old')
        print('  - Dropped old table')
        
        # 5. Recreate index
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_asset_timestamp ON transactions (asset_id, timestamp)')
        print('  - Recreated index')
        
        # 6. Update schema version
        cursor.execute('INSERT OR REPLACE INTO schema_version (version, updated_at) VALUES (?, datetime("now"))', (2,))
        print('  - Updated schema version to 2')
        
        conn.commit()
        print('\n✅ Schema migration completed successfully!')
        print('\nNext steps:')
        print('1. Run a full sync to fetch your P2P sell history')
        print('2. Verify your invested capital is now correct')
        return True
        
    except Exception as e:
        conn.rollback()
        print(f'\n❌ Migration failed: {e}')
        print('Your database has been rolled back to its previous state.')
        return False
        
    finally:
        conn.close()


if __name__ == '__main__':
    success = migrate_schema()
    sys.exit(0 if success else 1)
