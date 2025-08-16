import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from crypto_portfolio_tracker.dashboard import utils as ui_utils


def render_database_page(dashboard):
    """Render database page."""

    # Clear previous page state if coming from another page
    keys_to_clear = [
        "preview_file", "cleanup_confirmed", "import_holdings", "import_tx"
    ]

    ui_utils.initialize_page_state("database", keys_to_clear)

    st.markdown("## ️ Data Management")

    tracker = dashboard.initialize_tracker()
    db_path = tracker.db_manager.db_path
    backup_dir = tracker.db_manager.backup_dir

    tab1, tab2, tab3, tab4 = st.tabs(
        ["💾 Backup & Restore", "⬆️ Import / ⬇️ Export", "📸 Snapshots", "ℹ️ Info"]
    )

    # --- 1. Backup & Restore ---
    with tab1:
        st.header("💾 Backup & Restore")
        if st.button("Create Backup"):
            backup_path = tracker.db_manager.backup_database()
            if backup_path:
                st.success(f"Backup created: {backup_path}")
            else:
                st.error("Backup failed. See logs for details.")

        backups = tracker.db_manager.list_backups()
        if not backups:
            st.info("No backups found.")
        else:
            backup_names = [b.name for b in backups]
            selected_backup = st.selectbox("Select backup", backup_names)
            backup_path = backup_dir / selected_backup

            col1, col2, col3 = st.columns(3)
            with col1:
                with open(backup_path, "rb") as f:
                    st.download_button(
                        "Download",
                        f,
                        file_name=selected_backup,
                        use_container_width=True,
                    )
            with col2:
                if st.button(
                    "Delete",
                    key=f"delete_{selected_backup}",
                    use_container_width=True,
                ):
                    try:
                        backup_path.unlink()
                        st.success(f"Deleted {selected_backup}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")
            with col3:
                if st.button(
                    "Restore",
                    key=f"restore_{selected_backup}",
                    use_container_width=True,
                ):
                    st.warning(
                        "⚠️ This will overwrite your current database. This action is irreversible!"
                    )
                    if st.button(
                        "Confirm Restore", key=f"confirm_restore_{selected_backup}"
                    ):
                        success = tracker.db_manager.restore_from_backup(backup_path)
                        if success:
                            st.success(
                                "Database restored. Please restart the app to use the restored data."
                            )
                        else:
                            st.error("Restore failed. See logs for details.")

    # --- 2. Import/Export ---
    with tab2:
        st.header("⬆️ Import / ⬇️ Export Raw Data")
        st.markdown("#### Create New Export")

        col1, col2 = st.columns(2)
        with col1:
            export_type = st.selectbox(
                "Select data to export", ["Holdings", "Transactions"]
            )
        with col2:
            export_format = st.radio("Select format", ["CSV", "Excel"], horizontal=True)

        if st.button("🚀 Generate Export", use_container_width=True, type="primary"):
            with st.spinner(f"Generating {export_type} export..."):
                try:
                    result = tracker.export_data_backup(
                        export_type.lower(), export_format.lower()
                    )
                    if result:
                        st.success(f"Successfully created {export_type} export!")
                    else:
                        st.error(f"Failed to create {export_type} export.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Export failed: {e}")

        st.markdown("---")
        st.markdown("#### My Exports")

        all_files = sorted(
            list(
                Path(
                    dashboard.config_manager.config.get("exports", {}).get(
                        "path", "data/exports/"
                    )
                ).glob("*_backup_*.*")
            ),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not all_files:
            st.info("No exports found. Generate one above!")
        else:
            selected_export = st.selectbox(
                "Select an export file:",
                options=all_files,
                format_func=lambda x: f"{x.name} ({x.stat().st_size / 1024:.1f} KB, {datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})",
            )

            if "preview_file" not in st.session_state:
                st.session_state.preview_file = None

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👁️ Preview", use_container_width=True):
                    st.session_state.preview_file = selected_export
                    st.rerun()
            with col2:
                with open(selected_export, "rb") as f:
                    file_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download",
                        data=file_bytes,
                        file_name=selected_export.name,
                        mime=f"text/{selected_export.suffix[1:].lower()}"
                        if selected_export.suffix == ".csv"
                        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
            with col3:
                if st.button("🗑️ Delete", use_container_width=True):
                    try:
                        selected_export.unlink()
                        if st.session_state.preview_file == selected_export:
                            st.session_state.preview_file = None
                        st.success(f"Deleted {selected_export.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")

            if st.session_state.preview_file:
                st.markdown("---")
                st.markdown(f"### 👁️ Preview: {st.session_state.preview_file.name}")
                try:
                    if st.session_state.preview_file.name.endswith(".csv"):
                        st.dataframe(pd.read_csv(st.session_state.preview_file))
                    else:
                        st.dataframe(pd.read_excel(st.session_state.preview_file))
                except Exception as e:
                    st.error(f"Could not preview file: {e}")

                if st.button("❌ Close Preview"):
                    st.session_state.preview_file = None
                    st.rerun()

        st.markdown("---")
        st.markdown("### Import")
        uploaded_holdings = st.file_uploader(
            "Import Holdings (CSV/Excel)",
            type=["csv", "xlsx"],
            key="import_holdings",
        )
        if uploaded_holdings:
            try:
                if uploaded_holdings.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_holdings)
                else:
                    df = pd.read_excel(uploaded_holdings)

                # Validate required columns for holdings
                required_holdings_cols = [
                    "symbol",
                    "quantity",
                    "average_cost_basis",
                ]
                missing_cols = [
                    col for col in required_holdings_cols if col not in df.columns
                ]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info("💡 Required columns: symbol, quantity, average_cost_basis")
                    st.info("💡 All other columns will be imported if present")
                    st.stop()

                # Import ALL columns - let the database methods handle what they need
                st.dataframe(df)
                if st.button("Import Holdings"):
                    tracker.db_manager.update_holdings(df)
                    st.success("Holdings imported successfully!")
            except Exception as e:
                st.error(f"Failed to import holdings: {e}")

        uploaded_tx = st.file_uploader(
            "Import Transactions (CSV/Excel)", type=["csv", "xlsx"], key="import_tx"
        )
        if uploaded_tx:
            try:
                if uploaded_tx.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_tx)
                else:
                    df = pd.read_excel(uploaded_tx)

                # Validate required columns but import everything
                required_tx_cols = ["symbol", "timestamp", "type", "quantity"]
                missing_cols = [
                    col for col in required_tx_cols if col not in df.columns
                ]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info("💡 Required columns: symbol, timestamp, type, quantity")
                    st.info("💡 All other columns will be imported if present")
                    st.stop()

                # Validate transaction types
                valid_types = ["BUY", "SELL", "DEPOSIT", "WITHDRAWAL", "TRANSFER"]
                invalid_types = df[~df["type"].isin(valid_types)]["type"].unique()
                if len(invalid_types) > 0:
                    st.error(f"❌ Invalid transaction types found: {invalid_types}")
                    st.info(f"💡 Valid types: {valid_types}")
                    st.stop()

                # Import ALL columns - let the database methods handle what they need
                st.dataframe(df)
                if st.button("Import Transactions"):
                    st.warning(
                        "⚠️ This will update existing transactions and may create duplicates if transaction_hash is missing."
                    )
                    if st.button("Confirm Import"):
                        try:
                            transactions_list = df.to_dict(orient="records")
                            rows_affected = tracker.db_manager.bulk_insert_transactions(
                                transactions_list
                            )
                            st.success(
                                f"✅ Successfully imported {rows_affected} transactions!"
                            )
                        except Exception as e:
                            st.error(f"❌ Failed to import transactions: {e}")
            except Exception as e:
                st.error(f"Failed to read transaction file: {e}")

    # --- 3. Manage Snapshots ---
    with tab3:
        st.header("📸 Snapshots")
        try:
            snapshots = tracker.db_manager.get_all_snapshots()
            # Don't filter out snapshots with NaN timestamps - let the chart handle it
            # snapshots = snapshots[~pd.isna(snapshots["timestamp"])].copy()
            snapshots = snapshots.drop_duplicates(subset=["timestamp"])

            # Set timestamp as index for proper charting
            if not snapshots.empty and "timestamp" in snapshots.columns:
                # Convert timestamp to datetime first
                snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"])
                snapshots = snapshots.set_index("timestamp")
            else:
                st.write("No timestamp column found or empty snapshots")

        except Exception as e:
            st.error(f"Failed to load snapshots: {e}")
            snapshots = None

        if snapshots.empty:
            st.info("No snapshots found. Take a snapshot from the dashboard first.")
        else:
            # Sort by timestamp, handling None values properly
            snapshots = snapshots.sort_values("timestamp", na_position="first")

            # Include ALL snapshots (including problematic ones) so users can delete them
            snapshot_labels = []
            for idx, row in snapshots.iterrows():
                # Check for invalid snapshots: no timestamp OR all zero values
                # Use row.name to access the timestamp since it's now the index
                timestamp_value = row.name
                is_no_timestamp = (
                    pd.isna(timestamp_value)
                    or timestamp_value is None
                    or str(timestamp_value).lower()
                    in ["none", "nan", "nat", "null", ""]
                )
                is_zero_values = (
                    row["total_value_usd"] == 0.0
                    and row["total_cost_basis_usd"] == 0.0
                    and row["unrealized_pl_usd"] == 0.0
                    and row["unrealized_pl_percent"] == 0.0
                )

                if is_no_timestamp:
                    label = f"⚠️ Invalid Snapshot (No Timestamp) | Value: ${row['total_value_usd']:,.2f}"
                elif is_zero_values:
                    label = f"⚠️ Invalid Snapshot (Zero Values) | {timestamp_value} | Value: ${row['total_value_usd']:,.2f}"
                else:
                    label = f"{timestamp_value} | Value: ${row['total_value_usd']:,.2f}"
                snapshot_labels.append(label)

            selected_idx = st.selectbox(
                "Select Snapshot",
                range(len(snapshots)),
                format_func=lambda i: snapshot_labels[i],
            )
            selected_row = snapshots.iloc[selected_idx]

            st.write("### Snapshot Details")
            display_dict = {
                "Timestamp": str(selected_row.name),  # Use .name to access the index
                "Total Value (USD)": f"${selected_row['total_value_usd']:,.2f}",
                "Total Cost Basis (USD)": f"${selected_row['total_cost_basis_usd']:,.2f}",
                "Unrealized P/L (USD)": f"${selected_row['unrealized_pl_usd']:,.2f}",
                "Unrealized P/L (%)": f"{selected_row['unrealized_pl_percent']:.2f}%",
            }
            st.table(pd.DataFrame([display_dict]))

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Snapshot (CSV)",
                    selected_row.to_frame().T.to_csv(index=False),
                    "portfolio_snapshot.csv",
                )
            with col2:
                if st.button("Delete Selected Snapshot"):
                    try:
                        rows_deleted = tracker.db_manager.delete_snapshot(
                            selected_row.name,  # Use .name to access the timestamp index
                            selected_row["total_value_usd"],
                            selected_row["total_cost_basis_usd"],
                            selected_row["unrealized_pl_usd"],
                            selected_row["unrealized_pl_percent"],
                        )
                        if rows_deleted > 0:
                            st.success("✅ Successfully deleted snapshot.")
                            st.rerun()
                        else:
                            st.warning(
                                "⚠️ No rows were deleted. The snapshot may have already been removed or the query didn't match any records."
                            )
                    except Exception as e:
                        st.error(f"❌ Failed to delete snapshot: {e}")

        st.markdown("---")

        # --- Improved Data Cleanup with Context and Confirmation ---
        st.subheader("🗑️ Data Cleanup")

        # Get cleanup configuration
        cleanup_days = tracker.config.get("database", {}).get("cleanup_days", 90)

        # Show current configuration
        st.info(f"**Current Retention Period:** {cleanup_days} days")
        if cleanup_days <= 0:
            st.warning("⚠️ Data cleanup is currently disabled (cleanup_days = 0)")
            st.stop()

        # Calculate what would be deleted
        cutoff_date = datetime.now() - timedelta(days=cleanup_days)

        # Get cleanup statistics
        stats = tracker.db_manager.get_cleanup_statistics()

        if not stats["cleanup_enabled"]:
            st.warning("⚠️ Data cleanup is currently disabled (cleanup_days = 0)")
            st.stop()

        if "error" in stats:
            st.error(f"Could not analyze database: {stats['error']}")
            st.stop()

        old_transactions = stats["old_transactions"]
        old_snapshots = stats["old_snapshots"]
        total_transactions = stats["total_transactions"]
        total_snapshots = stats["total_snapshots"]
        cutoff_date = stats["cutoff_date"]

        # Display what will be deleted
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "📊 Old Transactions",
                f"{old_transactions:,}",
                f"of {total_transactions:,} total",
            )
        with col2:
            st.metric(
                "📸 Old Snapshots",
                f"{old_snapshots:,}",
                f"of {total_snapshots:,} total",
            )

        # Show cutoff date
        st.write(f"**Cutoff Date:** {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")

        # Warning about what this affects
        if old_transactions > 0 or old_snapshots > 0:
            st.warning("""
            ⚠️ **This will permanently delete:**
            - Historical transaction data older than the retention period
            - Portfolio snapshots older than the retention period
            - **Impact:** This may affect tax reporting, historical analysis, and portfolio tracking
            """)

            # Confirmation section
            st.markdown("---")
            st.subheader("🔐 Confirmation Required")

            # Two-step confirmation
            if "cleanup_confirmed" not in st.session_state:
                st.session_state.cleanup_confirmed = False

            if not st.session_state.cleanup_confirmed:
                if st.button(
                    "🗑️ I understand - Show Final Confirmation", type="secondary"
                ):
                    st.session_state.cleanup_confirmed = True
                    st.rerun()
            else:
                st.error("""
                🚨 **FINAL WARNING:**
                - This action is **IRREVERSIBLE**
                - No backup will be created automatically
                - Consider creating a backup first
                """)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ CONFIRM DELETION", type="primary"):
                        try:
                            # Create backup before deletion
                            backup_path = tracker.db_manager.backup_database()
                            if backup_path:
                                st.success(f"✅ Backup created: {backup_path}")

                            # Perform cleanup
                            tracker.cleanup_old_data()
                            st.success("✅ Data cleanup completed successfully!")
                            st.session_state.cleanup_confirmed = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Cleanup failed: {e}")
                            st.session_state.cleanup_confirmed = False

                with col2:
                    if st.button("❌ Cancel", type="secondary"):
                        st.session_state.cleanup_confirmed = False
                        st.rerun()
        else:
            st.success("✅ No old data to clean up!")
            st.info("All your data is within the retention period.")

    # --- 4. Database Info ---
    with tab4:
        st.header("ℹ️ Database Info")
        st.write(f"**Database Path:** `{db_path}`")
        if os.path.exists(db_path):
            st.write(f"**Size:** {os.path.getsize(db_path) / 1024:.1f} KB")
            st.write(
                f"**Last Modified:** {pd.to_datetime(os.path.getmtime(db_path), unit='s')}"
            )
        else:
            st.warning("Database file not found.")
        st.write(f"**Backup Directory:** `{backup_dir}`")
        if backup_dir.exists():
            st.write(f"**Backups Available:** {len(list(backup_dir.glob('*.bak')))}")
        else:
            st.warning("Backup directory not found.")

    # Add this debug section in the snapshots tab, right after loading snapshots:

    # Debug section - add this after loading snapshots
    with st.expander("🔍 Debug: Raw Database Data"):
        try:
            # Get raw data from database
            with tracker.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM portfolio_snapshots ORDER BY timestamp")
                raw_rows = cursor.fetchall()

            st.write(f"**Total snapshots in database:** {len(raw_rows)}")

            if raw_rows:
                st.write("**Raw database rows:**")
                for i, row in enumerate(raw_rows):
                    st.write(
                        f"Row {i + 1}: timestamp='{row[0]}', value={row[1]}, cost={row[2]}, pl={row[3]}, pl_pct={row[4]}"
                    )

            st.write("**Pandas DataFrame after parsing:**")
            st.dataframe(snapshots)

        except Exception as e:
            st.error(f"Debug failed: {e}")
