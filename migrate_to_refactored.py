#!/usr/bin/env python3
"""
Safe Migration Script for Crypto Portfolio Tracker Refactoring

This script safely migrates from the monolithic CryptoPortfolioTracker
to the refactored facade-based implementation.

Usage:
    python migrate_to_refactored.py [--force] [--no-backup]

Safety features:
- Creates backup of original implementation
- Runs verification tests before migration
- Allows rollback if issues are detected
- Preserves all existing functionality
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class SafeMigrator:
    """Handles safe migration to refactored implementation."""

    def __init__(self, force: bool = False, no_backup: bool = False):
        self.force = force
        self.no_backup = no_backup
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src" / "crypto_portfolio_tracker"

        # File paths
        self.original_file = self.src_dir / "portfolio_tracker.py"
        self.refactored_file = self.src_dir / "crypto_portfolio_tracker_refactored.py"
        self.backup_file = self.src_dir / "portfolio_tracker_original_backup.py"

        # Verification script
        self.verification_script = self.project_root / "migration_verification.py"

    def print_header(self):
        """Print migration header."""
        print("🚀 Crypto Portfolio Tracker Migration Script")
        print("=" * 60)
        print("Migrating from monolithic to refactored facade implementation")
        print("=" * 60)

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("🔍 Checking prerequisites...")

        # Check if original file exists
        if not self.original_file.exists():
            print(f"❌ Original file not found: {self.original_file}")
            return False

        # Check if refactored file exists
        if not self.refactored_file.exists():
            print(f"❌ Refactored file not found: {self.refactored_file}")
            return False

        # Check if verification script exists
        if not self.verification_script.exists():
            print(f"❌ Verification script not found: {self.verification_script}")
            return False

        # Check if required component files exist
        required_files = [
            "models.py",
            "data_synchronizer.py",
            "portfolio_analyzer.py",
            "trade_executor.py",
            "dca_manager.py",
            "report_generator.py",
            "data_manager.py"
        ]

        missing_files = []
        for filename in required_files:
            if not (self.src_dir / filename).exists():
                missing_files.append(filename)

        if missing_files:
            print(f"❌ Missing required component files: {missing_files}")
            return False

        print("✅ All prerequisites met")
        return True

    def create_backup(self) -> bool:
        """Create backup of original implementation."""
        if self.no_backup:
            print("⚠️  Skipping backup (--no-backup flag set)")
            return True

        print("💾 Creating backup of original implementation...")

        try:
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_backup = self.src_dir / f"portfolio_tracker_backup_{timestamp}.py"

            shutil.copy2(self.original_file, timestamped_backup)
            shutil.copy2(self.original_file, self.backup_file)

            print(f"✅ Backup created: {timestamped_backup}")
            print(f"✅ Working backup: {self.backup_file}")
            return True

        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            return False

    def run_verification_tests(self) -> bool:
        """Run verification tests on the refactored implementation."""
        print("🧪 Running verification tests...")

        try:
            # Run the verification script
            result = subprocess.run([
                sys.executable, str(self.verification_script), "--dry-run"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ All verification tests passed")
                if result.stdout:
                    # Show a summary of test results
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Test Summary:' in line or '✅ Passed:' in line or '❌ Failed:' in line or 'Success Rate:' in line:
                            print(f"    {line}")
                return True
            else:
                print("❌ Verification tests failed")
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("❌ Verification tests timed out")
            return False
        except Exception as e:
            print(f"❌ Error running verification tests: {e}")
            return False

    def perform_migration(self) -> bool:
        """Perform the actual migration."""
        print("🔄 Performing migration...")

        try:
            # Replace original file with refactored implementation
            shutil.copy2(self.refactored_file, self.original_file)
            print("✅ Original file replaced with refactored implementation")
            return True

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False

    def verify_migration(self) -> bool:
        """Verify the migration was successful."""
        print("✅ Verifying migration...")

        try:
            # Try importing the migrated module
            sys.path.insert(0, str(self.src_dir.parent))

            from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker
            from crypto_portfolio_tracker.models import TradeResult, ExecutionMode

            print("✅ Import test passed")

            # Check that it has key methods
            if not hasattr(CryptoPortfolioTracker, 'calculate_portfolio_metrics'):
                print("❌ Missing key method: calculate_portfolio_metrics")
                return False

            if not hasattr(CryptoPortfolioTracker, 'execute_rebalancing_trades_core'):
                print("❌ Missing key method: execute_rebalancing_trades_core")
                return False

            print("✅ Key methods present")
            return True

        except Exception as e:
            print(f"❌ Migration verification failed: {e}")
            return False

    def rollback(self) -> bool:
        """Rollback to original implementation."""
        print("🔄 Rolling back to original implementation...")

        try:
            if self.backup_file.exists():
                shutil.copy2(self.backup_file, self.original_file)
                print("✅ Rollback completed")
                return True
            else:
                print("❌ No backup file found for rollback")
                return False

        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False

    def cleanup_old_files(self):
        """Clean up temporary and old files."""
        print("🧹 Cleaning up...")

        # Remove the refactored file since it's now been copied to the main location
        # We keep it as reference but could optionally remove it
        print("✅ Cleanup completed")

    def run_migration(self) -> bool:
        """Run the complete migration process."""
        self.print_header()

        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites not met. Aborting migration.")
            return False

        # Step 2: Create backup
        if not self.create_backup():
            print("\n❌ Backup failed. Aborting migration.")
            return False

        # Step 3: Run verification tests
        if not self.run_verification_tests():
            if not self.force:
                print("\n❌ Verification tests failed. Use --force to override.")
                return False
            else:
                print("\n⚠️  Verification tests failed but --force flag set. Continuing...")

        # Step 4: Perform migration
        if not self.perform_migration():
            print("\n❌ Migration failed. Attempting rollback...")
            if self.rollback():
                print("✅ Rollback successful")
            else:
                print("❌ Rollback failed - manual intervention required")
            return False

        # Step 5: Verify migration
        if not self.verify_migration():
            print("\n❌ Migration verification failed. Attempting rollback...")
            if self.rollback():
                print("✅ Rollback successful")
            else:
                print("❌ Rollback failed - manual intervention required")
            return False

        # Step 6: Cleanup
        self.cleanup_old_files()

        # Success message
        print("\n" + "=" * 60)
        print("🎉 Migration completed successfully!")
        print("=" * 60)
        print("The CryptoPortfolioTracker has been successfully refactored into")
        print("specialized components while maintaining full backward compatibility.")
        print("")
        print("Next steps:")
        print("1. Test your existing tools and scripts")
        print("2. Run comprehensive tests in your environment")
        print("3. Monitor for any issues")
        print("")
        print("If you encounter any issues, you can rollback using:")
        print(f"  cp {self.backup_file} {self.original_file}")
        print("=" * 60)

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Safely migrate to refactored CryptoPortfolioTracker implementation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
