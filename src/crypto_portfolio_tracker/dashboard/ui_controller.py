import logging
import asyncio
import streamlit as st
from typing import Optional
from datetime import datetime

from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker
from crypto_portfolio_tracker.exceptions import NetworkUnavailableError


class Dashboard:
    """Main stateful controller for the UI."""

    def __init__(self):
        self.config_manager = None  # Initialize as None, like the original
        self.tracker = None
        self.offline_mode = False
        self.initialize_session_state()

    def setup_logging(self, level_override: Optional[str] = None):
        try:
            # Initialize config_manager if not already done
            if self.config_manager is None:
                self.config_manager = ConfigManager()

            logging_config = self.config_manager.config.get("logging", {})
            # Do not show logging to console if disabled in config.
            if logging_config.get("console_config", {}).get("enabled", True):
                log_level_str = (
                    level_override or logging_config.get("level", "INFO")
                ).upper()
                log_level = getattr(logging, log_level_str, logging.INFO)
                logging.basicConfig(
                    level=log_level,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
            logging.getLogger("httpx").setLevel(logging.WARNING)
        except Exception as e:
            st.error(f"Failed to setup logging: {str(e)}")

    def initialize_tracker(self):
        if self.tracker_initialized and st.session_state.get("tracker"):
            return st.session_state.tracker

        try:
            with st.spinner("Initializing Portfolio Tracker..."):
                # Initialize config_manager if not already done (like original)
                if self.config_manager is None:
                    self.config_manager = ConfigManager()

                tracker = CryptoPortfolioTracker(self.config_manager)
                st.session_state.tracker_initialized = True
                st.session_state.offline_mode = False
                st.session_state.tracker = tracker
                self.tracker = tracker  # Store as instance variable like original
                return tracker
        except NetworkUnavailableError:
            st.warning("⚠️ Network appears unavailable. Running in offline mode.")
            try:
                if self.config_manager is None:
                    self.config_manager = ConfigManager()
                tracker = CryptoPortfolioTracker(
                    self.config_manager, force_offline=True
                )
                st.session_state.tracker_initialized = True
                st.session_state.offline_mode = True
                st.session_state.tracker = tracker
                self.tracker = tracker  # Store as instance variable like original
                self.offline_mode = True
                return tracker
            except Exception as e:
                st.error(f"Failed to initialize tracker in offline mode: {str(e)}")
                return None
        except Exception as e:
            st.error(f"Failed to initialize tracker: {str(e)}")
            return None

    def initialize_session_state(self):
        defaults = {
            "tracker_initialized": False,
            "offline_mode": False,
            "last_sync": None,
            "portfolio_metrics": None,
            "confirm_delete": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @property
    def tracker_initialized(self):
        return st.session_state.get("tracker_initialized", False)

    @property
    def last_sync(self):
        return st.session_state.get("last_sync", None)

    @property
    def portfolio_metrics(self):
        return st.session_state.get("portfolio_metrics", None)

    def run_full_sync(self):
        tracker = self.initialize_tracker()
        if not tracker:
            st.error("Tracker not initialized")
            return
        try:
            with st.spinner("Running full sync and analysis..."):
                metrics = asyncio.run(tracker.run_full_sync())
                st.session_state.portfolio_metrics = metrics
                st.session_state.last_sync = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.success("✅ Full sync completed successfully!")
        except Exception as e:
            st.error(f"Sync failed: {str(e)}")

    def save_snapshot(self):
        tracker = self.initialize_tracker()
        try:
            if self.portfolio_metrics:
                tracker.save_snapshot(self.portfolio_metrics)
                st.success("📸 Portfolio snapshot saved!")
            else:
                st.warning("No portfolio metrics available to save")
        except Exception as e:
            st.error(f"Failed to save snapshot: {str(e)}")

    def reload(self):
        """
        Reload config and force tracker reinitialization so changes (e.g., testnet)
        take effect immediately on next render.
        """
        # Recreate ConfigManager to re-read the saved file
        self.config_manager = ConfigManager()
        # Drop cached tracker so initialize_tracker() rebuilds with new config
        self.tracker = None
        st.session_state.tracker = None
        st.session_state.tracker_initialized = False
