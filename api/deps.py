"""Process-wide singletons over the untouched core.

CryptoPortfolioTracker opens SQLite and initializes the Binance client in
its constructor, so it is built once per process and reused.
"""

from typing import Optional

from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker

_config_manager: Optional[ConfigManager] = None
_tracker: Optional[CryptoPortfolioTracker] = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_tracker() -> CryptoPortfolioTracker:
    global _tracker
    if _tracker is None:
        _tracker = CryptoPortfolioTracker(get_config_manager())
    return _tracker


def set_tracker_for_testing(tracker) -> None:
    global _tracker
    _tracker = tracker


def reset_singletons() -> None:
    global _config_manager, _tracker
    _config_manager = None
    _tracker = None
