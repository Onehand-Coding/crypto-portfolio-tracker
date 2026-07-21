"""Process-wide singletons over the untouched core.

CryptoPortfolioTracker opens SQLite and initializes the Binance client in
its constructor, so it is built once per process and reused.
"""

from dataclasses import dataclass
from typing import Optional

from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.database import DatabaseManager
from crypto_portfolio_tracker.portfolio_analyzer import PortfolioAnalyzer
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker

_config_manager: Optional[ConfigManager] = None
_tracker: Optional[CryptoPortfolioTracker] = None


@dataclass
class ReadContext:
    """Everything a read endpoint needs, and nothing that touches the network.

    CryptoPortfolioTracker.__init__ pings Binance and syncs server time, so
    read paths must not construct it. They get SQLite and an offline analyzer
    instead. get_tracker() stays reserved for sync.
    """

    config_manager: ConfigManager
    db_manager: DatabaseManager
    portfolio_analyzer: PortfolioAnalyzer


_read_context: Optional[ReadContext] = None


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


def get_read_context() -> ReadContext:
    global _read_context
    if _read_context is None:
        config_manager = get_config_manager()
        db_manager = DatabaseManager(
            db_path=config_manager.get_database_path(),
            backup_dir=config_manager.get_backup_dir(),
        )
        _read_context = ReadContext(
            config_manager=config_manager,
            db_manager=db_manager,
            portfolio_analyzer=PortfolioAnalyzer(
                config=config_manager.config,
                db_manager=db_manager,
                binance_client=None,
                fetcher=None,
                enricher=None,
                offline_mode=True,
                config_manager=config_manager,
            ),
        )
    return _read_context


def set_read_context_for_testing(context) -> None:
    global _read_context
    _read_context = context


def reset_singletons() -> None:
    global _config_manager, _tracker, _read_context
    _config_manager = None
    _tracker = None
    _read_context = None
