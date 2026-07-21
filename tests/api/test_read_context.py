from unittest.mock import Mock, patch

from api import deps


def test_read_context_never_constructs_the_binance_client():
    """The whole point: reads must not touch the network."""
    with patch("api.deps.CryptoPortfolioTracker") as tracker_ctor, \
         patch("api.deps.DatabaseManager"), \
         patch("api.deps.PortfolioAnalyzer"), \
         patch("api.deps.ConfigManager"):
        deps.get_read_context()

    tracker_ctor.assert_not_called()


def test_read_context_is_a_singleton():
    with patch("api.deps.DatabaseManager"), \
         patch("api.deps.PortfolioAnalyzer"), \
         patch("api.deps.ConfigManager"):
        first = deps.get_read_context()
        second = deps.get_read_context()

    assert first is second


def test_read_context_builds_analyzer_in_offline_mode():
    with patch("api.deps.DatabaseManager"), \
         patch("api.deps.PortfolioAnalyzer") as analyzer_ctor, \
         patch("api.deps.ConfigManager"):
        deps.get_read_context()

    assert analyzer_ctor.call_args.kwargs["offline_mode"] is True
    assert analyzer_ctor.call_args.kwargs["binance_client"] is None
    assert analyzer_ctor.call_args.kwargs["fetcher"] is None


def test_read_context_uses_the_configured_database_path():
    with patch("api.deps.DatabaseManager") as db_ctor, \
         patch("api.deps.PortfolioAnalyzer"), \
         patch("api.deps.ConfigManager") as cfg_ctor:
        cfg = Mock()
        cfg.get_database_path.return_value = "data/testnet_portfolio.db"
        cfg_ctor.return_value = cfg
        deps.get_read_context()

    assert db_ctor.call_args.kwargs["db_path"] == "data/testnet_portfolio.db"


def test_reset_singletons_clears_the_read_context():
    with patch("api.deps.DatabaseManager"), \
         patch("api.deps.PortfolioAnalyzer"), \
         patch("api.deps.ConfigManager"):
        first = deps.get_read_context()
        deps.reset_singletons()
        second = deps.get_read_context()

    assert first is not second
