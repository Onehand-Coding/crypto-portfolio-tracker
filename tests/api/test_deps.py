from unittest.mock import Mock, patch

from api import deps


def test_get_tracker_returns_same_instance_across_calls():
    with patch("api.deps.CryptoPortfolioTracker") as ctor:
        ctor.return_value = Mock()
        first = deps.get_tracker()
        second = deps.get_tracker()

    assert first is second
    assert ctor.call_count == 1


def test_get_tracker_passes_config_manager():
    with patch("api.deps.CryptoPortfolioTracker") as ctor, \
         patch("api.deps.ConfigManager") as cfg_ctor:
        cfg = Mock()
        cfg_ctor.return_value = cfg
        deps.get_tracker()

    ctor.assert_called_once_with(cfg)
