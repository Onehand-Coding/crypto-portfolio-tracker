"""Backtest configuration is whitelisted in the adapter.

The core backtester validates neither period nor frequency and silently
degrades on a bad value, so _backtest_config is the gate. These pin the
defaults and the whitelisting.
"""

from api.analysis_runner import _backtest_config


def test_defaults_when_no_params():
    assert _backtest_config(None) == {
        "initial_capital": 10000.0, "period": "2y", "frequency": "monthly"
    }


def test_accepts_valid_config():
    cfg = _backtest_config(
        {"initial_capital": 5000, "period": "1y", "frequency": "weekly"}
    )
    assert cfg == {"initial_capital": 5000.0, "period": "1y", "frequency": "weekly"}


def test_rejects_unknown_period_and_frequency():
    cfg = _backtest_config({"period": "17y", "frequency": "hourly"})
    assert cfg["period"] == "2y"
    assert cfg["frequency"] == "monthly"


def test_non_positive_or_garbage_capital_falls_back():
    assert _backtest_config({"initial_capital": -5})["initial_capital"] == 10000.0
    assert _backtest_config({"initial_capital": 0})["initial_capital"] == 10000.0
    assert _backtest_config({"initial_capital": "abc"})["initial_capital"] == 10000.0
