"""Backtest configuration is whitelisted in the adapter.

The core backtester validates neither period nor frequency and silently
degrades on a bad value, so _backtest_config is the gate. These pin the
defaults and the whitelisting.
"""

from api.analysis_runner import _backtest_config

DEFAULTS = {
    "initial_capital": 10000.0, "period": "2y", "frequency": "monthly",
    "custom_allocation": None,
    "majors_drift": 3.0, "alts_drift": 3.5,
    "majors_sell": 0.5, "majors_buy": 0.75,
    "alts_sell": 0.5, "alts_buy": 1.0,
    "suppress_bear": True,
}


def test_defaults_when_no_params():
    assert _backtest_config(None) == DEFAULTS


def test_accepts_valid_config():
    cfg = _backtest_config(
        {"initial_capital": 5000, "period": "1y", "frequency": "weekly"}
    )
    assert cfg == {**DEFAULTS, "initial_capital": 5000.0,
                   "period": "1y", "frequency": "weekly"}


def test_rejects_unknown_period_and_frequency():
    cfg = _backtest_config({"period": "decade", "frequency": "hourly"})
    assert cfg["period"] == "2y"
    assert cfg["frequency"] == "monthly"


def test_non_positive_or_garbage_capital_falls_back():
    assert _backtest_config({"initial_capital": -5})["initial_capital"] == 10000.0
    assert _backtest_config({"initial_capital": 0})["initial_capital"] == 10000.0
    assert _backtest_config({"initial_capital": "abc"})["initial_capital"] == 10000.0


def test_accepts_custom_year_period():
    # Streamlit parity: freeform "<N>y" periods are passed through to the core.
    assert _backtest_config({"period": "6y"})["period"] == "6y"
    assert _backtest_config({"period": "4y"})["period"] == "4y"
    # The existing whitelist still works alongside.
    assert _backtest_config({"period": "3y"})["period"] == "3y"
    assert _backtest_config({"period": "max"})["period"] == "max"


def test_rejects_non_year_period():
    assert _backtest_config({"period": "decade"})["period"] == "2y"
    assert _backtest_config({"period": ""})["period"] == "2y"
    assert _backtest_config({"period": "6m"})["period"] == "2y"
    assert _backtest_config({"period": None})["period"] == "2y"


def test_drift_thresholds_clamped_to_streamlit_range():
    cfg = _backtest_config({"custom": {"majors_drift": 0, "alts_drift": 99}})
    assert cfg["majors_drift"] == 1.0
    assert cfg["alts_drift"] == 20.0
    cfg = _backtest_config({"custom": {"majors_drift": "garbage", "alts_drift": None}})
    assert cfg["majors_drift"] == 3.0
    assert cfg["alts_drift"] == 3.5


def test_multipliers_clamped_to_streamlit_range():
    cfg = _backtest_config({"custom": {
        "majors_sell": 0, "majors_buy": 9, "alts_sell": 0, "alts_buy": 9,
    }})
    assert cfg["majors_sell"] == 0.1
    assert cfg["majors_buy"] == 2.0
    assert cfg["alts_sell"] == 0.1
    assert cfg["alts_buy"] == 2.0
    cfg = _backtest_config({"custom": {
        "majors_sell": "x", "majors_buy": None, "alts_sell": [], "alts_buy": {},
    }})
    assert cfg["majors_sell"] == 0.5
    assert cfg["majors_buy"] == 0.75
    assert cfg["alts_sell"] == 0.5
    assert cfg["alts_buy"] == 1.0


def test_custom_allocation_must_sum_to_one():
    cfg = _backtest_config({"custom": {"allocation": {"BTC": 0.6, "ETH": 0.4}}})
    assert cfg["custom_allocation"] == {"BTC": 0.6, "ETH": 0.4}
    assert _backtest_config({"custom": {"allocation": {"BTC": 0.5}}})[
        "custom_allocation"] is None
    assert _backtest_config({"custom": {"allocation": {"BTC": 0.9, "ETH": 0.9}}})[
        "custom_allocation"] is None
    assert _backtest_config({"custom": {"allocation": "BTC"}})[
        "custom_allocation"] is None
    assert _backtest_config({"custom": {"allocation": {"BTC": 2.0, "ETH": -1.0}}})[
        "custom_allocation"] is None
    assert _backtest_config({})["custom_allocation"] is None
    assert _backtest_config({"custom": "nope"})["custom_allocation"] is None


def test_suppress_bear_coerces_with_bool():
    assert _backtest_config({})["suppress_bear"] is True
    assert _backtest_config({"custom": {"suppress_bear": False}})["suppress_bear"] is False
    assert _backtest_config({"custom": {"suppress_bear": 0}})["suppress_bear"] is False
    assert _backtest_config({"custom": {"suppress_bear": 1}})["suppress_bear"] is True


def test_live_defaults_override_file_defaults():
    # Plain runs stay behavior-identical when the operator saved non-default
    # thresholds: missing custom fields fall back to the live config.
    live = {"majors_drift": 7.0, "alts_drift": 9.0, "suppress_bear": False}
    cfg = _backtest_config({"custom": {}}, defaults=live)
    assert cfg["majors_drift"] == 7.0
    assert cfg["alts_drift"] == 9.0
    assert cfg["suppress_bear"] is False
    # Explicit custom values still win over the live config.
    cfg = _backtest_config({"custom": {"majors_drift": 5.0}}, defaults=live)
    assert cfg["majors_drift"] == 5.0
