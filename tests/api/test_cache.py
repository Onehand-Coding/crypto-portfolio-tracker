import datetime

import pandas as pd

from api.cache import MetricsCache
from api.serialization import df_to_records, jsonable


def test_df_to_records_converts_nan_to_none():
    df = pd.DataFrame({"symbol": ["BTC"], "value_usd": [float("nan")]})
    assert df_to_records(df) == [{"symbol": "BTC", "value_usd": None}]


def test_df_to_records_on_empty_frame_returns_empty_list():
    assert df_to_records(pd.DataFrame()) == []


def test_jsonable_converts_timestamps_to_iso_strings():
    ts = datetime.datetime(2026, 7, 21, 9, 30, 0)
    assert jsonable(ts) == "2026-07-21T09:30:00"


def test_cache_round_trips_metrics_containing_dataframes(tmp_path):
    cache = MetricsCache(tmp_path / "metrics.json")
    cache.write({
        "total_value_usd": 57.78,
        "holdings_df": pd.DataFrame({"symbol": ["BTC"], "value_usd": [57.78]}),
        "timestamp": datetime.datetime(2026, 7, 21, 9, 30, 0),
    })

    loaded = cache.read()

    assert loaded["total_value_usd"] == 57.78
    assert loaded["holdings_df"] == [{"symbol": "BTC", "value_usd": 57.78}]
    assert loaded["timestamp"] == "2026-07-21T09:30:00"


def test_read_returns_none_when_cache_absent(tmp_path):
    assert MetricsCache(tmp_path / "missing.json").read() is None


def test_read_returns_none_when_cache_corrupt(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{ not json")
    assert MetricsCache(path).read() is None


def test_age_seconds_is_none_when_cache_absent(tmp_path):
    assert MetricsCache(tmp_path / "missing.json").age_seconds() is None


def test_age_seconds_is_small_immediately_after_write(tmp_path):
    cache = MetricsCache(tmp_path / "metrics.json")
    cache.write({"total_value_usd": 1.0})
    assert cache.age_seconds() < 5.0


def test_jsonable_converts_missing_timestamps_to_none():
    """pd.NaT is an instance of datetime.datetime. If the missing-value check
    runs after the datetime check, a missing timestamp serializes to the
    string "NaT" and renders in the UI as though it were a real value."""
    assert jsonable(pd.NaT) is None


def test_df_to_records_converts_missing_timestamps_to_none():
    df = pd.DataFrame({"symbol": ["BTC"], "last_trade": [pd.NaT]})
    assert df_to_records(df) == [{"symbol": "BTC", "last_trade": None}]
