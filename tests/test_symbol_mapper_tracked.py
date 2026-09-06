"""Testnet track-only list: the mapper never spends a CoinGecko discovery
call on an asset the user chose not to follow.

No network: SymbolMapper is built over a plain dict and discover_mappings
is stubbed to explode if touched.
"""

import pytest

from crypto_portfolio_tracker.symbol_mapper import SymbolMapper


def _mapper(tmp_path, portfolio, targets=None):
    config = {
        "portfolio": portfolio,
        "target_allocation": targets or {},
        "database": {"path": str(tmp_path / "db.sqlite")},
        "apis": {"coingecko": {}},
        "symbol_mappings": {"coingecko_ids": {"BTC": "bitcoin"}},
    }
    return SymbolMapper(config)


def _record(mapper, monkeypatch, plant=None):
    """Stub discovery to record calls; optionally plant a mapping."""
    calls = []
    def _discover(symbols):
        calls.append(list(symbols))
        if plant:
            for s in symbols:
                mapper._mappings[s] = plant
    monkeypatch.setattr(mapper, "discover_mappings", _discover)
    return calls


def _explode(mapper, monkeypatch):
    calls = []
    def _discover(symbols):
        calls.append(list(symbols))
        raise AssertionError(f"discovery must not run for {symbols}")
    monkeypatch.setattr(mapper, "discover_mappings", _discover)
    return calls


def test_get_skips_discovery_for_non_tracked_in_testnet(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True,
                                "tracked_symbols": ["BTC", "USDT"]})
    calls = _explode(mapper, monkeypatch)
    assert mapper.get("THISISATESTCOIN") is None
    assert mapper.get("THISISATESTCOIN", "fallback") == "fallback"
    assert calls == []


def test_getitem_skips_discovery_for_non_tracked_in_testnet(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True,
                                "tracked_symbols": ["BTC"]})
    _explode(mapper, monkeypatch)
    with pytest.raises(KeyError):
        mapper["THISISATESTCOIN"]


def test_tracked_symbol_still_discovers(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True,
                                "tracked_symbols": ["BTC", "ETH"]})
    calls = []
    def _discover(symbols):
        calls.append(list(symbols))
        mapper._mappings["ETH"] = "ethereum"
    monkeypatch.setattr(mapper, "discover_mappings", _discover)
    assert mapper.get("ETH") == "ethereum"
    assert calls == [["ETH"]]


def test_explicit_list_also_applies_to_live(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": False,
                                "tracked_symbols": ["BTC"]})
    _explode(mapper, monkeypatch)
    assert mapper.get("THISISATESTCOIN") is None


def test_absent_or_empty_list_means_unrestricted_without_targets(tmp_path, monkeypatch):
    for portfolio in ({"testnet_mode": True},
                      {"testnet_mode": True, "tracked_symbols": []}):
        mapper = _mapper(tmp_path, portfolio)
        calls = _record(mapper, monkeypatch)
        assert mapper.get("THISISATESTCOIN") is None
        assert calls == [["THISISATESTCOIN"]]


def test_empty_list_falls_back_to_core_on_testnet(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True, "tracked_symbols": []},
                     targets={"BTC": 0.5, "ETH": 0.5})
    _explode(mapper, monkeypatch)
    assert mapper.get("THISISATESTCOIN") is None
    # ...while a core asset still resolves (via discovery here).
    mapper._mappings.pop("BTC", None)
    calls = _record(mapper, monkeypatch, plant="resolved-id")
    assert mapper.get("BTC") == "resolved-id"
    assert calls == [["BTC"]]


def test_live_empty_list_ignores_targets(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": False, "tracked_symbols": []},
                     targets={"BTC": 0.5})
    calls = _record(mapper, monkeypatch)
    assert mapper.get("THISISATESTCOIN") is None
    assert calls == [["THISISATESTCOIN"]]


def test_explicit_list_wins_over_targets(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True, "tracked_symbols": ["BTC"]},
                     targets={"BTC": 0.5, "ETH": 0.5})
    _explode(mapper, monkeypatch)
    with pytest.raises(KeyError):
        mapper["ETH"]


def test_list_matching_is_case_insensitive(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True,
                                "tracked_symbols": ["btc", " Usdt "]})
    calls = []
    def _discover(symbols):
        calls.append(list(symbols))
        mapper._mappings[symbols[0]] = "some-id"
    monkeypatch.setattr(mapper, "discover_mappings", _discover)
    assert mapper.get("BTC") == "bitcoin"  # base mapping, no discovery
    assert mapper.get("usdt") == "some-id"
    assert calls == [["USDT"]]
