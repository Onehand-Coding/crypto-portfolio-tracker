"""Testnet track-only list: the mapper never spends a CoinGecko discovery
call on an asset the user chose not to follow.

No network: SymbolMapper is built over a plain dict and discover_mappings
is stubbed to explode if touched.
"""

import pytest

from crypto_portfolio_tracker.symbol_mapper import SymbolMapper


def _mapper(tmp_path, portfolio):
    config = {
        "portfolio": portfolio,
        "database": {"path": str(tmp_path / "db.sqlite")},
        "apis": {"coingecko": {}},
        "symbol_mappings": {"coingecko_ids": {"BTC": "bitcoin"}},
    }
    return SymbolMapper(config)


def _explode(mapper, monkeypatch):
    calls = []
    def _discover(symbols):
        calls.append(list(symbols))
        raise AssertionError(f"discovery must not run for {symbols}")
    monkeypatch.setattr(mapper, "discover_mappings", _discover)
    return calls


def test_get_skips_discovery_for_non_tracked_in_testnet(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True,
                                "testnet_tracked_symbols": ["BTC", "USDT"]})
    calls = _explode(mapper, monkeypatch)
    assert mapper.get("THISISATESTCOIN") is None
    assert mapper.get("THISISATESTCOIN", "fallback") == "fallback"
    assert calls == []


def test_getitem_skips_discovery_for_non_tracked_in_testnet(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True,
                                "testnet_tracked_symbols": ["BTC"]})
    _explode(mapper, monkeypatch)
    with pytest.raises(KeyError):
        mapper["THISISATESTCOIN"]


def test_tracked_symbol_still_discovers(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True,
                                "testnet_tracked_symbols": ["BTC", "ETH"]})
    calls = []
    def _discover(symbols):
        calls.append(list(symbols))
        mapper._mappings["ETH"] = "ethereum"
    monkeypatch.setattr(mapper, "discover_mappings", _discover)
    assert mapper.get("ETH") == "ethereum"
    assert calls == [["ETH"]]


def test_live_mode_ignores_the_list(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": False,
                                "testnet_tracked_symbols": ["BTC"]})
    calls = []
    monkeypatch.setattr(mapper, "discover_mappings",
                        lambda symbols: calls.append(list(symbols)))
    assert mapper.get("THISISATESTCOIN") is None
    assert calls == [["THISISATESTCOIN"]]


def test_absent_or_empty_list_means_unrestricted(tmp_path, monkeypatch):
    for portfolio in ({"testnet_mode": True},
                      {"testnet_mode": True, "testnet_tracked_symbols": []}):
        mapper = _mapper(tmp_path, portfolio)
        calls = []
        monkeypatch.setattr(mapper, "discover_mappings",
                            lambda symbols: calls.append(list(symbols)))
        assert mapper.get("THISISATESTCOIN") is None
        assert calls == [["THISISATESTCOIN"]]


def test_list_matching_is_case_insensitive(tmp_path, monkeypatch):
    mapper = _mapper(tmp_path, {"testnet_mode": True,
                                "testnet_tracked_symbols": ["btc", " Usdt "]})
    calls = []
    def _discover(symbols):
        calls.append(list(symbols))
        mapper._mappings[symbols[0]] = "some-id"
    monkeypatch.setattr(mapper, "discover_mappings", _discover)
    assert mapper.get("BTC") == "bitcoin"  # base mapping, no discovery
    assert mapper.get("usdt") == "some-id"
    assert calls == [["USDT"]]
