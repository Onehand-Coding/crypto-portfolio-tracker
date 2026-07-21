import time

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


def _client():
    return TestClient(app)


def test_cockpit_returns_empty_state_when_never_synced(mock_read_context, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routes.portfolio.cache_path_for",
                        lambda cm: tmp_path / "metrics.json")

    response = _client().get("/api/portfolio/cockpit")

    assert response.status_code == 200
    body = response.json()
    assert body["has_data"] is False
    assert body["total_value_usd"] == 0.0
    assert body["staleness"]["age_seconds"] is None


def test_cockpit_reports_both_bases_distinctly(mock_read_context, tmp_path, monkeypatch):
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)

    from api.cache import MetricsCache
    MetricsCache(cache_file).write({
        "total_value_usd": 57.78,
        "total_invested_capital": 76.41,
        "holdings_df": pd.DataFrame([{
            "symbol": "BTC", "total_quantity": 0.001, "value_usd": 57.78,
            "average_cost_basis": 100.0, "cost_basis_total": 199.75,
        }]),
    })

    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame([
        {"symbol": "BTC", "timestamp": "2026-01-01", "type": "BUY",
         "quantity": 1.0, "price_usd": 199.75, "fee_usd": 0.0},
    ])

    body = _client().get("/api/portfolio/cockpit").json()

    assert body["total_value_usd"] == 57.78
    assert body["net_invested"]["basis_usd"] == 76.41
    assert body["fifo"]["basis_usd"] == 199.75
    # The defect this test exists to prevent: the two bases printed as equal.
    assert body["net_invested"]["pl_usd"] != body["fifo"]["pl_usd"]


def test_cockpit_pl_math_matches_the_real_portfolio(mock_read_context, tmp_path, monkeypatch):
    """Golden values from spec section 8.1."""
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)

    from api.cache import MetricsCache
    MetricsCache(cache_file).write({
        "total_value_usd": 57.78,
        "total_invested_capital": 76.41,
        "holdings_df": pd.DataFrame(),
    })
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame([
        {"symbol": "BTC", "timestamp": "2026-01-01", "type": "BUY",
         "quantity": 1.0, "price_usd": 199.75, "fee_usd": 0.0},
    ])

    body = _client().get("/api/portfolio/cockpit").json()

    assert round(body["net_invested"]["pl_usd"], 2) == -18.63
    assert round(body["net_invested"]["pl_percent"], 2) == -24.38
    assert round(body["fifo"]["pl_usd"], 2) == -141.97
    assert round(body["fifo"]["pl_percent"], 2) == -71.07


def test_cockpit_marks_data_stale_past_threshold(mock_read_context, tmp_path, monkeypatch):
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)

    from api.cache import MetricsCache
    MetricsCache(cache_file).write({"total_value_usd": 1.0, "holdings_df": pd.DataFrame()})
    stale = __import__("json").loads(cache_file.read_text())
    stale["_cached_at"] = time.time() - 7200
    cache_file.write_text(__import__("json").dumps(stale))

    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = _client().get("/api/portfolio/cockpit").json()
    assert body["staleness"]["is_stale"] is True


def test_cockpit_never_constructs_the_networked_tracker(mock_read_context, tmp_path,
                                                        monkeypatch):
    """CryptoPortfolioTracker's constructor pings Binance. A read that reaches
    for it would block on the network and 500 while offline."""
    from unittest.mock import patch

    monkeypatch.setattr("api.routes.portfolio.cache_path_for",
                        lambda cm: tmp_path / "metrics.json")

    with patch("api.deps.CryptoPortfolioTracker") as tracker_ctor:
        response = _client().get("/api/portfolio/cockpit")

    assert response.status_code == 200
    tracker_ctor.assert_not_called()


def _write_holdings(cache_file, rows):
    from api.cache import MetricsCache
    MetricsCache(cache_file).write({
        "total_value_usd": 57.78,
        "total_invested_capital": 76.41,
        "holdings_df": pd.DataFrame(rows),
    })


def test_cockpit_reports_a_failed_price_lookup_as_unknown_not_zero(
    mock_read_context, tmp_path, monkeypatch
):
    """portfolio_analyzer seeds prices with 0.0 and only overwrites on success,
    so a failed lookup is indistinguishable from a real zero by the time it
    reaches the cache. Passed through, a material holding renders as $0.00 and
    the UI collapses it into dust -- the position vanishes."""
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)
    _write_holdings(cache_file, [
        {"symbol": "BTC", "total_quantity": 0.0007386, "current_price": 0.0,
         "value_usd": 0.0, "unrealized_pl_usd": -72.37,
         "unrealized_pl_percent": -100.0},
    ])
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = _client().get("/api/portfolio/cockpit").json()
    btc = body["holdings"][0]

    assert btc["price_unavailable"] is True
    assert btc["current_price"] is None
    assert btc["value_usd"] is None
    # Derived from the missing price: reporting -100% would be a fabricated loss.
    assert btc["unrealized_pl_usd"] is None
    assert btc["unrealized_pl_percent"] is None
    assert body["unpriced_count"] == 1


def test_cockpit_treats_a_null_price_the_same_as_zero(
    mock_read_context, tmp_path, monkeypatch
):
    """jsonable() normalises NaN to None, so the same failure arrives as null
    whenever the core's price column is float-typed."""
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)
    _write_holdings(cache_file, [
        {"symbol": "BTC", "total_quantity": 0.5, "current_price": None,
         "value_usd": None},
    ])
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = _client().get("/api/portfolio/cockpit").json()
    assert body["holdings"][0]["price_unavailable"] is True


def test_cockpit_does_not_flag_a_priced_holding(mock_read_context, tmp_path, monkeypatch):
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)
    _write_holdings(cache_file, [
        {"symbol": "BTC", "total_quantity": 0.0007386, "current_price": 76589.0,
         "value_usd": 56.57, "unrealized_pl_usd": -15.81},
    ])
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = _client().get("/api/portfolio/cockpit").json()

    assert body["holdings"][0]["price_unavailable"] is False
    assert body["holdings"][0]["value_usd"] == 56.57
    assert body["unpriced_count"] == 0


def test_cockpit_does_not_flag_an_empty_position(mock_read_context, tmp_path, monkeypatch):
    """A zero-quantity holding has no price to fetch; flagging it would produce
    a 'price unavailable' warning for something that is not held at all."""
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)
    _write_holdings(cache_file, [
        {"symbol": "DUST", "total_quantity": 0.0, "current_price": 0.0, "value_usd": 0.0},
    ])
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = _client().get("/api/portfolio/cockpit").json()

    assert body["holdings"][0]["price_unavailable"] is False
    assert body["unpriced_count"] == 0


def test_cockpit_reports_undefined_percentage_when_basis_is_zero(
    mock_read_context, tmp_path, monkeypatch
):
    """A portfolio built from deposits or rewards has zero net invested. The
    percentage is undefined there -- reporting 0% would read as 'unchanged'
    while the portfolio is actually up."""
    cache_file = tmp_path / "metrics.json"
    monkeypatch.setattr("api.routes.portfolio.cache_path_for", lambda cm: cache_file)

    from api.cache import MetricsCache
    MetricsCache(cache_file).write({
        "total_value_usd": 57.78,
        "total_invested_capital": 0.0,
        "holdings_df": pd.DataFrame(),
    })
    mock_read_context.db_manager.get_all_transactions.return_value = pd.DataFrame()

    body = _client().get("/api/portfolio/cockpit").json()

    assert body["net_invested"]["pl_usd"] == 57.78
    assert body["net_invested"]["pl_percent"] is None
