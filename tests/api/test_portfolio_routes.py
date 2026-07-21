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
