import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


def test_capital_flow_classifies_inflows_and_outflows(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 0.9},
        {"source": "Binance P2P Sell", "type": "SELL", "quantity": 20.0, "price_usd": 1.0},
        {"source": "Binance", "type": "WITHDRAWAL", "quantity": 5.0, "price_usd": 2.0},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 60.0

    body = TestClient(app).get("/api/capital/flow").json()

    assert body["net_invested_usd"] == 60.0
    assert [r["direction"] for r in body["rows"]] == ["in", "out", "out"]
    assert body["total_in_usd"] == 90.0
    assert body["total_out_usd"] == 30.0


def test_capital_flow_flags_peg_fallback_provenance(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 1.0},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 100.0

    row = TestClient(app).get("/api/capital/flow").json()["rows"][0]

    assert row["provenance"] == "usdt_peg_fallback"
    assert row["is_suspect"] is True


def test_capital_flow_flags_zero_price_as_failed_lookup(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 0.0},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 0.0

    row = TestClient(app).get("/api/capital/flow").json()["rows"][0]

    assert row["provenance"] == "failed_lookup"
    assert row["is_suspect"] is True
    assert row["value_usd"] == 0.0


def test_capital_flow_marks_computed_rates_as_trusted(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 0.0179},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 1.79

    row = TestClient(app).get("/api/capital/flow").json()["rows"][0]

    assert row["provenance"] == "computed"
    assert row["is_suspect"] is False


def test_capital_flow_empty_when_no_transactions(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame()
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 0.0

    body = TestClient(app).get("/api/capital/flow").json()

    assert body["rows"] == []
    assert body["net_invested_usd"] == 0.0


def test_capital_flow_flags_nan_price_as_failed_lookup(mock_read_context):
    """A SQL NULL in a column that also holds numbers arrives as NaN, not None.
    NaN is truthy, so a naive `value or 0.0` passes it through and the row
    classifies as 'computed' -- an unpriced row presented as trustworthy, which
    is the precise failure this endpoint exists to surface."""
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 0.0179},
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 50.0, "price_usd": None},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 1.79

    body = TestClient(app).get("/api/capital/flow").json()
    unpriced = body["rows"][1]

    assert unpriced["provenance"] == "failed_lookup"
    assert unpriced["is_suspect"] is True
    assert unpriced["value_usd"] == 0.0
    assert body["suspect_count"] == 1


def test_capital_flow_totals_are_json_safe_when_a_price_is_missing(mock_read_context):
    """NaN is not valid JSON. A missing price must not poison the totals."""
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 100.0, "price_usd": 0.0179},
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 50.0, "price_usd": None},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 1.79

    response = TestClient(app).get("/api/capital/flow")

    assert "NaN" not in response.text
    assert round(response.json()["total_in_usd"], 4) == 1.79


def test_capital_flow_treats_negative_price_as_failed_lookup(mock_read_context):
    mock_read_context.db_manager.get_invested_capital_transactions.return_value = pd.DataFrame([
        {"source": "Binance P2P Buy", "type": "BUY", "quantity": 10.0, "price_usd": -5.0},
    ])
    mock_read_context.portfolio_analyzer.calculate_total_invested_capital.return_value = 0.0

    row = TestClient(app).get("/api/capital/flow").json()["rows"][0]

    assert row["provenance"] == "failed_lookup"
    assert row["is_suspect"] is True
