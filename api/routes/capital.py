"""Capital flow: fiat -> USDT -> asset provenance.

A failed yfinance fiat lookup silently zeroes price_usd, which once hid
$50.41 of inflow and inverted reported P/L. Provenance makes that class of
failure visible in the UI rather than invisible in the total.
"""

import math

from fastapi import APIRouter, Depends

from api.deps import get_read_context
from api.schemas.capital import CapitalFlowResponse, CapitalFlowRow

router = APIRouter(prefix="/api/capital", tags=["capital"])

INFLOW_SOURCES = {"Binance P2P Buy"}


def _safe_float(value) -> float:
    """Coerce a DataFrame cell to a real float, mapping missing to 0.0.

    A SQL NULL in a column that also holds numbers arrives as NaN, not None,
    and NaN is truthy -- so `value or 0.0` passes it straight through. That
    matters here: an unpriced row would then classify as 'computed' and be
    presented as trustworthy, which is the exact failure this endpoint exists
    to surface. NaN also serializes to invalid JSON.
    """
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(result) or math.isinf(result) else result


def _provenance(price_usd: float) -> str:
    """Classify how a row's USD rate was arrived at.

    A real USDT/USD transaction legitimately has a rate of 1.0 and will be
    flagged as a peg fallback. That false positive is deliberate: prompting a
    check on a good row is cheap, presenting a bad row as good is what cost
    $50.41 of hidden inflow.
    """
    if not price_usd or price_usd < 0:
        return "failed_lookup"
    if price_usd == 1.0:
        return "usdt_peg_fallback"
    return "computed"


@router.get("/flow", response_model=CapitalFlowResponse)
def capital_flow(ctx=Depends(get_read_context)) -> CapitalFlowResponse:
    df = ctx.db_manager.get_invested_capital_transactions()
    net_invested = float(
        ctx.portfolio_analyzer.calculate_total_invested_capital()
    )

    rows: list[CapitalFlowRow] = []
    if df is not None and not df.empty:
        for record in df.to_dict(orient="records"):
            source = str(record.get("source") or "")
            tx_type = str(record.get("type") or "")
            price_usd = _safe_float(record.get("price_usd"))
            quantity = _safe_float(record.get("quantity"))
            direction = "in" if source in INFLOW_SOURCES else "out"
            provenance = _provenance(price_usd)

            rows.append(CapitalFlowRow(
                source=source,
                type=tx_type,
                direction=direction,
                quantity=quantity,
                price_usd=price_usd,
                value_usd=quantity * price_usd,
                provenance=provenance,
                is_suspect=provenance != "computed",
            ))

    return CapitalFlowResponse(
        rows=rows,
        total_in_usd=sum(r.value_usd for r in rows if r.direction == "in"),
        total_out_usd=sum(r.value_usd for r in rows if r.direction == "out"),
        net_invested_usd=net_invested,
        suspect_count=sum(1 for r in rows if r.is_suspect),
    )
