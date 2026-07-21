"""Capital flow: fiat -> USDT -> asset provenance.

A failed yfinance fiat lookup silently zeroes price_usd, which once hid
$50.41 of inflow and inverted reported P/L. Provenance makes that class of
failure visible in the UI rather than invisible in the total.
"""

from fastapi import APIRouter, Depends

from api.deps import get_read_context
from api.schemas.capital import CapitalFlowResponse, CapitalFlowRow

router = APIRouter(prefix="/api/capital", tags=["capital"])

INFLOW_SOURCES = {"Binance P2P Buy"}
OUTFLOW_SOURCES = {"Binance P2P Sell"}


def _provenance(price_usd: float) -> str:
    if not price_usd:
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
            price_usd = float(record.get("price_usd") or 0.0)
            quantity = float(record.get("quantity") or 0.0)
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
