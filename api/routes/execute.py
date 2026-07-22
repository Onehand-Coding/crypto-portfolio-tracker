"""Order execution -- deliberately testnet-only.

Every other route reads or writes local state. These place real orders on the
exchange, so they are hard-gated: unless the process is in testnet mode the
endpoints refuse (403). Flipping to live is a deliberate, separate act -- this
layer never places a live-money order on its own.
"""

from fastapi import APIRouter, HTTPException

from api.deps import get_config_manager, get_tracker
from api.schemas.screens import TradeExecuteRequest, TradeExecuteResponse

router = APIRouter(prefix="/api/execute", tags=["execute"])


def _require_testnet():
    """Refuse to execute anything unless we are on testnet."""
    cm = get_config_manager()
    if not cm.is_testnet_mode:
        raise HTTPException(
            status_code=403,
            detail="Execution is disabled outside testnet mode. Enable testnet to trade.",
        )
    return cm


@router.get("/status")
def execution_status() -> dict:
    """Whether execution is currently possible. Lets the UI gate its own buttons."""
    return {"testnet": bool(get_config_manager().is_testnet_mode)}


@router.post("/trade", response_model=TradeExecuteResponse)
async def execute_trade(payload: TradeExecuteRequest) -> TradeExecuteResponse:
    """Place a market order on the (testnet) exchange.

    Two gates before anything is sent: the process must be in testnet mode, and
    the request must carry confirm=true.
    """
    _require_testnet()
    if not payload.confirm:
        raise HTTPException(
            status_code=400, detail="Execution requires explicit confirmation."
        )

    trade_type = payload.trade_type.strip().upper()
    if trade_type not in ("BUY", "SELL"):
        raise HTTPException(status_code=422, detail="trade_type must be BUY or SELL.")
    symbol = payload.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="A symbol is required.")
    if not payload.amount > 0:
        raise HTTPException(status_code=422, detail="Amount must be positive.")

    ticker = f"{symbol}USDT"
    try:
        tracker = get_tracker()
        result = await tracker.execute_manual_trade_core(
            trade_type, symbol, ticker, float(payload.amount), payload.is_quote_qty, True
        )
    except Exception as exc:  # a failed order must report, not 500 the page
        return TradeExecuteResponse(success=False, testnet=True, errors=[str(exc)])

    return TradeExecuteResponse(
        success=bool(result.success),
        testnet=True,
        messages=list(result.messages),
        errors=list(result.errors),
    )
