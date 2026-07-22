"""Order execution -- honours the same two config switches the core does.

Every other route reads or writes local state. These place orders on the
exchange, so two independent switches decide what actually happens, exactly as
in the CLI/Streamlit path:

  * portfolio.testnet_mode      -- which exchange endpoint (testnet vs mainnet)
  * portfolio.live_trading_enabled -- arm real orders; off means a dry run

With live trading disabled every route still runs, but the core simulates
rather than sending an order -- so the whole UI is usable on mainnet in
simulation before anything is armed. On top of that, each route still requires
an explicit confirm=true from the caller.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.cache import MetricsCache, analysis_cache_path
from api.deps import get_config_manager, get_tracker
from api.schemas.screens import (
    DcaExecuteRequest,
    ExecuteSelectionRequest,
    RedeemRequest,
    TradeExecuteRequest,
    TradeExecuteResponse,
    TransferRequest,
)

router = APIRouter(prefix="/api/execute", tags=["execute"])


def _cm():
    """The config manager -- the source of the testnet and live-trading switches."""
    return get_config_manager()


def _result(result, testnet: bool) -> TradeExecuteResponse:
    return TradeExecuteResponse(
        success=bool(getattr(result, "success", False)),
        testnet=testnet,
        messages=list(getattr(result, "messages", []) or []),
        errors=list(getattr(result, "errors", []) or []),
    )


@router.get("/status")
def execution_status() -> dict:
    """The current execution posture, so the UI can label its own gate honestly."""
    cm = _cm()
    return {"testnet": bool(cm.is_testnet_mode), "is_live": bool(cm.is_live)}


@router.post("/trade", response_model=TradeExecuteResponse)
async def execute_trade(payload: TradeExecuteRequest) -> TradeExecuteResponse:
    """Place a market order. Real only when live trading is enabled; else a dry run."""
    cm = _cm()
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
            trade_type, symbol, ticker, float(payload.amount), payload.is_quote_qty,
            cm.is_live,
        )
    except Exception as exc:  # a failed order must report, not 500 the page
        return TradeExecuteResponse(
            success=False, testnet=bool(cm.is_testnet_mode), errors=[str(exc)]
        )

    return _result(result, bool(cm.is_testnet_mode))


@router.post("/rebalance", response_model=TradeExecuteResponse)
async def execute_rebalance(payload: ExecuteSelectionRequest) -> TradeExecuteResponse:
    """Execute the cached rebalance suggestions (or a chosen subset).

    The core reads live_trading_enabled itself, so a dry run needs no flag here.
    """
    cm = _cm()
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Execution requires explicit confirmation.")

    cached = MetricsCache(analysis_cache_path(cm, "rebalance")).read() or {}
    records = cached.get("suggestions") or []
    if not records:
        raise HTTPException(
            status_code=409,
            detail="No rebalance analysis to execute. Run the analysis first.",
        )
    if payload.symbols:
        wanted = {s.upper() for s in payload.symbols}
        records = [r for r in records if str(r.get("Symbol", "")).upper() in wanted]
    if not records:
        raise HTTPException(status_code=409, detail="None of the chosen symbols are in the analysis.")

    from crypto_portfolio_tracker.models import ExecutionMode

    tracker = get_tracker()
    try:
        # earn_balances is empty here, mirroring the Streamlit path; AUTO
        # + no callback runs every BUY/SELL without a per-trade prompt.
        result = await tracker.execute_rebalancing_trades(
            pd.DataFrame(records), {}, confirmation_callback=None,
            execution_mode=ExecutionMode.AUTO,
        )
    except Exception as exc:
        return TradeExecuteResponse(
            success=False, testnet=bool(cm.is_testnet_mode), errors=[str(exc)]
        )
    return _result(result, bool(cm.is_testnet_mode))


@router.post("/dca", response_model=TradeExecuteResponse)
async def execute_dca(payload: DcaExecuteRequest) -> TradeExecuteResponse:
    """Execute a DCA plan (from the preview). Real only when live trading is enabled."""
    cm = _cm()
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Execution requires explicit confirmation.")
    if not payload.trades:
        raise HTTPException(status_code=409, detail="No DCA trades to execute.")

    trades = [
        {"asset": str(t.get("asset", "")).upper(),
         "amount": float(t.get("amount", 0.0)),
         "method": payload.strategy}
        for t in payload.trades
        if str(t.get("asset", "")).strip() and float(t.get("amount", 0.0)) > 0
    ]
    if not trades:
        raise HTTPException(status_code=422, detail="No valid DCA trades in the request.")

    tracker = get_tracker()
    try:
        result = await tracker.execute_dca_trades(trades, payload.strategy, cm.is_live)
    except Exception as exc:
        return TradeExecuteResponse(
            success=False, testnet=bool(cm.is_testnet_mode), errors=[str(exc)]
        )
    return _result(result, bool(cm.is_testnet_mode))


@router.post("/profit", response_model=TradeExecuteResponse)
async def execute_profit(payload: ExecuteSelectionRequest) -> TradeExecuteResponse:
    """Execute profit-taking on the cached opportunities (or a subset).

    The per-trade sell size is recomputed exactly as the core does -- a share
    of the unrealized gain, capped by the configured max -- from figures the
    cached analysis already holds, so no live re-fetch is needed to size it.
    Real only when live trading is enabled; otherwise the core simulates.
    """
    cm = _cm()
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Execution requires explicit confirmation.")

    cached = MetricsCache(analysis_cache_path(cm, "profit")).read() or {}
    opportunities = cached.get("opportunities") or []
    if not opportunities:
        raise HTTPException(
            status_code=409,
            detail="No profit-taking analysis to execute. Run the analysis first.",
        )
    if payload.symbols:
        wanted = {s.upper() for s in payload.symbols}
        opportunities = [o for o in opportunities if str(o.get("symbol", "")).upper() in wanted]

    pt = cm.config.get("profit_taking", {}) or {}
    take_pct = max(1.0, min(float(pt.get("default_take_percentage", 30)),
                            float(pt.get("max_gain_take_pct", 50))))

    trades = []
    for o in opportunities:
        gain = o.get("unrealized_gain_usd")
        price = o.get("current_price")
        if gain is None or price is None or price <= 0:
            continue
        usd = gain * take_pct / 100.0
        if usd <= 0:
            continue
        trades.append({
            "symbol": o["symbol"], "usd_amount": usd,
            "coin_quantity": usd / price, "take_percentage": take_pct,
        })
    if not trades:
        raise HTTPException(status_code=409, detail="No sellable profit-taking trades.")

    tracker = get_tracker()
    try:
        result = await tracker.execute_profit_taking_trades(trades, cm.is_live)
    except Exception as exc:
        return TradeExecuteResponse(
            success=False, testnet=bool(cm.is_testnet_mode), errors=[str(exc)]
        )
    return _result(result, bool(cm.is_testnet_mode))


# The six directional transfer methods, keyed by (from, to).
_TRANSFER_METHODS = {
    ("FUNDING", "SPOT"): "transfer_funding_to_spot",
    ("SPOT", "FUNDING"): "transfer_spot_to_funding",
    ("SPOT", "FUTURES"): "transfer_spot_to_futures",
    ("FUTURES", "SPOT"): "transfer_futures_to_spot",
    ("FUNDING", "FUTURES"): "transfer_funding_to_futures",
    ("FUTURES", "FUNDING"): "transfer_futures_to_funding",
}

TRANSFER_ROUTES = [f"{a} -> {b}" for (a, b) in _TRANSFER_METHODS]


@router.get("/transfer/routes")
def transfer_routes() -> dict:
    """The supported wallet-to-wallet directions, for the UI's selector."""
    return {"routes": [{"from": a, "to": b} for (a, b) in _TRANSFER_METHODS]}


@router.post("/transfer", response_model=TradeExecuteResponse)
async def execute_transfer(payload: TransferRequest) -> TradeExecuteResponse:
    """Move an asset between Spot / Funding / Futures wallets."""
    cm = _cm()
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Transfer requires explicit confirmation.")
    if not payload.amount > 0:
        raise HTTPException(status_code=422, detail="Amount must be positive.")

    key = (payload.from_wallet.strip().upper(), payload.to_wallet.strip().upper())
    method = _TRANSFER_METHODS.get(key)
    if method is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported transfer {key[0]} -> {key[1]}. One of: {TRANSFER_ROUTES}.",
        )

    tracker = get_tracker()
    try:
        result = await getattr(tracker, method)(
            float(payload.amount), payload.asset.strip().upper(), cm.is_live
        )
    except Exception as exc:
        return TradeExecuteResponse(
            success=False, testnet=bool(cm.is_testnet_mode), errors=[str(exc)]
        )
    return _result(result, bool(cm.is_testnet_mode))


@router.post("/redeem", response_model=TradeExecuteResponse)
async def execute_redeem(payload: RedeemRequest) -> TradeExecuteResponse:
    """Redeem an asset from Binance Simple Earn back to Spot."""
    cm = _cm()
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Redeem requires explicit confirmation.")
    if not payload.amount > 0:
        raise HTTPException(status_code=422, detail="Amount must be positive.")

    tracker = get_tracker()
    try:
        # redeem_from_earn is synchronous.
        result = tracker.redeem_from_earn(
            payload.asset.strip().upper(), float(payload.amount), cm.is_live
        )
    except Exception as exc:
        return TradeExecuteResponse(
            success=False, testnet=bool(cm.is_testnet_mode), errors=[str(exc)]
        )
    return _result(result, bool(cm.is_testnet_mode))
