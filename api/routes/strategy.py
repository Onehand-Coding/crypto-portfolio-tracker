"""Live-analysis screens: rebalancing, DCA, profit-taking, technical.

Each is a pair: POST runs the analysis against Binance, GET serves the last
cached result with its age. A GET never touches the network, so these pages
open instantly and say plainly when their figures are old.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.analysis_runner import get_analysis_runner
from api.cache import MetricsCache, analysis_cache_path, cache_path_for
from api.deps import get_read_context
from api.routes.common import num, opt, staleness_for
from api.schemas.screens import (
    DcaAllocation,
    DcaPreviewRequest,
    DcaPreviewResponse,
    DcaResponse,
    IndicatorRow,
    ProfitOpportunityOut,
    ProfitResponse,
    RebalanceResponse,
    RebalanceSuggestion,
    TechnicalResponse,
)

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


def _cache(ctx, kind: str) -> MetricsCache:
    return MetricsCache(analysis_cache_path(ctx.config_manager, kind))


def _norm(name: str) -> str:
    """Reduce a column name to letters and digits, lowercased.

    The core labels its rebalance columns for humans -- "Target %",
    "Drift (pts)", "Current Value (USD)" -- while other methods return
    snake_case. Normalising both sides means one alias matches either spelling
    instead of the alias list having to guess at punctuation.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _bare_symbol(symbol) -> str:
    """Strip the quote-currency suffix the trend analyzer carries.

    It reports yfinance tickers -- "BTC-USD" -- while every other screen keys
    on "BTC". Left unstripped, joins against holdings match nothing and the
    indicator columns silently render blank rather than erroring.
    """
    text = str(symbol).upper()
    for suffix in ("-USD", "-USDT", "USDT"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def _pick(row: dict, *names):
    """First present key among aliases, compared on normalised names.

    Falling back to None keeps a renamed column showing as unknown rather than
    as zero.
    """
    normalised = {_norm(str(k)): v for k, v in row.items()}
    for name in names:
        value = normalised.get(_norm(name))
        if value is not None:
            return value
    return None


@router.post("/{kind}/run")
async def run_analysis(kind: str) -> dict:
    runner = get_analysis_runner()
    try:
        started = runner.start(kind)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown analysis: {kind}")
    if not started:
        raise HTTPException(status_code=409, detail=f"{kind} analysis already running")
    return {"status": "started"}


@router.get("/rebalance", response_model=RebalanceResponse)
def rebalance(ctx=Depends(get_read_context)) -> RebalanceResponse:
    runner = get_analysis_runner()
    cache = _cache(ctx, "rebalance")
    cached = cache.read()

    suggestions = []
    for row in (cached or {}).get("suggestions", []) or []:
        if not isinstance(row, dict):
            continue
        suggestions.append(
            RebalanceSuggestion(
                # The first alias in each list is the name the core actually
                # emits today, read off a real run rather than guessed at.
                symbol=str(_pick(row, "Symbol", "asset") or "?"),
                action=(str(_pick(row, "Signal", "action", "suggestion"))
                        if _pick(row, "Signal", "action", "suggestion") else None),
                current_value_usd=opt(_pick(row, "Current Value (USD)",
                                            "current_value_usd", "value_usd")),
                current_allocation_pct=opt(_pick(row, "Current %",
                                                 "current_allocation_pct", "current_pct")),
                target_allocation_pct=opt(_pick(row, "Target %",
                                                "target_allocation_pct", "target_pct")),
                drift_pct=opt(_pick(row, "Drift (pts)", "drift_pct", "drift")),
                action_amount_usd=opt(_pick(row, "action_usd_value",
                                            "action_amount_usd", "amount_usd")),
                action_quantity=opt(_pick(row, "action_coin_quantity",
                                          "action_quantity", "quantity")),
                reason=(str(_pick(row, "Suggested Action Detail", "reason", "reasoning"))
                        if _pick(row, "Suggested Action Detail", "reason", "reasoning")
                        else None),
                raw={str(k): v for k, v in row.items()},
            )
        )

    return RebalanceResponse(
        has_data=cached is not None,
        is_running=runner.is_running("rebalance"),
        error=runner.last_error("rebalance"),
        staleness=staleness_for(cache),
        suggestions=suggestions,
    )


@router.get("/profit", response_model=ProfitResponse)
def profit(ctx=Depends(get_read_context)) -> ProfitResponse:
    runner = get_analysis_runner()
    cache = _cache(ctx, "profit")
    cached = cache.read()

    opportunities = [
        ProfitOpportunityOut(
            symbol=str(row.get("symbol") or "?"),
            unrealized_gain_usd=opt(row.get("unrealized_gain_usd")),
            unrealized_gain_pct=opt(row.get("unrealized_gain_pct")),
            opportunity_score=opt(row.get("opportunity_score")),
            rsi_score=opt(row.get("rsi_score")),
            pl_score=opt(row.get("pl_score")),
            resistance_score=opt(row.get("resistance_score")),
            market_context_score=opt(row.get("market_context_score")),
            current_price=opt(row.get("current_price")),
            support_level=opt(row.get("support_level")),
            resistance_level=opt(row.get("resistance_level")),
            reasons=[str(r) for r in (row.get("reasons") or [])],
        )
        for row in (cached or {}).get("opportunities", []) or []
        if isinstance(row, dict)
    ]

    return ProfitResponse(
        has_data=cached is not None,
        is_running=runner.is_running("profit"),
        error=runner.last_error("profit"),
        staleness=staleness_for(cache),
        opportunities=opportunities,
    )


@router.get("/dca", response_model=DcaResponse)
def dca(ctx=Depends(get_read_context)) -> DcaResponse:
    runner = get_analysis_runner()
    cache = _cache(ctx, "dca")
    cached = cache.read()
    available = (cached or {}).get("available") or {}
    portfolio = ctx.config_manager.config.get("portfolio", {}) or {}

    return DcaResponse(
        has_data=cached is not None,
        is_running=runner.is_running("dca"),
        error=runner.last_error("dca"),
        staleness=staleness_for(cache),
        available_usdt=opt(available.get("total") if isinstance(available, dict) else None),
        spot_usdt=opt(available.get("spot") if isinstance(available, dict) else None),
        earn_usdt=opt(available.get("earn") if isinstance(available, dict) else None),
        minimum_trade_usd=num(portfolio.get("minimum_trade_usd"), 5.0),
    )


@router.post("/dca/preview", response_model=DcaPreviewResponse)
def dca_preview(request: DcaPreviewRequest, ctx=Depends(get_read_context)) -> DcaPreviewResponse:
    """Allocation preview only. This never places an order.

    Runs offline against the cached holdings and the configured target weights,
    so the user can see where money would go before running anything live.
    """
    config = ctx.config_manager.config
    portfolio = config.get("portfolio", {}) or {}
    minimum = num(portfolio.get("minimum_trade_usd"), 5.0)
    target = config.get("target_allocation", {}) or {}

    if request.amount_usd <= 0:
        return DcaPreviewResponse(strategy=request.strategy, amount_usd=request.amount_usd,
                                  valid=False, message="Amount must be greater than zero.")
    if request.amount_usd < minimum:
        return DcaPreviewResponse(
            strategy=request.strategy, amount_usd=request.amount_usd, valid=False,
            message=f"Amount is below the ${minimum:,.2f} minimum trade size.")
    if not target:
        return DcaPreviewResponse(strategy=request.strategy, amount_usd=request.amount_usd,
                                  valid=False, message="No target allocation configured.")

    metrics = MetricsCache(cache_path_for(ctx.config_manager)).read() or {}
    holdings = {str(r.get("symbol")).upper(): r
                for r in (metrics.get("holdings_df") or []) if isinstance(r, dict)}
    core_value = sum(num(holdings.get(s, {}).get("value_usd")) for s in target)

    allocations: list[DcaAllocation] = []
    if request.strategy == "proportional":
        # Split in proportion to what is already held, preserving current shape.
        basis = {s: num(holdings.get(s, {}).get("value_usd")) for s in target}
        total = sum(basis.values())
        if total <= 0:
            return DcaPreviewResponse(
                strategy=request.strategy, amount_usd=request.amount_usd, valid=False,
                message="Proportional DCA needs existing core holdings to scale from. "
                        "Use target-weight instead.")
        for symbol in target:
            share = basis[symbol] / total
            allocations.append(_allocation(symbol, request.amount_usd * share,
                                           holdings, target, core_value, request.amount_usd))
    else:
        # Target weight: buy the assets furthest below their target first.
        after_total = core_value + request.amount_usd
        deficits = {}
        for symbol, weight in target.items():
            desired = after_total * float(weight)
            deficits[symbol] = max(0.0, desired - num(holdings.get(symbol, {}).get("value_usd")))
        total_deficit = sum(deficits.values())
        if total_deficit <= 0:
            return DcaPreviewResponse(
                strategy=request.strategy, amount_usd=request.amount_usd, valid=False,
                message="Every core asset is already at or above its target weight.")
        for symbol, deficit in deficits.items():
            if deficit <= 0:
                continue
            allocations.append(_allocation(symbol,
                                           request.amount_usd * (deficit / total_deficit),
                                           holdings, target, core_value, request.amount_usd))

    allocations.sort(key=lambda a: a.amount_usd, reverse=True)
    return DcaPreviewResponse(strategy=request.strategy, amount_usd=request.amount_usd,
                              valid=True, allocations=allocations)


def _allocation(symbol, amount, holdings, target, core_value, total_amount) -> DcaAllocation:
    price = opt(holdings.get(symbol, {}).get("current_price"))
    value = num(holdings.get(symbol, {}).get("value_usd"))
    after = core_value + total_amount
    return DcaAllocation(
        symbol=symbol,
        amount_usd=round(amount, 2),
        # None, not zero, when the price is unknown: a quantity of 0 would read
        # as "buys nothing".
        quantity=(amount / price if price else None),
        current_allocation_pct=(value / core_value * 100.0) if core_value else None,
        target_allocation_pct=float(target[symbol]) * 100.0,
    ) if after else DcaAllocation(symbol=symbol, amount_usd=round(amount, 2))


@router.get("/technical", response_model=TechnicalResponse)
def technical(ctx=Depends(get_read_context)) -> TechnicalResponse:
    runner = get_analysis_runner()
    cache = _cache(ctx, "technical")
    cached = cache.read()
    reports = (cached or {}).get("reports") or {}

    timeframes: dict[str, list[IndicatorRow]] = {}
    bear = None
    for timeframe, report in reports.items():
        if not isinstance(report, dict):
            continue
        rows = []
        analyses = report.get("analyses") or report.get("coin_analyses") or {}
        if isinstance(analyses, dict):
            items = analyses.items()
        else:
            items = [(str(a.get("symbol")), a) for a in analyses if isinstance(a, dict)]
        for symbol, analysis in items:
            if not isinstance(analysis, dict):
                continue
            rows.append(
                IndicatorRow(
                    symbol=_bare_symbol(symbol),
                    price=opt(analysis.get("current_price") or analysis.get("price")),
                    rsi=opt(analysis.get("rsi")),
                    sma_short=opt(analysis.get("sma_short")),
                    sma_long=opt(analysis.get("sma_long")),
                    # The core names these *_level; reading the bare names
                    # returned None for every row while the response stayed 200.
                    support=opt(analysis.get("support_level") or analysis.get("support")),
                    resistance=opt(
                        analysis.get("resistance_level") or analysis.get("resistance")
                    ),
                    conditions=[str(c) for c in (analysis.get("active_conditions") or [])],
                )
            )
        timeframes[str(timeframe)] = rows

    long_term = reports.get("long_term")
    if isinstance(long_term, dict):
        benchmark = long_term.get("benchmark_analysis") or {}
        conditions = benchmark.get("active_conditions") or []
        bear = any("SMA200" in str(c) and "BELOW" in str(c).upper() for c in conditions)

    return TechnicalResponse(
        has_data=cached is not None,
        is_running=runner.is_running("technical"),
        error=runner.last_error("technical"),
        staleness=staleness_for(cache),
        timeframes=timeframes,
        bear_market=bear,
    )


@router.get("/backtest")
def backtest(ctx=Depends(get_read_context)) -> dict:
    """Backtest result. Shape is whatever the core's backtester returned.

    Left untyped on purpose: RebalancingBacktester.run's return value is not a
    documented contract, and modelling it strictly would drop fields silently
    the moment it changes.
    """
    runner = get_analysis_runner()
    cache = _cache(ctx, "backtest")
    cached = cache.read() or {}
    staleness = staleness_for(cache)
    return {
        "has_data": bool(cached),
        "is_running": runner.is_running("backtest"),
        "error": runner.last_error("backtest"),
        "staleness": staleness.model_dump(),
        "result": cached.get("result"),
        "report": cached.get("report"),
    }
