"""Portfolio read endpoints. These never contact Binance."""

import datetime
from typing import Optional

from fastapi import APIRouter, Depends

from api.accounting import portfolio_fifo_cost_basis
from api.cache import MetricsCache, cache_path_for
from api.deps import get_read_context
from api.schemas.portfolio import AccountingBasis, CockpitResponse, Holding
from api.schemas.system import Environment, Staleness

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

STALE_AFTER_SECONDS = 3600.0


def _basis(label: str, question: str, value: float, basis_usd: float) -> AccountingBasis:
    pl = value - basis_usd
    # A zero basis makes the percentage undefined, not zero. Reporting 0.0 would
    # render as "unchanged" for a portfolio built entirely from deposits or
    # rewards, which is a lie in the direction that costs money.
    percent = (pl / basis_usd * 100.0) if basis_usd else None
    return AccountingBasis(
        label=label, question=question, basis_usd=basis_usd,
        pl_usd=pl, pl_percent=percent,
    )


def _holding(row: dict) -> Holding:
    """Build a Holding, distinguishing an unknown price from a zero one.

    portfolio_analyzer pre-seeds its price map with 0.0 and only overwrites on a
    successful lookup, so a failed fetch is indistinguishable from a real price
    of zero by the time it reaches the cache. Passed through, the position
    renders as "$0.00", the holdings table folds it into the dust aggregate, and
    a material holding disappears while the total silently understates. Reporting
    it as unknown keeps the row visible and the loss of information explicit.
    """
    holding = Holding(**{k: v for k, v in row.items() if k in Holding.model_fields})
    # `not current_price` covers both 0.0 and None (jsonable normalises NaN to
    # None). Guarded on quantity so a genuinely empty position is not flagged.
    if holding.total_quantity > 0 and not holding.current_price:
        return holding.model_copy(update={
            "price_unavailable": True,
            "current_price": None,
            # Every figure below is derived from the missing price. Leaving them
            # at their computed values would report a fabricated total loss.
            "value_usd": None,
            "unrealized_pl_usd": None,
            "unrealized_pl_percent": None,
        })
    return holding


def _staleness(age: Optional[float], cached_at: Optional[float]) -> Staleness:
    return Staleness(
        cached_at=(datetime.datetime.fromtimestamp(cached_at).isoformat()
                   if cached_at else None),
        age_seconds=age,
        is_stale=(age is None or age > STALE_AFTER_SECONDS),
    )


def _environment(config_manager) -> Environment:
    is_testnet = bool(config_manager.is_testnet_mode)
    return Environment(
        is_testnet=is_testnet,
        database_path=str(config_manager.get_database_path()),
        label="TESTNET" if is_testnet else "LIVE",
    )


@router.get("/cockpit", response_model=CockpitResponse)
def cockpit(ctx=Depends(get_read_context)) -> CockpitResponse:
    cache = MetricsCache(cache_path_for(ctx.config_manager))
    cached = cache.read()
    environment = _environment(ctx.config_manager)

    if cached is None:
        empty = _basis("", "", 0.0, 0.0)
        return CockpitResponse(
            total_value_usd=0.0,
            net_invested=empty.model_copy(update={
                "label": "Cash profit", "question": "did I make money?"}),
            fifo=empty.model_copy(update={
                "label": "Holdings profit (FIFO)", "question": "are my holdings underwater?"}),
            holdings=[],
            staleness=_staleness(None, None),
            environment=environment,
            has_data=False,
        )

    total_value = float(cached.get("total_value_usd") or 0.0)
    net_invested_basis = float(cached.get("total_invested_capital") or 0.0)
    fifo_basis = portfolio_fifo_cost_basis(ctx.db_manager.get_holdings())

    holdings = [_holding(row) for row in (cached.get("holdings_df") or [])]

    return CockpitResponse(
        total_value_usd=total_value,
        net_invested=_basis(
            "Cash profit", "did I make money?", total_value, net_invested_basis),
        fifo=_basis(
            "Holdings profit (FIFO)", "are my holdings underwater?", total_value, fifo_basis),
        holdings=holdings,
        staleness=_staleness(cache.age_seconds(), cached.get("_cached_at")),
        environment=environment,
        has_data=True,
        unpriced_count=sum(1 for h in holdings if h.price_unavailable),
    )
