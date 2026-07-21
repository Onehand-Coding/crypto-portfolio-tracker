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
    percent = (pl / basis_usd * 100.0) if basis_usd else 0.0
    return AccountingBasis(
        label=label, question=question, basis_usd=basis_usd,
        pl_usd=pl, pl_percent=percent,
    )


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
                "label": "NET INVESTED BASIS", "question": "did I make money?"}),
            fifo=empty.model_copy(update={
                "label": "FIFO BASIS", "question": "are my holdings underwater?"}),
            holdings=[],
            staleness=_staleness(None, None),
            environment=environment,
            has_data=False,
        )

    total_value = float(cached.get("total_value_usd") or 0.0)
    net_invested_basis = float(cached.get("total_invested_capital") or 0.0)
    fifo_basis = portfolio_fifo_cost_basis(ctx.db_manager.get_all_transactions())

    holdings = [
        Holding(**{k: v for k, v in row.items() if k in Holding.model_fields})
        for row in (cached.get("holdings_df") or [])
    ]

    return CockpitResponse(
        total_value_usd=total_value,
        net_invested=_basis(
            "NET INVESTED BASIS", "did I make money?", total_value, net_invested_basis),
        fifo=_basis(
            "FIFO BASIS", "are my holdings underwater?", total_value, fifo_basis),
        holdings=holdings,
        staleness=_staleness(cache.age_seconds(), cached.get("_cached_at")),
        environment=environment,
        has_data=True,
    )
