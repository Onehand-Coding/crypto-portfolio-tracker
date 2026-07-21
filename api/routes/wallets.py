"""Wallet breakdown. Served from the cached metrics -- no network."""

from fastapi import APIRouter, Depends

from api.cache import MetricsCache, cache_path_for
from api.deps import get_read_context
from api.routes.common import staleness_for
from api.schemas.wallets import WalletBalance, WalletsResponse

router = APIRouter(prefix="/api/wallets", tags=["wallets"])


def _balances(rows) -> list[WalletBalance]:
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        out.append(
            WalletBalance(
                symbol=str(row.get("symbol") or row.get("asset") or "?"),
                quantity=_num(row.get("quantity") or row.get("total_quantity")),
                value_usd=_opt(row.get("value_usd")),
            )
        )
    return out


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _opt(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@router.get("", response_model=WalletsResponse)
def wallets(ctx=Depends(get_read_context)) -> WalletsResponse:
    cache = MetricsCache(cache_path_for(ctx.config_manager))
    cached = cache.read()

    if cached is None:
        return WalletsResponse(
            has_data=False,
            spot_earn_value_usd=0.0,
            futures_value_usd=0.0,
            funding_value_usd=0.0,
            total_value_usd=0.0,
            spot_holdings=[],
            futures_balances=[],
            funding_balances=[],
            staleness=staleness_for(cache),
        )

    return WalletsResponse(
        has_data=True,
        spot_earn_value_usd=_num(cached.get("spot_earn_value_usd")),
        futures_value_usd=_num(cached.get("futures_value_usd")),
        funding_value_usd=_num(cached.get("funding_value_usd")),
        total_value_usd=_num(cached.get("total_value_usd")),
        spot_holdings=_balances(cached.get("holdings_df")),
        futures_balances=_balances(cached.get("futures_balances")),
        funding_balances=_balances(cached.get("funding_balances")),
        staleness=staleness_for(cache),
    )
