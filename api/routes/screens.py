"""Overview, Asset Detail, Reports and System Health.

All offline: cached metrics plus SQLite. None of these construct a tracker.
"""

import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends

from api.cache import MetricsCache, cache_path_for
from api.deps import get_read_context
from api.routes.common import num, opt, staleness_for
from api.schemas.screens import (
    AssetDetailResponse,
    AssetTransaction,
    BackupInfo,
    ExportFile,
    OverviewResponse,
    RealizedGainRow,
    RealizedGainSummary,
    RealizedResponse,
    ReportsResponse,
    SnapshotPoint,
    SystemHealthResponse,
)
from crypto_portfolio_tracker.utils import calculate_fifo_realized_gains

router = APIRouter(prefix="/api", tags=["screens"])


def _str_or_none(value):
    if value is None:
        return None
    text = str(value)
    # pandas renders missing values as these literals; passing them through
    # would print "NaT" where a blank belongs.
    return None if text in ("nan", "NaT", "None", "") else text


@router.get("/overview", response_model=OverviewResponse)
def overview(ctx=Depends(get_read_context)) -> OverviewResponse:
    cache = MetricsCache(cache_path_for(ctx.config_manager))
    snapshots = ctx.db_manager.get_all_snapshots()

    points: list[SnapshotPoint] = []
    if isinstance(snapshots, pd.DataFrame) and not snapshots.empty:
        for row in snapshots.to_dict(orient="records"):
            points.append(
                SnapshotPoint(
                    timestamp=_str_or_none(row.get("timestamp")),
                    total_value_usd=opt(row.get("total_value_usd")),
                    total_cost_basis_usd=opt(row.get("total_cost_basis_usd")),
                    unrealized_pl_usd=opt(row.get("unrealized_pl_usd")),
                    unrealized_pl_percent=opt(row.get("unrealized_pl_percent")),
                )
            )

    return OverviewResponse(
        has_data=len(points) > 0,
        points=points,
        staleness=staleness_for(cache),
    )


@router.get("/assets/{symbol}", response_model=AssetDetailResponse)
def asset_detail(symbol: str, ctx=Depends(get_read_context)) -> AssetDetailResponse:
    symbol = symbol.upper()
    cache = MetricsCache(cache_path_for(ctx.config_manager))
    cached = cache.read() or {}

    holding = None
    for row in cached.get("holdings_df") or []:
        if isinstance(row, dict) and str(row.get("symbol", "")).upper() == symbol:
            holding = row
            break

    transactions: list[AssetTransaction] = []
    all_tx = ctx.db_manager.get_all_transactions()
    if isinstance(all_tx, pd.DataFrame) and not all_tx.empty and "symbol" in all_tx:
        rows = all_tx[all_tx["symbol"].astype(str).str.upper() == symbol]
        for row in rows.to_dict(orient="records"):
            quantity = opt(row.get("quantity"))
            price = opt(row.get("price_usd"))
            transactions.append(
                AssetTransaction(
                    timestamp=_str_or_none(row.get("timestamp")),
                    type=str(row.get("type") or "?"),
                    quantity=quantity,
                    price_usd=price,
                    # Unknown times unknown is unknown, not zero.
                    value_usd=(quantity * price
                               if quantity is not None and price is not None else None),
                    source=_str_or_none(row.get("source")),
                    notes=_str_or_none(row.get("notes")),
                )
            )
        transactions.reverse()

    target = ctx.config_manager.config.get("target_allocation", {}) or {}
    price = opt(holding.get("current_price")) if holding else None
    quantity = opt(holding.get("total_quantity")) if holding else None
    unpriced = bool(holding) and bool(quantity) and not price

    return AssetDetailResponse(
        symbol=symbol,
        found=holding is not None or len(transactions) > 0,
        total_quantity=quantity,
        current_price=None if unpriced else price,
        value_usd=None if unpriced else (opt(holding.get("value_usd")) if holding else None),
        average_cost_basis=opt(holding.get("average_cost_basis")) if holding else None,
        cost_basis_total=opt(holding.get("cost_basis_total")) if holding else None,
        unrealized_pl_usd=(None if unpriced
                           else (opt(holding.get("unrealized_pl_usd")) if holding else None)),
        unrealized_pl_percent=(None if unpriced
                               else (opt(holding.get("unrealized_pl_percent"))
                                     if holding else None)),
        price_unavailable=unpriced,
        is_core=symbol in target,
        target_allocation_pct=(float(target[symbol]) * 100.0 if symbol in target else None),
        transactions=transactions,
        staleness=staleness_for(cache),
    )


@router.get("/reports", response_model=ReportsResponse)
def reports(ctx=Depends(get_read_context)) -> ReportsResponse:
    export_dir = Path(
        (ctx.config_manager.config.get("paths", {}) or {}).get("export_dir", "data/exports")
    )
    files: list[ExportFile] = []
    if export_dir.is_dir():
        for path in sorted(export_dir.iterdir(), key=lambda p: p.name):
            if not path.is_file():
                continue
            stat = path.stat()
            files.append(
                ExportFile(
                    name=path.name,
                    path=str(path),
                    size_bytes=stat.st_size,
                    modified=datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
            )
    files.sort(key=lambda f: f.modified, reverse=True)
    return ReportsResponse(files=files, export_dir=str(export_dir))


@router.get("/realized", response_model=RealizedResponse)
def realized(ctx=Depends(get_read_context)) -> RealizedResponse:
    """Realized gains by FIFO -- the taxable half of the accounting.

    This wraps the same core function the Streamlit tax report uses
    (calculate_fifo_realized_gains) so the two agree by construction. Reads
    only: it computes over the transaction table and never touches the wire.
    """
    cache = MetricsCache(cache_path_for(ctx.config_manager))
    transactions = ctx.db_manager.get_all_transactions()

    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        return RealizedResponse(has_data=False, staleness=staleness_for(cache))

    tax_df = calculate_fifo_realized_gains(transactions)
    if tax_df.empty:
        # Transactions exist but none of them are taxable events yet. That is a
        # real, distinct state from "no data" -- has_data stays True.
        return RealizedResponse(has_data=True, staleness=staleness_for(cache))

    rows: list[RealizedGainRow] = []
    for record in tax_df.to_dict(orient="records"):
        date = pd.to_datetime(record.get("date"), errors="coerce")
        rows.append(
            RealizedGainRow(
                date=_str_or_none(date),
                year=(int(date.year) if pd.notna(date) else None),
                symbol=str(record.get("symbol")),
                quantity=opt(record.get("quantity")),
                proceeds_usd=opt(record.get("proceeds_usd")),
                cost_basis_usd=opt(record.get("cost_basis_usd")),
                gain_usd=opt(record.get("gain_usd")),
            )
        )
    # Newest first, matching every other dated table in the app.
    rows.sort(key=lambda r: r.date or "", reverse=True)

    summary = (
        tax_df.groupby("symbol")
        .agg(
            total_gain_usd=("gain_usd", "sum"),
            total_proceeds_usd=("proceeds_usd", "sum"),
            total_cost_basis_usd=("cost_basis_usd", "sum"),
        )
        .reset_index()
    )
    by_asset = [
        RealizedGainSummary(
            symbol=str(record.get("symbol")),
            total_gain_usd=opt(record.get("total_gain_usd")),
            total_proceeds_usd=opt(record.get("total_proceeds_usd")),
            total_cost_basis_usd=opt(record.get("total_cost_basis_usd")),
        )
        for record in summary.to_dict(orient="records")
    ]
    by_asset.sort(key=lambda s: s.total_gain_usd or 0.0, reverse=True)

    return RealizedResponse(
        has_data=True,
        rows=rows,
        by_asset=by_asset,
        total_gain_usd=opt(tax_df["gain_usd"].sum()),
        total_proceeds_usd=opt(tax_df["proceeds_usd"].sum()),
        total_cost_basis_usd=opt(tax_df["cost_basis_usd"].sum()),
        staleness=staleness_for(cache),
    )


@router.get("/system/health", response_model=SystemHealthResponse)
def system_health(ctx=Depends(get_read_context)) -> SystemHealthResponse:
    config = ctx.config_manager.config
    db_path = Path(str(ctx.config_manager.get_database_path()))
    portfolio = config.get("portfolio", {}) or {}

    transactions = ctx.db_manager.get_all_transactions()
    tx_count = int(len(transactions)) if isinstance(transactions, pd.DataFrame) else 0
    asset_count = 0
    if isinstance(transactions, pd.DataFrame) and "symbol" in transactions:
        asset_count = int(transactions["symbol"].nunique())
    snapshots = ctx.db_manager.get_all_snapshots()
    snapshot_count = int(len(snapshots)) if isinstance(snapshots, pd.DataFrame) else 0

    backups: list[BackupInfo] = []
    try:
        for path in ctx.db_manager.list_backups() or []:
            stat = Path(path).stat()
            backups.append(
                BackupInfo(
                    name=Path(path).name,
                    size_bytes=stat.st_size,
                    modified=datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
            )
    except Exception:  # backup listing is diagnostic; never fail the page for it
        backups = []

    is_testnet = bool(ctx.config_manager.is_testnet_mode)
    keys = ("TESTNET_API_KEY", "TESTNET_API_SECRET") if is_testnet \
        else ("MAIN_API_KEY", "MAIN_API_SECRET")
    import os
    # Presence only. The values are never read into a response.
    binance_configured = all(bool(os.environ.get(k)) for k in keys)

    return SystemHealthResponse(
        environment_label="TESTNET" if is_testnet else "LIVE",
        is_testnet=is_testnet,
        database_path=str(db_path),
        database_exists=db_path.is_file(),
        database_size_bytes=(db_path.stat().st_size if db_path.is_file() else 0),
        transaction_count=tx_count,
        asset_count=asset_count,
        snapshot_count=snapshot_count,
        live_trading_enabled=bool(portfolio.get("live_trading_enabled", False)),
        minimum_trade_usd=num(portfolio.get("minimum_trade_usd"), 5.0),
        target_allocation={
            str(k): float(v) for k, v in (config.get("target_allocation", {}) or {}).items()
        },
        backups=backups[:10],
        metrics_cache_age_seconds=MetricsCache(cache_path_for(ctx.config_manager)).age_seconds(),
        binance_configured=binance_configured,
    )
