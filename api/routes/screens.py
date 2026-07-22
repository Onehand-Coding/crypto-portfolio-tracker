"""Overview, Asset Detail, Reports and System Health.

All offline: cached metrics plus SQLite. None of these construct a tracker.
"""

import datetime
import io
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.cache import MetricsCache, cache_path_for
from api.deps import get_read_context
from api.routes.common import num, opt, staleness_for
from api.schemas.screens import (
    AssetDetailResponse,
    AssetTransaction,
    BackupCreateResponse,
    BackupInfo,
    CleanupRequest,
    CleanupResponse,
    CleanupStatsResponse,
    ExportFile,
    GenerateExportRequest,
    GenerateExportResponse,
    ImportResponse,
    OverviewResponse,
    ProfitTakingSettings,
    RealizedGainRow,
    RealizedGainSummary,
    RealizedResponse,
    ReportsResponse,
    RestoreRequest,
    RestoreResponse,
    SettingsResponse,
    SettingsUpdate,
    SnapshotDeleteRequest,
    SnapshotDeleteResponse,
    SnapshotPoint,
    SnapshotRow,
    SnapshotsResponse,
    SystemHealthResponse,
    TargetAllocationRequest,
    TargetAllocationResponse,
    TransactionRow,
    TransactionsResponse,
    TrendAnalyzerSettings,
)
from crypto_portfolio_tracker.utils import calculate_fifo_realized_gains

router = APIRouter(prefix="/api", tags=["screens"])


def _save_config_preserving_secrets(cm) -> None:
    """Persist config, then restore the secrets save_config() strips.

    save_config() deletes main_api_keys and the coingecko key from the *live*
    config dict before writing (so they never reach disk -- correct). But that
    dict is the process-wide singleton the tracker also holds, so the keys are
    put straight back afterwards, leaving the file clean and the running
    process intact.
    """
    try:
        cm.save_config()
    finally:
        cm.config["main_api_keys"] = getattr(cm, "main_api_keys", None)
        cm.config.setdefault("apis", {}).setdefault("coingecko", {})[
            "api_key"
        ] = os.getenv("COINGECKO_API_KEY")


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


def _export_dir(ctx) -> Path:
    return Path(
        (ctx.config_manager.config.get("paths", {}) or {}).get("export_dir", "data/exports")
    )


@router.get("/reports", response_model=ReportsResponse)
def reports(ctx=Depends(get_read_context)) -> ReportsResponse:
    export_dir = _export_dir(ctx)
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


@router.post("/reports/generate", response_model=GenerateExportResponse)
def generate_export(
    payload: GenerateExportRequest, ctx=Depends(get_read_context)
) -> GenerateExportResponse:
    """Write a CSV/Excel export of transactions or holdings. Offline: it reads
    SQLite and writes to the export dir, no tracker and no network."""
    data_type = payload.data_type.strip().lower()
    fmt = payload.format.strip().lower()
    if data_type not in ("transactions", "holdings"):
        raise HTTPException(status_code=422, detail="data_type must be transactions or holdings.")
    if fmt not in ("csv", "excel"):
        raise HTTPException(status_code=422, detail="format must be csv or excel.")

    if data_type == "transactions":
        frame = ctx.db_manager.get_all_transactions()
    else:
        frame = ctx.db_manager.get_holdings()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame()

    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "csv" if fmt == "csv" else "xlsx"
    name = f"{data_type}_{stamp}.{ext}"
    path = export_dir / name
    try:
        if fmt == "csv":
            frame.to_csv(path, index=False)
        else:
            # Excel cannot store timezone-aware datetimes, so any such column is
            # made naive first rather than letting the write 500.
            safe = frame.copy()
            for col in safe.columns:
                if isinstance(safe[col].dtype, pd.DatetimeTZDtype):
                    safe[col] = safe[col].dt.tz_localize(None)
            safe.to_excel(path, index=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write export: {exc}")
    return GenerateExportResponse(name=name, path=str(path))


@router.get("/reports/download")
def download_report(name: str, ctx=Depends(get_read_context)) -> FileResponse:
    """Serve a generated export. The name must be a plain filename that exists
    in the export dir -- no path separators, so this cannot escape it."""
    if name != Path(name).name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    target = _export_dir(ctx) / name
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No such export: {name}")
    return FileResponse(target, filename=name)


@router.get("/transactions", response_model=TransactionsResponse)
def transactions(ctx=Depends(get_read_context)) -> TransactionsResponse:
    """The full trade log across every asset, newest first.

    Reads only: the whole transaction table with a computed value per row.
    Filtering and CSV export are done client-side off this single payload.
    """
    cache = MetricsCache(cache_path_for(ctx.config_manager))
    all_tx = ctx.db_manager.get_all_transactions()

    if not isinstance(all_tx, pd.DataFrame) or all_tx.empty:
        return TransactionsResponse(has_data=False, count=0, staleness=staleness_for(cache))

    rows: list[TransactionRow] = []
    for record in all_tx.to_dict(orient="records"):
        quantity = opt(record.get("quantity"))
        price = opt(record.get("price_usd"))
        rows.append(
            TransactionRow(
                timestamp=_str_or_none(record.get("timestamp")),
                symbol=str(record.get("symbol") or "?"),
                type=str(record.get("type") or "?"),
                quantity=quantity,
                price_usd=price,
                # Unknown times unknown is unknown, not zero.
                value_usd=(quantity * price
                           if quantity is not None and price is not None else None),
                fee_usd=opt(record.get("fee_usd")),
                source=_str_or_none(record.get("source")),
                notes=_str_or_none(record.get("notes")),
            )
        )
    rows.sort(key=lambda r: r.timestamp or "", reverse=True)

    return TransactionsResponse(
        has_data=True, count=len(rows), rows=rows, staleness=staleness_for(cache),
    )


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


@router.post("/system/backup", response_model=BackupCreateResponse)
def create_backup(ctx=Depends(get_read_context)) -> BackupCreateResponse:
    """Create a timestamped copy of the database. Additive: never touches the
    live file, only reads it, so this cannot lose data.

    force=True because auto-backup may be disabled in config, and an explicit
    request from the user is exactly the case that should always run.
    """
    try:
        path = ctx.db_manager.backup_database(reason="manual", force=True)
    except Exception as exc:  # a failed backup must report, not 500 the page
        return BackupCreateResponse(created=False, error=str(exc))
    if not path:
        return BackupCreateResponse(
            created=False,
            error="Backup could not be created -- the database file may be missing.",
        )
    return BackupCreateResponse(created=True, name=Path(path).name, path=path)


@router.post("/system/restore", response_model=RestoreResponse)
def restore_backup(
    payload: RestoreRequest, ctx=Depends(get_read_context)
) -> RestoreResponse:
    """Restore the database from a named backup. Destructive, but reversible.

    Only a file already listed by the core counts, which keeps this from
    touching anything outside the backup directory. Before overwriting, the
    current database is itself backed up, so a mistaken restore can be undone.
    """
    backups = ctx.db_manager.list_backups() or []
    match = next((p for p in backups if Path(p).name == payload.name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No such backup: {payload.name}")

    safety = None
    try:
        safety_path = ctx.db_manager.backup_database(reason="pre_restore", force=True)
        safety = Path(safety_path).name if safety_path else None
        restored = ctx.db_manager.restore_from_backup(Path(match))
    except Exception as exc:  # a failed restore must report, not 500 the page
        return RestoreResponse(
            restored=False, name=payload.name, safety_backup=safety, error=str(exc)
        )

    return RestoreResponse(
        restored=bool(restored),
        name=payload.name,
        safety_backup=safety,
        error=None if restored else "Restore failed -- see server logs.",
    )


@router.get("/system/snapshots", response_model=SnapshotsResponse)
def list_snapshots(ctx=Depends(get_read_context)) -> SnapshotsResponse:
    """Every portfolio snapshot, newest first, with the fields delete needs."""
    snaps = ctx.db_manager.get_all_snapshots()
    rows: list[SnapshotRow] = []
    if isinstance(snaps, pd.DataFrame) and not snaps.empty:
        for r in snaps.to_dict(orient="records"):
            rows.append(SnapshotRow(
                timestamp=_str_or_none(r.get("timestamp")),
                total_value_usd=opt(r.get("total_value_usd")),
                total_cost_basis_usd=opt(r.get("total_cost_basis_usd")),
                unrealized_pl_usd=opt(r.get("unrealized_pl_usd")),
                unrealized_pl_percent=opt(r.get("unrealized_pl_percent")),
            ))
    rows.sort(key=lambda r: r.timestamp or "", reverse=True)
    return SnapshotsResponse(count=len(rows), rows=rows)


@router.post("/system/snapshots/delete", response_model=SnapshotDeleteResponse)
def delete_snapshot(
    payload: SnapshotDeleteRequest, ctx=Depends(get_read_context)
) -> SnapshotDeleteResponse:
    """Delete one snapshot, matched on its exact values. Confirmation required."""
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Deletion requires explicit confirmation.")
    try:
        deleted = ctx.db_manager.delete_snapshot(
            payload.timestamp, payload.total_value_usd, payload.total_cost_basis_usd,
            payload.unrealized_pl_usd, payload.unrealized_pl_percent,
        )
    except Exception as exc:  # a failed delete must report, not 500
        return SnapshotDeleteResponse(deleted=0, error=str(exc))
    return SnapshotDeleteResponse(deleted=int(deleted or 0))


@router.get("/system/cleanup", response_model=CleanupStatsResponse)
def cleanup_stats(ctx=Depends(get_read_context)) -> CleanupStatsResponse:
    database = ctx.config_manager.config.get("database", {}) or {}
    days = int(num(database.get("cleanup_days"), 90))
    try:
        raw = ctx.db_manager.get_cleanup_statistics() or {}
    except Exception:  # stats are diagnostic; never fail the page for them
        raw = {}
    # Coerce to JSON-safe scalars (timestamps etc. become strings).
    stats = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
             for k, v in raw.items()}
    return CleanupStatsResponse(cleanup_days=days, enabled=days > 0, stats=stats)


@router.post("/system/cleanup", response_model=CleanupResponse)
def run_cleanup(payload: CleanupRequest, ctx=Depends(get_read_context)) -> CleanupResponse:
    """Delete data older than the configured retention. Confirmation required.

    The core snapshots the database inside cleanup_old_data, so this is
    recoverable from the pre-cleanup backup.
    """
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Cleanup requires explicit confirmation.")
    try:
        result = ctx.db_manager.cleanup_old_data()
    except Exception as exc:
        return CleanupResponse(success=False, error=str(exc))
    message = "Cleanup complete."
    if isinstance(result, (int, str)):
        message = f"Cleanup complete: {result}"
    return CleanupResponse(success=True, message=message)


@router.post("/system/import/{data_type}", response_model=ImportResponse)
async def import_data(
    data_type: str, file: UploadFile = File(...), ctx=Depends(get_read_context)
) -> ImportResponse:
    """Import transactions or holdings from a CSV/Excel file.

    Reversible: a pre-import backup is taken before anything is written (the
    core does this for transactions; it is done explicitly for holdings too).
    """
    kind = data_type.strip().lower()
    if kind not in ("transactions", "holdings"):
        raise HTTPException(status_code=422, detail="data_type must be transactions or holdings.")

    content = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(content))
        else:
            frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        return ImportResponse(success=False, error=f"Could not parse file: {exc}")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return ImportResponse(success=False, error="The file has no rows.")

    try:
        if kind == "transactions":
            # bulk_insert_transactions takes its own backup first.
            affected = ctx.db_manager.bulk_insert_transactions(frame.to_dict(orient="records"))
        else:
            ctx.db_manager.backup_database(reason="pre_import", force=True)
            ctx.db_manager.update_holdings(frame)
            affected = len(frame)
    except Exception as exc:
        return ImportResponse(success=False, error=str(exc))
    return ImportResponse(success=True, rows_affected=int(affected or 0))


@router.put("/system/target-allocation", response_model=TargetAllocationResponse)
def set_target_allocation(
    payload: TargetAllocationRequest, ctx=Depends(get_read_context)
) -> TargetAllocationResponse:
    """Replace the target allocation and persist it to the config file.

    The config manager is a process-wide singleton shared by the read context,
    the analyzer and the tracker, all holding the same config dict -- so
    mutating it in place makes the change live everywhere at once, and
    save_config() writes it to disk.
    """
    cleaned: dict[str, float] = {}
    for symbol, weight in payload.allocation.items():
        name = str(symbol).strip().upper()
        if not name:
            raise HTTPException(status_code=422, detail="An asset symbol was blank.")
        try:
            value = float(weight)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422, detail=f"Weight for {name} is not a number."
            )
        # Fractions: a weight below 0 or above 1 (100%) is a data-entry error,
        # not a valid allocation.
        if value < 0 or value > 1.0001:
            raise HTTPException(
                status_code=422,
                detail=f"Weight for {name} must be between 0 and 1 (got {value}).",
            )
        cleaned[name] = value

    cm = ctx.config_manager
    cm.config["target_allocation"] = cleaned
    _save_config_preserving_secrets(cm)

    total = sum(cleaned.values())
    return TargetAllocationResponse(
        allocation=cleaned, sum=total, sums_to_one=abs(total - 1.0) <= 0.001
    )


def _read_settings(config) -> SettingsResponse:
    portfolio = config.get("portfolio", {}) or {}
    pt = config.get("profit_taking", {}) or {}
    ta = config.get("trend_analyzer", {}) or {}
    database = config.get("database", {}) or {}
    return SettingsResponse(
        minimum_trade_usd=num(portfolio.get("minimum_trade_usd"), 5.0),
        profit_taking=ProfitTakingSettings(
            enabled=bool(pt.get("enabled", False)),
            min_opportunity_score=num(pt.get("min_opportunity_score")),
            min_unrealized_gain_pct=num(pt.get("min_unrealized_gain_pct")),
            min_unrealized_gain_usd=num(pt.get("min_unrealized_gain_usd")),
            max_gain_take_pct=num(pt.get("max_gain_take_pct")),
            default_take_percentage=num(pt.get("default_take_percentage")),
        ),
        p2p_fiat_currency=str(portfolio.get("p2p_fiat_currency", "") or ""),
        crypto_quotes=[str(x) for x in (portfolio.get("crypto_quotes") or [])],
        stablecoin_symbols=[str(x) for x in (portfolio.get("stablecoin_symbols") or [])],
        trend_analyzer=TrendAnalyzerSettings(
            rsi_period=int(num(ta.get("rsi_period"), 14)),
            rsi_oversold=num(ta.get("rsi_oversold"), 30),
            rsi_overbought=num(ta.get("rsi_overbought"), 70),
            cryptocurrencies=[str(x) for x in (ta.get("cryptocurrencies") or [])],
        ),
        cleanup_days=int(num(database.get("cleanup_days"), 90)),
    )


@router.get("/system/settings", response_model=SettingsResponse)
def get_settings(ctx=Depends(get_read_context)) -> SettingsResponse:
    """The subset of config the UI can edit. Reads only."""
    return _read_settings(ctx.config_manager.config)


@router.put("/system/settings", response_model=SettingsResponse)
def update_settings(payload: SettingsUpdate, ctx=Depends(get_read_context)) -> SettingsResponse:
    """Apply a partial settings patch and persist it. Only present fields change."""
    cm = ctx.config_manager
    config = cm.config
    portfolio = config.setdefault("portfolio", {})

    if payload.minimum_trade_usd is not None:
        if not payload.minimum_trade_usd > 0:
            raise HTTPException(status_code=422, detail="Minimum trade must be positive.")
        portfolio["minimum_trade_usd"] = float(payload.minimum_trade_usd)

    if payload.profit_taking is not None:
        pt = payload.profit_taking
        for label, value, hi in (
            ("default_take_percentage", pt.default_take_percentage, 100.0),
            ("max_gain_take_pct", pt.max_gain_take_pct, 100.0),
            ("min_opportunity_score", pt.min_opportunity_score, 100.0),
        ):
            if not 0 <= value <= hi:
                raise HTTPException(
                    status_code=422, detail=f"{label} must be between 0 and {hi:g}."
                )
        if pt.min_unrealized_gain_pct < 0 or pt.min_unrealized_gain_usd < 0:
            raise HTTPException(
                status_code=422, detail="Minimum unrealized gain cannot be negative."
            )
        config["profit_taking"] = pt.model_dump()

    if payload.p2p_fiat_currency is not None:
        portfolio["p2p_fiat_currency"] = payload.p2p_fiat_currency.strip().upper()
    if payload.crypto_quotes is not None:
        portfolio["crypto_quotes"] = [
            s.strip().upper() for s in payload.crypto_quotes if s.strip()
        ]
    if payload.stablecoin_symbols is not None:
        portfolio["stablecoin_symbols"] = [
            s.strip().upper() for s in payload.stablecoin_symbols if s.strip()
        ]

    if payload.trend_analyzer is not None:
        ta = payload.trend_analyzer
        if ta.rsi_period < 1:
            raise HTTPException(status_code=422, detail="RSI period must be at least 1.")
        for label, value in (("rsi_oversold", ta.rsi_oversold),
                             ("rsi_overbought", ta.rsi_overbought)):
            if not 0 <= value <= 100:
                raise HTTPException(status_code=422, detail=f"{label} must be between 0 and 100.")
        tacfg = config.setdefault("trend_analyzer", {})
        tacfg["rsi_period"] = int(ta.rsi_period)
        tacfg["rsi_oversold"] = float(ta.rsi_oversold)
        tacfg["rsi_overbought"] = float(ta.rsi_overbought)
        tacfg["cryptocurrencies"] = [s.strip().upper() for s in ta.cryptocurrencies if s.strip()]

    if payload.cleanup_days is not None:
        if payload.cleanup_days < 0:
            raise HTTPException(status_code=422, detail="Cleanup days cannot be negative.")
        config.setdefault("database", {})["cleanup_days"] = int(payload.cleanup_days)

    _save_config_preserving_secrets(cm)
    return _read_settings(cm.config)
