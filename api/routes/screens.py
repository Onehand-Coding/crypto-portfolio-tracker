"""Overview, Asset Detail, Reports and System Health.

Mostly offline: cached metrics plus SQLite. The connection probe is the one
exception -- it constructs a tracker and touches the network, so it is POST
like sync and execute, never GET.
"""

import copy
import datetime
import io
import json
import os
import platform
import re
from pathlib import Path

import pandas as pd
import psutil
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.cache import MetricsCache, analysis_cache_path, cache_path_for
from api.deps import get_read_context, get_tracker
from api.routes.common import num, opt, staleness_for
from api.schemas.screens import (
    DISPOSAL_KIND_LABELS,
    FREQUENCIES,
    LOG_LEVELS,
    LOOKBACK_KEYS,
    ApiSettings,
    AssetDetailResponse,
    AssetTransaction,
    AutomationSettings,
    BackupCreateResponse,
    BackupDeleteRequest,
    BackupDeleteResponse,
    BackupInfo,
    CleanupRequest,
    CleanupResponse,
    CleanupStatsResponse,
    ConnectionsResponse,
    ConnectionStatus,
    DeleteExportRequest,
    DeleteExportResponse,
    ExportFile,
    GenerateExportRequest,
    GenerateExportResponse,
    ImportResponse,
    LoggingSettings,
    LogPreviewResponse,
    OverviewResponse,
    PreviewResponse,
    ProfitTakingSettings,
    RealizedExportRequest,
    RealizedGainRow,
    RealizedGainSummary,
    RealizedKindSummary,
    RealizedResponse,
    ReportsResponse,
    ResourcesResponse,
    RestoreRequest,
    RestoreResponse,
    SettingsResponse,
    SettingsUpdate,
    SnapshotDeleteRequest,
    SnapshotDeleteResponse,
    SnapshotPoint,
    SnapshotRow,
    SnapshotSaveResponse,
    SnapshotsResponse,
    SummaryExportRequest,
    SystemHealthResponse,
    TargetAllocationRequest,
    TargetAllocationResponse,
    TimeframeWindows,
    TransactionRow,
    TransactionsResponse,
    TrendAnalyzerSettings,
    TrendExportRequest,
    TrendTimeframes,
)
from api.serialization import jsonable
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


def _new_files_since(export_dir: Path, before: set[str]) -> list[Path]:
    """Files the export just wrote. The core exporters timestamp their own
    filenames, so the newcomer is found by diffing, not by guessing names."""
    return sorted(
        (p for p in export_dir.iterdir()
         if p.is_file() and p.name not in before),
        key=lambda p: p.stat().st_mtime, reverse=True)


def _report_generator(ctx, export_dir: Path):
    """Core exporters pointed at the API export dir. Constructing this never
    touches the network; the tracker is not involved."""
    from crypto_portfolio_tracker.exporters import (
        CsvExporter,
        ExcelExporter,
        HtmlExporter,
    )
    from crypto_portfolio_tracker.report_generator import ReportGenerator
    config = dict(ctx.config_manager.config or {})
    exports = (ctx.config_manager.config or {}).get("exports", {}) or {}
    config["exports"] = {**exports, "path": str(export_dir)}
    return ReportGenerator(
        config=config,
        db_manager=ctx.db_manager,
        excel_exporter=ExcelExporter(config),
        html_exporter=HtmlExporter(config),
        csv_exporter=CsvExporter(config),
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


@router.post("/reports/summary", response_model=GenerateExportResponse)
def export_summary(
    payload: SummaryExportRequest, ctx=Depends(get_read_context)
) -> GenerateExportResponse:
    """Portfolio summary via the core exporters (same output as the CLI)."""
    fmt = payload.format.strip().lower()
    if fmt not in ("csv", "excel", "html"):
        raise HTTPException(status_code=422, detail="format must be csv, excel or html.")
    metrics = MetricsCache(cache_path_for(ctx.config_manager)).read()
    if not metrics:
        raise HTTPException(status_code=422, detail="No synced metrics yet -- run a sync first.")
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in export_dir.iterdir() if p.is_file()}
    full = dict(metrics)
    full["holdings_df"] = pd.DataFrame(metrics.get("holdings_df") or [])
    try:
        _report_generator(ctx, export_dir).export_portfolio_summary(full, fmt)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write export: {exc}")
    fresh = _new_files_since(export_dir, before)
    if not fresh:
        raise HTTPException(status_code=500, detail="Exporter reported success but wrote no file.")
    return GenerateExportResponse(name=fresh[0].name, path=str(fresh[0]))


@router.post("/reports/charts", response_model=GenerateExportResponse)
def export_charts(ctx=Depends(get_read_context)) -> GenerateExportResponse:
    """All portfolio charts as PNGs via the core visualizer (same files as the CLI chart menu)."""
    # Headless backend (Agg) is pinned at process start via MPLBACKEND -- never
    # switch it per request: use() only works pre-import and pyplot is already
    # imported by then via deps -> portfolio_tracker -> visualizations.
    from crypto_portfolio_tracker.visualizations import Visualizer

    metrics = MetricsCache(cache_path_for(ctx.config_manager)).read()
    if not metrics:
        raise HTTPException(status_code=422, detail="No synced metrics yet -- run a sync first.")
    holdings = pd.DataFrame(metrics.get("holdings_df") or [])
    if holdings.empty:
        raise HTTPException(status_code=422, detail="No holdings to chart -- run a sync first.")
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = export_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in charts_dir.iterdir() if p.is_file()}
    config = dict(ctx.config_manager.config or {})
    exports = (ctx.config_manager.config or {}).get("exports", {}) or {}
    config["exports"] = {**exports, "path": str(export_dir)}
    full = dict(metrics)
    full["holdings_df"] = holdings
    snapshots = ctx.db_manager.get_all_snapshots()
    if not isinstance(snapshots, pd.DataFrame):
        snapshots = pd.DataFrame()
    try:
        Visualizer(config).generate_all_charts(
            holdings, full, config.get("target_allocation", {}) or {}, snapshots
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write charts: {exc}")
    fresh = sorted(
        (p for p in charts_dir.iterdir() if p.is_file() and p.name not in before),
        key=lambda p: p.stat().st_mtime, reverse=True)
    fresh = [p for p in fresh if p.suffix == ".png"]
    if not fresh:
        raise HTTPException(status_code=500, detail="Chart generation wrote no file.")
    # Filenames carry second-resolution stamps, so a same-second regeneration
    # overwrites the same name -- harmless because the output is regenerable.
    # The move keeps PNGs alongside the other generated files where the
    # existing list and download links find them.
    try:
        for path in fresh:
            path.rename(export_dir / path.name)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write export: {exc}")
    newest = export_dir / fresh[0].name
    return GenerateExportResponse(name=newest.name, path=str(newest))


@router.post("/reports/trend", response_model=GenerateExportResponse)
def export_trend(
    payload: TrendExportRequest, ctx=Depends(get_read_context)
) -> GenerateExportResponse:
    """A cached technical report via the core trend exporter."""
    fmt = payload.format.strip().lower()
    if fmt not in ("csv", "json", "html"):
        raise HTTPException(status_code=422, detail="format must be csv, json or html.")
    timeframe = payload.timeframe.strip().lower()
    cache = MetricsCache(analysis_cache_path(ctx.config_manager, "technical"))
    reports = (cache.read() or {}).get("reports") or {}
    report = reports.get(timeframe)
    if not isinstance(report, dict) or "coin_analyses" not in report:
        raise HTTPException(
            status_code=422,
            detail=f"No {timeframe} technical report cached -- run the analysis first.")
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in export_dir.iterdir() if p.is_file()}
    try:
        _report_generator(ctx, export_dir).export_trend_report(report, timeframe, fmt.upper())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write export: {exc}")
    fresh = _new_files_since(export_dir, before)
    if not fresh:
        raise HTTPException(status_code=500, detail="Exporter reported success but wrote no file.")
    return GenerateExportResponse(name=fresh[0].name, path=str(fresh[0]))


@router.post("/reports/realized", response_model=GenerateExportResponse)
def export_realized(
    payload: RealizedExportRequest, ctx=Depends(get_read_context)
) -> GenerateExportResponse:
    """Realized FIFO gains as a file. Same tax_df the /realized screen shows,
    written with the same tz-naive guard as generate_export."""
    fmt = payload.format.strip().lower()
    if fmt not in ("csv", "excel"):
        raise HTTPException(status_code=422, detail="format must be csv or excel.")
    transactions = ctx.db_manager.get_all_transactions()
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise HTTPException(status_code=422, detail="No transactions to export.")
    tax_df = calculate_fifo_realized_gains(transactions)
    if tax_df.empty:
        raise HTTPException(status_code=422, detail="No realized gains to export yet.")
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"realized_{stamp}.{'csv' if fmt == 'csv' else 'xlsx'}"
    path = export_dir / name
    try:
        if fmt == "csv":
            tax_df.to_csv(path, index=False)
        else:
            safe = tax_df.copy()
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


@router.get("/reports/preview", response_model=PreviewResponse)
def preview_report(name: str, ctx=Depends(get_read_context)) -> PreviewResponse:
    """On-screen preview of a generated export, shaped by file type.

    Mirrors the Streamlit export viewer: tabular files (CSV/spreadsheets)
    come back as columns plus rows for a real table, HTML reports are
    flagged for rendering (they are Jinja templates meant to be viewed, not
    read as source), JSON is flagged for pretty-printing, and images return
    a download URL for an <img> tag -- reading their bytes as text would
    splatter binary mojibake across the page. Anything else is refused
    plainly rather than guessed at.
    """
    if name != Path(name).name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    target = _export_dir(ctx) / name
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No such export: {name}")
    suffix = target.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        return PreviewResponse(
            name=name, kind="image",
            image_url=f"/api/reports/download?name={name}")
    if suffix in (".csv", ".xlsx", ".xls"):
        try:
            frame = (pd.read_excel(target) if suffix != ".csv"
                     else pd.read_csv(target))
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Could not preview table: {exc}")
        head = frame.head(50)
        return PreviewResponse(
            name=name, kind="table",
            columns=[str(col) for col in head.columns],
            rows=[[jsonable(value) for value in record]
                  for record in head.itertuples(index=False, name=None)],
            truncated=len(frame) > 50, total_lines=len(frame))
    if suffix == ".html":
        # Returned whole, not truncated: the viewer renders it via srcDoc,
        # not by navigating to the download URL (which serves attachment
        # and would leave the frame blank). Reports are ~10KB templates.
        try:
            text = target.read_text(errors="replace")
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not read export: {exc}")
        if len(text) > 200_000:
            text = text[:200_000]
        return PreviewResponse(name=name, kind="html", lines=text.splitlines(),
                               total_lines=len(text.splitlines()))
    if suffix == ".json":
        try:
            text = target.read_text(errors="replace")
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not read export: {exc}")
        return PreviewResponse(
            name=name, kind="json", lines=text.splitlines()[:50],
            truncated=len(text.splitlines()) > 50,
            total_lines=len(text.splitlines()))
    if suffix not in (".txt", ".md", ".log"):
        raise HTTPException(
            status_code=422,
            detail=f"Preview is not available for {suffix or 'extensionless'} "
                   "files. Download it instead.")
    try:
        text = target.read_text(errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read export: {exc}")
    if len(text) > 200_000:
        text = text[:200_000]
    lines = text.splitlines()
    return PreviewResponse(
        name=name, lines=lines[:50],
        truncated=len(lines) > 50, total_lines=len(lines))


@router.post("/reports/delete", response_model=DeleteExportResponse)
def delete_report(
    payload: DeleteExportRequest, ctx=Depends(get_read_context)
) -> DeleteExportResponse:
    """Delete a generated export. Confirmation required, same as snapshots."""
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Deletion requires explicit confirmation.")
    if payload.name != Path(payload.name).name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    target = _export_dir(ctx) / payload.name
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No such export: {payload.name}")
    try:
        target.unlink()
    except OSError as exc:
        return DeleteExportResponse(deleted=False, name=payload.name, error=str(exc))
    return DeleteExportResponse(deleted=True, name=payload.name)


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
                kind=str(record.get("kind") or "OTHER"),
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

    # Same roll-up by economic kind of disposal, so the UI can show where the
    # gross proceeds actually come from (trades vs Earn sweeps vs ...). The
    # kind groups must add back up to the headline totals exactly -- a test
    # pins that, because two disagreeing totals would be worse than one lump.
    kind_summary = (
        tax_df.groupby("kind")
        .agg(
            event_count=("gain_usd", "size"),
            total_gain_usd=("gain_usd", "sum"),
            total_proceeds_usd=("proceeds_usd", "sum"),
            total_cost_basis_usd=("cost_basis_usd", "sum"),
        )
        .reset_index()
    )
    by_kind = [
        RealizedKindSummary(
            kind=str(record.get("kind")),
            label=DISPOSAL_KIND_LABELS.get(str(record.get("kind")), "Other"),
            event_count=int(record.get("event_count")),
            total_gain_usd=opt(record.get("total_gain_usd")),
            total_proceeds_usd=opt(record.get("total_proceeds_usd")),
            total_cost_basis_usd=opt(record.get("total_cost_basis_usd")),
        )
        for record in kind_summary.to_dict(orient="records")
    ]
    by_kind.sort(key=lambda s: s.total_proceeds_usd or 0.0, reverse=True)

    return RealizedResponse(
        has_data=True,
        rows=rows,
        by_asset=by_asset,
        by_kind=by_kind,
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


def _match_backup(ctx, name: str) -> Path:
    """A listed backup path by plain filename, else 404. Same trust model
    as restore: only files the core already reports can be touched."""
    if name != Path(name).name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    backups = ctx.db_manager.list_backups() or []
    match = next((p for p in backups if Path(p).name == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No such backup: {name}")
    return Path(match)


@router.get("/system/backup/download")
def download_backup(name: str, ctx=Depends(get_read_context)) -> FileResponse:
    """Serve a listed database backup for download."""
    target = _match_backup(ctx, name)
    return FileResponse(target, filename=target.name)


@router.post("/system/backup/delete", response_model=BackupDeleteResponse)
def delete_backup(
    payload: BackupDeleteRequest, ctx=Depends(get_read_context)
) -> BackupDeleteResponse:
    """Delete a listed database backup. Confirmation required."""
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Deletion requires explicit confirmation.")
    target = _match_backup(ctx, payload.name)
    try:
        target.unlink()
    except OSError as exc:
        return BackupDeleteResponse(deleted=False, name=payload.name, error=str(exc))
    return BackupDeleteResponse(deleted=True, name=payload.name)


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
    auto = config.get("automation", {}) or {}
    dca = auto.get("dca", {}) or {}
    rb = auto.get("rebalancing", {}) or {}
    sy = auto.get("auto_sync", {}) or {}
    apis = config.get("apis", {}) or {}
    cg = apis.get("coingecko", {}) or {}
    bi = apis.get("binance", {}) or {}
    # Only the keys the Streamlit lookback widgets manage; anything else in
    # the mapping (e.g. the legacy "transfers") is not surfaced.
    raw_lb = config.get("history_lookback_days", {}) or {}
    lookback = {key: int(num(raw_lb.get(key), 90)) for key in LOOKBACK_KEYS}
    log = config.get("logging", {}) or {}
    log_file = log.get("file_config", {}) or {}
    log_console = log.get("console_config", {}) or {}
    tf = ta.get("timeframe_settings", {}) or {}

    def _windows(name: str, default_period: str) -> TimeframeWindows:
        # Fallbacks match the Streamlit timeframe widgets (period per
        # timeframe, windows 10/30); the shipped config carries its own
        # per-timeframe values, which win when present.
        slot = tf.get(name, {}) or {}
        return TimeframeWindows(
            period=str(slot.get("period", default_period) or default_period),
            sma_short_window=int(num(slot.get("sma_short_window"), 10)),
            sma_long_window=int(num(slot.get("sma_long_window"), 30)),
        )

    return SettingsResponse(
        minimum_trade_usd=num(portfolio.get("minimum_trade_usd"), 5.0),
        testnet_mode=bool(portfolio.get("testnet_mode", False)),
        live_trading_enabled=bool(portfolio.get("live_trading_enabled", False)),
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
        automation=AutomationSettings(
            dca_frequency=str(dca.get("frequency", "monthly") or "monthly"),
            rebalancing_frequency=str(rb.get("frequency", "weekly") or "weekly"),
            auto_sync_enabled=bool(sy.get("enabled", False)),
            auto_sync_interval_minutes=int(num(sy.get("interval_minutes"), 5)),
        ),
        apis=ApiSettings(
            coingecko_timeout=num(cg.get("timeout"), 30),
            binance_timeout=num(bi.get("timeout"), 60),
            binance_recv_window=int(num(bi.get("recv_window"), 20000)),
            binance_delay_ms=num(bi.get("request_delay_ms"), 500),
            coingecko_delay_ms=num(cg.get("request_delay_ms"), 1500),
        ),
        history_lookback_days=lookback,
        logging=LoggingSettings(
            level=str(log.get("level", "INFO") or "INFO"),
            file_enabled=bool(log_file.get("enabled", True)),
            file_path=str(log_file.get("path", "logs/portfolio_tracker.log")
                          or "logs/portfolio_tracker.log"),
            console_enabled=bool(log_console.get("enabled", True)),
        ),
        trend_timeframes=TrendTimeframes(
            long_term=_windows("long_term", "1y"),
            swing=_windows("swing", "90d"),
            day=_windows("day", "7d"),
        ),
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

    # The two exchange switches, mirroring the CLI/Streamlit trading-mode block.
    # testnet_mode selects the endpoint (and DB); a running server keeps its
    # cached tracker, so a flip here needs a restart to take full effect.
    if payload.testnet_mode is not None:
        portfolio["testnet_mode"] = bool(payload.testnet_mode)
    if payload.live_trading_enabled is not None:
        portfolio["live_trading_enabled"] = bool(payload.live_trading_enabled)

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

    if payload.automation is not None:
        # exclude_unset: a partial group patches only the sent fields
        # instead of resetting the others to their schema defaults.
        auto = payload.automation.model_dump(exclude_unset=True)
        for key in ("dca_frequency", "rebalancing_frequency"):
            if key in auto and str(auto[key]).strip().lower() not in FREQUENCIES:
                raise HTTPException(
                    status_code=422,
                    detail=f"{key} must be one of {', '.join(FREQUENCIES)}.")
        auto_cfg = config.setdefault("automation", {})
        if "dca_frequency" in auto:
            auto_cfg.setdefault("dca", {})["frequency"] = str(auto["dca_frequency"]).strip().lower()
        if "rebalancing_frequency" in auto:
            auto_cfg.setdefault("rebalancing", {})["frequency"] = (
                str(auto["rebalancing_frequency"]).strip().lower())
        if "auto_sync_enabled" in auto:
            auto_cfg.setdefault("auto_sync", {})["enabled"] = bool(auto["auto_sync_enabled"])
        if "auto_sync_interval_minutes" in auto:
            minutes = int(auto["auto_sync_interval_minutes"])
            # Bounds: each run costs Binance + CoinGecko calls, so sub-2-minute
            # cadences burn rate limit for no freshness gain; 1440 = once daily.
            if not 2 <= minutes <= 1440:
                raise HTTPException(
                    status_code=422,
                    detail="auto_sync_interval_minutes must be between 2 and 1440.")
            auto_cfg.setdefault("auto_sync", {})["interval_minutes"] = minutes

    if payload.apis is not None:
        ap = payload.apis.model_dump(exclude_unset=True)
        for key in ("coingecko_timeout", "binance_timeout"):
            if key in ap and not ap[key] > 0:
                raise HTTPException(
                    status_code=422, detail=f"{key} must be positive.")
        if "binance_recv_window" in ap and not ap["binance_recv_window"] > 0:
            raise HTTPException(
                status_code=422, detail="binance_recv_window must be positive.")
        for key in ("binance_delay_ms", "coingecko_delay_ms"):
            if key in ap and not ap[key] >= 0:
                raise HTTPException(
                    status_code=422, detail=f"{key} cannot be negative.")
        apis_cfg = config.setdefault("apis", {})
        if "coingecko_timeout" in ap:
            apis_cfg.setdefault("coingecko", {})["timeout"] = float(ap["coingecko_timeout"])
        if "binance_timeout" in ap:
            apis_cfg.setdefault("binance", {})["timeout"] = float(ap["binance_timeout"])
        if "binance_recv_window" in ap:
            apis_cfg.setdefault("binance", {})["recv_window"] = int(ap["binance_recv_window"])
        if "binance_delay_ms" in ap:
            apis_cfg.setdefault("binance", {})["request_delay_ms"] = float(ap["binance_delay_ms"])
        if "coingecko_delay_ms" in ap:
            apis_cfg.setdefault("coingecko", {})["request_delay_ms"] = float(
                ap["coingecko_delay_ms"])

    if payload.history_lookback_days is not None:
        for key, value in payload.history_lookback_days.items():
            if key not in LOOKBACK_KEYS:
                raise HTTPException(
                    status_code=422, detail=f"Unknown lookback key: {key}.")
            # Mirror the Streamlit number_input min_value=1.
            if int(value) < 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Lookback days for {key} must be at least 1.")
        lb_cfg = config.setdefault("history_lookback_days", {})
        for key, value in payload.history_lookback_days.items():
            lb_cfg[key] = int(value)

    if payload.logging is not None:
        lg = payload.logging.model_dump(exclude_unset=True)
        if "level" in lg and str(lg["level"]).strip().upper() not in LOG_LEVELS:
            raise HTTPException(
                status_code=422,
                detail=f"level must be one of {', '.join(LOG_LEVELS)}.")
        if "file_path" in lg:
            new_path = str(lg["file_path"] or "").strip()
            if not new_path:
                raise HTTPException(
                    status_code=422, detail="Log file path cannot be empty.")
            # Validate writability without creating anything: a later block
            # (e.g. trend windows) can still reject the request, and that
            # rejection must not leave a created directory behind.
            ancestor = Path(new_path).parent
            while not ancestor.exists():
                if ancestor.parent == ancestor:
                    break
                ancestor = ancestor.parent
            if not os.access(ancestor, os.W_OK):
                raise HTTPException(
                    status_code=422, detail="Log file directory is not writable.")
        log_cfg = config.setdefault("logging", {})
        if "level" in lg:
            log_cfg["level"] = str(lg["level"]).strip().upper()
        if "file_enabled" in lg:
            log_cfg.setdefault("file_config", {})["enabled"] = bool(lg["file_enabled"])
        if "file_path" in lg:
            log_cfg.setdefault("file_config", {})["path"] = str(lg["file_path"]).strip()
        if "console_enabled" in lg:
            log_cfg.setdefault("console_config", {})["enabled"] = bool(lg["console_enabled"])

    if payload.trend_timeframes is not None:
        for name in ("long_term", "swing", "day"):
            group = getattr(payload.trend_timeframes, name)
            # Mirror the Streamlit widgets: min 1, max 200, short below long.
            for label, value in (("sma_short_window", group.sma_short_window),
                                 ("sma_long_window", group.sma_long_window)):
                if not 1 <= value <= 200:
                    raise HTTPException(
                        status_code=422,
                        detail=f"{label} for {name} must be between 1 and 200.")
            if group.sma_short_window >= group.sma_long_window:
                raise HTTPException(
                    status_code=422,
                    detail=f"Short SMA must be less than Long SMA for {name}.")
            # Mirror the Streamlit period check: non-empty Xy/Xd/Xmo.
            period = str(group.period or "").strip()
            if not period:
                raise HTTPException(
                    status_code=422,
                    detail=f"Period for {name} cannot be empty.")
            if not re.match(r"^\d+(y|d|mo)$", period):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid period format for {name}: Use Xy, Xd, or Xmo.")
        tf_cfg = config.setdefault("trend_analyzer", {}).setdefault(
            "timeframe_settings", {})
        provided = payload.trend_timeframes.model_dump(exclude_unset=True)
        for name in ("long_term", "swing", "day"):
            group = getattr(payload.trend_timeframes, name)
            slot = tf_cfg.setdefault(name, {})
            # A windows-only patch must not clobber a stored period string.
            if "period" in provided.get(name, {}):
                slot["period"] = str(group.period).strip()
            slot["sma_short_window"] = int(group.sma_short_window)
            slot["sma_long_window"] = int(group.sma_long_window)

    # Deferred side effect: the log directory is created only after every
    # validation branch passed, using the final configured path.
    if payload.logging is not None:
        if "file_path" in payload.logging.model_dump(exclude_unset=True):
            final_path = str(
                ((config.get("logging", {}) or {}).get("file_config", {}) or {}).get(
                    "path", "") or "")
            if final_path:
                try:
                    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise HTTPException(
                        status_code=422, detail=f"Cannot create log directory: {exc}")
                if not os.access(Path(final_path).parent, os.W_OK):
                    raise HTTPException(
                        status_code=422, detail="Log file directory is not writable.")

    _save_config_preserving_secrets(cm)
    return _read_settings(cm.config)


def _sanitized_config_copy(config) -> dict:
    """A disk-safe deep copy of the config: main keys masked, CoinGecko key
    dropped. Mirrors what save_config() strips, without mutating the live dict
    the way save_config() does (the restore happens in its wrapper)."""
    snapshot = copy.deepcopy(config or {})
    if isinstance(snapshot.get("main_api_keys"), dict):
        snapshot["main_api_keys"] = {
            key: "********" for key in snapshot["main_api_keys"]
        }
    try:
        apis = snapshot.get("apis")
        if isinstance(apis, dict) and isinstance(apis.get("coingecko"), dict):
            apis["coingecko"].pop("api_key", None)
    except (KeyError, TypeError, AttributeError):
        pass
    return snapshot


@router.get("/system/config/export")
def export_config(ctx=Depends(get_read_context)) -> FileResponse:
    """Download the config as JSON. Secrets are masked/dropped in the file;
    the live process keeps the real values (deep copy, never the live dict)."""
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"config_export_{stamp}.json"
    path = export_dir / name
    try:
        path.write_text(
            json.dumps(_sanitized_config_copy(ctx.config_manager.config),
                       indent=2, default=str))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write export: {exc}")
    return FileResponse(path, filename=name, media_type="application/json")


def _deep_merge_live(base: dict, incoming: dict) -> dict:
    """Recursive merge for config import: dict-vs-dict merges, everything
    else overwrites. Unmentioned nested keys (e.g. apis.yfinance) survive
    a partial import instead of being silently dropped."""
    for key, value in incoming.items():
        if (isinstance(value, dict) and isinstance(base.get(key), dict)):
            _deep_merge_live(base[key], value)
        else:
            base[key] = value
    return base


@router.post("/system/config/import", response_model=SettingsResponse)
async def import_config(
    file: UploadFile = File(...), ctx=Depends(get_read_context)
) -> SettingsResponse:
    """Apply a previously exported config file. The live secrets are preserved:
    an export only carries masked placeholders, so both secret keys are dropped
    from the upload unconditionally. A sanitized backup of the live config is
    written first, so a bad import is reversible."""
    content = await file.read()
    if len(content) > 1_000_000:
        raise HTTPException(
            status_code=422, detail="Config file too large (max 1MB).")
    try:
        new_config = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON file: {exc}")
    if not isinstance(new_config, dict):
        raise HTTPException(
            status_code=422, detail="Config must be a JSON object.")

    new_config.pop("main_api_keys", None)
    try:
        apis = new_config.get("apis")
        if isinstance(apis, dict) and isinstance(apis.get("coingecko"), dict):
            apis["coingecko"].pop("api_key", None)
    except (KeyError, TypeError, AttributeError):
        pass

    cm = ctx.config_manager
    export_dir = _export_dir(ctx)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = export_dir / f"config_backup_{stamp}.json"
    try:
        backup_path.write_text(
            json.dumps(_sanitized_config_copy(cm.config), indent=2, default=str))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write backup: {exc}")

    _deep_merge_live(cm.config, new_config)
    _save_config_preserving_secrets(cm)
    return _read_settings(cm.config)


@router.get("/system/logs/preview", response_model=LogPreviewResponse)
def log_preview(
    lines: int = 50, ctx=Depends(get_read_context)
) -> LogPreviewResponse:
    """Last N lines of the configured log file. The path comes from the config
    only, never from the caller, so this cannot be aimed at other files."""
    config = ctx.config_manager.config or {}
    path_str = str(
        ((config.get("logging", {}) or {}).get("file_config", {}) or {}).get(
            "path", "logs/portfolio_tracker.log")
        or "logs/portfolio_tracker.log")
    count = max(1, min(500, lines))
    target = Path(path_str)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No log file at {path_str}")
    try:
        with open(target, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 200_000))
            text = handle.read().decode(errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read log file: {exc}")
    all_lines = text.splitlines()
    total = len(all_lines)
    return LogPreviewResponse(
        path=path_str,
        lines=all_lines[-count:] if total else [],
        truncated=total > count,
        total_lines=total,
    )


@router.post("/system/snapshot/save", response_model=SnapshotSaveResponse)
def save_snapshot(ctx=Depends(get_read_context)) -> SnapshotSaveResponse:
    """Persist the cached metrics as a portfolio snapshot (same row the CLI writes after a sync)."""
    metrics = MetricsCache(cache_path_for(ctx.config_manager)).read()
    if not metrics or opt(metrics.get("total_value_usd")) is None:
        raise HTTPException(status_code=422, detail="No synced metrics yet -- run a sync first.")
    # Fresh stamp, not the metrics' own: this records when the snapshot was
    # taken, the same way the core stamps its post-sync row with now.
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        # Keyword names mirror DatabaseManager.save_portfolio_snapshot exactly;
        # the remaining figures default to 0.0 just like the core's own save.
        ctx.db_manager.save_portfolio_snapshot(
            timestamp=timestamp,
            total_value=num(metrics.get("total_value_usd"), 0.0),
            total_cost_basis=num(metrics.get("total_cost_basis_usd"), 0.0),
            unrealized_pl=num(metrics.get("unrealized_pl_usd"), 0.0),
            unrealized_pl_percent=num(metrics.get("unrealized_pl_percent"), 0.0),
        )
    except Exception as exc:  # a failed save must report, not 500 the page
        return SnapshotSaveResponse(saved=False, error=str(exc))
    return SnapshotSaveResponse(saved=True, timestamp=timestamp)


@router.get("/system/resources", response_model=ResourcesResponse)
def system_resources(ctx=Depends(get_read_context)) -> ResourcesResponse:
    """App + host figures. Any single psutil failure nulls that field, never 500s."""
    try:
        python_version = platform.python_version()
    except Exception:
        python_version = ""
    try:
        # Blocks ~1s by design: cpu_percent measures usage over the interval,
        # same as the Streamlit status tab.
        cpu_percent = opt(psutil.cpu_percent(interval=1))
    except Exception:
        cpu_percent = None
    try:
        ram = psutil.virtual_memory()
        ram_percent = opt(ram.percent)
        ram_used_gb = opt(ram.used / (1024 ** 3))
    except Exception:
        ram_percent = None
        ram_used_gb = None
    try:
        disk_percent = opt(psutil.disk_usage("/").percent)
    except Exception:
        disk_percent = None

    version = ctx.config_manager.config.get("version")
    return ResourcesResponse(
        app_version=str(version) if version else None,
        python_version=python_version,
        cpu_percent=cpu_percent,
        ram_percent=ram_percent,
        ram_used_gb=ram_used_gb,
        disk_percent=disk_percent,
    )


@router.post("/system/connections", response_model=ConnectionsResponse)
async def test_connections() -> ConnectionsResponse:
    """Live connectivity probe (Binance ping + CoinGecko BTC price). POST because
    it touches the network -- same rule as sync and execute."""
    try:
        tracker = get_tracker()
    except Exception as exc:
        return ConnectionsResponse(
            binance=ConnectionStatus(ok=False, detail=f"No client: {exc}"),
            coingecko=ConnectionStatus(ok=False, detail="Skipped without tracker."))
    # Mirrors the CLI test_connections: ping when a client exists, skip when
    # there are no keys, then a BTC price as the CoinGecko probe.
    client = getattr(tracker, "binance_client", None)
    if client is None:
        binance = ConnectionStatus(
            ok=False, detail="SKIPPED (No API keys or client init failed)")
    else:
        try:
            client.ping()
            binance = ConnectionStatus(ok=True, detail="SUCCESS")
        except Exception as exc:
            binance = ConnectionStatus(ok=False, detail=f"FAILED ({exc})")
    btc_price_usd = None
    try:
        prices = await tracker.enricher.get_current_prices(["BTC"])
        raw = prices.get("BTC") if isinstance(prices, dict) else None
        btc_price_usd = opt(raw)
        if btc_price_usd is None:
            coingecko = ConnectionStatus(
                ok=False, detail="FAILED (No price data returned)")
        else:
            coingecko = ConnectionStatus(
                ok=True, detail=f"SUCCESS (BTC price: ${btc_price_usd})")
    except Exception as exc:
        btc_price_usd = None
        coingecko = ConnectionStatus(ok=False, detail=f"FAILED ({exc})")
    return ConnectionsResponse(
        binance=binance, coingecko=coingecko, btc_price_usd=btc_price_usd)
