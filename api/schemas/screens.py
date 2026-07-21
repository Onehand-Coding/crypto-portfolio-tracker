"""Response models for the remaining screens.

Optional[float] throughout means genuinely unknown. Nothing here substitutes
zero for a missing figure -- that distinction is the whole point of this UI.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from api.schemas.system import Staleness


class SnapshotPoint(BaseModel):
    timestamp: Optional[str]
    total_value_usd: Optional[float]
    total_cost_basis_usd: Optional[float]
    unrealized_pl_usd: Optional[float]
    unrealized_pl_percent: Optional[float]


class OverviewResponse(BaseModel):
    has_data: bool
    points: list[SnapshotPoint]
    staleness: Staleness


class AssetTransaction(BaseModel):
    timestamp: Optional[str]
    type: str
    quantity: Optional[float]
    price_usd: Optional[float]
    value_usd: Optional[float]
    source: Optional[str]
    notes: Optional[str]


class AssetDetailResponse(BaseModel):
    symbol: str
    found: bool = Field(description="False when the symbol is not held and has no history")
    total_quantity: Optional[float] = None
    current_price: Optional[float] = None
    value_usd: Optional[float] = None
    average_cost_basis: Optional[float] = None
    cost_basis_total: Optional[float] = None
    unrealized_pl_usd: Optional[float] = None
    unrealized_pl_percent: Optional[float] = None
    price_unavailable: bool = False
    is_core: bool = False
    target_allocation_pct: Optional[float] = None
    transactions: list[AssetTransaction] = []
    staleness: Staleness


class TransactionRow(BaseModel):
    """One transaction across all assets, for the global trade log."""

    timestamp: Optional[str]
    symbol: str
    type: str
    quantity: Optional[float]
    price_usd: Optional[float]
    value_usd: Optional[float]
    fee_usd: Optional[float]
    source: Optional[str]
    notes: Optional[str]


class TransactionsResponse(BaseModel):
    has_data: bool
    count: int
    rows: list[TransactionRow] = []
    staleness: Staleness


class RealizedGainRow(BaseModel):
    """One taxable event (a SELL/WITHDRAWAL) priced against its FIFO lots."""

    date: Optional[str]
    year: Optional[int]
    symbol: str
    quantity: Optional[float]
    proceeds_usd: Optional[float]
    cost_basis_usd: Optional[float]
    gain_usd: Optional[float]


class RealizedGainSummary(BaseModel):
    """Per-asset roll-up of realized gains."""

    symbol: str
    total_gain_usd: Optional[float]
    total_proceeds_usd: Optional[float]
    total_cost_basis_usd: Optional[float]


class RealizedResponse(BaseModel):
    # False when there are no transactions at all; has_data True with an empty
    # rows list is the distinct, legitimate "no taxable events yet" state.
    has_data: bool
    rows: list[RealizedGainRow] = []
    by_asset: list[RealizedGainSummary] = []
    total_gain_usd: Optional[float] = None
    total_proceeds_usd: Optional[float] = None
    total_cost_basis_usd: Optional[float] = None
    staleness: Staleness


class AnalysisState(BaseModel):
    """Wrapper for every live analysis: the result plus how it is doing."""

    has_data: bool
    is_running: bool
    error: Optional[str] = Field(
        None, description="Last failure. Present means the figures shown are the "
                          "previous run's, not this one's."
    )
    staleness: Staleness


class RebalanceSuggestion(BaseModel):
    symbol: str
    action: Optional[str] = None
    current_value_usd: Optional[float] = None
    current_allocation_pct: Optional[float] = None
    target_allocation_pct: Optional[float] = None
    drift_pct: Optional[float] = None
    action_amount_usd: Optional[float] = None
    action_quantity: Optional[float] = None
    reason: Optional[str] = None
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Every column the core returned, so a field this schema does "
                    "not model yet is visible rather than silently dropped.",
    )


class RebalanceResponse(AnalysisState):
    suggestions: list[RebalanceSuggestion] = []


class ProfitOpportunityOut(BaseModel):
    symbol: str
    unrealized_gain_usd: Optional[float] = None
    unrealized_gain_pct: Optional[float] = None
    opportunity_score: Optional[float] = None
    rsi_score: Optional[float] = None
    pl_score: Optional[float] = None
    resistance_score: Optional[float] = None
    market_context_score: Optional[float] = None
    current_price: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    reasons: list[str] = []


class ProfitResponse(AnalysisState):
    opportunities: list[ProfitOpportunityOut] = []


class DcaAllocation(BaseModel):
    symbol: str
    amount_usd: float
    quantity: Optional[float] = None
    current_allocation_pct: Optional[float] = None
    target_allocation_pct: Optional[float] = None


class DcaResponse(AnalysisState):
    available_usdt: Optional[float] = None
    spot_usdt: Optional[float] = None
    earn_usdt: Optional[float] = None
    minimum_trade_usd: float = 5.0


class DcaPreviewRequest(BaseModel):
    amount_usd: float
    strategy: str = Field("target_weight", description="'proportional' or 'target_weight'")


class DcaPreviewResponse(BaseModel):
    strategy: str
    amount_usd: float
    valid: bool
    message: Optional[str] = None
    allocations: list[DcaAllocation] = []


class IndicatorRow(BaseModel):
    symbol: str
    price: Optional[float] = None
    rsi: Optional[float] = None
    sma_short: Optional[float] = None
    sma_long: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    conditions: list[str] = []


class TechnicalResponse(AnalysisState):
    timeframes: dict[str, list[IndicatorRow]] = {}
    bear_market: Optional[bool] = Field(
        None, description="BTC below SMA200. None when the report is unavailable."
    )


class BacktestPoint(BaseModel):
    date: str
    value_usd: Optional[float] = None


class BacktestResponse(BaseModel):
    available: bool
    message: Optional[str] = None
    points: list[BacktestPoint] = []


class ExportFile(BaseModel):
    name: str
    path: str
    size_bytes: int
    modified: str


class ReportsResponse(BaseModel):
    files: list[ExportFile] = []
    export_dir: str


class BackupInfo(BaseModel):
    name: str
    size_bytes: int
    modified: str


class SystemHealthResponse(BaseModel):
    environment_label: str
    is_testnet: bool
    database_path: str
    database_exists: bool
    database_size_bytes: int
    transaction_count: int
    asset_count: int
    snapshot_count: int
    live_trading_enabled: bool
    minimum_trade_usd: float
    target_allocation: dict[str, float]
    backups: list[BackupInfo] = []
    metrics_cache_age_seconds: Optional[float] = None
    binance_configured: bool
