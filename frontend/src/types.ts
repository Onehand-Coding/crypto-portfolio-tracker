export interface Staleness {
  cached_at: string | null;
  age_seconds: number | null;
  is_stale: boolean;
}

export interface Environment {
  is_testnet: boolean;
  database_path: string;
  label: string;
}

export interface AccountingBasis {
  label: string;
  question: string;
  basis_usd: number;
  pl_usd: number;
  /** null when basis_usd is zero: the percentage is undefined, not zero. */
  pl_percent: number | null;
}

export interface Holding {
  symbol: string;
  total_quantity: number;
  spot_quantity: number | null;
  earn_quantity: number | null;
  current_price: number | null;
  value_usd: number | null;
  average_cost_basis: number | null;
  cost_basis_total: number | null;
  unrealized_pl_usd: number | null;
  unrealized_pl_percent: number | null;
  is_core: boolean;
  /** Price lookup failed: value_usd is unknown, not zero. Never dust-collapse. */
  price_unavailable: boolean;
}

export interface CockpitResponse {
  total_value_usd: number;
  net_invested: AccountingBasis;
  fifo: AccountingBasis;
  holdings: Holding[];
  staleness: Staleness;
  environment: Environment;
  has_data: boolean;
  /** >0 means total_value_usd is understated by an unknown amount. */
  unpriced_count: number;
}

export interface CapitalFlowRow {
  source: string;
  type: string;
  direction: 'in' | 'out';
  quantity: number;
  price_usd: number;
  value_usd: number;
  provenance: 'computed' | 'usdt_peg_fallback' | 'failed_lookup';
  is_suspect: boolean;
}

export interface CapitalFlowResponse {
  rows: CapitalFlowRow[];
  total_in_usd: number;
  total_out_usd: number;
  net_invested_usd: number;
  suspect_count: number;
}

export interface WalletBalance {
  symbol: string;
  quantity: number;
  value_usd: number | null;
}

export interface WalletsResponse {
  has_data: boolean;
  spot_earn_value_usd: number;
  futures_value_usd: number;
  funding_value_usd: number;
  total_value_usd: number;
  spot_holdings: WalletBalance[];
  futures_balances: WalletBalance[];
  funding_balances: WalletBalance[];
  staleness: Staleness;
}

export interface SnapshotPoint {
  timestamp: string | null;
  total_value_usd: number | null;
  total_cost_basis_usd: number | null;
  unrealized_pl_usd: number | null;
  unrealized_pl_percent: number | null;
}

export interface OverviewResponse {
  has_data: boolean;
  points: SnapshotPoint[];
  staleness: Staleness;
}

export interface TransactionRow {
  timestamp: string | null;
  symbol: string;
  type: string;
  quantity: number | null;
  price_usd: number | null;
  value_usd: number | null;
  fee_usd: number | null;
  source: string | null;
  notes: string | null;
}

export interface TransactionsResponse {
  has_data: boolean;
  count: number;
  rows: TransactionRow[];
  staleness: Staleness;
}

export interface RealizedGainRow {
  date: string | null;
  year: number | null;
  symbol: string;
  quantity: number | null;
  proceeds_usd: number | null;
  cost_basis_usd: number | null;
  gain_usd: number | null;
}

export interface RealizedGainSummary {
  symbol: string;
  total_gain_usd: number | null;
  total_proceeds_usd: number | null;
  total_cost_basis_usd: number | null;
}

export interface RealizedResponse {
  has_data: boolean;
  rows: RealizedGainRow[];
  by_asset: RealizedGainSummary[];
  total_gain_usd: number | null;
  total_proceeds_usd: number | null;
  total_cost_basis_usd: number | null;
  staleness: Staleness;
}

export interface AssetTransaction {
  timestamp: string | null;
  type: string;
  quantity: number | null;
  price_usd: number | null;
  value_usd: number | null;
  source: string | null;
  notes: string | null;
}

export interface AssetDetailResponse {
  symbol: string;
  found: boolean;
  total_quantity: number | null;
  current_price: number | null;
  value_usd: number | null;
  average_cost_basis: number | null;
  cost_basis_total: number | null;
  unrealized_pl_usd: number | null;
  unrealized_pl_percent: number | null;
  price_unavailable: boolean;
  is_core: boolean;
  target_allocation_pct: number | null;
  transactions: AssetTransaction[];
  staleness: Staleness;
}

/** Every live analysis carries its own run state alongside its result. */
export interface AnalysisState {
  has_data: boolean;
  is_running: boolean;
  error: string | null;
  staleness: Staleness;
}

export interface BacktestConfig {
  initial_capital: number;
  period: string;
  frequency: string;
}

export interface BacktestPoint {
  date: string;
  value: number | null;
}

export interface BacktestResponse extends AnalysisState {
  result: Record<string, number> | null;
  trade_log: string[] | null;
  value_history: BacktestPoint[] | null;
  config: BacktestConfig | null;
}

export interface RebalanceSuggestion {
  symbol: string;
  action: string | null;
  current_value_usd: number | null;
  current_allocation_pct: number | null;
  target_allocation_pct: number | null;
  drift_pct: number | null;
  action_amount_usd: number | null;
  action_quantity: number | null;
  reason: string | null;
  raw: Record<string, unknown>;
}

export interface RebalanceResponse extends AnalysisState {
  suggestions: RebalanceSuggestion[];
}

export interface ProfitOpportunity {
  symbol: string;
  unrealized_gain_usd: number | null;
  unrealized_gain_pct: number | null;
  opportunity_score: number | null;
  rsi_score: number | null;
  pl_score: number | null;
  resistance_score: number | null;
  market_context_score: number | null;
  current_price: number | null;
  support_level: number | null;
  resistance_level: number | null;
  reasons: string[];
}

export interface ProfitResponse extends AnalysisState {
  opportunities: ProfitOpportunity[];
}

export interface DcaResponse extends AnalysisState {
  available_usdt: number | null;
  spot_usdt: number | null;
  earn_usdt: number | null;
  minimum_trade_usd: number;
}

export interface DcaAllocation {
  symbol: string;
  amount_usd: number;
  quantity: number | null;
  current_allocation_pct: number | null;
  target_allocation_pct: number | null;
}

export interface DcaPreviewResponse {
  strategy: string;
  amount_usd: number;
  valid: boolean;
  message: string | null;
  allocations: DcaAllocation[];
}

export interface CompletionRow {
  symbol: string;
  target_allocation_pct: number;
  target_value_usd: number;
  current_value_usd: number | null;
  need_usd: number;
}

export interface CompletionResponse {
  valid: boolean;
  message: string | null;
  anchor_symbol: string | null;
  implied_total_usd: number | null;
  additional_total_usd: number;
  rows: CompletionRow[];
}

export interface IndicatorRow {
  symbol: string;
  price: number | null;
  rsi: number | null;
  sma_short: number | null;
  sma_long: number | null;
  support: number | null;
  resistance: number | null;
  conditions: string[];
}

export interface TechnicalResponse extends AnalysisState {
  timeframes: Record<string, IndicatorRow[]>;
  bear_market: boolean | null;
}

export interface IndicatorPoint {
  date: string;
  close: number | null;
  sma_short: number | null;
  sma_long: number | null;
  rsi: number | null;
  macd: number | null;
  macd_signal: number | null;
  macd_hist: number | null;
}

export interface IndicatorsResponse extends AnalysisState {
  symbol: string;
  timeframe: string;
  points: IndicatorPoint[];
}

export interface ExportFile {
  name: string;
  path: string;
  size_bytes: number;
  modified: string;
}

export interface ReportsResponse {
  files: ExportFile[];
  export_dir: string;
}

export interface BackupInfo {
  name: string;
  size_bytes: number;
  modified: string;
}

export interface BackupCreateResponse {
  created: boolean;
  name: string | null;
  path: string | null;
  error: string | null;
}

export interface GenerateExportResponse {
  name: string;
  path: string;
}

export interface ExecutionStatus {
  testnet: boolean;
  is_live: boolean;
}

export interface TradeExecuteResponse {
  success: boolean;
  testnet: boolean;
  messages: string[];
  errors: string[];
}

export interface RestoreResponse {
  restored: boolean;
  name: string;
  safety_backup: string | null;
  error: string | null;
}

export interface TargetAllocationResponse {
  allocation: Record<string, number>;
  sum: number;
  sums_to_one: boolean;
}

export interface ProfitTakingSettings {
  enabled: boolean;
  min_opportunity_score: number;
  min_unrealized_gain_pct: number;
  min_unrealized_gain_usd: number;
  max_gain_take_pct: number;
  default_take_percentage: number;
}

export interface TrendAnalyzerSettings {
  rsi_period: number;
  rsi_oversold: number;
  rsi_overbought: number;
  cryptocurrencies: string[];
}

export interface AutomationSettings {
  dca_frequency: string;
  rebalancing_frequency: string;
}

export interface ApiSettings {
  coingecko_timeout: number;
  binance_timeout: number;
  binance_recv_window: number;
  binance_delay_ms: number;
  coingecko_delay_ms: number;
}

export interface LoggingSettings {
  level: string;
  file_enabled: boolean;
  file_path: string;
  console_enabled: boolean;
}

export interface TimeframeWindows {
  sma_short_window: number;
  sma_long_window: number;
}

export interface TrendTimeframes {
  long_term: TimeframeWindows;
  swing: TimeframeWindows;
  day: TimeframeWindows;
}

export interface LogPreviewResponse {
  path: string;
  lines: string[];
  truncated: boolean;
  total_lines: number;
}

export interface SettingsResponse {
  minimum_trade_usd: number;
  testnet_mode: boolean;
  live_trading_enabled: boolean;
  profit_taking: ProfitTakingSettings;
  p2p_fiat_currency: string;
  crypto_quotes: string[];
  stablecoin_symbols: string[];
  trend_analyzer: TrendAnalyzerSettings;
  cleanup_days: number;
  automation: AutomationSettings;
  apis: ApiSettings;
  history_lookback_days: Record<string, number>;
  logging: LoggingSettings;
  trend_timeframes: TrendTimeframes;
}

export interface SnapshotRow {
  timestamp: string | null;
  total_value_usd: number | null;
  total_cost_basis_usd: number | null;
  unrealized_pl_usd: number | null;
  unrealized_pl_percent: number | null;
}

export interface SnapshotsResponse {
  count: number;
  rows: SnapshotRow[];
}

export interface SnapshotDeleteResponse {
  deleted: number;
  error: string | null;
}

export interface CleanupStatsResponse {
  cleanup_days: number;
  enabled: boolean;
  stats: Record<string, string | number | boolean | null>;
}

export interface CleanupResponse {
  success: boolean;
  message: string | null;
  error: string | null;
}

export interface ImportResponse {
  success: boolean;
  rows_affected: number;
  error: string | null;
}

export interface SystemHealthResponse {
  environment_label: string;
  is_testnet: boolean;
  database_path: string;
  database_exists: boolean;
  database_size_bytes: number;
  transaction_count: number;
  asset_count: number;
  snapshot_count: number;
  live_trading_enabled: boolean;
  minimum_trade_usd: number;
  target_allocation: Record<string, number>;
  backups: BackupInfo[];
  metrics_cache_age_seconds: number | null;
  binance_configured: boolean;
}
