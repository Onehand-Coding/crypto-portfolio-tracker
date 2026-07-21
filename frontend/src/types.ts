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
