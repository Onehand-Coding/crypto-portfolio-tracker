"""Runs one live analysis in the background and caches its result.

Rebalancing, DCA, profit-taking and technical analysis all need a live Binance
client plus price and kline fetches. Running them inside a GET would make every
page load slow and network-dependent, and would break the rule that reads never
touch the network. They work like sync instead: an explicit POST computes and
caches, and the GET serves the cached result with its age attached.

The core is not modified. Each kind is a thin adapter over a method the core
already exposes.
"""

import asyncio
import copy
import logging
import re
from pathlib import Path
from typing import Awaitable, Callable, Optional

import pandas as pd

from api.cache import MetricsCache, analysis_cache_path
from api.deps import get_tracker
from api.serialization import df_to_records

logger = logging.getLogger(__name__)


async def _rebalance(tracker, params=None) -> dict:
    df = await tracker.portfolio_analyzer.get_core_portfolio_rebalance_suggestions_technical()
    # None means live balances could not be fetched. That is a failure, not an
    # empty portfolio, and the two must not render identically.
    if df is None:
        raise RuntimeError(
            "Could not fetch live balances from Binance. Rebalance suggestions "
            "are unavailable -- this is a connection failure, not a 'nothing to do'."
        )
    return {"suggestions": df_to_records(df)}


async def _profit_taking(tracker, params=None) -> dict:
    opportunities = await tracker.portfolio_analyzer.get_profit_taking_opportunities()
    return {
        "opportunities": [
            {
                "symbol": o.symbol,
                "unrealized_gain_usd": o.unrealized_gain_usd,
                "unrealized_gain_pct": o.unrealized_gain_pct,
                "opportunity_score": o.opportunity_score,
                "rsi_score": o.rsi_score,
                "pl_score": o.pl_score,
                "resistance_score": o.resistance_score,
                "market_context_score": o.market_context_score,
                "current_price": o.current_price,
                "support_level": o.support_level,
                "resistance_level": o.resistance_level,
                "reasons": list(o.reasons),
            }
            for o in (opportunities or [])
        ]
    }


async def _dca(tracker, params=None) -> dict:
    balance = await asyncio.to_thread(tracker.dca_manager.get_available_usdt_balance)
    return {"available": balance}


async def _technical(tracker, params=None) -> dict:
    # The tracker exposes no trend analyzer; the core builds one on demand
    # inside its rebalance method (portfolio_analyzer.py). Same construction
    # here rather than inventing an attribute that does not exist.
    from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer

    analyzer = CryptoTrendAnalyzer(
        config=tracker.config, binance_client=tracker.binance_client
    )
    reports = {}
    for timeframe in ("swing", "long_term"):
        try:
            reports[timeframe] = await analyzer.generate_report(timeframe)
        except Exception as exc:  # one timeframe failing must not lose the other
            logger.warning("Technical report failed for %s: %s", timeframe, exc)
            reports[timeframe] = None
    return {"reports": reports}


def _num_or_none(value) -> Optional[float]:
    """Float or None. NaN and None both mean 'not a number here'."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result


_INDICATOR_TIMEFRAMES = {"long_term", "swing", "day"}


def _write_indicators_cache(config_manager, payload) -> None:
    """Per-symbol cache file (the generic kind file cannot hold every coin)."""
    suffix = "testnet" if config_manager.is_testnet_mode else "live"
    MetricsCache(Path("data") / "api_cache" / f"indicators_{payload['symbol']}_{payload['timeframe']}_{suffix}.json").write(payload)


async def _indicators(tracker, params=None) -> dict:
    """Per-coin indicator history for charting.

    Same fetch + indicator path as the Streamlit coin viewer: period/interval
    rule, 30-row minimum, 500-row tail. Writes its own per-symbol cache file
    and returns a summary for the generic kind file.
    """
    from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer

    params = params or {}
    symbol = str(params.get("symbol", "")).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,10}", symbol):
        raise ValueError(f"Unknown symbol: {params.get('symbol')!r}")
    timeframe = str(params.get("timeframe", "swing"))
    if timeframe not in _INDICATOR_TIMEFRAMES:
        raise ValueError(f"Unknown timeframe: {params.get('timeframe')!r}")

    # The tracker exposes no trend analyzer; the core builds one on demand
    # inside its rebalance method (portfolio_analyzer.py). Same construction
    # here rather than inventing an attribute that does not exist.
    analyzer = CryptoTrendAnalyzer(
        config=tracker.config, binance_client=tracker.binance_client
    )
    settings = analyzer.timeframe_settings.get(timeframe) or {}
    period = settings.get("period", "1mo")
    interval = "1wk" if timeframe == "long_term" else "1d"
    data = await analyzer.fetch_crypto_data_async(symbol, period, interval)
    if data is None or data.empty or len(data) < 30:
        summary = {"symbol": symbol, "timeframe": timeframe, "points": []}
        _write_indicators_cache(tracker.config_manager, summary)
        return summary

    # _calculate_indicators is underscore-private; used here deliberately -
    # it is the exact computation the viewer plots, and duplicating the
    # Study assembly would fork the indicator logic.
    frame = analyzer._calculate_indicators(data.copy(), settings)
    if not pd.api.types.is_datetime64_any_dtype(frame.index):
        raise ValueError(f"No datetime index for {symbol}.")
    frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame[~frame.index.isna()].sort_index().tail(500)

    short_len = settings.get("sma_short_window")
    long_len = settings.get("sma_long_window")
    rsi_col = f"RSI_{analyzer.rsi_period}"
    points = []
    for stamp, row in frame.iterrows():
        points.append({
            "date": str(stamp.date()) if hasattr(stamp, "date") else str(stamp),
            "close": _num_or_none(row.get("Close")),
            "sma_short": _num_or_none(row.get(f"SMA_{short_len}")) if short_len else None,
            "sma_long": _num_or_none(row.get(f"SMA_{long_len}")) if long_len else None,
            "rsi": _num_or_none(row.get(rsi_col)),
            "macd": _num_or_none(row.get("MACD_12_26_9")),
            "macd_signal": _num_or_none(row.get("MACDs_12_26_9")),
            "macd_hist": _num_or_none(row.get("MACDh_12_26_9")),
        })
    summary = {"symbol": symbol, "timeframe": timeframe, "points": points}
    _write_indicators_cache(tracker.config_manager, summary)
    return summary


# The core validates neither, and a bad period/frequency silently degrades the
# run rather than erroring, so the adapter is the place to whitelist them.
# Freeform "<N>y" periods are the Streamlit custom-period parity path.
_BACKTEST_PERIODS = {"1y", "2y", "3y", "5y", "max"}
_BACKTEST_FREQUENCIES = {"weekly", "monthly", "quarterly"}
_BACKTEST_CUSTOM_PERIOD = re.compile(r"^\d+y$")

# Clamp fallbacks mirror config/default_config.json rebalance_technical, so a
# custom run that omits a field rebuilds the value the core would read anyway.
_DEFAULT_MAJORS_DRIFT = 3.0
_DEFAULT_ALTS_DRIFT = 3.5
_DEFAULT_MAJORS_SELL = 0.5
_DEFAULT_MAJORS_BUY = 0.75
_DEFAULT_ALTS_SELL = 0.5
_DEFAULT_ALTS_BUY = 1.0


def _clamp(value, lo, hi, default):
    """Float within [lo, hi]; garbage becomes the Streamlit default."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, result))


def _live_backtest_defaults(tracker_config) -> dict:
    """Current configured rebalance_technical values, as clamp fallbacks.

    A plain run (no custom block) must behave byte-identically to the old
    config path, even if the operator saved non-default thresholds via
    Streamlit. Missing custom fields therefore fall back to the live config,
    not the file defaults above.
    """
    tracker_config = tracker_config or {}
    rt = tracker_config.get("rebalance_technical", {}) or {}
    majors = rt.get("majors", {}) or {}
    alts = rt.get("alts", {}) or {}
    rules = rt.get("market_regime_rules", {}) or {}

    def _num(source, key, fallback):
        try:
            return float(source.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    return {
        "majors_drift": _num(majors, "allocation_drift_threshold_pct", _DEFAULT_MAJORS_DRIFT),
        "alts_drift": _num(alts, "allocation_drift_threshold_pct", _DEFAULT_ALTS_DRIFT),
        "majors_sell": _num(majors, "sell_percentage_multiplier", _DEFAULT_MAJORS_SELL),
        "majors_buy": _num(majors, "buy_amount_multiplier", _DEFAULT_MAJORS_BUY),
        "alts_sell": _num(alts, "sell_percentage_multiplier", _DEFAULT_ALTS_SELL),
        "alts_buy": _num(alts, "buy_amount_multiplier", _DEFAULT_ALTS_BUY),
        "suppress_bear": rules.get("suppress_buys_in_bear", True),
    }


def _valid_custom_allocation(value) -> Optional[dict]:
    """Validated allocation override, or None when it must be ignored.

    Mirrors Streamlit's create_custom_config: the override applies only when
    the weights sum to 1.0 within tolerance. Non-numeric or out-of-range
    weights are also rejected rather than passed to the core.
    """
    if not isinstance(value, dict):
        return None
    try:
        weights = {str(symbol): float(weight) for symbol, weight in value.items()}
    except (TypeError, ValueError):
        return None
    if any(weight < 0 or weight > 1 for weight in weights.values()):
        return None
    if abs(sum(weights.values()) - 1.0) >= 0.001:
        return None
    return weights


def _backtest_config(params, defaults=None) -> dict:
    params = params or {}
    try:
        capital = float(params.get("initial_capital", 10000.0))
    except (TypeError, ValueError):
        capital = 10000.0
    if not capital > 0:
        capital = 10000.0
    period = params.get("period", "2y")
    if period not in _BACKTEST_PERIODS and not (
        isinstance(period, str) and _BACKTEST_CUSTOM_PERIOD.match(period)
    ):
        period = "2y"
    frequency = params.get("frequency", "monthly")
    if frequency not in _BACKTEST_FREQUENCIES:
        frequency = "monthly"
    custom = params.get("custom")
    if not isinstance(custom, dict):
        custom = {}
    base = defaults or {}

    def _custom(key, lo, hi, fallback):
        return _clamp(custom.get(key), lo, hi, base.get(key, fallback))

    return {
        "initial_capital": capital,
        "period": period,
        "frequency": frequency,
        "custom_allocation": _valid_custom_allocation(custom.get("allocation")),
        "majors_drift": _custom("majors_drift", 1.0, 20.0, _DEFAULT_MAJORS_DRIFT),
        "alts_drift": _custom("alts_drift", 1.0, 20.0, _DEFAULT_ALTS_DRIFT),
        "majors_sell": _custom("majors_sell", 0.1, 2.0, _DEFAULT_MAJORS_SELL),
        "majors_buy": _custom("majors_buy", 0.1, 2.0, _DEFAULT_MAJORS_BUY),
        "alts_sell": _custom("alts_sell", 0.1, 2.0, _DEFAULT_ALTS_SELL),
        "alts_buy": _custom("alts_buy", 0.1, 2.0, _DEFAULT_ALTS_BUY),
        "suppress_bear": bool(custom.get("suppress_bear", base.get("suppress_bear", True))),
    }


async def _backtest(tracker, params=None) -> dict:
    """Run the rebalancing backtest with the requested configuration.

    RebalancingBacktester.run is synchronous and fetches years of price history,
    so it goes to a thread rather than blocking the event loop and stalling
    every other request for the duration. run() itself returns None and stores
    its figures on the instance -- the metrics live in summary_stats, the equity
    curve in portfolio_value_history -- so they are read back off the backtester
    rather than from a (non-existent) return value.
    """
    from crypto_portfolio_tracker.rebalancing_backtester import RebalancingBacktester

    config = _backtest_config(params, defaults=_live_backtest_defaults(tracker.config))
    # Mirrors Streamlit's create_custom_config: the backtester reads its
    # thresholds, allocation and frequency from the config dict it is built
    # with. Deepcopy, not .copy(): the tracker is a process-wide singleton and
    # setdefault below would otherwise mutate the live automation dict.
    merged = copy.deepcopy(tracker.config)
    merged["rebalance_technical"] = {
        "market_regime_rules": {"suppress_buys_in_bear": config["suppress_bear"]},
        "majors": {
            "allocation_drift_threshold_pct": config["majors_drift"],
            "sell_percentage_multiplier": config["majors_sell"],
            "buy_amount_multiplier": config["majors_buy"],
        },
        "alts": {
            "allocation_drift_threshold_pct": config["alts_drift"],
            "sell_percentage_multiplier": config["alts_sell"],
            "buy_amount_multiplier": config["alts_buy"],
        },
    }
    if config["custom_allocation"] is not None:
        merged["target_allocation"] = config["custom_allocation"]
    automation = merged.setdefault("automation", {})
    rebalancing = automation.setdefault("rebalancing", {})
    rebalancing["frequency"] = config["frequency"]
    backtester = RebalancingBacktester(config=merged)
    await asyncio.to_thread(
        backtester.run, config["initial_capital"], config["period"], config["frequency"]
    )

    stats = dict(getattr(backtester, "summary_stats", None) or {})
    # The trade log is surfaced at top level; drop the duplicate the core tucks
    # inside summary_stats so the metrics table stays scalars only.
    trade_log = list(stats.pop("Trade Log", []) or [])
    value_history = [
        {"date": str(point.get("date")), "value": point.get("value")}
        for point in (getattr(backtester, "portfolio_value_history", None) or [])
    ]
    return {
        "result": stats,
        "trade_log": trade_log,
        "value_history": value_history,
        "config": config,
    }


KINDS: dict[str, Callable[..., Awaitable[dict]]] = {
    "rebalance": _rebalance,
    "profit": _profit_taking,
    "dca": _dca,
    "technical": _technical,
    "backtest": _backtest,
    "indicators": _indicators,
}


class AnalysisRunner:
    """One in-flight analysis per kind."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._errors: dict[str, Optional[str]] = {}

    def is_running(self, kind: str) -> bool:
        task = self._tasks.get(kind)
        return task is not None and not task.done()

    def last_error(self, kind: str) -> Optional[str]:
        return self._errors.get(kind)

    def start(self, kind: str, params: Optional[dict] = None) -> bool:
        if kind not in KINDS:
            raise KeyError(kind)
        if self.is_running(kind):
            return False
        self._errors[kind] = None
        # get_running_loop, not get_event_loop: the route must be async so a
        # loop exists here. A plain `def` endpoint runs in a threadpool and
        # this raises -- the exact bug that made POST /api/sync 500.
        self._tasks[kind] = asyncio.get_running_loop().create_task(
            self._run(kind, params)
        )
        return True

    async def _run(self, kind: str, params: Optional[dict] = None) -> None:
        try:
            # Built inside the try: the tracker constructor pings Binance and
            # can raise, and that failure must be reported rather than escaping.
            #
            # In a thread, not inline: construction is synchronous network I/O
            # (server-time ping plus client setup, ~12s observed). Inline it
            # wedges the event loop, so no other request is answered until it
            # returns -- the UI's own reload cannot learn the run started, and
            # every Run button sits dead for the whole stall, then flashes
            # "Running…" for a moment at the end. Same reason _backtest pushes
            # its synchronous work to a thread below.
            tracker = await asyncio.to_thread(get_tracker)
            result = await KINDS[kind](tracker, params)
            MetricsCache(analysis_cache_path(tracker.config_manager, kind)).write(result)
        except Exception as exc:
            logger.exception("Analysis %s failed", kind)
            self._errors[kind] = str(exc)


_runner: Optional[AnalysisRunner] = None


def get_analysis_runner() -> AnalysisRunner:
    global _runner
    if _runner is None:
        _runner = AnalysisRunner()
    return _runner
