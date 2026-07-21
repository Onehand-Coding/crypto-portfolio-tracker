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
import logging
from typing import Any, Awaitable, Callable, Optional

from api.cache import MetricsCache, analysis_cache_path
from api.deps import get_tracker
from api.serialization import df_to_records

logger = logging.getLogger(__name__)


async def _rebalance(tracker) -> dict:
    df = await tracker.portfolio_analyzer.get_core_portfolio_rebalance_suggestions_technical()
    # None means live balances could not be fetched. That is a failure, not an
    # empty portfolio, and the two must not render identically.
    if df is None:
        raise RuntimeError(
            "Could not fetch live balances from Binance. Rebalance suggestions "
            "are unavailable -- this is a connection failure, not a 'nothing to do'."
        )
    return {"suggestions": df_to_records(df)}


async def _profit_taking(tracker) -> dict:
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


async def _dca(tracker) -> dict:
    balance = tracker.dca_manager.get_available_usdt_balance()
    return {"available": balance}


async def _technical(tracker) -> dict:
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


async def _backtest(tracker) -> dict:
    """Run the rebalancing backtest.

    RebalancingBacktester.run is synchronous and fetches years of price history,
    so it goes to a thread rather than blocking the event loop and stalling
    every other request for the duration.
    """
    from crypto_portfolio_tracker.rebalancing_backtester import RebalancingBacktester

    backtester = RebalancingBacktester(config=tracker.config)
    result = await asyncio.to_thread(backtester.run, 10000.0, "2y", "monthly")
    report = backtester.generate_report() if hasattr(backtester, "generate_report") else None
    return {"result": result, "report": report}


KINDS: dict[str, Callable[[Any], Awaitable[dict]]] = {
    "rebalance": _rebalance,
    "profit": _profit_taking,
    "dca": _dca,
    "technical": _technical,
    "backtest": _backtest,
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

    def start(self, kind: str) -> bool:
        if kind not in KINDS:
            raise KeyError(kind)
        if self.is_running(kind):
            return False
        self._errors[kind] = None
        # get_running_loop, not get_event_loop: the route must be async so a
        # loop exists here. A plain `def` endpoint runs in a threadpool and
        # this raises -- the exact bug that made POST /api/sync 500.
        self._tasks[kind] = asyncio.get_running_loop().create_task(self._run(kind))
        return True

    async def _run(self, kind: str) -> None:
        try:
            # Built inside the try: the tracker constructor pings Binance and
            # can raise, and that failure must be reported rather than escaping.
            tracker = get_tracker()
            result = await KINDS[kind](tracker)
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
