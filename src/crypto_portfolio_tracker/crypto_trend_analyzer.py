import logging
import asyncio
import functools
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import yfinance as yf
import pandas_ta as ta
from binance.client import Client

from .config import ConfigManager

# Suppress yfinance verbose error logging
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("yfinance").propagate = False


class TrendCondition(Enum):
    GOLDEN_CROSS = "Golden Cross"
    DEATH_CROSS = "Death Cross"
    PRICE_ABOVE_SMA200 = "Price > SMA200"
    PRICE_BELOW_SMA200 = "Price < SMA200"
    RSI_OVERBOUGHT = "RSI Overbought (>70)"
    RSI_OVERSOLD = "RSI Oversold (<30)"
    NEUTRAL_RSI = "Neutral RSI (40-60)"
    MACD_BULLISH_CROSS = "MACD Bullish Crossover"
    MACD_BEARISH_CROSS = "MACD Bearish Crossover"
    VOLUME_SPIKE = "High Volume Spike"
    INSUFFICIENT_DATA = "Insufficient Historical Data"


@dataclass
class TrendAnalysis:
    symbol: str
    timeframe: str
    current_price: float
    price_change_pct: float
    active_conditions: List[str]
    rsi: float
    support_level: float
    resistance_level: float
    ma_periods: Optional[Tuple[int, int]] = None


def get_market_regime(long_term_report: Dict[str, Any]) -> bool:
    """
    Determines the market regime based on the benchmark asset's long-term trend.

    Args:
        long_term_report: The long-term trend analysis report.

    Returns:
        True if the market is considered to be in a bear regime, False otherwise.
    """
    btc_analysis = long_term_report.get("benchmark_analysis")
    if not btc_analysis:
        # Default to False (normal market) if benchmark data is unavailable
        return False

    # The single source of truth for our "bear market" definition
    return TrendCondition.PRICE_BELOW_SMA200.value in btc_analysis.get(
        "active_conditions", []
    )


class CryptoTrendAnalyzer:
    """
    Analyzes cryptocurrency market trends using technical indicators.
    
    This class provides comprehensive technical analysis of cryptocurrency markets
    using various indicators including moving averages, RSI, MACD, and support/resistance levels.
    It can generate reports for different timeframes (long-term, swing, day) and is used
    by both the main application for live analysis and by backtesting engines.
    
    The analyzer supports data fetching from both yfinance and Binance as fallback,
    with caching to minimize API calls and improve performance.
    """
    
    def __init__(self, config: Dict[str, Any], binance_client: Optional[Client] = None):
        """
        Initialize the crypto trend analyzer.
        
        Args:
            config: Configuration dictionary containing analyzer settings
            binance_client: Optional Binance client for fallback data fetching
        """
        self.config = config
        self.analyzer_config = config.get("trend_analyzer", {})
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}
        self.binance_client = binance_client

        self.cryptocurrencies = self.analyzer_config.get("cryptocurrencies", [])
        target_coins = list(config.get("target_allocation", {}).keys())
        for coin in target_coins:
            yf_ticker = f"{coin}-USD"
            if yf_ticker not in self.cryptocurrencies:
                self.cryptocurrencies.append(yf_ticker)

        self.timeframe_settings = self.analyzer_config.get("timeframe_settings", {})
        self.rsi_period = self.analyzer_config.get("rsi_period", 14)
        self.rsi_oversold = self.analyzer_config.get("rsi_oversold", 30)
        self.rsi_overbought = self.analyzer_config.get("rsi_overbought", 70)

    async def _fetch_data_source(
        self, symbol: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Unified data fetching logic with yfinance and Binance fallback."""
        yf_ticker = f"{symbol.split('-')[0]}-USD"

        # 1. Try yfinance
        try:
            self.logger.info(
                f"Fetching {period} of {interval} data for {yf_ticker} from yfinance..."
            )
            data = yf.download(
                tickers=yf_ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                # yfinance data is typically already tz-aware
                return data
        except Exception as e:
            self.logger.warning(f"yfinance download for {yf_ticker} failed: {e}")

        # 2. Fallback to Binance if yfinance fails and a client is available
        if self.binance_client:
            self.logger.warning(f"Falling back to Binance API for {symbol}.")
            try:
                interval_map = {
                    "1m": Client.KLINE_INTERVAL_1MINUTE,
                    "3m": Client.KLINE_INTERVAL_3MINUTE,
                    "5m": Client.KLINE_INTERVAL_5MINUTE,
                    "15m": Client.KLINE_INTERVAL_15MINUTE,
                    "30m": Client.KLINE_INTERVAL_30MINUTE,
                    "1h": Client.KLINE_INTERVAL_1HOUR,
                    "2h": Client.KLINE_INTERVAL_2HOUR,
                    "4h": Client.KLINE_INTERVAL_4HOUR,
                    "6h": Client.KLINE_INTERVAL_6HOUR,
                    "8h": Client.KLINE_INTERVAL_8HOUR,
                    "12h": Client.KLINE_INTERVAL_12HOUR,
                    "1d": Client.KLINE_INTERVAL_1DAY,
                    "3d": Client.KLINE_INTERVAL_3DAY,
                    "1wk": Client.KLINE_INTERVAL_1WEEK,
                    "1mo": Client.KLINE_INTERVAL_1MONTH,
                }
                binance_interval = interval_map.get(interval)
                if not binance_interval:
                    self.logger.error(
                        f"Binance fallback does not support interval: {interval}"
                    )
                    return None

                start_str = (
                    f"{period.replace('y', ' years').replace('mo', ' months')} ago UTC"
                )
                binance_pair = f"{symbol.split('-')[0].upper()}USDT"

                loop = asyncio.get_event_loop()
                klines = await loop.run_in_executor(
                    None,
                    self.binance_client.get_historical_klines,
                    binance_pair,
                    binance_interval,
                    start_str,
                )

                if not klines:
                    return None

                cols = [
                    "timestamp",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "close_time",
                    "quote_av",
                    "trades",
                    "tb_base_av",
                    "tb_quote_av",
                    "ignore",
                ]
                data = pd.DataFrame(klines, columns=cols)

                # --- THIS IS THE ONLY REQUIRED CHANGE ---
                # This ensures the timestamp is made timezone-aware (UTC), preventing the crash.
                data["timestamp"] = pd.to_datetime(
                    data["timestamp"], unit="ms", utc=True
                )

                data.set_index("timestamp", inplace=True)
                data = data[["Open", "High", "Low", "Close", "Volume"]].apply(
                    pd.to_numeric, errors="coerce"
                )
                self.logger.info(
                    f"Successfully fetched {len(data)} data points for {symbol} from Binance API fallback."
                )
                return data
            except Exception as e:
                self.logger.error(f"Binance API fallback also failed for {symbol}: {e}")

        return None

    async def fetch_crypto_data_async(
        self, symbol: str, period: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetches and caches historical crypto data."""
        cache_key = f"historical_data_{symbol}_{period}_{interval}"
        if cache_key in self.data_cache:
            self.logger.debug(
                f"Loading {symbol} data from cache for {interval} interval."
            )
            return self.data_cache[cache_key]

        data = await self._fetch_data_source(symbol, period, interval)
        self.data_cache[cache_key] = data
        return data

    def _calculate_indicators(
        self, data: pd.DataFrame, settings: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculates indicators and returns them as a NEW DataFrame."""
        if data.empty:
            return pd.DataFrame()

        # Use a temporary copy to avoid modifying the original cache
        temp_data = data.copy()

        sma_short = settings.get("sma_short_window")
        sma_long = settings.get("sma_long_window")

        # Create a list of indicators to apply based on data availability
        ta_list = []

        # --- FIX: More robust checks for data length ---
        if sma_short and len(temp_data) > sma_short:
            ta_list.append({"kind": "sma", "length": sma_short})

        if sma_long and len(temp_data) > sma_long:
            ta_list.append({"kind": "sma", "length": sma_long})

        if self.rsi_period and len(temp_data) > self.rsi_period:
            ta_list.append({"kind": "rsi", "length": self.rsi_period})

        # Standard MACD needs at least 26 periods for its slowest EMA. A buffer is safer.
        if len(temp_data) > 35:
            ta_list.append({"kind": "macd"})

        # Only create and run the strategy if there are indicators to apply
        if ta_list:
            ta_strategy = ta.Strategy(name="Crypto Trend Analysis", ta=ta_list)
            temp_data.ta.strategy(ta_strategy)

        return temp_data

    def analyze_coin(
        self, data: pd.DataFrame, symbol: str, timeframe: str
    ) -> TrendAnalysis:
        """Analyzes a single coin's data, handling cases with insufficient history."""

        # A safe buffer to decide if full analysis is possible
        if data.empty or len(data) < 35:
            self.logger.warning(
                f"Insufficient data for {symbol} ({len(data)} periods). Cannot perform full analysis."
            )
            current_price = data.iloc[-1]["Close"] if not data.empty else 0.0
            return TrendAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                price_change_pct=0,
                active_conditions=[TrendCondition.INSUFFICIENT_DATA.value],
                rsi=50.0,
                support_level=0,
                resistance_level=0,
                ma_periods=None,
            )

        timeframe_cfg = self.timeframe_settings.get(timeframe, {})
        indicator_df = self._calculate_indicators(data.copy(), timeframe_cfg)

        if indicator_df.empty:
            return TrendAnalysis(symbol, timeframe, 0, 0, [], 50.0, 0, 0, None)

        latest = indicator_df.iloc[-1]
        previous = indicator_df.iloc[-2]

        current_price = latest.get("Close", 0.0)
        price_change_pct = (
            ((current_price / indicator_df["Close"].iloc[0]) - 1) * 100
            if indicator_df["Close"].iloc[0] > 0
            else 0
        )

        sma_short_len = timeframe_cfg.get("sma_short_window")
        sma_long_len = timeframe_cfg.get("sma_long_window")

        sma_short_val = latest.get(f"SMA_{sma_short_len}", 0.0)
        sma_long_val = latest.get(f"SMA_{sma_long_len}", 0.0)
        rsi_val = latest.get(f"RSI_{self.rsi_period}", 50.0)

        # --- FIX: Pass the correct arguments to the corrected function signature ---
        active_conditions = self._evaluate_conditions(
            latest, previous, sma_short_val, sma_long_val, rsi_val
        )

        if (
            "Volume" in indicator_df.columns
            and len(indicator_df) > 20
            and pd.notna(indicator_df["Volume"].iloc[-1])
        ):
            if (
                latest.get("Volume", 0)
                > indicator_df["Volume"].rolling(window=20).mean().iloc[-1] * 2
            ):
                active_conditions.append(TrendCondition.VOLUME_SPIKE)

        support, resistance = self._calculate_support_resistance(indicator_df)

        return TrendAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            price_change_pct=price_change_pct,
            active_conditions=[condition.value for condition in active_conditions],
            rsi=rsi_val,
            support_level=support,
            resistance_level=resistance,
            ma_periods=(sma_short_len, sma_long_len)
            if sma_short_len and sma_long_len
            else None,
        )

    def analyze_historical_row(
        self, data: pd.DataFrame, symbol: str, timeframe: str, sim_date: pd.Timestamp
    ) -> TrendAnalysis:
        """
        Analyzes a single row of historical data for a specific simulation date.
        This method is designed to be called by the backtester.
        """
        timeframe_cfg = self.timeframe_settings.get(timeframe, {})
        # Assume indicators are already calculated on the 'data' DataFrame
        indicator_df = data

        if indicator_df.empty or sim_date not in indicator_df.index:
            self.logger.warning(
                f"Backtest: Data for {symbol} on {sim_date} not available."
            )
            return TrendAnalysis(
                symbol,
                timeframe,
                0,
                0,
                [TrendCondition.INSUFFICIENT_DATA.value],
                50.0,
                0,
                0,
            )

        # Get the integer index of the simulation date to find the previous day
        try:
            date_idx = indicator_df.index.get_loc(sim_date)
            if date_idx == 0:  # Not enough data for a 'previous' row
                return TrendAnalysis(
                    symbol,
                    timeframe,
                    0,
                    0,
                    [TrendCondition.INSUFFICIENT_DATA.value],
                    50.0,
                    0,
                    0,
                )
            latest = indicator_df.iloc[date_idx]
            previous = indicator_df.iloc[date_idx - 1]
        except (KeyError, IndexError) as e:
            self.logger.error(
                f"Backtest: Could not locate date {sim_date} or its predecessor for {symbol}. Error: {e}"
            )
            return TrendAnalysis(
                symbol,
                timeframe,
                0,
                0,
                [TrendCondition.INSUFFICIENT_DATA.value],
                50.0,
                0,
                0,
            )

        current_price = latest.get("Close", 0.0)
        # Note: price_change_pct is not meaningful in a backtest context like this, setting to 0.
        price_change_pct = 0.0

        sma_short_len = timeframe_cfg.get("sma_short_window")
        sma_long_len = timeframe_cfg.get("sma_long_window")

        sma_short_val = latest.get(f"SMA_{sma_short_len}", 0.0)
        sma_long_val = latest.get(f"SMA_{sma_long_len}", 0.0)
        rsi_val = latest.get(f"RSI_{self.rsi_period}", 50.0)

        active_conditions = self._evaluate_conditions(
            latest, previous, sma_short_val, sma_long_val, rsi_val
        )

        if "Volume" in indicator_df.columns and pd.notna(latest.get("Volume")):
            # Calculate rolling volume mean on data up to the current simulation date
            volume_window = (
                indicator_df.loc[:sim_date, "Volume"].rolling(window=20).mean()
            )
            if (
                not volume_window.empty
                and latest.get("Volume", 0) > volume_window.iloc[-1] * 2
            ):
                active_conditions.append(TrendCondition.VOLUME_SPIKE)

        # For historical S/R, calculate based on the window ending on the sim_date
        historical_slice = indicator_df.loc[:sim_date]
        support, resistance = self._calculate_support_resistance(historical_slice)

        return TrendAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            price_change_pct=price_change_pct,
            active_conditions=[condition.value for condition in active_conditions],
            rsi=rsi_val,
            support_level=support,
            resistance_level=resistance,
            ma_periods=(sma_short_len, sma_long_len)
            if sma_short_len and sma_long_len
            else None,
        )

    def _evaluate_conditions(
        self,
        latest: pd.Series,
        previous: pd.Series,
        sma_short: float,
        sma_long: float,
        rsi: float,
    ) -> List[TrendCondition]:
        """Evaluates and returns a list of active technical conditions from a data row."""
        conditions = []

        if sma_short > 0 and sma_long > 0:
            if sma_short > sma_long:
                conditions.append(TrendCondition.GOLDEN_CROSS)
            else:
                conditions.append(TrendCondition.DEATH_CROSS)

        if sma_long > 0:
            if latest.get("Close", 0) > sma_long:
                conditions.append(TrendCondition.PRICE_ABOVE_SMA200)
            else:
                conditions.append(TrendCondition.PRICE_BELOW_SMA200)

        if rsi > self.rsi_overbought:
            conditions.append(TrendCondition.RSI_OVERBOUGHT)
        elif rsi < self.rsi_oversold:
            conditions.append(TrendCondition.RSI_OVERSOLD)
        elif 40 <= rsi <= 60:
            conditions.append(TrendCondition.NEUTRAL_RSI)

        macd_line = latest.get("MACD_12_26_9")
        macd_signal = latest.get("MACDs_12_26_9")
        macd_line_prev = previous.get("MACD_12_26_9")
        macd_signal_prev = previous.get("MACDs_12_26_9")

        if (
            pd.notna(macd_line)
            and pd.notna(macd_signal)
            and pd.notna(macd_line_prev)
            and pd.notna(macd_signal_prev)
        ):
            if macd_line_prev < macd_signal_prev and macd_line > macd_signal:
                conditions.append(TrendCondition.MACD_BULLISH_CROSS)
            if macd_line_prev > macd_signal_prev and macd_line < macd_signal:
                conditions.append(TrendCondition.MACD_BEARISH_CROSS)

        return conditions

    def _calculate_support_resistance(
        self, data: pd.DataFrame, window: int = 30
    ) -> Tuple[float, float]:
        if data.empty or len(data) < window:
            return (0, 0)
        recent_data = data.tail(window)
        return float(recent_data["Low"].min()), float(recent_data["High"].max())

    async def generate_report(self, timeframe: str) -> Optional[Dict[str, Any]]:
        self.logger.info(f"Generating trend report for timeframe: {timeframe}")
        settings = self.timeframe_settings.get(timeframe)
        if not settings:
            self.logger.error(f"Invalid timeframe '{timeframe}'. Not found in config.")
            return None

        interval = "1wk" if timeframe == "long_term" else "1d"
        report = {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "interval": interval,
        }

        benchmark_symbol = "BTC-USD"
        if benchmark_symbol not in self.cryptocurrencies:
            self.cryptocurrencies.insert(0, benchmark_symbol)

        tasks = [
            self.fetch_crypto_data_async(coin, settings["period"], interval)
            for coin in self.cryptocurrencies
        ]
        results = await asyncio.gather(*tasks)

        coin_data_map = {
            coin: data for coin, data in zip(self.cryptocurrencies, results)
        }
        valid_coin_data = {
            k: v for k, v in coin_data_map.items() if v is not None and not v.empty
        }

        if not valid_coin_data:
            self.logger.error("Failed to fetch data for any coins. Aborting report.")
            return None

        analyses = {
            coin: self.analyze_coin(data, coin, timeframe)
            for coin, data in valid_coin_data.items()
        }
        report["coin_analyses"] = {s: asdict(a) for s, a in analyses.items()}

        if analyses.get(benchmark_symbol):
            report["benchmark_analysis"] = asdict(analyses[benchmark_symbol])

        all_conditions = [
            cond
            for analysis in analyses.values()
            for cond in analysis.active_conditions
        ]
        bullish_conditions = [
            TrendCondition.GOLDEN_CROSS.value,
            TrendCondition.PRICE_ABOVE_SMA200.value,
            TrendCondition.MACD_BULLISH_CROSS.value,
        ]
        bull_count = sum(
            1
            for analysis in analyses.values()
            if any(c in analysis.active_conditions for c in bullish_conditions)
        )

        report["market_summary"] = {
            "total_coins": len(analyses),
            "most_common_condition": max(set(all_conditions), key=all_conditions.count)
            if all_conditions
            else "None",
            "bull_count": bull_count,
            "bear_count": len(analyses) - bull_count,
        }
        self.logger.info(
            f"Successfully generated trend analysis report for {timeframe}."
        )
        return report

    def set_symbol(self, symbol: str):
        """Sets the symbol for the current analysis context."""
        self.symbol = symbol


# Standalone testing block
if __name__ == "__main__":

    async def main_test():
        try:
            from config import ConfigManager

            print("Running CryptoTrendAnalyzer in standalone test mode...")
            config_manager = ConfigManager()
            analyzer = CryptoTrendAnalyzer(
                config=config_manager.config, binance_client=None
            )
            swing_report = await analyzer.generate_report("swing")
            if swing_report:
                import json

                print("\n✅ Successfully generated report for 'swing' timeframe.")
                print(json.dumps(swing_report, indent=4))
            else:
                print(f"\n❌ Failed to generate report for 'swing'.")
        except ImportError:
            print(
                "Could not import ConfigManager. Make sure this script is run from the 'src' directory."
            )
        except Exception as e:
            print(f"An error occurred during standalone execution: {e}")

    asyncio.run(main_test())
