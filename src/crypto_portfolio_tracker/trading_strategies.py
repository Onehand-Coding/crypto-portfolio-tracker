import pandas as pd
from typing import Tuple, Optional, List, Dict, Any

from .crypto_trend_analyzer import CryptoTrendAnalyzer, TrendCondition


class Strategy:
    """A base class for all strategies, with a consistent configuration method."""

    valid_intervals: List[str] = ["1d"]
    strategy_type: str = "general"

    def __init__(
        self,
        analyzer: Optional[CryptoTrendAnalyzer] = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.analyzer = analyzer
        self._state = state or {}

    @property
    def name(self):
        raise NotImplementedError("Strategy must have a name.")

    @classmethod
    def get_user_params(cls) -> Dict[str, Any]:
        """Class method to get user-defined parameters. Can be overridden by subclasses."""
        return {}

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the strategy for persistence."""
        return self._state

    async def generate_signal(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Returns a signal, a trade size (as a fraction of capital), and a reason.
        """
        raise NotImplementedError


class GoldenCrossStrategy(Strategy):
    """A simple trend-following strategy based on MA crossovers."""

    valid_intervals = ["1d"]
    strategy_type = "swing"

    def __init__(self, fast_period: int = 50, slow_period: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self):
        return f"Golden Cross ({self.fast_period}/{self.slow_period})"

    async def generate_signal(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[str], float, Optional[str]]:
        if len(data) < self.slow_period + 1:
            return "HOLD", 0.0, "Insufficient data"
        df = data.copy()
        df.ta.sma(length=self.fast_period, append=True)
        df.ta.sma(length=self.slow_period, append=True)
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        fast_ma_col = f"SMA_{self.fast_period}"
        slow_ma_col = f"SMA_{self.slow_period}"
        if (
            previous[fast_ma_col] <= previous[slow_ma_col]
            and latest[fast_ma_col] > latest[slow_ma_col]
        ):
            return (
                "BUY",
                1.0,
                f"Golden Cross: {self.fast_period}d > {self.slow_period}d",
            )
        elif (
            previous[fast_ma_col] >= previous[slow_ma_col]
            and latest[fast_ma_col] < latest[slow_ma_col]
        ):
            return (
                "SELL",
                1.0,
                f"Death Cross: {self.fast_period}d < {self.slow_period}d",
            )
        return "HOLD", 0.0, "No crossover"


class MultiTimeframeMomentumStrategy(Strategy):
    """
    A sophisticated strategy that requires trend confirmation from multiple timeframes
    before entering and uses a trailing stop-loss for exits.
    """

    valid_intervals = ["1d"]
    strategy_type = "swing"
    strategy_param_specs = {
        "trailing_stop_pct": {
            "type": "float",
            "label": "Trailing Stop (%)",
            "default": 20.0,
            "min_value": 1.0,
            "max_value": 99.0,
        }
    }

    def __init__(
        self,
        analyzer: CryptoTrendAnalyzer,
        state: Optional[Dict[str, Any]] = None,
        trailing_stop_pct: float = 0.20,
        **kwargs,
    ):
        super().__init__(analyzer=analyzer, state=state, **kwargs)
        self.fast_period = 50
        self.slow_period = 200
        self.rsi_overbought = 70
        self.trailing_stop_pct = trailing_stop_pct
        self._state.setdefault("in_position", False)
        self._state.setdefault("highest_price_since_entry", 0)

    @property
    def name(self):
        return f"MTF Momentum w/ {self.trailing_stop_pct:.0%} T-Stop"

    @classmethod
    def get_user_params(cls) -> Dict[str, Any]:
        """Class method to get user-defined parameters interactively."""
        try:
            pct_str = input(
                f"Enter trailing stop percentage (e.g., 20 for 20%, default: 20): "
            ).strip()
            trailing_pct = float(pct_str) / 100 if pct_str else 0.20
        except ValueError:
            print("Invalid input. Using default 20%.")
            trailing_pct = 0.20
        return {"trailing_stop_pct": trailing_pct}

    async def generate_signal(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[str], float, Optional[str]]:
        symbol = self.analyzer.symbol
        current_price = data["Close"].iloc[-1]

        # --- Exit Logic (only applies if we are in a position) ---
        if self._state["in_position"]:
            self._state["highest_price_since_entry"] = max(
                self._state["highest_price_since_entry"], current_price
            )
            stop_price = self._state["highest_price_since_entry"] * (
                1 - self.trailing_stop_pct
            )
            if current_price < stop_price:
                self._state["in_position"] = False
                self._state["highest_price_since_entry"] = 0
                return "SELL", 1.0, f"Trailing Stop Triggered at ${stop_price:,.2f}"
            else:
                return "HOLD", 0.0, f"Trailing Stop at ${stop_price:,.2f}"

        # --- Entry Logic (only if not in a position) ---
        if len(data) < self.slow_period:
            return "HOLD", 0.0, "Insufficient daily data"

        daily_data = data.copy()
        daily_data.ta.sma(length=self.fast_period, append=True)
        daily_data.ta.sma(length=self.slow_period, append=True)
        daily_data.ta.rsi(length=14, append=True)

        daily_fast_ma = daily_data[f"SMA_{self.fast_period}"].iloc[-1]
        daily_slow_ma = daily_data[f"SMA_{self.slow_period}"].iloc[-1]
        daily_rsi = daily_data["RSI_14"].iloc[-1]

        if daily_fast_ma < daily_slow_ma or daily_rsi >= self.rsi_overbought:
            return "HOLD", 0.0, "Daily trend not bullish or RSI overbought"

        weekly_data = await self.analyzer.fetch_crypto_data_async(
            symbol, period="4y", interval="1wk"
        )
        if weekly_data is None or len(weekly_data) < self.slow_period:
            return "HOLD", 0.0, "Insufficient weekly data"

        weekly_data.ta.sma(length=self.fast_period, append=True)
        weekly_data.ta.sma(length=self.slow_period, append=True)
        weekly_fast_ma = weekly_data[f"SMA_{self.fast_period}"].iloc[-1]
        weekly_slow_ma = weekly_data[f"SMA_{self.slow_period}"].iloc[-1]

        if weekly_fast_ma < weekly_slow_ma:
            return "HOLD", 0.0, "Weekly trend not bullish"

        self._state["in_position"] = True
        self._state["highest_price_since_entry"] = current_price
        return "BUY", 1.0, "Daily & Weekly trends align"


class OpeningRangeBreakoutStrategy(Strategy):
    """
    A classic day trading strategy that trades breakouts from the session's opening range.
    """

    valid_intervals = ["15m", "1h"]
    strategy_type = "day"
    strategy_param_specs = {
        "range_duration": {
            "type": "text",
            "label": "Opening Range Duration (e.g., '1h', '30m')",
            "default": "1h",
        },
        "risk_per_trade_pct": {
            "type": "float",
            "label": "Capital to Risk per Trade (%)",
            "default": 20.0,
            "min_value": 1.0,
            "max_value": 100.0,
        },
    }

    def __init__(
        self,
        analyzer: CryptoTrendAnalyzer,
        state: Optional[Dict[str, Any]] = None,
        range_duration: str = "1h",
        risk_per_trade_pct: float = 0.20,
        **kwargs,
    ):
        super().__init__(analyzer=analyzer, state=state, **kwargs)
        self.range_duration = range_duration.upper()
        self.risk_per_trade_pct = risk_per_trade_pct
        self._state.setdefault("last_day_processed", None)
        self._state.setdefault("trade_taken_today", False)
        self._state.setdefault("in_position", False)
        self._state.setdefault("stop_loss", 0)

    @property
    def name(self):
        return f"ORB ({self.range_duration}, {self.risk_per_trade_pct:.1%} Risk)"

    @classmethod
    def get_user_params(cls) -> Dict[str, Any]:
        """Get user-defined parameters for the ORB strategy."""
        range_duration = (
            input(
                "Enter opening range duration (e.g., '1h', '30m', default: '1h'): "
            ).strip()
            or "1h"
        )
        try:
            risk_pct_str = input(
                "Enter capital to risk per trade (e.g., 20 for 20%, default: 20): "
            ).strip()
            risk_per_trade_pct = float(risk_pct_str) / 100 if risk_pct_str else 0.20
        except ValueError:
            print("Invalid percentage. Using default 20%.")
            risk_per_trade_pct = 0.20
        return {
            "range_duration": range_duration,
            "risk_per_trade_pct": risk_per_trade_pct,
        }

    async def generate_signal(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[str], float, Optional[str]]:
        if data.empty:
            return "HOLD", 0.0, "No data available"

        latest_bar = data.iloc[-1]
        current_price = latest_bar["Close"]
        # Fix: Remove utc=True to make current_time timezone-naive
        current_time = pd.to_datetime(latest_bar.name)
        current_day_str = str(current_time.date())

        if self._state.get("last_day_processed") != current_day_str:
            self._state.update(
                {
                    "last_day_processed": current_day_str,
                    "in_position": False,
                    "trade_taken_today": False,
                    "opening_range_high": 0,
                    "opening_range_low": 0,
                    "stop_loss": 0,
                }
            )

        if self._state["in_position"] and current_price < self._state["stop_loss"]:
            self._state["in_position"] = False
            return "SELL", 1.0, f"Stop loss hit at ${self._state['stop_loss']:.2f}"

        if self._state["in_position"] and current_time.hour >= 23:
            self._state["in_position"] = False
            return "SELL", 1.0, "End-of-day position close"

        if self._state["trade_taken_today"]:
            return "HOLD", 0.0, "Trade already attempted for today."

        today_data = data[data.index.date == current_time.date()]
        if today_data.empty:
            return "HOLD", 0.0, "Awaiting data for current day."

        opening_time = today_data.index[0]
        range_end_time = opening_time + pd.Timedelta(self.range_duration.lower())

        if current_time < range_end_time:
            return "HOLD", 0.0, "In opening range period."

        opening_range_bars = today_data[today_data.index < range_end_time]
        if opening_range_bars.empty:
            return "HOLD", 0.0, "Awaiting opening range bar data."

        opening_range_high = opening_range_bars["High"].max()
        opening_range_low = opening_range_bars["Low"].min()

        if current_price > opening_range_high:
            self._state["in_position"] = True
            self._state["trade_taken_today"] = True
            self._state["stop_loss"] = opening_range_low
            return (
                "BUY",
                self.risk_per_trade_pct,
                f"Breakout above {self.range_duration} range high ${opening_range_high:.2f}",
            )

        return "HOLD", 0.0, "Awaiting breakout."


class VolatilityAdaptiveEMASwing(Strategy):
    """
    A swing strategy that enters on pullbacks in an established trend,
    with a stop-loss based on market volatility (ATR).
    - Trend Filter: Only enters long if the price is above the 200-day EMA.
    - Entry: Buys on a dip when the fast EMA (20) crosses above the slow EMA (50).
    - Confirmation: RSI must not be overbought (>70) to avoid buying into euphoria.
    - Risk Management: Stop-loss is set at 2x ATR below the entry price. Take-profit is set at a 2:1 risk/reward ratio.
    """

    valid_intervals = ["1d"]
    strategy_type = "swing"

    def __init__(
        self,
        analyzer: CryptoTrendAnalyzer,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(analyzer=analyzer, state=state, **kwargs)
        # Default state
        self._state.setdefault("in_position", False)
        self._state.setdefault("stop_loss", 0)
        self._state.setdefault("take_profit", 0)

    @property
    def name(self):
        return "Volatility-Adaptive EMA Swing Strategy"

    async def generate_signal(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[str], float, Optional[str]]:
        if len(data) < 200:
            return "HOLD", 0.0, "Insufficient data for 200-day EMA trend filter"

        df = data.copy()
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # --- EXIT LOGIC ---
        if self._state["in_position"]:
            if latest["Close"] < self._state["stop_loss"]:
                self._state["in_position"] = False
                return (
                    "SELL",
                    1.0,
                    f"Stop-loss triggered at ${self._state['stop_loss']:.2f}",
                )
            if latest["Close"] > self._state["take_profit"]:
                self._state["in_position"] = False
                return (
                    "SELL",
                    1.0,
                    f"Take-profit triggered at ${self._state['take_profit']:.2f}",
                )
            return (
                "HOLD",
                0.0,
                f"Position held. SL=${self._state['stop_loss']:.2f}, TP=${self._state['take_profit']:.2f}",
            )

        # --- ENTRY LOGIC ---
        # 1. Trend Filter: Must be in a long-term uptrend
        if latest["Close"] < latest["EMA_200"]:
            return "HOLD", 0.0, "Price is below 200-day EMA (long-term downtrend)"

        # 2. Entry Signal: A bullish crossover of short-term EMAs
        ema_20_crossed_above_50 = (
            previous["EMA_20"] <= previous["EMA_50"]
            and latest["EMA_20"] > latest["EMA_50"]
        )

        # 3. Confirmation Filters
        rsi_ok = latest["RSI_14"] < 70  # Not overbought
        atr_valid = (
            "ATRr_14" in latest
            and pd.notna(latest["ATRr_14"])
            and latest["ATRr_14"] > 0
        )

        if ema_20_crossed_above_50 and rsi_ok and atr_valid:
            # Set dynamic stop-loss and take-profit based on volatility
            stop_loss_price = latest["Close"] - (latest["ATRr_14"] * 2)
            profit_target_price = latest["Close"] + (
                latest["ATRr_14"] * 4
            )  # 2:1 Risk/Reward

            self._state["in_position"] = True
            self._state["stop_loss"] = stop_loss_price
            self._state["take_profit"] = profit_target_price

            return (
                "BUY",
                1.0,
                f"EMA 20/50 cross with trend confirmation. SL=${stop_loss_price:.2f}",
            )

        return "HOLD", 0.0, "Awaiting valid entry signal"


class ORBWithVolumeFilter(Strategy):
    """
    A day trading strategy that trades breakouts from the session's opening range,
    but only if the breakout occurs on significant volume.
    - Opening Range: Defines high/low of the first hour of the trading day.
    - Breakout: Enters when price closes above the opening range high.
    - Volume Filter: Breakout bar's volume must be > 1.5x the average volume of the opening range.
    - Risk Management: Stop-loss is placed at the low of the opening range. Position is closed at the end of the day.
    """

    valid_intervals = ["15m", "1h"]
    strategy_type = "day"

    def __init__(
        self,
        analyzer: CryptoTrendAnalyzer,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(analyzer=analyzer, state=state, **kwargs)
        self._state.setdefault("last_day_processed", None)
        self._state.setdefault("in_position", False)
        self._state.setdefault("opening_range_high", 0)
        self._state.setdefault("opening_range_low", 0)

    @property
    def name(self):
        return "Opening Range Breakout with Volume Filter"

    async def generate_signal(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[str], float, Optional[str]]:
        if data.empty or len(data) < 5:  # Need at least a few bars
            return "HOLD", 0.0, "No data"

        latest_bar = data.iloc[-1]
        # Fix: Remove utc=True to make current_time timezone-naive
        current_time = pd.to_datetime(latest_bar.name)
        current_day_str = str(current_time.date())

        # Reset daily variables at the start of a new day
        if self._state.get("last_day_processed") != current_day_str:
            self._state.update(
                {
                    "last_day_processed": current_day_str,
                    "in_position": False,
                    "opening_range_high": 0,
                    "opening_range_low": 0,
                }
            )

        # End-of-day exit logic
        if self._state["in_position"] and current_time.hour >= 23:
            self._state["in_position"] = False
            return "SELL", 1.0, "End-of-day exit"

        # Stop-loss exit logic
        if (
            self._state["in_position"]
            and latest_bar["Close"] < self._state["opening_range_low"]
        ):
            self._state["in_position"] = False
            return (
                "SELL",
                1.0,
                f"Stop-loss triggered at ${self._state['opening_range_low']:.2f}",
            )

        # If already in a position, just hold until exit signal
        if self._state["in_position"]:
            return "HOLD", 0.0, "Position held"

        # Define Opening Range (first hour, i.e., first four 15m bars)
        # Assumes UTC day starts at 00:00
        opening_range_end = current_time.replace(
            hour=1, minute=0, second=0, microsecond=0
        )

        # Do not trade before the opening range is set
        if current_time < opening_range_end:
            return "HOLD", 0.0, "Awaiting opening range formation"

        # Calculate opening range metrics only once per day
        if self._state["opening_range_high"] == 0:
            today_data = data[data.index.date == current_time.date()]
            range_bars = today_data[today_data.index < opening_range_end]
            if range_bars.empty:
                return "HOLD", 0.0, "Waiting for opening range bars"

            self._state["opening_range_high"] = range_bars["High"].max()
            self._state["opening_range_low"] = range_bars["Low"].min()
            self._state["opening_range_avg_volume"] = range_bars["Volume"].mean()

        # --- ENTRY LOGIC ---
        is_breakout = latest_bar["Close"] > self._state["opening_range_high"]
        is_volume_confirmed = latest_bar["Volume"] > (
            self._state["opening_range_avg_volume"] * 1.5
        )

        if is_breakout and is_volume_confirmed:
            self._state["in_position"] = True
            return (
                "BUY",
                1.0,
                f"Volume-confirmed breakout above ${self._state['opening_range_high']:.2f}",
            )

        return "HOLD", 0.0, "Awaiting valid breakout"
