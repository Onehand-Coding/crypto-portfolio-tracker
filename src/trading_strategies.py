import pandas as pd
import pandas_ta as ta
from typing import Tuple, Optional, List, Dict, Any

from crypto_trend_analyzer import CryptoTrendAnalyzer, TrendCondition


class Strategy:
    """A base class for all strategies, now with a consistent configuration method."""
    valid_intervals: List[str] = ['1d']
    strategy_type: str = "general"

    def __init__(self, analyzer: Optional[CryptoTrendAnalyzer] = None, state: Optional[Dict[str, Any]] = None, **kwargs):
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

    async def generate_signal(self, data: pd.DataFrame) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Returns a signal, a trade size (as a fraction of capital), and a reason.
        """
        raise NotImplementedError


class GoldenCrossStrategy(Strategy):
    """A simple trend-following strategy based on MA crossovers."""
    valid_intervals = ['1d']
    strategy_type = 'swing'

    def __init__(self, fast_period: int = 50, slow_period: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self):
        return f"Golden Cross ({self.fast_period}/{self.slow_period})"

    async def generate_signal(self, data: pd.DataFrame) -> Tuple[Optional[str], float, Optional[str]]:
        if len(data) < self.slow_period + 1: return "HOLD", 0.0, "Insufficient data"
        df = data.copy()
        df.ta.sma(length=self.fast_period, append=True)
        df.ta.sma(length=self.slow_period, append=True)
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        fast_ma_col = f"SMA_{self.fast_period}"
        slow_ma_col = f"SMA_{self.slow_period}"
        if previous[fast_ma_col] <= previous[slow_ma_col] and latest[fast_ma_col] > latest[slow_ma_col]:
            return "BUY", 1.0, f"Golden Cross: {self.fast_period}d > {self.slow_period}d"
        elif previous[fast_ma_col] >= previous[slow_ma_col] and latest[fast_ma_col] < latest[slow_ma_col]:
            return "SELL", 1.0, f"Death Cross: {self.fast_period}d < {self.slow_period}d"
        return "HOLD", 0.0, "No crossover"


class MultiTimeframeMomentumStrategy(Strategy):
    """
    A sophisticated strategy that requires trend confirmation from multiple timeframes
    before entering and uses a trailing stop-loss for exits.
    """
    valid_intervals = ['1d']
    strategy_type = 'swing'

    def __init__(self, analyzer: CryptoTrendAnalyzer, state: Optional[Dict[str, Any]] = None, trailing_stop_pct: float = 0.20, **kwargs):
        super().__init__(analyzer=analyzer, state=state, **kwargs)
        self.fast_period = 50
        self.slow_period = 200
        self.rsi_overbought = 70
        self.trailing_stop_pct = trailing_stop_pct
        self._state.setdefault('in_position', False)
        self._state.setdefault('highest_price_since_entry', 0)

    @property
    def name(self):
        return f"MTF Momentum w/ {self.trailing_stop_pct:.0%} T-Stop"

    @classmethod
    def get_user_params(cls) -> Dict[str, Any]:
        """Class method to get user-defined parameters interactively."""
        try:
            pct_str = input(f"Enter trailing stop percentage (e.g., 20 for 20%, default: 20): ").strip()
            trailing_pct = float(pct_str) / 100 if pct_str else 0.20
        except ValueError:
            print("Invalid input. Using default 20%.")
            trailing_pct = 0.20
        return {"trailing_stop_pct": trailing_pct}

    async def generate_signal(self, data: pd.DataFrame) -> Tuple[Optional[str], float, Optional[str]]:
        symbol = self.analyzer.symbol
        current_price = data['Close'].iloc[-1]

        # --- Exit Logic (only applies if we are in a position) ---
        if self._state['in_position']:
            self._state['highest_price_since_entry'] = max(self._state['highest_price_since_entry'], current_price)
            stop_price = self._state['highest_price_since_entry'] * (1 - self.trailing_stop_pct)
            if current_price < stop_price:
                self._state['in_position'] = False
                self._state['highest_price_since_entry'] = 0
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

        daily_fast_ma = daily_data[f'SMA_{self.fast_period}'].iloc[-1]
        daily_slow_ma = daily_data[f'SMA_{self.slow_period}'].iloc[-1]
        daily_rsi = daily_data['RSI_14'].iloc[-1]

        if daily_fast_ma < daily_slow_ma or daily_rsi >= self.rsi_overbought:
            return "HOLD", 0.0, "Daily trend not bullish or RSI overbought"

        weekly_data = await self.analyzer.fetch_crypto_data_async(symbol, period="4y", interval="1wk")
        if weekly_data is None or len(weekly_data) < self.slow_period:
            return "HOLD", 0.0, "Insufficient weekly data"

        weekly_data.ta.sma(length=self.fast_period, append=True)
        weekly_data.ta.sma(length=self.slow_period, append=True)
        weekly_fast_ma = weekly_data[f'SMA_{self.fast_period}'].iloc[-1]
        weekly_slow_ma = weekly_data[f'SMA_{self.slow_period}'].iloc[-1]

        if weekly_fast_ma < weekly_slow_ma:
            return "HOLD", 0.0, "Weekly trend not bullish"

        self._state['in_position'] = True
        self._state['highest_price_since_entry'] = current_price
        return "BUY", 1.0, "Daily & Weekly trends align"


class OpeningRangeBreakoutStrategy(Strategy):
    """
    A classic day trading strategy that trades breakouts from the session's opening range.
    """
    valid_intervals = ['15m', '1h']
    strategy_type = 'day'

    def __init__(self, analyzer: CryptoTrendAnalyzer, state: Optional[Dict[str, Any]] = None, range_duration: str = "1h", risk_per_trade_pct: float = 0.20, **kwargs):
        super().__init__(analyzer=analyzer, state=state, **kwargs)
        self.range_duration = range_duration.upper()
        self.risk_per_trade_pct = risk_per_trade_pct
        self._state.setdefault('last_day_processed', None)
        self._state.setdefault('trade_taken_today', False)
        self._state.setdefault('in_position', False)
        self._state.setdefault('stop_loss', 0)

    @property
    def name(self):
        return f"ORB ({self.range_duration}, {self.risk_per_trade_pct:.1%} Risk)"

    @classmethod
    def get_user_params(cls) -> Dict[str, Any]:
        """Get user-defined parameters for the ORB strategy."""
        range_duration = input("Enter opening range duration (e.g., '1h', '30m', default: '1h'): ").strip() or "1h"
        try:
            risk_pct_str = input("Enter capital to risk per trade (e.g., 20 for 20%, default: 20): ").strip()
            risk_per_trade_pct = float(risk_pct_str) / 100 if risk_pct_str else 0.20
        except ValueError:
            print("Invalid percentage. Using default 20%.")
            risk_per_trade_pct = 0.20
        return {"range_duration": range_duration, "risk_per_trade_pct": risk_per_trade_pct}

    async def generate_signal(self, data: pd.DataFrame) -> Tuple[Optional[str], float, Optional[str]]:
        if data.empty:
            return "HOLD", 0.0, "No data available"

        latest_bar = data.iloc[-1]
        current_price = latest_bar['Close']
        current_time = pd.to_datetime(latest_bar.name, utc=True)
        current_day_str = str(current_time.date())

        if self._state.get('last_day_processed') != current_day_str:
            self._state.update({
                'last_day_processed': current_day_str,
                'in_position': False, 'trade_taken_today': False,
                'opening_range_high': 0, 'opening_range_low': 0, 'stop_loss': 0
            })

        if self._state['in_position'] and current_price < self._state['stop_loss']:
            self._state['in_position'] = False
            return "SELL", 1.0, f"Stop loss hit at ${self._state['stop_loss']:.2f}"

        if self._state['in_position'] and current_time.hour >= 23:
            self._state['in_position'] = False
            return "SELL", 1.0, "End-of-day position close"

        if self._state['trade_taken_today']:
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

        opening_range_high = opening_range_bars['High'].max()
        opening_range_low = opening_range_bars['Low'].min()

        if current_price > opening_range_high:
            self._state['in_position'] = True
            self._state['trade_taken_today'] = True
            self._state['stop_loss'] = opening_range_low
            return "BUY", self.risk_per_trade_pct, f"Breakout above {self.range_duration} range high ${opening_range_high:.2f}"

        return "HOLD", 0.0, "Awaiting breakout."#
