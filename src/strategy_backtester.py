import logging
import pandas as pd
from typing import Dict, Any, Optional

from . crypto_trend_analyzer import CryptoTrendAnalyzer


class StrategyBacktester:
    """
    A modular engine for backtesting directional trading strategies.
    It now relies on an injected TrendAnalyzer instance for all data fetching.
    """

    def __init__(self, config: Dict[str, Any], analyzer):
        self.config = config
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
        self.initial_capital = 0
        self.cash = 0
        self.position_size = 0
        self.trade_log = []
        self.portfolio_value_history = []
        self.data = None

    def _reset_state(self, initial_capital: float):
        """Resets the backtester to its initial state for a new run."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size = 0
        self.trade_log = []
        self.portfolio_value_history = []
        self.data = None
        self.logger.info(f"Backtester state reset. Initial capital: ${self.initial_capital:,.2f}")

    def _execute_trade(self, date, signal: str, price: float, reason: str, size: float):
        """Executes a simulated buy or sell trade using a specified size."""
        trade_fee_pct = self.config.get("portfolio", {}).get("transaction_fee_percent", 0.1) / 100
        current_portfolio_value = self.cash + (self.position_size * price)

        if signal == "BUY" and self.cash > 0:
            # --- FIX: Calculate trade value based on strategy size ---
            trade_value = current_portfolio_value * size
            if trade_value > self.cash:
                trade_value = self.cash # Can't spend more cash than we have

            fee = trade_value * trade_fee_pct
            amount_to_invest = trade_value - fee
            if price > 0:
                quantity_to_buy = amount_to_invest / price
                self.position_size += quantity_to_buy
                self.cash -= trade_value
                log_msg = f"{date.strftime('%Y-%m-%d %H:%M')}: BUY {quantity_to_buy:,.4f} of {self.symbol} @ ${price:,.2f} ({reason})"
                self.trade_log.append(log_msg)
                self.logger.info(log_msg)

        elif signal == "SELL" and self.position_size > 0:
            # --- FIX: Calculate sell quantity based on strategy size ---
            quantity_to_sell = self.position_size * size
            if quantity_to_sell > self.position_size:
                quantity_to_sell = self.position_size

            trade_value = quantity_to_sell * price
            fee = trade_value * trade_fee_pct

            log_msg = f"{date.strftime('%Y-%m-%d %H:%M')}: SELL {quantity_to_sell:,.4f} of {self.symbol} @ ${price:,.2f} ({reason})"
            self.trade_log.append(log_msg)
            self.logger.info(log_msg)

            self.cash += (trade_value - fee)
            self.position_size -= quantity_to_sell

    async def run(self, strategy, symbol: str, initial_capital: float = 10000.0, period: str = "3y", interval: str = "1d"):
        """Runs the backtest for a given strategy and symbol."""
        self._reset_state(initial_capital)
        self.symbol = symbol
        self.strategy_name = strategy.name

        if hasattr(self.analyzer, 'set_symbol'):
            self.analyzer.set_symbol(symbol)

        self.data = await self.analyzer.fetch_crypto_data_async(symbol, period, interval)
        if self.data is None or self.data.empty:
            self.logger.error(f"Failed to fetch any data for {symbol}, cannot run backtest.")
            return

        self.logger.info(f"Starting simulation for {symbol} with strategy: {strategy.name}...")

        live_strategy_instance = strategy

        for i in range(len(self.data)):
            current_date = self.data.index[i]
            current_price = self.data['Close'].iloc[i]
            historical_data_slice = self.data.iloc[:i+1]

            signal, size, reason = await live_strategy_instance.generate_signal(historical_data_slice)

            if signal in ["BUY", "SELL"] and size > 0:
                self._execute_trade(current_date, signal, current_price, reason, size)

            current_portfolio_value = self.cash + (self.position_size * current_price)
            self.portfolio_value_history.append(current_portfolio_value)

        self.logger.info("Backtest run finished.")

    def generate_report(self):
        """Calculates and prints the performance report."""
        if not self.portfolio_value_history:
            print("No backtest data to generate a report.")
            return

        final_value = self.portfolio_value_history[-1]
        total_return_pct = (final_value / self.initial_capital - 1) * 100

        buy_hold_start_price = self.data['Close'].iloc[0]
        buy_hold_end_price = self.data['Close'].iloc[-1]
        buy_hold_return_pct = (buy_hold_end_price / buy_hold_start_price - 1) * 100

        outperformance = total_return_pct - buy_hold_return_pct

        print("\n" + "="*80)
        print(f"📈 BACKTEST PERFORMANCE REPORT: {self.symbol} ({self.data.index[0].year} - {self.data.index[-1].year})")
        print(f"Strategy: {self.strategy_name}")
        print("="*80)
        print(f"Initial Capital:         ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value:   ${final_value:,.2f}")
        print("-" * 40)
        print(f"Strategy Total Return:   {total_return_pct:,.2f}%")
        print(f"Buy & Hold Return:       {buy_hold_return_pct:,.2f}%")
        print(f"Strategy Outperformance: {outperformance:+.2f}%")
        print("-" * 40)
        print(f"Total Trades Executed:   {len(self.trade_log)}")
        print("="*80)

        print("\n--- Trade Log (First 15) ---")
        for trade in self.trade_log[:15]:
            print(trade)
        if len(self.trade_log) > 15:
            print(f"... and {len(self.trade_log) - 15} more trades.")
        print("="*80)
