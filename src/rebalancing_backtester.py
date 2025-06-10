# src/rebalancing_backtester.py

import logging
import pandas as pd
import numpy as np
import re
import yfinance as yf
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass

# Solves the circular import for type hinting
if TYPE_CHECKING:
    from .portfolio_tracker import CryptoPortfolioTracker

@dataclass
class BacktestSuggestion:
    """A dataclass to hold a single rebalancing suggestion during a backtest."""
    symbol: str
    signal: str
    action_text: str

class RebalancingBacktester:
    """
    Simulates the performance of the dynamic rebalancing strategy over historical data.
    This version is optimized to fetch all data once and process it in memory.
    """
    def __init__(self, config: Dict[str, Any], tracker: "CryptoPortfolioTracker"):
        self.config = config
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
        self.trade_log: List[str] = []
        self.portfolio_history: Dict[pd.Timestamp, float] = {}
        self.initial_capital: float = 0.0
        self.hist_prices_df: Optional[pd.DataFrame] = None
        self.normalized_alloc: Dict[str, float] = {}

    def _reset_state(self, initial_capital: float):
        """Resets the state for a new backtest run."""
        self.initial_capital = initial_capital
        self.trade_log = []
        self.portfolio_history = {}
        self.logger.info(f"Backtester state reset. Initial capital: ${initial_capital:,.2f}")

    async def run(self, initial_capital: float = 10000.0, backtest_period: str = "3y"):
        """Runs the rebalancing backtest with an efficient, single-fetch strategy."""
        self._reset_state(initial_capital)

        target_allocation = self.config.get("target_allocation", {})
        if not target_allocation:
            self.logger.error("No 'target_allocation' found in config. Aborting backtest.")
            return

        all_symbols = list(target_allocation.keys())
        yf_tickers = [self.tracker._get_yfinance_ticker(s) for s in all_symbols]

        self.logger.info(f"Fetching {backtest_period} of historical price data for all {len(all_symbols)} assets...")
        hist_prices_df = yf.download(yf_tickers, period=backtest_period, auto_adjust=True, group_by='ticker')

        # --- ROBUST DATA CLEANING ---
        # First forward-fill gaps, then backward-fill gaps at the start.
        hist_prices_df.ffill(inplace=True)
        hist_prices_df.bfill(inplace=True)
        # --- END OF FIX ---

        valid_symbols = []
        for symbol, ticker in zip(all_symbols, yf_tickers):
            if ticker in hist_prices_df.columns and not hist_prices_df[(ticker, 'Close')].isna().all():
                valid_symbols.append(symbol)
            else:
                self.logger.warning(f"Failed to fetch any valid data for {symbol} ({ticker}). It will be excluded from this backtest.")

        if not valid_symbols:
            self.logger.error("Could not fetch valid data for any target assets. Aborting backtest.")
            return

        self.hist_prices_df = hist_prices_df[[(self.tracker._get_yfinance_ticker(s), 'Close') for s in valid_symbols]]

        self.logger.info("Pre-calculating technical indicators for the backtest period...")
        indicator_data = {}
        rebal_config = self.config.get("rebalance_technical", {})
        for symbol in valid_symbols:
            ticker = self.tracker._get_yfinance_ticker(symbol)
            df = pd.DataFrame(self.hist_prices_df[(ticker, 'Close')].copy())
            df.columns = ['Close'] # Ensure single-level column for pandas_ta

            df.ta.rsi(length=rebal_config.get("rsi_period_weekly", 14), append=True)
            df.ta.sma(length=200, append=True)
            indicator_data[symbol] = df

        active_target_alloc = {s: target_allocation[s] for s in valid_symbols}
        total_valid_pct = sum(active_target_alloc.values())
        self.normalized_alloc = {s: pct / total_valid_pct for s, pct in active_target_alloc.items()}

        # --- SAFER INITIALIZATION ---
        # Find the first date where we have price data for all assets
        first_valid_index = self.hist_prices_df.dropna().first_valid_index()
        if first_valid_index is None:
            self.logger.error("No date found with valid prices for all assets after cleaning. Aborting.")
            return

        start_prices = self.hist_prices_df.loc[first_valid_index]
        portfolio_qty = {s: (initial_capital * pct) / start_prices[(self.tracker._get_yfinance_ticker(s), 'Close')] for s, pct in self.normalized_alloc.items()}

        self.logger.info(f"Starting monthly rebalancing simulation from {first_valid_index.date()}...")
        for i in range(self.hist_prices_df.index.get_loc(first_valid_index), len(self.hist_prices_df)):
            current_date = self.hist_prices_df.index[i]
            prev_date = self.hist_prices_df.index[i-1]

            if current_date.month != prev_date.month:
                suggestions = self._get_rebalance_suggestions_from_history(current_date, portfolio_qty, indicator_data)

                for suggestion in suggestions:
                    self._execute_simulated_trade(suggestion, portfolio_qty, current_date, indicator_data)

            current_value = sum(qty * self.hist_prices_df.loc[current_date, (self.tracker._get_yfinance_ticker(symbol), 'Close')] for symbol, qty in portfolio_qty.items())
            self.portfolio_history[current_date] = current_value

        self.logger.info("Backtest run finished.")

    def _get_rebalance_suggestions_from_history(self, current_date, portfolio_qty, indicator_data):
        """A lightweight version of the rebalancing logic that uses pre-fetched data."""
        suggestions = []
        rebal_config = self.config.get("rebalance_technical", {})
        drift_threshold = rebal_config.get("allocation_drift_threshold", 0.1)
        rsi_overbought = rebal_config.get("rsi_overbought", 70)
        rsi_oversold = rebal_config.get("rsi_oversold", 30)
        price_vs_ma_above = rebal_config.get("price_vs_ma_above", 25)
        price_vs_ma_near_below = rebal_config.get("price_vs_ma_near_below", 0)
        sell_multiplier = rebal_config.get("sell_percentage_multiplier", 0.15)
        buy_multiplier = rebal_config.get("buy_amount_multiplier", 0.75)

        current_prices = {s: self.hist_prices_df.loc[current_date, (self.tracker._get_yfinance_ticker(s), 'Close')] for s in portfolio_qty}
        current_values = {s: qty * current_prices[s] for s, qty in portfolio_qty.items()}
        total_portfolio_value = sum(current_values.values())

        if total_portfolio_value == 0: return []

        for symbol in portfolio_qty.keys():
            rsi_col_name = f"RSI_{rebal_config.get('rsi_period_weekly', 14)}"
            ma_col_name = 'SMA_200'

            rsi = indicator_data[symbol].loc[current_date, rsi_col_name]
            ma_200 = indicator_data[symbol].loc[current_date, ma_col_name]
            price = current_prices[symbol]
            if pd.isna(rsi) or pd.isna(ma_200): continue

            price_vs_200w_ma = ((price - ma_200) / ma_200) * 100 if ma_200 > 0 else 0
            current_pct = current_values[symbol] / total_portfolio_value
            target_pct = self.normalized_alloc[symbol]
            drift = current_pct - target_pct

            signal = "HOLD"
            action_text = ""
            if drift > drift_threshold or (rsi > rsi_overbought and price_vs_200w_ma > price_vs_ma_above):
                signal = "SELL"
                sell_qty_pct = min(0.1, drift * sell_multiplier)
                action_text = f"Suggest SELL {sell_qty_pct * 100:.1f}% of position"
            elif drift < -drift_threshold or (rsi < rsi_oversold and price_vs_200w_ma <= price_vs_ma_near_below):
                signal = "BUY"
                underweight_usd = (target_pct - current_pct) * total_portfolio_value
                buy_value = underweight_usd * buy_multiplier
                action_text = f"Suggest BUY ${buy_value:,.2f} worth"

            if signal != "HOLD":
                suggestions.append(BacktestSuggestion(symbol, signal, action_text))
        return suggestions

    def _execute_simulated_trade(self, suggestion, portfolio_qty, date, indicator_data):
        """Executes a simulated trade based on the text suggestion."""
        symbol = suggestion.symbol
        price = self.hist_prices_df.loc[date, (self.tracker._get_yfinance_ticker(symbol), 'Close')]

        if suggestion.signal == "SELL":
            pct_match = re.search(r"([0-9]+\.?[0-9]*)%", suggestion.action_text)
            if pct_match:
                sell_pct = float(pct_match.group(1)) / 100
                trade_qty = portfolio_qty[symbol] * sell_pct
                portfolio_qty[symbol] -= trade_qty
                log_entry = f"{date.date()}: SELL {trade_qty:,.4f} {symbol} @ ${price:,.2f}"
                self.trade_log.append(log_entry)
                self.logger.info(f"SIMULATED TRADE: {log_entry}")
        elif suggestion.signal == "BUY":
            usd_match = re.search(r"\$([0-9,]+\.?[0-9]*)", suggestion.action_text)
            if usd_match:
                buy_value_usd = float(usd_match.group(1).replace(",", ""))
                if price > 0:
                    trade_qty = buy_value_usd / price
                    portfolio_qty[symbol] += trade_qty
                    log_entry = f"{date.date()}: BUY {trade_qty:,.4f} {symbol} @ ${price:,.2f}"
                    self.trade_log.append(log_entry)
                    self.logger.info(f"SIMULATED TRADE: {log_entry}")

    def generate_report(self):
        """Generates the final performance report, including Buy & Hold comparison."""
        if not self.portfolio_history:
            print("\n❌ Backtest did not run or produced no history.")
            return

        final_value = list(self.portfolio_history.values())[-1]
        strategy_return_pct = ((final_value / self.initial_capital) - 1) * 100

        first_valid_index = self.hist_prices_df.dropna().first_valid_index()
        start_prices = self.hist_prices_df.loc[first_valid_index]
        end_prices = self.hist_prices_df.iloc[-1]

        buy_and_hold_value = 0
        for symbol, pct in self.normalized_alloc.items():
            ticker = self.tracker._get_yfinance_ticker(symbol)
            start_price = start_prices.get((ticker, 'Close'), 0)
            end_price = end_prices.get((ticker, 'Close'), 0)
            if start_price > 0:
                initial_investment = self.initial_capital * pct
                shares_bought = initial_investment / start_price
                final_value_of_shares = shares_bought * end_price
                buy_and_hold_value += final_value_of_shares

        buy_and_hold_return_pct = ((buy_and_hold_value / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else 0
        vs_hold_pct = strategy_return_pct - buy_and_hold_return_pct

        print("\n" + "="*80)
        print("DYNAMIC REBALANCING STRATEGY - BACKTEST REPORT")
        print("="*80)
        print(f"Initial Capital:          ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value:    ${final_value:,.2f}")
        print("-" * 40)
        print(f"Strategy Total Return:    {strategy_return_pct:,.2f}%")
        print(f"Buy & Hold Return:        {buy_and_hold_return_pct:,.2f}%")
        print(f"Strategy Outperformance:  {vs_hold_pct:+.2f}%")
        print("-" * 40)
        print(f"Total Trades Executed:    {len(self.trade_log)}")
        print("="*80)
        print("\n--- Trade Log (First 15) ---")
        for log in self.trade_log[:15]: print(log)
        if len(self.trade_log) > 15: print(f"... and {len(self.trade_log) - 15} more trades.")
        print("="*80)
