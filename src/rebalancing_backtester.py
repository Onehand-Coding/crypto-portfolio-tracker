import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from . rebalancing_logic import get_backtest_rebalance_suggestions
from . crypto_trend_analyzer import CryptoTrendAnalyzer


class RebalancingBacktester:
    """
    Performs a 1:1 backtest of the live rebalancing strategy.
    It pre-calculates all indicators for performance and then calls the central logic engine.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analyzer = CryptoTrendAnalyzer(config=self.config)
        self.reset_state()

    def reset_state(self, initial_capital: float = 10000.0):
        """
        Resets the backtester's state for a new run.
        Accepts initial_capital and sets up instance variables.
        """
        self.initial_capital = initial_capital
        self.portfolio_value_history = []
        self.trade_log = []
        self.data = pd.DataFrame()
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        self.executed_allocation = {}
        self.logger.info("Backtester state reset.")

    def _pre_calculate_indicators(self, assets_to_calculate: List[str]):
        """
        Pre-calculates all technical indicators for the assets that have valid data.
        """
        self.logger.info("Pre-calculating all technical indicators for the backtest period...")
        long_term_settings = self.analyzer.timeframe_settings.get('long_term', {})
        swing_settings = self.analyzer.timeframe_settings.get('swing', {})
        rsi_period = self.analyzer.rsi_period

        for asset in assets_to_calculate:
            close_col = f"{asset}_Close"
            low_col = f"{asset}_Low"
            high_col = f"{asset}_High"

            if close_col in self.data.columns:
                sma_long_len = long_term_settings.get('sma_long_window', 200)
                self.data[f'{asset}_SMA_L_long'] = ta.sma(self.data[close_col], length=sma_long_len)
                sma_swing_len = swing_settings.get('sma_long_window', 30)
                self.data[f'{asset}_SMA_L_swing'] = ta.sma(self.data[close_col], length=sma_swing_len)
                self.data[f'{asset}_RSI'] = ta.rsi(self.data[close_col], length=rsi_period)
                sr_window = 30
                if high_col in self.data.columns and low_col in self.data.columns:
                    self.data[f'{asset}_Support'] = self.data[low_col].rolling(window=sr_window).min()
                    self.data[f'{asset}_Resistance'] = self.data[high_col].rolling(window=sr_window).max()

        self.data.ffill(inplace=True)
        self.logger.info("Indicator pre-calculation complete.")

    def _fetch_and_prepare_data(self, symbols: list, period: str) -> bool:
        """
        Fetches and prepares historical price data from yfinance.
        """
        self.logger.info(f"Fetching {period} of historical data for {len(symbols)} assets...")
        tickers = [f"{s.upper()}-USD" for s in symbols]
        try:
            self.data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, timeout=60)
            if self.data.empty:
                return False

            if len(symbols) > 1:
                self.data.columns = [f"{col[1].replace('-USD', '')}_{col[0]}" for col in self.data.columns]
            else:
                self.data.columns = [f"{symbols[0]}_{col}" for col in self.data.columns]

            self.data.ffill(inplace=True)
            self.data.bfill(inplace=True)
        except Exception as e:
            self.logger.error(f"Critical error during yfinance download: {e}", exc_info=True)
            return False

        if "BTC_Close" not in self.data.columns or self.data["BTC_Close"].isna().all():
            self.logger.warning("Failed to fetch benchmark (BTC) data. Buy & Hold comparison might be inaccurate.")

        self.logger.info("Successfully fetched and prepared historical data.")
        return True

    def run(self, initial_capital: float, period: str):
        """
        The main orchestration method for the backtest. This is self-contained and runs the whole process.
        """
        self.reset_state(initial_capital)

        original_assets = list(self.config.get("target_allocation", {}).keys())

        if not self._fetch_and_prepare_data(symbols=original_assets, period=period):
            self.logger.error("Failed to fetch any historical data. Aborting backtest.")
            return

        assets_to_backtest = []
        for asset in original_assets:
            close_col = f"{asset}_Close"
            if close_col in self.data.columns and not self.data[close_col].dropna().empty:
                assets_to_backtest.append(asset)
            else:
                self.logger.warning(
                    f"Excluding '{asset}' from this backtest due to missing historical data for the '{period}' period.")

        if not assets_to_backtest:
            self.logger.error("No valid historical data for any target assets for the selected period. Aborting backtest.")
            return

        original_target_allocation = self.config.get("target_allocation", {})
        adjusted_target_allocation = {asset: original_target_allocation[asset] for asset in assets_to_backtest}
        total_adj_alloc = sum(adjusted_target_allocation.values())

        if total_adj_alloc > 0:
            self.executed_allocation = {asset: alloc / total_adj_alloc for asset, alloc in adjusted_target_allocation.items()}
        else:
            self.executed_allocation = {asset: 1.0 / len(assets_to_backtest) for asset in assets_to_backtest}

        self.logger.info(f"Running backtest with adjusted allocation for {len(assets_to_backtest)} assets.")

        self._pre_calculate_indicators(assets_to_backtest)
        self.run_simulation(self.executed_allocation)
        self.generate_report()

    def run_simulation(self, target_allocation: Dict[str, float]):
        """
        Executes the core simulation logic month by month.
        """
        self.logger.info(f"Starting monthly rebalancing simulation from {self.data.index[0].date()} to {self.data.index[-1].date()}...")

        portfolio = {'USDT': self.initial_capital}
        for asset in target_allocation.keys():
            portfolio[asset] = 0.0

        # --- FIX: Define a minimum price threshold to avoid dust trades ---
        min_price_threshold = 0.0001

        rebalance_dates = self.data.resample('MS').first().index

        for date in rebalance_dates:
            if date not in self.data.index:
                continue

            current_prices = {asset: self.data.loc[date, f"{asset}_Close"] for asset in target_allocation.keys() if f"{asset}_Close" in self.data.columns}

            current_value = portfolio['USDT']
            for asset, quantity in portfolio.items():
                if asset != 'USDT':
                    current_value += quantity * current_prices.get(asset, 0)

            if np.isnan(current_value): current_value = self.initial_capital

            self.portfolio_value_history.append({'date': date, 'value': current_value})

            self.peak_value = max(self.peak_value, current_value)
            drawdown = (self.peak_value - current_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)

            suggestions = get_backtest_rebalance_suggestions(
                full_historical_data_with_indicators=self.data,
                portfolio_state=portfolio,
                sim_date=date,
                config=self.config,
                analyzer_config=self.config.get('trend_analyzer', {})
            )

            if not suggestions.empty:
                for index, suggestion_row in suggestions.iterrows():
                    asset = suggestion_row['Symbol']
                    action = suggestion_row['Signal']
                    amount_usd = suggestion_row['action_usd_value']
                    price = current_prices.get(asset)

                    # --- FIX: Use the min_price_threshold in the check ---
                    if price is None or price < min_price_threshold or np.isnan(price):
                        self.logger.warning(f"SIM: {date.date()}: Skipping {action} for {asset} due to invalid price (${price:.6f}).")
                        continue

                    quantity = amount_usd / price

                    if action == 'BUY' and portfolio['USDT'] >= amount_usd:
                        portfolio['USDT'] -= amount_usd
                        portfolio[asset] = portfolio.get(asset, 0) + quantity
                        self.trade_log.append(f"SIM: {date.date()}: BUY {quantity:.6f} {asset} @ ${price:,.2f}")
                    elif action == 'SELL' and portfolio.get(asset, 0) >= quantity:
                        portfolio[asset] -= quantity
                        portfolio['USDT'] += amount_usd
                        self.trade_log.append(f"SIM: {date.date()}: SELL {quantity:.6f} {asset} @ ${price:,.2f}")

        # Final portfolio value update
        last_date = self.data.index[-1]
        last_prices = {asset: self.data.loc[last_date, f"{asset}_Close"] for asset in target_allocation.keys() if f"{asset}_Close" in self.data.columns}
        final_value = portfolio['USDT']
        for asset, quantity in portfolio.items():
            if asset != 'USDT':
                final_value += quantity * last_prices.get(asset, 0)
        self.portfolio_value_history.append({'date': last_date, 'value': final_value})

        self.logger.info("Backtest completed.")

    def generate_report(self):
        """
        Generates and prints the final performance report.
        """
        if not self.portfolio_value_history or len(self.portfolio_value_history) < 2:
            print("\n--- Not enough backtest data to generate a report ---")
            return

        results_df = pd.DataFrame(self.portfolio_value_history).set_index('date').dropna(subset=['value'])
        if results_df.empty:
            print("\n--- Portfolio value calculation resulted in no valid data. Cannot generate report. ---")
            return

        final_value = results_df['value'].iloc[-1]
        strategy_return = (final_value - self.initial_capital) / self.initial_capital

        buy_hold_returns = pd.Series(0.0, index=self.data.index)
        for asset, target_pct in self.executed_allocation.items():
            col = f'{asset}_Close'
            if col in self.data.columns and not self.data[col].isnull().all():
                asset_returns = self.data[col].pct_change().fillna(0)
                buy_hold_returns += asset_returns * target_pct

        buy_hold_equity = (1 + buy_hold_returns).cumprod() * self.initial_capital
        buy_hold_return = (buy_hold_equity.iloc[-1] - self.initial_capital) / self.initial_capital if not buy_hold_equity.empty else 0

        strategy_returns_pct = results_df['value'].pct_change().fillna(0)
        strategy_volatility = strategy_returns_pct.std() * np.sqrt(252)
        sharpe_ratio = (strategy_returns_pct.mean() * 252) / strategy_volatility if strategy_volatility > 0 else 0

        print("\n" + "=" * 80)
        print("DYNAMIC REBALANCING STRATEGY - BACKTEST REPORT")
        print("=" * 80)
        print(f"Initial Capital:         ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value:   ${final_value:,.2f}")
        print("----------------------------------------")
        print(f"Strategy Total Return:   {strategy_return:,.2%}")
        print(f"Buy & Hold Return:       {buy_hold_return:,.2%}")
        print(f"Strategy Outperformance: {strategy_return - buy_hold_return:+.2%}")
        print(f"Maximum Drawdown:        {-self.max_drawdown:,.2%}")
        print(f"Annualized Volatility:   {strategy_volatility:,.2%}")
        print(f"Sharpe Ratio:            {sharpe_ratio:.2f}")
        print("----------------------------------------")
        print(f"Total Trades Executed:   {len(self.trade_log)}")
        print("=" * 80)
        print("\n--- Recent Trade Log (Last 15) ---")
        for log_entry in self.trade_log[-15:]:
            print(log_entry)
        if len(self.trade_log) > 15:
            print(f"... (showing last 15 of {len(self.trade_log)} total trades)")
        print("=" * 80)
