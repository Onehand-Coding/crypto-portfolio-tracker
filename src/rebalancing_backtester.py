import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

from rebalancing_logic import get_backtest_rebalance_suggestions
from crypto_trend_analyzer import CryptoTrendAnalyzer

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

    def reset_state(self):
        self.initial_capital = 10000.0
        self.portfolio_value_history = []
        self.trade_log = []
        self.data = pd.DataFrame()
        self.max_drawdown = 0.0
        self.logger.info("Backtester state reset.")

    def _pre_calculate_indicators(self):
        """Pre-calculates all technical indicators for all assets at once."""
        self.logger.info("Pre-calculating all technical indicators for the backtest period...")
        long_term_settings = self.analyzer.timeframe_settings.get('long_term', {})
        swing_settings = self.analyzer.timeframe_settings.get('swing', {})
        rsi_period = self.analyzer.rsi_period

        for asset in self.config.get("target_allocation", {}).keys():
            close_col = f"{asset}_Close"
            low_col = f"{asset}_Low"
            high_col = f"{asset}_High"

            if close_col in self.data.columns:
                # Long-Term Indicators
                sma_long_len = long_term_settings.get('sma_long_window', 200)
                self.data[f'{asset}_SMA_L_long'] = ta.sma(self.data[close_col], length=sma_long_len)

                # Swing-Term Indicators
                sma_swing_len = swing_settings.get('sma_long_window', 30)
                self.data[f'{asset}_SMA_L_swing'] = ta.sma(self.data[close_col], length=sma_swing_len)

                # RSI
                self.data[f'{asset}_RSI'] = ta.rsi(self.data[close_col], length=rsi_period)

                # Pre-calculate rolling min/max for Support/Resistance
                sr_window = 30 # Matching the default window in the analyzer
                if high_col in self.data.columns and low_col in self.data.columns:
                    self.data[f'{asset}_Support'] = self.data[low_col].rolling(window=sr_window).min()
                    self.data[f'{asset}_Resistance'] = self.data[high_col].rolling(window=sr_window).max()

        self.data.ffill(inplace=True)
        self.logger.info("Indicator pre-calculation complete.")

    def _fetch_and_prepare_data(self, symbols: list, period: str) -> bool:
        self.logger.info(f"Fetching {period} of historical data for {len(symbols)} assets...")
        tickers = [f"{s.upper()}-USD" for s in symbols]
        try:
            self.data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, timeout=60)
            if self.data.empty: return False
            self.data.columns = [f"{col[1].replace('-USD', '')}_{col[0]}" for col in self.data.columns]
            self.data.ffill(inplace=True)
            self.data.bfill(inplace=True)
        except Exception as e:
            self.logger.error(f"Critical error during yfinance download: {e}", exc_info=True)
            return False

        if "BTC_Close" not in self.data.columns or self.data["BTC_Close"].isna().all():
            self.logger.error("Failed to fetch benchmark (BTC) data.")
            return False

        self._pre_calculate_indicators()
        self.logger.info("Successfully fetched and prepared historical data with indicators.")
        return True

    def run(self, initial_capital: float = 10000.0, period: str = '3y'):
        """Synchronously runs the complete backtesting simulation."""
        self.reset_state()
        self.initial_capital = initial_capital
        target_allocation = self.config.get("target_allocation", {})
        if not target_allocation:
            self.logger.error("Target allocation not found in config."); return

        symbols_to_fetch = list(target_allocation.keys())
        if "BTC" not in symbols_to_fetch: symbols_to_fetch.append("BTC")

        if not self._fetch_and_prepare_data(symbols_to_fetch, period): return

        portfolio = {asset: 0.0 for asset in target_allocation}
        portfolio['USDT'] = self.initial_capital
        peak_value = self.initial_capital

        start_date = self.data.index.min()
        end_date = self.data.index.max()
        self.logger.info(f"Starting monthly rebalancing simulation from {start_date.date()} to {end_date.date()}...")

        for date in pd.date_range(start_date, end_date, freq='MS'):
            sim_date = self.data.index.asof(date)
            if pd.isna(sim_date): continue

            suggestions_df = get_backtest_rebalance_suggestions(
                full_historical_data_with_indicators=self.data,
                portfolio_state=portfolio,
                sim_date=sim_date,
                config=self.config,
                analyzer_config=self.analyzer.analyzer_config
            )

            if not suggestions_df.empty:
                trades_to_execute = suggestions_df[suggestions_df['Signal'].isin(['BUY', 'SELL'])]
                trades_to_execute = trades_to_execute.sort_values(by=['Signal'], ascending=False)
                for _, trade in trades_to_execute.iterrows():
                    asset = trade['Symbol']
                    signal = trade['Signal']
                    price = trade['TA_Price']
                    action_value_usd = trade['action_usd_value']
                    min_trade_usd = self.config.get("portfolio", {}).get("minimum_trade_usd", 10.0)

                    # A price of zero or None means the asset likely didn't exist yet on this simulation date.
                    if price is None or price < 0.00001:
                        self.logger.warning(f"SIM: {sim_date.date()}: Skipping {signal} for {asset} due to missing or invalid price data (${price}).")
                        continue

                    if action_value_usd < min_trade_usd: continue

                    if signal == 'SELL' and portfolio.get(asset, 0) > 0 and price > 0:
                        sell_quantity = min(action_value_usd / price, portfolio[asset])
                        portfolio[asset] -= sell_quantity
                        portfolio['USDT'] += sell_quantity * price
                        self.trade_log.append(f"SIM: {sim_date.date()}: SELL {sell_quantity:.6f} {asset} @ ${price:.2f}")

                    elif signal == 'BUY' and portfolio['USDT'] >= action_value_usd and price > 0:
                        buy_quantity = action_value_usd / price
                        portfolio[asset] += buy_quantity
                        portfolio['USDT'] -= action_value_usd
                        self.trade_log.append(f"SIM: {sim_date.date()}: BUY {buy_quantity:.6f} {asset} @ ${price:.2f}")

            current_portfolio_value = portfolio['USDT']
            for asset, quantity in portfolio.items():
                if asset != 'USDT':
                    price_col = f"{asset}_Close"
                    if price_col in self.data.columns:
                        current_portfolio_value += quantity * self.data.loc[sim_date, price_col]

            peak_value = max(peak_value, current_portfolio_value)
            drawdown = (current_portfolio_value - peak_value) / peak_value if peak_value > 0 else 0
            self.max_drawdown = min(self.max_drawdown, drawdown)
            self.portfolio_value_history.append({'date': sim_date, 'value': current_portfolio_value})

        self.logger.info(f"Backtest completed.")

    def generate_report(self):
        """Generates and prints the final performance report."""
        if not self.portfolio_value_history:
            print("\n--- No backtest data to generate a report ---")
            return

        results_df = pd.DataFrame(self.portfolio_value_history).set_index('date')
        final_value = results_df['value'].iloc[-1]
        strategy_return = (final_value - self.initial_capital) / self.initial_capital

        target_allocation = self.config.get("target_allocation", {})
        buy_hold_returns = pd.Series(0.0, index=self.data.index)
        for asset, target_pct in target_allocation.items():
            col = f'{asset}_Close'
            if col in self.data.columns:
                asset_returns = self.data[col].pct_change().fillna(0)
                buy_hold_returns += asset_returns * target_pct

        buy_hold_equity = (1 + buy_hold_returns).cumprod() * self.initial_capital
        buy_hold_return = (buy_hold_equity.iloc[-1] - self.initial_capital) / self.initial_capital

        if len(results_df) > 1:
            strategy_returns_pct = results_df['value'].pct_change().fillna(0)
            strategy_volatility = strategy_returns_pct.std() * np.sqrt(252)
            sharpe_ratio = (strategy_returns_pct.mean() * 252) / strategy_volatility if strategy_volatility > 0 else 0
        else:
            strategy_volatility = 0; sharpe_ratio = 0

        print("\n" + "="*80)
        print("DYNAMIC REBALANCING STRATEGY - BACKTEST REPORT")
        print("="*80)
        print(f"Initial Capital:         ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value:   ${final_value:,.2f}")
        print("----------------------------------------")
        print(f"Strategy Total Return:   {strategy_return:,.2%}")
        print(f"Buy & Hold Return:       {buy_hold_return:,.2%}")
        print(f"Strategy Outperformance: {strategy_return - buy_hold_return:+.2%}")
        print(f"Maximum Drawdown:        {self.max_drawdown:,.2%}")
        print(f"Annualized Volatility:   {strategy_volatility:,.2%}")
        print(f"Sharpe Ratio:            {sharpe_ratio:.2f}")
        print("----------------------------------------")
        print(f"Total Trades Executed:   {len(self.trade_log)}")
        print("="*80)
        print("\n--- Recent Trade Log (Last 15) ---")
        for log_entry in self.trade_log[-15:]:
            print(log_entry)
        if len(self.trade_log) > 15:
            print(f"... (showing last 15 of {len(self.trade_log)} total trades)")
        print("="*80)
