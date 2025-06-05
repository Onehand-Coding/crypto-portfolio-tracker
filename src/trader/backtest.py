import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveBacktester:
    def __init__(self):
        """
        Complete backtesting system for swing trading strategy
        """
        # Strategy parameters
        self.ema_fast = 12
        self.ema_slow = 26
        self.rsi_period = 14
        self.atr_period = 14
        self.volume_ma_period = 20
        
        # Signal thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.atr_multiplier = 2.0
        
        # Risk management
        self.risk_per_trade = 0.02  # 2%
        self.min_rr_ratio = 2.0
        self.max_drawdown_limit = 0.15  # 15%
        
        # Trading costs
        self.commission = 0.001  # 0.1% per trade
        self.slippage = 0.0005   # 0.05% slippage
    
    def fetch_binance_data(self, symbol='BTCUSDT', interval='4h', start_date='2022-01-01', end_date=None):
        """
        Fetch historical data from Binance API (no API key required for historical data)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        url = "https://api.binance.com/api/v3/klines"
        
        all_data = []
        current_start = start_ts
        
        print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        
        while current_start < end_ts:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': min(current_start + (1000 * self.interval_to_ms(interval)), end_ts),
                'limit': 1000
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                current_start = data[-1][6] + 1  # Next start time
                
                print(f"Fetched {len(data)} candles, total: {len(all_data)}")
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[numeric_columns]  # Keep only OHLCV
        
        print(f"Successfully fetched {len(df)} candles")
        return df
    
    def interval_to_ms(self, interval):
        """Convert interval string to milliseconds"""
        intervals = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return intervals.get(interval, 4 * 60 * 60 * 1000)
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        df = df.copy()
        
        # EMAs
        df['EMA_fast'] = talib.EMA(df['close'], timeperiod=self.ema_fast)
        df['EMA_slow'] = talib.EMA(df['close'], timeperiod=self.ema_slow)
        
        # RSI
        df['RSI'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # ATR
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        
        # Volume MA
        df['Volume_MA'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        
        # Additional indicators for analysis
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals"""
        df = df.copy()
        df['signal'] = 0
        
        # EMA crossover conditions
        df['ema_bullish'] = (df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))
        df['ema_bearish'] = (df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))
        
        # Trend confirmation
        df['uptrend'] = df['close'] > df['EMA_slow']
        df['downtrend'] = df['close'] < df['EMA_slow']
        
        # Volume confirmation
        df['volume_confirm'] = df['volume'] > df['Volume_MA']
        
        # RSI filter
        df['rsi_neutral'] = (df['RSI'] > self.rsi_oversold) & (df['RSI'] < self.rsi_overbought)
        
        # Generate signals
        buy_conditions = (
            df['ema_bullish'] &
            df['uptrend'] &
            df['rsi_neutral'] &
            df['volume_confirm']
        )
        
        sell_conditions = (
            df['ema_bearish'] &
            df['downtrend'] &
            df['rsi_neutral'] &
            df['volume_confirm']
        )
        
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        return df
    
    def backtest_strategy(self, df, initial_capital=10000):
        """
        Comprehensive backtesting with realistic trading conditions
        """
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        
        # Initialize tracking variables
        capital = initial_capital
        position = None
        trades = []
        equity_curve = [initial_capital]
        max_equity = initial_capital
        max_drawdown = 0
        
        print(f"Starting backtest with ${initial_capital:,.2f}")
        print(f"Found {len(df[df['signal'] != 0])} signals to analyze")
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            current_equity = capital
            
            # Update max equity and drawdown
            if current_equity > max_equity:
                max_equity = current_equity
            
            current_drawdown = (max_equity - current_equity) / max_equity
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
            
            # Check for new signals when no position
            if current_bar['signal'] != 0 and position is None and not pd.isna(current_bar['ATR']):
                
                # Calculate position size based on risk
                entry_price = current_bar['close']
                atr = current_bar['ATR']
                
                if current_bar['signal'] == 1:  # Long
                    stop_loss = entry_price - (atr * self.atr_multiplier)
                    take_profit = entry_price + (atr * self.atr_multiplier * self.min_rr_ratio)
                else:  # Short
                    stop_loss = entry_price + (atr * self.atr_multiplier)
                    take_profit = entry_price - (atr * self.atr_multiplier * self.min_rr_ratio)
                
                # Calculate position size
                risk_amount = capital * self.risk_per_trade
                stop_distance = abs(entry_price - stop_loss)
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    position_value = position_size * entry_price
                    
                    # Check if we have enough capital
                    if position_value <= capital * 0.95:  # Use max 95% of capital
                        
                        # Apply slippage and commission
                        actual_entry = entry_price * (1 + self.slippage if current_bar['signal'] == 1 else 1 - self.slippage)
                        commission_cost = position_value * self.commission
                        
                        position = {
                            'entry_time': current_bar.name,
                            'entry_price': actual_entry,
                            'direction': 'long' if current_bar['signal'] == 1 else 'short',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': position_size,
                            'commission_paid': commission_cost
                        }
                        
                        capital -= commission_cost
            
            # Check exit conditions
            elif position is not None:
                exit_trade = False
                exit_reason = ""
                exit_price = current_bar['close']
                
                if position['direction'] == 'long':
                    # Check stop loss
                    if current_bar['low'] <= position['stop_loss']:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                        exit_price = position['stop_loss']
                    # Check take profit
                    elif current_bar['high'] >= position['take_profit']:
                        exit_trade = True
                        exit_reason = "Take Profit"
                        exit_price = position['take_profit']
                    # Check signal reversal
                    elif current_bar['signal'] == -1:
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                
                else:  # Short position
                    if current_bar['high'] >= position['stop_loss']:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                        exit_price = position['stop_loss']
                    elif current_bar['low'] <= position['take_profit']:
                        exit_trade = True
                        exit_reason = "Take Profit"
                        exit_price = position['take_profit']
                    elif current_bar['signal'] == 1:
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                
                if exit_trade:
                    # Apply slippage and commission
                    actual_exit = exit_price * (1 - self.slippage if position['direction'] == 'long' else 1 + self.slippage)
                    commission_cost = position['position_size'] * actual_exit * self.commission
                    
                    # Calculate P&L
                    if position['direction'] == 'long':
                        pnl = position['position_size'] * (actual_exit - position['entry_price'])
                    else:
                        pnl = position['position_size'] * (position['entry_price'] - actual_exit)
                    
                    # Update capital
                    capital += pnl - commission_cost
                    
                    # Record trade
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_bar.name,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': actual_exit,
                        'position_size': position['position_size'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / (position['position_size'] * position['entry_price'])) * 100,
                        'commission_total': position['commission_paid'] + commission_cost,
                        'exit_reason': exit_reason,
                        'duration': (current_bar.name - position['entry_time']).total_seconds() / 3600  # hours
                    }
                    
                    trades.append(trade_record)
                    position = None
            
            equity_curve.append(capital)
        
        # Create results summary
        trades_df = pd.DataFrame(trades)
        results = self.analyze_performance(trades_df, initial_capital, max_drawdown)
        
        return trades_df, results, equity_curve, df
    
    def analyze_performance(self, trades_df, initial_capital, max_drawdown):
        """Comprehensive performance analysis"""
        if trades_df.empty:
            return {"error": "No trades executed"}
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = (total_pnl / initial_capital) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        # Risk metrics
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = trades_df['pnl_pct'] / 100
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Trading frequency
        avg_trade_duration = trades_df['duration'].mean()
        
        # Commission analysis
        total_commission = trades_df['commission_total'].sum()
        commission_pct = (total_commission / initial_capital) * 100
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_return': round(total_return, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'avg_trade_duration_hours': round(avg_trade_duration, 1),
            'total_commission': round(total_commission, 2),
            'commission_pct': round(commission_pct, 2)
        }
        
        return results
    
    def plot_comprehensive_results(self, df, trades_df, equity_curve, results):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price chart with signals and trades
        ax1 = plt.subplot(3, 2, (1, 2))
        ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7, linewidth=1)
        ax1.plot(df.index, df['EMA_fast'], label=f'EMA {self.ema_fast}', alpha=0.8)
        ax1.plot(df.index, df['EMA_slow'], label=f'EMA {self.ema_slow}', alpha=0.8)
        
        # Plot trade entries and exits
        if not trades_df.empty:
            for _, trade in trades_df.iterrows():
                color = 'green' if trade['direction'] == 'long' else 'red'
                ax1.scatter(trade['entry_time'], trade['entry_price'], color=color, marker='^' if trade['direction'] == 'long' else 'v', s=100, alpha=0.8)
                ax1.scatter(trade['exit_time'], trade['exit_price'], color='black', marker='x', s=80, alpha=0.8)
        
        ax1.set_title('Price Chart with Trading Signals', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Equity curve
        ax2 = plt.subplot(3, 2, 3)
        equity_dates = df.index[:len(equity_curve)]
        ax2.plot(equity_dates, equity_curve, label='Portfolio Value', color='blue', linewidth=2)
        ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade P&L distribution
        ax3 = plt.subplot(3, 2, 4)
        if not trades_df.empty:
            ax3.hist(trades_df['pnl'], bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            ax3.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('P&L ($)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. RSI
        ax4 = plt.subplot(3, 2, 5)
        ax4.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1)
        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        ax4.set_title('RSI Indicator', fontsize=12, fontweight='bold')
        ax4.set_ylabel('RSI')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance summary table
        ax5 = plt.subplot(3, 2, 6)
        ax5.axis('off')
        
        # Create performance summary text
        summary_text = f"""
BACKTEST RESULTS SUMMARY

Total Trades: {results['total_trades']}
Win Rate: {results['win_rate']}%
Total Return: {results['total_return']}%

Profit Factor: {results['profit_factor']}
Max Drawdown: {results['max_drawdown']}%
Sharpe Ratio: {results['sharpe_ratio']}

Avg Win: ${results['avg_win']:.2f}
Avg Loss: ${results['avg_loss']:.2f}
Max Win: ${results['max_win']:.2f}
Max Loss: ${results['max_loss']:.2f}

Avg Trade Duration: {results['avg_trade_duration_hours']:.1f}h
Total Commission: ${results['total_commission']:.2f}
Commission %: {results['commission_pct']}%
        """
        
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_backtest(self, symbol='BTCUSDT', start_date='2022-01-01', end_date=None, initial_capital=10000):
        """Run complete backtesting process"""
        print("="*60)
        print(f"COMPREHENSIVE BACKTEST: {symbol}")
        print("="*60)
        
        # Fetch data
        df = self.fetch_binance_data(symbol, '4h', start_date, end_date)
        
        if df.empty:
            print("No data fetched. Please check symbol and dates.")
            return None
        
        # Run backtest
        trades_df, results, equity_curve, df_with_signals = self.backtest_strategy(df, initial_capital)
        
        # Display results
        print("\n" + "="*40)
        print("BACKTEST RESULTS")
        print("="*40)
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                if 'pct' in key or 'rate' in key or 'ratio' in key:
                    print(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value:,.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Show recent trades
        if not trades_df.empty:
            print(f"\nRECENT TRADES (Last 10):")
            print("-"*40)
            recent_trades = trades_df.tail(10)[['entry_time', 'direction', 'pnl', 'pnl_pct', 'exit_reason']].copy()
            recent_trades['entry_time'] = recent_trades['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
            print(recent_trades.to_string(index=False))
        
        # Plot results
        self.plot_comprehensive_results(df_with_signals, trades_df, equity_curve, results)
        
        return {
            'trades': trades_df,
            'results': results,
            'equity_curve': equity_curve,
            'data': df_with_signals
        }

# Quick execution function
def quick_backtest(symbol='BTCUSDT', start_date='2023-01-01', capital=10000):
    """Quick backtest function"""
    backtester = ComprehensiveBacktester()
    return backtester.run_comprehensive_backtest(symbol, start_date, initial_capital=capital)

# Example usage
if __name__ == "__main__":
    # Run backtest on BTCUSDT
    print("Starting comprehensive backtest...")
    
    backtest_results = quick_backtest(
        symbol='BTCUSDT',
        start_date='2023-01-01',
        capital=10000
    )
    
    print("\nBacktest completed! Check the charts and results above.")
