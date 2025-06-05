import pandas as pd
import numpy as np
import talib
from binance.client import Client
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SwingTradingStrategy:
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """
        Trend Following Swing Trading Strategy
        
        Parameters:
        - EMA Fast: 12 periods
        - EMA Slow: 26 periods  
        - RSI: 14 periods (filter: 30-70 range)
        - ATR: 14 periods (for stop loss)
        - Volume MA: 20 periods
        """
        self.client = None
        if api_key and api_secret:
            self.client = Client(api_key, api_secret, testnet=testnet)
        
        # Strategy parameters
        self.ema_fast = 12
        self.ema_slow = 26
        self.rsi_period = 14
        self.atr_period = 14
        self.volume_ma_period = 20
        
        # Signal thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.atr_multiplier = 2.0  # For stop loss
        
        # Trade management
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.min_rr_ratio = 2.0     # Minimum risk/reward ratio
        
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        # EMAs
        df['EMA_fast'] = talib.EMA(df['close'], timeperiod=self.ema_fast)
        df['EMA_slow'] = talib.EMA(df['close'], timeperiod=self.ema_slow)
        
        # RSI
        df['RSI'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # ATR for stop loss
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        
        # Volume MA
        df['Volume_MA'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        
        return df
    
    def generate_signals(self, df):
        """Generate buy/sell signals"""
        df['signal'] = 0
        df['position'] = 0
        
        # EMA Crossover conditions
        df['ema_bullish'] = (df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))
        df['ema_bearish'] = (df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))
        
        # Trend confirmation (price above/below slow EMA)
        df['uptrend'] = df['close'] > df['EMA_slow']
        df['downtrend'] = df['close'] < df['EMA_slow']
        
        # Volume confirmation
        df['volume_confirm'] = df['volume'] > df['Volume_MA']
        
        # Generate signals
        buy_conditions = (
            df['ema_bullish'] &
            df['uptrend'] &
            (df['RSI'] > self.rsi_oversold) &
            (df['RSI'] < self.rsi_overbought) &
            df['volume_confirm']
        )
        
        sell_conditions = (
            df['ema_bearish'] &
            df['downtrend'] &
            (df['RSI'] < self.rsi_overbought) &
            (df['RSI'] > self.rsi_oversold) &
            df['volume_confirm']
        )
        
        df.loc[buy_conditions, 'signal'] = 1   # Buy signal
        df.loc[sell_conditions, 'signal'] = -1  # Sell signal
        
        return df
    
    def calculate_positions(self, df):
        """Calculate position sizes and stop losses"""
        df['stop_loss'] = 0.0
        df['take_profit'] = 0.0
        df['position_size'] = 0.0
        
        for i in range(len(df)):
            if df.iloc[i]['signal'] == 1:  # Buy signal
                entry_price = df.iloc[i]['close']
                atr = df.iloc[i]['ATR']
                
                # Calculate stop loss (below entry - ATR * multiplier)
                stop_loss = entry_price - (atr * self.atr_multiplier)
                
                # Calculate position size based on risk
                risk_amount = entry_price * self.risk_per_trade
                stop_distance = entry_price - stop_loss
                position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                
                # Calculate take profit (risk/reward ratio)
                take_profit = entry_price + (stop_distance * self.min_rr_ratio)
                
                df.loc[i, 'stop_loss'] = stop_loss
                df.loc[i, 'take_profit'] = take_profit
                df.loc[i, 'position_size'] = position_size
                
            elif df.iloc[i]['signal'] == -1:  # Sell signal
                entry_price = df.iloc[i]['close']
                atr = df.iloc[i]['ATR']
                
                # Calculate stop loss (above entry + ATR * multiplier)
                stop_loss = entry_price + (atr * self.atr_multiplier)
                
                # Calculate position size based on risk
                risk_amount = entry_price * self.risk_per_trade
                stop_distance = stop_loss - entry_price
                position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                
                # Calculate take profit
                take_profit = entry_price - (stop_distance * self.min_rr_ratio)
                
                df.loc[i, 'stop_loss'] = stop_loss
                df.loc[i, 'take_profit'] = take_profit
                df.loc[i, 'position_size'] = position_size
        
        return df
    
    def backtest(self, df):
        """Run backtest simulation"""
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        df = self.calculate_positions(df)
        
        # Simulate trades
        trades = []
        position = None
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            
            # Check for new signals
            if current_bar['signal'] != 0 and position is None:
                position = {
                    'entry_time': current_bar.name,
                    'entry_price': current_bar['close'],
                    'direction': 'long' if current_bar['signal'] == 1 else 'short',
                    'stop_loss': current_bar['stop_loss'],
                    'take_profit': current_bar['take_profit'],
                    'position_size': current_bar['position_size']
                }
            
            # Check exit conditions
            elif position is not None:
                exit_trade = False
                exit_reason = ""
                exit_price = current_bar['close']
                
                if position['direction'] == 'long':
                    if current_bar['low'] <= position['stop_loss']:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                        exit_price = position['stop_loss']
                    elif current_bar['high'] >= position['take_profit']:
                        exit_trade = True
                        exit_reason = "Take Profit"
                        exit_price = position['take_profit']
                    elif current_bar['signal'] == -1:
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                
                else:  # short position
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
                    # Calculate P&L
                    if position['direction'] == 'long':
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - exit_price) / position['entry_price']
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_bar.name,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl * 100,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
        
        return pd.DataFrame(trades), df
    
    def analyze_results(self, trades_df):
        """Analyze backtest results"""
        if trades_df.empty:
            return {"error": "No trades generated"}
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L statistics
        total_pnl = trades_df['pnl_pct'].sum()
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        max_win = trades_df['pnl_pct'].max()
        max_loss = trades_df['pnl_pct'].min()
        
        # Calculate cumulative returns
        trades_df['cumulative_pnl'] = trades_df['pnl_pct'].cumsum()
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'profit_factor': round(abs(avg_win * winning_trades / (avg_loss * losing_trades)), 2) if losing_trades > 0 and avg_loss != 0 else float('inf')
        }
        
        return results
    
    def plot_results(self, df, trades_df):
        """Plot backtest results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price chart with EMAs and signals
        ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7)
        ax1.plot(df.index, df['EMA_fast'], label=f'EMA {self.ema_fast}', alpha=0.8)
        ax1.plot(df.index, df['EMA_slow'], label=f'EMA {self.ema_slow}', alpha=0.8)
        
        # Plot buy/sell signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, label='Buy Signal')
        ax1.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, label='Sell Signal')
        
        ax1.set_title('Price Chart with Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cumulative P&L
        if not trades_df.empty:
            ax3.plot(range(len(trades_df)), trades_df['cumulative_pnl'], label='Cumulative P&L (%)', color='blue')
            ax3.set_title('Cumulative P&L')
            ax3.set_ylabel('P&L (%)')
            ax3.set_xlabel('Trade Number')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_live_signal(self, symbol='BTCUSDT', interval='4h', limit=100):
        """Get live trading signal"""
        if not self.client:
            return {"error": "Binance client not initialized"}
        
        # Get recent klines
        klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to proper data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators and signals
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        df = self.calculate_positions(df)
        
        # Get latest signal
        latest = df.iloc[-1]
        
        signal_info = {
            'timestamp': latest.name,
            'symbol': symbol,
            'price': latest['close'],
            'signal': latest['signal'],
            'ema_fast': latest['EMA_fast'],
            'ema_slow': latest['EMA_slow'],
            'rsi': latest['RSI'],
            'atr': latest['ATR'],
            'volume_confirm': latest['volume_confirm']
        }
        
        if latest['signal'] != 0:
            signal_info.update({
                'stop_loss': latest['stop_loss'],
                'take_profit': latest['take_profit'],
                'position_size': latest['position_size']
            })
        
        return signal_info

# Example usage and backtesting
def run_backtest_example():
    """Example of running a backtest"""
    # Create strategy instance
    strategy = SwingTradingStrategy()
    
    # For demo purposes, let's create some sample data
    # In real usage, you'd fetch this from Binance API
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='4H')
    
    # Generate sample OHLCV data (replace with real data)
    price = 50000
    data = []
    
    for date in dates:
        change = np.random.randn() * 0.02  # 2% volatility
        price = price * (1 + change)
        
        high = price * (1 + abs(np.random.randn() * 0.01))
        low = price * (1 - abs(np.random.randn() * 0.01))
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': date,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Run backtest
    trades_df, signals_df = strategy.backtest(df)
    results = strategy.analyze_results(trades_df)
    
    print("=== BACKTEST RESULTS ===")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print("\n=== RECENT TRADES ===")
    if not trades_df.empty:
        print(trades_df.tail().to_string())
    
    # Plot results
    strategy.plot_results(signals_df, trades_df)
    
    return strategy, trades_df, signals_df, results

if __name__ == "__main__":
    # Run example backtest
    strategy, trades, signals, results = run_backtest_example()
    
    # Example of getting live signal (requires API keys)
    # live_signal = strategy.get_live_signal('BTCUSDT', '4h')
    # print("Live Signal:", live_signal)
