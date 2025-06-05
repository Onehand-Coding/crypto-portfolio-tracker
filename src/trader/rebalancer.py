#!/usr/bin/env python3
"""
Crypto Portfolio Rebalancing Automation Script
Automatically fetches portfolio data and provides rebalancing suggestions
based on your 8-coin core portfolio strategy.
"""

import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from datetime import datetime
import os
from binance.client import Client
import ta  # Technical Analysis library

@dataclass
class CoinTarget:
    symbol: str
    name: str
    target_percentage: float

@dataclass
class PortfolioAnalysis:
    symbol: str
    current_value: float
    current_percentage: float
    target_percentage: float
    deviation: float
    rsi: Optional[float]
    price_vs_ma200: Optional[float]
    signal: str
    recommendation: str
    action_percentage: float  # % of position to buy/sell
    action_usd_value: float   # USD value to buy/sell
    action_coin_amount: float # Coin amount to buy/sell
    current_coin_amount: float # Current holdings in coins

class CryptoRebalancer:
    def __init__(self, binance_api_key: str = None, binance_secret: str = None):
        """
        Initialize the rebalancer with optional Binance API credentials.
        If not provided, will use demo mode with manual input.
        """
        self.binance_client = None
        if binance_api_key and binance_secret:
            try:
                self.binance_client = Client(binance_api_key, binance_secret)
                print("✅ Connected to Binance API")
            except Exception as e:
                print(f"⚠️ Binance API connection failed: {e}")
                print("Running in demo mode - you'll need to input balances manually")

        # Define your 8-coin core portfolio targets
        self.core_targets = [
            CoinTarget("BTC", "Bitcoin", 35.0),
            CoinTarget("ETH", "Ethereum", 20.0),
            CoinTarget("SOL", "Solana", 12.0),
            CoinTarget("RENDER", "Render", 8.0),
            CoinTarget("TAO", "Bittensor", 8.0),
            CoinTarget("AVAX", "Avalanche", 6.0),
            CoinTarget("LINK", "Chainlink", 6.0),
            CoinTarget("ONDO", "Ondo", 5.0)
        ]

        self.target_symbols = [coin.symbol for coin in self.core_targets]

    def get_current_prices(self) -> Dict[str, float]:
        """Fetch current prices for all target coins from CoinGecko API"""
        try:
            # CoinGecko API mapping
            coin_mapping = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'SOL': 'solana',
                'RENDER': 'render-token',
                'TAO': 'bittensor',
                'AVAX': 'avalanche-2',
                'LINK': 'chainlink',
                'ONDO': 'ondo-finance'
            }

            ids = ','.join(coin_mapping.values())
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"

            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            prices = {}
            for symbol, gecko_id in coin_mapping.items():
                if gecko_id in data:
                    prices[symbol] = data[gecko_id]['usd']

            print("✅ Current prices fetched successfully")
            return prices

        except Exception as e:
            print(f"❌ Error fetching prices: {e}")
            return {}

    def get_binance_balances(self) -> Dict[str, float]:
        """Fetch current balances from Binance account"""
        if not self.binance_client:
            return {}

        try:
            account = self.binance_client.get_account()
            balances = {}

            for balance in account['balances']:
                symbol = balance['asset']
                if symbol in self.target_symbols:
                    free_balance = float(balance['free'])
                    locked_balance = float(balance['locked'])
                    total_balance = free_balance + locked_balance

                    if total_balance > 0:
                        balances[symbol] = total_balance

            print("✅ Binance balances fetched successfully")
            return balances

        except Exception as e:
            print(f"❌ Error fetching Binance balances: {e}")
            return {}

    def get_manual_balances(self) -> Dict[str, float]:
        """Get balances through manual input"""
        print("\n📝 Please enter your current coin balances:")
        print("(Enter 0 if you don't hold that coin)")

        balances = {}
        for coin in self.core_targets:
            while True:
                try:
                    amount = input(f"{coin.name} ({coin.symbol}): ")
                    amount = float(amount) if amount.strip() else 0.0
                    if amount >= 0:
                        balances[coin.symbol] = amount
                        break
                    else:
                        print("Please enter a positive number or 0")
                except ValueError:
                    print("Please enter a valid number")

        return balances

    def get_technical_indicators(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get RSI and price vs 200-period MA for a given symbol
        Using Binance Kline data for technical analysis
        """
        try:
            if not self.binance_client:
                return None, None

            # Get daily klines for the last 300 days (enough for 200-day MA)
            klines = self.binance_client.get_historical_klines(
                f"{symbol}USDT",
                Client.KLINE_INTERVAL_1DAY,
                "300 day ago UTC"
            )

            if not klines:
                return None, None

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convert price columns to float
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)

            # Calculate RSI
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi().iloc[-1]

            # Calculate 200-day MA and current price deviation
            ma_200 = df['close'].rolling(window=200).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            price_vs_ma = ((current_price - ma_200) / ma_200) * 100

            return rsi, price_vs_ma

        except Exception as e:
            print(f"⚠️ Could not fetch technical indicators for {symbol}: {e}")
            return None, None

    def analyze_portfolio(self, balances: Dict[str, float], prices: Dict[str, float]) -> List[PortfolioAnalysis]:
        """Analyze current portfolio vs targets and generate recommendations"""
        
        # Calculate total portfolio value
        total_value = sum(balances.get(symbol, 0) * prices.get(symbol, 0)
                         for symbol in self.target_symbols)

        if total_value == 0:
            print("❌ Portfolio has no value or price data unavailable")
            return []

        analysis_results = []

        for coin in self.core_targets:
            symbol = coin.symbol
            current_amount = balances.get(symbol, 0)
            current_price = prices.get(symbol, 0)
            current_value = current_amount * current_price
            current_percentage = (current_value / total_value) * 100

            # Calculate deviation from target
            deviation = ((current_percentage - coin.target_percentage) / coin.target_percentage) * 100

            # Get technical indicators
            rsi, price_vs_ma = self.get_technical_indicators(symbol)

            # Generate signal and recommendation with action details
            signal, recommendation, action_pct, action_usd, action_coins = self._generate_recommendation_with_amounts(
                deviation, rsi, price_vs_ma, current_percentage, coin.target_percentage,
                total_value, current_value, current_amount, current_price
            )

            analysis_results.append(PortfolioAnalysis(
                symbol=symbol,
                current_value=current_value,
                current_percentage=current_percentage,
                target_percentage=coin.target_percentage,
                deviation=deviation,
                rsi=rsi,
                price_vs_ma=price_vs_ma,
                signal=signal,
                recommendation=recommendation,
                action_percentage=action_pct,
                action_usd_value=action_usd,
                action_coin_amount=action_coins,
                current_coin_amount=current_amount
            ))

        return analysis_results

    def _generate_recommendation_with_amounts(self, deviation: float, rsi: Optional[float],
                                           price_vs_ma: Optional[float], current_pct: float,
                                           target_pct: float, total_value: float, current_value: float,
                                           current_amount: float, current_price: float) -> Tuple[str, str, float, float, float]:
        """Generate trading signal and recommendation with specific amounts"""

        # Calculate target value and difference
        target_value = (target_pct / 100) * total_value
        value_difference = current_value - target_value

        action_percentage = 0.0
        action_usd_value = 0.0
        action_coin_amount = 0.0

        # Determine action intensity based on deviation and technical signals
        if abs(deviation) > 50:  # Strong deviation (>50% from target)
            if deviation > 0:  # Overweight - SELL
                signal = "🔴 STRONG SELL"
                # For strong overweight, sell 10-15% of position
                action_percentage = min(15.0, abs(deviation) * 0.2)  # Cap at 15%
                action_usd_value = abs(value_difference * 0.6)  # Move 60% toward target
                action_coin_amount = action_usd_value / current_price
                recommendation = f"SELL {action_percentage:.1f}% of position (${action_usd_value:,.0f} = {action_coin_amount:.4f} {current_pct > target_pct and 'coins' or 'USDT'})"
            else:  # Underweight - BUY
                signal = "🟢 STRONG BUY"
                action_usd_value = abs(value_difference * 0.6)  # Move 60% toward target
                action_percentage = (action_usd_value / current_value) * 100 if current_value > 0 else 100
                action_coin_amount = action_usd_value / current_price
                recommendation = f"BUY ${action_usd_value:,.0f} worth ({action_coin_amount:.4f} coins) - increase position by {action_percentage:.1f}%"

        elif abs(deviation) > 25:  # Moderate deviation
            # Check technical indicators for confirmation
            tech_signal_strength = 0
            if rsi is not None and price_vs_ma is not None:
                if (rsi > 75 and price_vs_ma > 25) or (rsi < 25 and abs(price_vs_ma) < 25):
                    tech_signal_strength = 2  # Strong technical signal
                elif (rsi > 65 and price_vs_ma > 15) or (rsi < 35 and price_vs_ma < 15):
                    tech_signal_strength = 1  # Moderate technical signal

            if deviation > 0:  # Overweight
                if tech_signal_strength >= 1:
                    signal = "🔴 SELL"
                    action_percentage = 5.0 + (tech_signal_strength * 2.5)  # 5-10%
                    action_usd_value = abs(value_difference * 0.4)  # Move 40% toward target
                    action_coin_amount = action_usd_value / current_price
                    tech_info = f"RSI: {rsi:.0f}, MA: {price_vs_ma:+.1f}%" if rsi else ""
                    recommendation = f"SELL {action_percentage:.1f}% (${action_usd_value:,.0f} = {action_coin_amount:.4f} coins) - {tech_info}"
                else:
                    signal = "🟡 REDUCE"
                    action_percentage = 3.0
                    action_usd_value = abs(value_difference * 0.25)  # Move 25% toward target
                    action_coin_amount = action_usd_value / current_price
                    recommendation = f"Consider SELLING {action_percentage:.1f}% (${action_usd_value:,.0f} = {action_coin_amount:.4f} coins)"
            else:  # Underweight
                if tech_signal_strength >= 1:
                    signal = "🟢 BUY"
                    action_usd_value = abs(value_difference * 0.4)  # Move 40% toward target
                    action_percentage = (action_usd_value / current_value) * 100 if current_value > 0 else 100
                    action_coin_amount = action_usd_value / current_price
                    tech_info = f"RSI: {rsi:.0f}, MA: {price_vs_ma:+.1f}%" if rsi else ""
                    recommendation = f"BUY ${action_usd_value:,.0f} worth ({action_coin_amount:.4f} coins) - {tech_info}"
                else:
                    signal = "🟡 ADD"
                    action_usd_value = abs(value_difference * 0.25)  # Move 25% toward target
                    action_percentage = (action_usd_value / current_value) * 100 if current_value > 0 else 100
                    action_coin_amount = action_usd_value / current_price
                    recommendation = f"Consider BUYING ${action_usd_value:,.0f} worth ({action_coin_amount:.4f} coins)"

        else:  # Minor deviation (<25%)
            signal = "✅ HOLD"
            recommendation = f"Well balanced - no action needed (within {abs(deviation):.1f}% of target)"

        return signal, recommendation, action_percentage, action_usd_value, action_coin_amount

    def print_analysis_report(self, analysis: List[PortfolioAnalysis], total_value: float):
        """Print a formatted analysis report"""

        print(f"\n{'='*80}")
        print(f"🎯 CRYPTO PORTFOLIO REBALANCING ANALYSIS")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Total Portfolio Value: ${total_value:,.2f}")
        print(f"{'='*80}")

        # Summary table
        print(f"\n{'COIN':<8} {'CURRENT':<10} {'TARGET':<8} {'DEVIATION':<12} {'RSI':<6} {'SIGNAL':<15}")
        print(f"{'-'*70}")

        for item in analysis:
            rsi_str = f"{item.rsi:.0f}" if item.rsi else "N/A"
            print(f"{item.symbol:<8} {item.current_percentage:<9.1f}% {item.target_percentage:<7.1f}% "
                  f"{item.deviation:<+11.1f}% {rsi_str:<6} {item.signal:<15}")

        print(f"\n{'DETAILED RECOMMENDATIONS'}")
        print(f"{'-'*50}")

        # Detailed recommendations
        for item in analysis:
            print(f"\n🪙 {item.symbol}")
            print(f"   Current: ${item.current_value:,.2f} ({item.current_percentage:.1f}%)")
            print(f"   Target:  {item.target_percentage:.1f}%")
            if item.rsi:
                print(f"   RSI: {item.rsi:.1f}")
            if item.price_vs_ma:
                print(f"   Price vs 200-day MA: {item.price_vs_ma:+.1f}%")
            print(f"   📋 {item.recommendation}")

        # Action summary
        strong_actions = [item for item in analysis if "STRONG" in item.signal]
        if strong_actions:
            print(f"\n🚨 PRIORITY ACTIONS:")
            for item in strong_actions:
                print(f"   • {item.symbol}: {item.recommendation}")

        print(f"\n{'='*80}")
        print("💡 Remember: Never sell more than 15% of any position at once!")
        print("🔄 Implement changes gradually over 1-2 weeks for large adjustments")
        print(f"{'='*80}")

    def save_analysis_to_file(self, analysis: List[PortfolioAnalysis], total_value: float, filename: str = None):
        """Save analysis results to a JSON file"""
        if not filename:
            filename = f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "total_portfolio_value": total_value,
            "analysis": [
                {
                    "symbol": item.symbol,
                    "current_value": item.current_value,
                    "current_percentage": item.current_percentage,
                    "target_percentage": item.target_percentage,
                    "deviation": item.deviation,
                    "rsi": item.rsi,
                    "price_vs_ma200": item.price_vs_ma,
                    "signal": item.signal,
                    "action_percentage": item.action_percentage,
                    "action_usd_value": item.action_usd_value,
                    "action_coin_amount": item.action_coin_amount,
                    "current_coin_amount": item.current_coin_amount
                }
                for item in analysis
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Analysis saved to: {filename}")

    def run_analysis(self, save_to_file: bool = True):
        """Main method to run the complete portfolio analysis"""

        print("🚀 Starting Crypto Portfolio Rebalancing Analysis...")

        # Step 1: Get current prices
        print("\n📈 Fetching current market prices...")
        prices = self.get_current_prices()
        if not prices:
            print("❌ Cannot proceed without price data")
            return

        # Step 2: Get portfolio balances
        print("\n💼 Getting portfolio balances...")
        balances = self.get_binance_balances() if self.binance_client else self.get_manual_balances()

        if not balances:
            print("❌ No portfolio data available")
            return

        # Step 3: Calculate total portfolio value
        total_value = sum(balances.get(symbol, 0) * prices.get(symbol, 0)
                         for symbol in self.target_symbols)

        # Step 4: Analyze portfolio
        print("\n🔍 Analyzing portfolio allocations and technical indicators...")
        analysis = self.analyze_portfolio(balances, prices)

        if not analysis:
            print("❌ Analysis failed")
            return

        # Step 5: Display results
        self.print_analysis_report(analysis, total_value)

        # Step 6: Save to file if requested
        if save_to_file:
            self.save_analysis_to_file(analysis, total_value)

        return analysis

def main():
    """Main execution function"""
    print("🎯 Crypto Portfolio Rebalancer")
    print("Based on your 8-coin core portfolio strategy\n")

    # Initialize with API keys (optional)
    api_key = os.getenv('BINANCE_API_KEY')  # Set these as environment variables
    secret_key = os.getenv('BINANCE_SECRET_KEY')

    if not api_key:
        print("💡 Tip: Set BINANCE_API_KEY and BINANCE_SECRET_KEY environment variables")
        print("   for automatic balance fetching. Otherwise, you'll input manually.\n")

    rebalancer = CryptoRebalancer(api_key, secret_key)

    try:
        analysis = rebalancer.run_analysis()

        if analysis:
            print("\n✅ Analysis complete! Check the recommendations above.")

            # Ask if user wants to run again
            while True:
                again = input("\n🔄 Run analysis again? (y/n): ").lower().strip()
                if again in ['y', 'yes']:
                    rebalancer.run_analysis()
                elif again in ['n', 'no']:
                    break
                else:
                    print("Please enter 'y' or 'n'")

    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your API credentials and internet connection.")

if __name__ == "__main__":
    main()
