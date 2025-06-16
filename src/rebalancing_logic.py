import pandas as pd
import asyncio
from typing import Dict, Any, List, Optional, Tuple

from src.crypto_trend_analyzer import CryptoTrendAnalyzer, TrendCondition

async def get_live_rebalance_suggestions(
    analyzer: CryptoTrendAnalyzer, portfolio_df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Orchestrates the entire rebalancing suggestion process using LIVE data."""
    print("🔄 Gathering live trend reports for rebalancing...")
    swing_report_task = analyzer.generate_report("swing")
    long_term_report_task = analyzer.generate_report("long_term")
    swing_report, long_term_report = await asyncio.gather(swing_report_task, long_term_report_task)

    if not swing_report or not long_term_report:
        analyzer.logger.error("Could not generate trend reports for rebalancing.")
        return pd.DataFrame()

    btc_analysis = long_term_report.get("benchmark_analysis", {})
    is_bear_market = "Price < SMA200" in btc_analysis.get("active_conditions", [])
    if is_bear_market:
        analyzer.logger.warning("BEARISH MARKET REGIME DETECTED. BUY signals may be suppressed.")

    return generate_rebalancing_suggestions(
        portfolio_df=portfolio_df,
        target_allocation=config.get("target_allocation", {}),
        long_term_report=long_term_report,
        swing_report=swing_report,
        asset_classes=config.get("asset_classes", {}),
        rebalance_config=config.get("rebalance_technical", {}),
        is_bear_market=is_bear_market,
        norm_map=config.get("symbol_normalization_map", {})
    )


def get_backtest_rebalance_suggestions(
    full_historical_data_with_indicators: pd.DataFrame,
    portfolio_state: Dict[str, float],
    sim_date: pd.Timestamp,
    config: Dict[str, Any],
    analyzer_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Orchestrates rebalancing suggestions by fully reconstructing analysis reports
    from pre-calculated indicator values for a true 1:1 backtest.
    """
    latest_row = full_historical_data_with_indicators.loc[sim_date]
    swing_report = {"coin_analyses": {}}
    long_term_report = {"coin_analyses": {}}

    target_assets = list(config.get("target_allocation", {}).keys())
    if "BTC" not in target_assets: target_assets.append("BTC")

    for asset in target_assets:
        yf_symbol = f"{asset}-USD"
        current_price = latest_row.get(f"{asset}_Close")
        if pd.isna(current_price): continue

        # Read all pre-calculated values for the current date
        sma_long = latest_row.get(f'{asset}_SMA_L_long')
        sma_short = latest_row.get(f'{asset}_SMA_S_long')
        rsi = latest_row.get(f'{asset}_RSI')
        support = latest_row.get(f'{asset}_Support', 0.0)
        resistance = latest_row.get(f'{asset}_Resistance', 0.0)

        # Reconstruct the active conditions exactly like the live analyzer
        active_conditions = []
        if not pd.isna(sma_long) and not pd.isna(sma_short):
            if sma_short > sma_long:
                active_conditions.append(TrendCondition.GOLDEN_CROSS.value)
            else:
                active_conditions.append(TrendCondition.DEATH_CROSS.value)

        if not pd.isna(sma_long):
            if current_price > sma_long:
                active_conditions.append(TrendCondition.PRICE_ABOVE_SMA200.value)
            else:
                active_conditions.append(TrendCondition.PRICE_BELOW_SMA200.value)

        if not pd.isna(rsi):
            if rsi > analyzer_config.get('rsi_overbought', 70):
                active_conditions.append(TrendCondition.RSI_OVERBOUGHT.value)
            elif rsi < analyzer_config.get('rsi_oversold', 30):
                active_conditions.append(TrendCondition.RSI_OVERSOLD.value)
            elif 40 <= rsi <= 60:
                active_conditions.append(TrendCondition.NEUTRAL_RSI.value)

        if not active_conditions:
            active_conditions.append(TrendCondition.INSUFFICIENT_DATA.value)

        # Build the report structures with all the required fields
        analysis_dict = {
            'symbol': yf_symbol,
            'current_price': current_price,
            'active_conditions': active_conditions,
            'rsi': rsi,
            'support_level': support,
            'resistance_level': resistance,
            'price_change_pct': 0 # Not needed for backtest logic
        }

        swing_report['coin_analyses'][yf_symbol] = analysis_dict
        long_term_report['coin_analyses'][yf_symbol] = analysis_dict
        if asset == "BTC":
            long_term_report['benchmark_analysis'] = analysis_dict

    btc_analysis = long_term_report.get("benchmark_analysis", {})
    is_bear_market = TrendCondition.PRICE_BELOW_SMA200.value in btc_analysis.get("active_conditions", [])

    portfolio_rows = [{'symbol': asset, 'value_usd': quantity * latest_row.get(f"{asset}_Close", 0)} for asset, quantity in portfolio_state.items() if asset != 'USDT']
    portfolio_rows.append({'symbol': 'USDT', 'value_usd': portfolio_state.get('USDT', 0)})
    portfolio_df = pd.DataFrame(portfolio_rows)

    return generate_rebalancing_suggestions(
        portfolio_df=portfolio_df,
        target_allocation=config.get("target_allocation", {}),
        long_term_report=long_term_report,
        swing_report=swing_report,
        asset_classes=config.get("asset_classes", {}),
        rebalance_config=config.get("rebalance_technical", {}),
        is_bear_market=is_bear_market,
        norm_map=config.get("symbol_normalization_map", {})
    )


def generate_rebalancing_suggestions(
    portfolio_df: pd.DataFrame,
    target_allocation: Dict[str, float],
    long_term_report: Dict[str, Any],
    swing_report: Dict[str, Any],
    asset_classes: Dict[str, List[str]],
    rebalance_config: Dict[str, Any],
    is_bear_market: bool,
    norm_map: Dict[str, str]
) -> pd.DataFrame:
    # This function remains unchanged
    total_portfolio_value = portfolio_df['value_usd'].sum()
    if total_portfolio_value == 0:
        return pd.DataFrame()

    suggestions_data = []
    major_assets = asset_classes.get("majors", ["BTC", "ETH"])
    target_allocation_normalized = {norm_map.get(k.upper(), k.upper()): v for k, v in target_allocation.items()}

    for symbol_simple, target_pct in target_allocation_normalized.items():
        yf_symbol = f"{symbol_simple}-USD"
        swing_analysis = swing_report['coin_analyses'].get(yf_symbol)
        long_term_analysis = long_term_report['coin_analyses'].get(yf_symbol)

        if not swing_analysis or not long_term_analysis: continue

        strategy_params = rebalance_config.get("majors", {}) if symbol_simple in major_assets else rebalance_config.get("alts", {})
        drift_threshold_pct = strategy_params.get("allocation_drift_threshold_pct", 5.0)
        current_value = portfolio_df[portfolio_df['symbol'] == symbol_simple]['value_usd'].sum()
        current_pct = (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        absolute_drift_pct = current_pct - (target_pct * 100)
        signal, action_text = "HOLD", "Hold: Allocation is within tolerance."
        action_value_usd, coin_amount = 0.0, 0.0
        is_overweight = absolute_drift_pct > drift_threshold_pct
        is_underweight = absolute_drift_pct < -drift_threshold_pct
        long_term_conditions = long_term_analysis.get('active_conditions', [])
        is_oversold_long_term = TrendCondition.RSI_OVERSOLD.value in long_term_conditions
        signal_strength = 1.0
        current_price = swing_analysis.get('current_price', 0)
        support = swing_analysis.get('support_level', 0)
        resistance = swing_analysis.get('resistance_level', 0)
        if is_underweight and support > 0 and (current_price - support) / support < 0.10: signal_strength = 1.5
        if is_overweight and resistance > 0 and (resistance - current_price) / resistance < 0.10: signal_strength = 1.5
        if is_overweight:
            signal = "SELL"
            overweight_usd = (absolute_drift_pct / 100) * total_portfolio_value
            action_value_usd = overweight_usd * strategy_params.get('sell_percentage_multiplier', 0.5) * signal_strength
            coin_amount = action_value_usd / current_price if current_price > 0 else 0
            action_text = f"Sell ~${action_value_usd:,.2f} worth, which is {f'{coin_amount:,.8g}'} {symbol_simple}"
        elif is_underweight or is_oversold_long_term:
            if is_bear_market and not is_oversold_long_term:
                signal = "HOLD"
                action_text = "BUY signal suppressed due to Bearish Market Regime."
            else:
                signal = "BUY"
                underweight_usd = abs(absolute_drift_pct / 100) * total_portfolio_value
                action_value_usd = underweight_usd * strategy_params.get('buy_amount_multiplier', 1.0) * signal_strength
                coin_amount = action_value_usd / current_price if current_price > 0 else 0
                action_text = f"Buy ~${action_value_usd:,.2f} worth ({f'{coin_amount:,.8g}'} {symbol_simple})"

        suggestions_data.append({
            "Symbol": symbol_simple, "Target %": target_pct * 100, "Current %": current_pct,
            "Drift (pts)": absolute_drift_pct, "Current Value (USD)": current_value, "TA_Price": current_price,
            "Support": support, "Resistance": resistance, "TA_Conditions": ", ".join(long_term_conditions) or "None",
            "Signal": signal, "Suggested Action Detail": action_text,
            "action_usd_value": action_value_usd, "action_coin_quantity": coin_amount
        })
    return pd.DataFrame(suggestions_data)
