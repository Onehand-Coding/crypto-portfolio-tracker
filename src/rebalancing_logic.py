import asyncio
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from src.crypto_trend_analyzer import CryptoTrendAnalyzer, TrendCondition, get_market_regime


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

    # --- Call the central function to determine market regime ---
    is_bear_market = get_market_regime(long_term_report)
    if is_bear_market:
        analyzer.logger.warning("BEARISH MARKET REGIME DETECTED. Rebalancing rules will apply.")

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
    analyzer: CryptoTrendAnalyzer,
    target_allocation: Dict[str, float]
) -> pd.DataFrame:
    """
    Orchestrates rebalancing suggestions for a backtest by using the centralized
    CryptoTrendAnalyzer to generate reports for a specific historical date.
    """
    swing_report = {"coin_analyses": {}}
    long_term_report = {"coin_analyses": {}}
    target_assets = list(target_allocation.keys())
    if "BTC" not in target_assets:
        target_assets.append("BTC")

    for asset in target_assets:
        yf_symbol = f"{asset}-USD"

        # Create a temporary DataFrame for the single asset with generic column names
        # that the analyzer expects ('Close', 'Low', 'High', etc.).

        # 1. Define the mapping from generic names to the specific asset's column names
        column_mapping = {
            'Close': f'{asset}_Close', 'Open': f'{asset}_Open',
            'High': f'{asset}_High', 'Low': f'{asset}_Low', 'Volume': f'{asset}_Volume'
        }

        # 2. Check if the essential 'Close' column exists for this asset in the main DataFrame
        if column_mapping['Close'] not in full_historical_data_with_indicators.columns:
            continue

        # 3. Create the temporary DataFrame
        single_asset_df = pd.DataFrame()

        # 4. Copy data from the main DataFrame and rename columns
        for generic_name, asset_specific_name in column_mapping.items():
            if asset_specific_name in full_historical_data_with_indicators:
                single_asset_df[generic_name] = full_historical_data_with_indicators[asset_specific_name]

        # 5. Copy over all the pre-calculated indicators for that specific asset
        for col_name in full_historical_data_with_indicators.columns:
            if col_name.startswith(f"{asset}_") and col_name not in column_mapping.values():
                single_asset_df[col_name] = full_historical_data_with_indicators[col_name]


        # Use the new historical analysis method with the correctly formatted DataFrame
        lt_analysis = analyzer.analyze_historical_row(single_asset_df, yf_symbol, "long_term", sim_date)
        sw_analysis = analyzer.analyze_historical_row(single_asset_df, yf_symbol, "swing", sim_date)

        long_term_report['coin_analyses'][yf_symbol] = asdict(lt_analysis)
        swing_report['coin_analyses'][yf_symbol] = asdict(sw_analysis)

        if asset == "BTC":
            long_term_report['benchmark_analysis'] = asdict(lt_analysis)

    is_bear_market = get_market_regime(long_term_report)

    latest_row = full_historical_data_with_indicators.loc[sim_date]
    portfolio_rows = [{'symbol': asset, 'value_usd': quantity * latest_row.get(f"{asset}_Close", 0)} for asset, quantity in portfolio_state.items() if asset != 'USDT']
    portfolio_rows.append({'symbol': 'USDT', 'value_usd': portfolio_state.get('USDT', 0)})
    portfolio_df = pd.DataFrame(portfolio_rows)

    return generate_rebalancing_suggestions(
        portfolio_df=portfolio_df,
        target_allocation=target_allocation,
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
    market_rules = rebalance_config.get("market_regime_rules", {})
    suppress_buys_in_bear = market_rules.get("suppress_buys_in_bear", False)

    total_portfolio_value = portfolio_df['value_usd'].sum()
    if total_portfolio_value == 0:
        return pd.DataFrame()

    suggestions_data = []
    major_assets = asset_classes.get("majors", ["BTC", "ETH"])
    target_allocation_normalized = {norm_map.get(k.upper(), k.upper()): v for k, v in target_allocation.items()}

    for symbol_simple, target_pct in target_allocation_normalized.items():
        yf_symbol = f"{symbol_simple}-USD"
        swing_analysis, long_term_analysis = swing_report['coin_analyses'].get(yf_symbol), long_term_report['coin_analyses'].get(yf_symbol)
        if not swing_analysis or not long_term_analysis: continue
        strategy_params = rebalance_config.get("majors", {}) if symbol_simple in major_assets else rebalance_config.get("alts", {})
        drift_threshold_pct, current_value = strategy_params.get("allocation_drift_threshold_pct", 5.0), portfolio_df[portfolio_df['symbol'] == symbol_simple]['value_usd'].sum()
        current_pct = (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        absolute_drift_pct = current_pct - (target_pct * 100)
        signal, action_text = "HOLD", "Hold: Allocation is within tolerance."
        action_value_usd, coin_amount = 0.0, 0.0
        is_overweight, is_underweight = absolute_drift_pct > drift_threshold_pct, absolute_drift_pct < -drift_threshold_pct
        long_term_conditions, is_oversold_long_term = long_term_analysis.get('active_conditions', []), TrendCondition.RSI_OVERSOLD.value in long_term_analysis.get('active_conditions', [])
        signal_strength, current_price = 1.0, swing_analysis.get('current_price', 0)
        support, resistance = swing_analysis.get('support_level', 0), swing_analysis.get('resistance_level', 0)
        if is_underweight and support > 0 and (current_price - support) / support < 0.10: signal_strength = 1.5
        if is_overweight and resistance > 0 and (resistance - current_price) / resistance < 0.10: signal_strength = 1.5

        if is_overweight:
            signal = "SELL"
            overweight_usd = (absolute_drift_pct / 100) * total_portfolio_value
            action_value_usd = overweight_usd * strategy_params.get('sell_percentage_multiplier', 0.5) * signal_strength
            coin_amount = action_value_usd / current_price if current_price > 0 else 0
            action_text = f"Sell ~${action_value_usd:,.2f} worth, which is {f'{coin_amount:,.8g}'} {symbol_simple}"
        elif is_underweight or is_oversold_long_term:

            if is_bear_market and suppress_buys_in_bear and not is_oversold_long_term:
                signal = "HOLD"
                action_text = "BUY signal suppressed due to Bearish Market Regime."
            else:
                signal = "BUY"
                underweight_usd = abs(absolute_drift_pct / 100) * total_portfolio_value
                action_value_usd = underweight_usd * strategy_params.get('buy_amount_multiplier', 1.0) * signal_strength
                coin_amount = action_value_usd / current_price if current_price > 0 else 0
                action_text = f"Buy ~${action_value_usd:,.2f} worth ({f'{coin_amount:,.8g}'} {symbol_simple})"

        suggestions_data.append({
            "Symbol": symbol_simple, "Target %": target_pct * 100, "Current %": current_pct, "Drift (pts)": absolute_drift_pct,
            "Current Value (USD)": current_value, "TA_Price": current_price, "Support": support, "Resistance": resistance,
            "TA_Conditions": ", ".join(long_term_conditions) or "None", "Signal": signal, "Suggested Action Detail": action_text,
            "action_usd_value": action_value_usd, "action_coin_quantity": coin_amount
        })
    return pd.DataFrame(suggestions_data)
