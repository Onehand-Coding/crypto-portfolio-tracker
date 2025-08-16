import json
import asyncio

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from crypto_portfolio_tracker.dashboard import utils as ui_utils
from crypto_portfolio_tracker.crypto_trend_analyzer import CryptoTrendAnalyzer


def render_market_page(dashboard):
    """Render market page."""

    # Clear previous page state if coming from another page
    keys_to_clear = [
        "preview_file", "current_chart_symbol", "unified_chart_coin_select"
    ]
    ui_utils.initialize_page_state("market", keys_to_clear)

    st.markdown("## 📈 Market Trends")
    export_dir = Path(
        dashboard.config_manager.config.get("exports", {}).get("path", "data/exports/")
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Define get_trend_report
    @st.cache_data(show_spinner="Generating trend report...", ttl=300)
    def get_trend_report(config, _binance_client, timeframe):
        analyzer = CryptoTrendAnalyzer(config=config, binance_client=_binance_client)
        return asyncio.run(analyzer.generate_report(timeframe))

    # 2. Define the chart plotting function
    def plot_coin_chart(symbol, timeframe, config, binance_client):
        analyzer = CryptoTrendAnalyzer(config=config, binance_client=binance_client)
        settings = analyzer.timeframe_settings.get(timeframe)
        if not settings:
            st.error("No settings for this timeframe.")
            return

        period = settings.get("period", "1mo")
        interval = "1wk" if timeframe == "long_term" else "1d"
        data = asyncio.run(analyzer.fetch_crypto_data_async(symbol, period, interval))
        if data is None or data.empty:
            st.warning(f"No data available for {symbol}.")
            return

        indicator_df = analyzer._calculate_indicators(data, settings)
        # Defensive: ensure index is datetime, sorted, unique, and not NaT
        if indicator_df is None or indicator_df.empty:
            st.warning("Not enough data for indicators.")
            return
        if not pd.api.types.is_datetime64_any_dtype(indicator_df.index):
            st.warning("Data index is not datetime. Cannot plot.")
            return
        indicator_df = indicator_df[~indicator_df.index.duplicated(keep="first")]
        indicator_df = indicator_df[~indicator_df.index.isna()]
        indicator_df = indicator_df.sort_index()

        # Require a minimum number of rows for meaningful plotting
        min_required_rows = 30
        if len(indicator_df) < min_required_rows:
            st.warning(
                f"Not enough data to plot a meaningful chart (need at least {min_required_rows} rows, got {len(indicator_df)})."
            )
            return

        # Check for all-empty rows (all columns are NaN)
        if indicator_df.dropna(how="all").empty:
            st.warning(
                "No valid data to plot (all rows are empty or all columns are NaN)."
            )
            return

        # Check for all-NaN in the columns you want to plot
        plot_cols = ["Close"]
        if all(
            indicator_df[col].dropna().empty
            for col in plot_cols
            if col in indicator_df.columns
        ):
            st.warning("No valid data to plot (all values in plot columns are NaN).")
            return

        # Limit to last 500 rows for safety
        if len(indicator_df) > 500:
            indicator_df = indicator_df.tail(500)

        # Only now is it safe to plot!
        fig, axs = plt.subplots(
            3,
            1,
            figsize=(10, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )
        fig.suptitle(
            f"{symbol} Price & Indicators ({timeframe.replace('_', ' ').title()})",
            fontsize=16,
        )

        # --- Price and SMAs ---
        axs[0].plot(
            indicator_df.index, indicator_df["Close"], label="Close", color="blue"
        )
        sma_short = settings.get("sma_short_window")
        sma_long = settings.get("sma_long_window")
        if sma_short and f"SMA_{sma_short}" in indicator_df.columns:
            axs[0].plot(
                indicator_df.index,
                indicator_df[f"SMA_{sma_short}"],
                label=f"SMA {sma_short}",
                color="orange",
            )
        if sma_long and f"SMA_{sma_long}" in indicator_df.columns:
            axs[0].plot(
                indicator_df.index,
                indicator_df[f"SMA_{sma_long}"],
                label=f"SMA {sma_long}",
                color="green",
            )
        # Support/Resistance (last value)
        if "Low" in indicator_df.columns and "High" in indicator_df.columns:
            window = 30 if len(indicator_df) > 30 else len(indicator_df)
            support = indicator_df["Low"].tail(window).min()
            resistance = indicator_df["High"].tail(window).max()
            axs[0].axhline(support, color="red", linestyle="--", label="Support")
            axs[0].axhline(
                resistance, color="purple", linestyle="--", label="Resistance"
            )
        axs[0].set_ylabel("Price")
        axs[0].legend()
        axs[0].grid(True)

        # --- RSI ---
        rsi_col = f"RSI_{analyzer.rsi_period}"
        if rsi_col in indicator_df.columns:
            axs[1].plot(
                indicator_df.index,
                indicator_df[rsi_col],
                label="RSI",
                color="magenta",
            )
            axs[1].axhline(70, color="red", linestyle="--", linewidth=1)
            axs[1].axhline(30, color="green", linestyle="--", linewidth=1)
            axs[1].set_ylabel("RSI")
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].text(0.5, 0.5, "No RSI data", ha="center", va="center")

        # --- MACD ---
        macd_col = "MACD_12_26_9"
        macds_col = "MACDs_12_26_9"
        if macd_col in indicator_df.columns and macds_col in indicator_df.columns:
            axs[2].plot(
                indicator_df.index,
                indicator_df[macd_col],
                label="MACD",
                color="blue",
            )
            axs[2].plot(
                indicator_df.index,
                indicator_df[macds_col],
                label="Signal",
                color="orange",
            )
            axs[2].set_ylabel("MACD")
            axs[2].legend()
            axs[2].grid(True)
        else:
            axs[2].text(0.5, 0.5, "No MACD data", ha="center", va="center")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig)

    # 3. Timeframe selector
    timeframe_map = {
        "Long-term (4 Years)": "long_term",
        "Swing (3 Months)": "swing",
        "Day (1 Month)": "day",
    }
    timeframe_label = st.radio(
        "Select timeframe for trend analysis:",
        list(timeframe_map.keys()),
        horizontal=True,
    )
    timeframe = timeframe_map[timeframe_label]

    tracker = dashboard.initialize_tracker()
    report = get_trend_report(
        dashboard.config_manager.config,
        getattr(tracker, "binance_client", None),
        timeframe,
    )

    if not report:
        st.error("Could not generate trend report. Please try again later.")
        return

    btc = report.get("benchmark_analysis", {})
    coin_analyses = report.get("coin_analyses", {})

    # --- Unified Chart Section ---
    st.markdown("### 📊 Coin Chart Viewer")

    # Get all available symbols (including BTC)
    all_symbols = list(coin_analyses.keys())
    if all_symbols:
        if "current_chart_symbol" not in st.session_state:
            st.session_state.current_chart_symbol = all_symbols[
                0
            ]  # Default to first coin

        # Dropdown for selecting coin
        selected_symbol = st.selectbox(
            "Select coin to view chart:",
            all_symbols,
            key="unified_chart_coin_select",
            index=all_symbols.index(st.session_state.current_chart_symbol),
        )

        # Update session state with selected symbol
        st.session_state.current_chart_symbol = selected_symbol

        # Always render the chart when coins are available
        plot_coin_chart(
            st.session_state.current_chart_symbol,
            timeframe,
            dashboard.config_manager.config,
            getattr(tracker, "binance_client", None),
        )
    else:
        st.info("No coin data available for charting.")

    # 5. Coin-by-Coin Table
    st.markdown("### 🪙 Coin Analysis")
    rows = []
    for symbol, analysis in coin_analyses.items():
        rows.append(
            {
                "Symbol": symbol,
                "Price": f"${analysis.get('current_price', 0):,.2f}",
                "Change (%)": f"{analysis.get('price_change_pct', 0):+.2f}",
                "RSI": f"{analysis.get('rsi', 0):,.2f}",
                "Support": f"${analysis.get('support_level', 0):,.2f}",
                "Resistance": f"${analysis.get('resistance_level', 0):,.2f}",
                "Active Conditions": ", ".join(analysis.get("active_conditions", [])),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No coin analysis data available.")

    # --- Prepare exportable data ---
    df_export = pd.DataFrame(
        [
            {
                "Symbol": symbol,
                "Price": analysis.get("current_price", 0),
                "Change (%)": analysis.get("price_change_pct", 0),
                "RSI": analysis.get("rsi", 0),
                "Support": analysis.get("support_level", 0),
                "Resistance": analysis.get("resistance_level", 0),
                "Active Conditions": ", ".join(analysis.get("active_conditions", [])),
            }
            for symbol, analysis in coin_analyses.items()
        ]
    )

    # --- Export Section ---
    st.markdown("### 📤 Analysis Exports")

    # Export creation controls
    with st.container():
        st.markdown("#### Create New Export")
        col_format, col_action = st.columns([2, 3])

        with col_format:
            export_format = st.radio(
                "Format",
                options=["CSV", "JSON", "HTML"],
                format_func=lambda x: {
                    "CSV": "📄 CSV",
                    "JSON": "{ } JSON",
                    "HTML": "🌐 HTML",
                }[x],
                horizontal=True,
            )

        with col_action:
            if st.button("🚀 Generate Export", use_container_width=True):
                with st.spinner(f"Creating {export_format} export..."):
                    try:
                        # Use the unified export method from tracker
                        exported_file = tracker.export_trend_report(
                            report, timeframe, export_format
                        )

                        if exported_file:
                            st.success(f"Successfully created {exported_file.name}")
                            st.rerun()
                        else:
                            st.error("Export failed")

                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")

    # Consolidated exports dropdown
    st.markdown("---")
    st.markdown("#### My Exports")

    # Get all export files sorted by modification time (newest first)
    all_files = sorted(
        list(export_dir.glob("trend_report_*.*")),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not all_files:
        st.info("No exports found. Generate one above!")
        return

    # Dropdown to select export file
    selected_export = st.selectbox(
        "Select an export file:",
        options=all_files,
        format_func=lambda x: f"{x.name} ({x.stat().st_size / 1024:.1f} KB, {datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})",
        index=0,
        help="Select an export file to view, download or delete",
    )

    # Initialize preview state if not exists
    if "preview_file" not in st.session_state:
        st.session_state.preview_file = None

    # File actions
    col_preview, col_download, col_delete = st.columns(3)

    with col_preview:
        if st.button("👁️ Preview", use_container_width=True):
            st.session_state.preview_file = selected_export
            st.rerun()

    with col_download:
        with open(selected_export, "rb") as f:
            file_bytes = f.read()
            st.download_button(
                label="⬇️ Download",
                data=file_bytes,
                file_name=selected_export.name,
                mime=f"text/{selected_export.suffix[1:].lower()}"
                if selected_export.suffix != ".json"
                else "application/json",
                use_container_width=True,
            )

    with col_delete:
        if st.button("🗑️ Delete", use_container_width=True):
            try:
                selected_export.unlink()
                if st.session_state.preview_file == selected_export:
                    st.session_state.preview_file = None
                st.success(f"Deleted {selected_export.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete: {e}")

    # Preview panel
    if st.session_state.preview_file:
        preview_file = st.session_state.preview_file
        filetype = preview_file.suffix[1:].upper()

        st.markdown("---")
        st.markdown(f"### 👁️ Preview: {preview_file.name}")

        try:
            if filetype == "CSV":
                st.dataframe(pd.read_csv(preview_file), use_container_width=True)
            elif filetype == "JSON":
                st.json(json.load(open(preview_file)))
            elif filetype == "HTML":
                st.components.v1.html(
                    preview_file.read_text(), height=800, scrolling=True
                )
        except Exception as e:
            st.error(f"Could not preview file: {str(e)}")

        if st.button("❌ Close Preview"):
            st.session_state.preview_file = None
            st.rerun()
