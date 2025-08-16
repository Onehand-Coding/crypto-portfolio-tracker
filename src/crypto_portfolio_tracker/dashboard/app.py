#!/usr/bin/env python3
import streamlit as st

from crypto_portfolio_tracker.dashboard.ui_controller import Dashboard
from crypto_portfolio_tracker.dashboard import (
    utils,
    main_dashboard,
    home_page,
    market_page,
    rebalancing_page,
    dca_page,
    trading_page,
    backtest_page,
    database_page,
    settings_page,
)


def main():
    st.set_page_config(
        page_title="Crypto Portfolio Tracker",
        page_icon="🪙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    dashboard = Dashboard()
    dashboard.initialize_session_state()
    dashboard.setup_logging()

    utils.inject_custom_css()

    main_dashboard.render_header()
    main_dashboard.render_status_indicator()
    page = main_dashboard.render_sidebar(dashboard)

    # Initialize page session state tracking.
    if "current_page" not in st.session_state:
        st.session_state.current_page = None

    if page == "🏠 Home":
        home_page.render_home_page(dashboard)
    elif page == "📈 Market":
        market_page.render_market_page(dashboard)
    elif page == "⚖️ Rebalance":
        rebalancing_page.render_rebalancing_page(dashboard)
    elif page == "💸 DCA":
        dca_page.render_dca_page(dashboard)
    elif page == "💰 Trade":
        trading_page.render_trading_page(dashboard)
    elif page == "🧪 Backtest":
        backtest_page.render_backtest_page(dashboard)
    elif page == "🗄️ Database":
        database_page.render_database_page(dashboard)
    elif page == "⚙️ Settings":
        settings_page.render_settings_page(dashboard)


if __name__ == "__main__":
    main()
