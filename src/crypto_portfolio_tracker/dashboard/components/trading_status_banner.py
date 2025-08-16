import streamlit as st


def render_trading_status_banner(is_live: bool, is_testnet: bool) -> None:
    """Render trading status banner which is common for some page."""
    col1, col2, col3 = st.columns(3)
    with col1:
        if is_live:
            st.error("🔴 LIVE TRADING ENABLED")
        else:
            st.warning("🟡 LIVE TRADING DISABLED")
    with col2:
        if is_testnet:
            st.info("🧪 TESTNET CONNECTION")
        else:
            st.info("🌐 MAINNET CONNECTION")
    with col3:
        if is_live:
            st.error("⚠️ ORDERS WILL BE PLACED")
        else:
            st.success("✅ SIMULATION MODE")
