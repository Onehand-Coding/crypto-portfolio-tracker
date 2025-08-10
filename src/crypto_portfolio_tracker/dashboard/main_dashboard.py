import streamlit as st

from . import utils
from crypto_portfolio_tracker import __version__


def render_header():
    st.markdown(
        f"""
    <div class="main-header">
        <h1><span style="font-size: 3rem;">🪙</span> Crypto Portfolio Tracker v{__version__}</h1>
        <p>Professional Portfolio Management & Analysis Dashboard</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_status_indicator():
    if st.session_state.offline_mode:
        st.markdown(
            """
        <div style="text-align: center; padding: 0.5rem; background: #f8d7da; border-radius: 8px; margin-bottom: 1rem;">
            <span class="status-offline">⚠️ OFFLINE MODE - Network features are disabled</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div style="text-align: center; padding: 0.5rem; background: #d4edda; border-radius: 8px; margin-bottom: 1rem;">
            <span class="status-online">✅ ONLINE - All features available</span>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_sidebar(dashboard):
    st.sidebar.markdown(
        """
    <h3 style="margin: 0; padding: 0.1rem 0 1rem 0; color: #fff; font-weight: 700; font-size: 1.15rem;">🗺️ Navigation</h3>
    """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "📈 Market",
            "⚖️ Rebalance",
            "💸 DCA",
            "💰 Trade",
            "🧪 Backtest",
            "🗄️ Database",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        """
    <div class="sidebar-section">
        <h4>⚡ Quick Actions</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🔄 Sync Portfolio"):
        dashboard.run_full_sync()

    if st.sidebar.button("💾 Save Snapshot"):
        if st.session_state.portfolio_metrics:
            dashboard.save_snapshot()
        else:
            st.sidebar.warning("No metrics available to save")

    if st.session_state.last_sync:
        st.sidebar.markdown(
            f"""
        <div class="sidebar-section">
            <small>Last Sync: {st.session_state.last_sync}</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    return page
