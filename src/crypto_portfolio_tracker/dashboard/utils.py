import streamlit as st


def inject_custom_css():
    """Inject the CSS styles used across the UI."""
    st.markdown(
        """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .status-online {
        color: #28a745;
    }
    .status-offline {
        color: #dc3545;
    }
    .sidebar-section {
        background: transparent;
        padding: 0.25rem 0;
        border-radius: 0;
        margin-bottom: 0.5rem;
    }
    .sidebar-section h3, .sidebar-section h4 {
        margin: 0;
        font-weight: 700;
        color: #fff;
        font-size: 1.15rem;
        letter-spacing: 0.01em;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .sidebar .stMarkdown hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .stRadio > div {
        margin-top: -10px !important;
    }
    .stRadio > div > label > div:first-child {
        margin-bottom: 0 !important;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def format_currency(value):
    """Format a number as USD currency."""
    try:
        return f"${value:,.2f}"
    except (TypeError, ValueError):
        return "$0.00"
