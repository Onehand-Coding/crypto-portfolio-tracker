import asyncio
from typing import Optional

import streamlit as st


def render_transfer_widget(
    dashboard,
    *,
    context: str = "",
    required_usdt: Optional[float] = None,
) -> None:
    """
    Contextual widget to transfer USDT from Funding to Spot.

    - Shows balances and allows a quick transfer.
    - Optionally accepts a required_usdt hint to propose a default amount.
    """
    tracker = dashboard.initialize_tracker()
    if not tracker:
        st.error("❌ Failed to initialize tracker.")
        return

    # Trading status
    is_live = dashboard.config_manager.is_live

    try:
        balances = tracker.get_available_usdt_balance()
        spot_earn = float(balances.get("spot_earn", 0.0))
        funding = float(balances.get("funding", 0.0))
    except Exception as e:
        st.info(f"USDT balance unavailable. Error: {e}")
        return

    # Only render when there is funding to transfer
    if funding <= 0:
        return

    title = "💵 Transfer from Funding"
    if context:
        title += f" (for {context})"

    with st.expander(title, expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spot + Earn", f"${spot_earn:,.2f}")
        with col2:
            st.metric("Funding", f"${funding:,.2f}")
        with col3:
            st.metric("Total", f"${spot_earn + funding:,.2f}")

        # Suggest amount when required_usdt provided
        suggested = min(max((required_usdt or 0.0) - spot_earn, 0.0), funding)
        default_amount = suggested if suggested > 0 else min(100.0, funding)

        with st.form(f"transfer_form_{context or 'default'}"):
            amount = st.number_input(
                "Amount to transfer (USDT)",
                min_value=0.01,
                max_value=float(funding),
                value=float(round(default_amount, 2)) if funding >= 0.01 else 0.01,
                step=0.01,
                help=f"Maximum available: ${funding:,.2f}",
            )

            if required_usdt is not None and required_usdt > 0:
                st.caption(
                    f"Required: ${required_usdt:,.2f} | In Spot+Earn: ${spot_earn:,.2f}"
                )

            submitted = st.form_submit_button("Transfer Funds")
            if submitted:
                if amount <= 0:
                    st.error("❌ Amount must be greater than zero.")
                    return

                if amount > funding:
                    st.error(
                        f"❌ Amount exceeds available funding balance (${funding:,.2f})."
                    )
                    return

                with st.spinner(" Executing transfer..."):
                    result = asyncio.run(
                        tracker.transfer_funding_to_spot(
                            asset="USDT", amount=float(amount), is_live=is_live
                        )
                    )

                if result.success:
                    st.success("✅ Transfer executed successfully!")
                    for msg in result.messages:
                        st.write(f"• {msg}")
                else:
                    st.error("❌ Transfer failed:")
                    for err in result.errors:
                        st.write(f"• {err}")

                st.rerun()
