import asyncio
from typing import Optional

import pandas as pd
import streamlit as st


def render_transfer_widget(
    dashboard,
    *,
    context: str = "",
    required_asset: Optional[str] = None,
    required_amount: Optional[float] = None,
) -> None:
    """
    Enhanced widget to transfer assets between any supported wallets.

    - Shows balances for all wallets
    - Allows transfers between any supported wallet combinations
    - Optionally accepts required asset and amount hints
    """
    # Handle case where dashboard is None
    if dashboard is None:
        st.warning("Transfer widget requires a valid dashboard instance.")
        return

    tracker = dashboard.initialize_tracker()
    if not tracker:
        st.error("❌ Failed to initialize tracker.")
        return

    # Trading status
    is_live = dashboard.config_manager.is_live

    try:
        # Get balances for all wallets
        balances = tracker.get_available_usdt_balance()
        spot_earn = float(balances.get("spot_earn", 0.0))
        funding = float(balances.get("funding", 0.0))

        # Get futures balance
        futures_balance = 0.0
        try:
            futures_balances = tracker.fetcher.fetch_futures_balance()
            for balance in futures_balances:
                if balance.get("asset") == "USDT":
                    futures_balance = float(balance.get("balance", 0.0))
                    break
        except Exception:
            futures_balance = 0.0

        # Get funding balance details
        funding_details = {}
        try:
            funding_balances = tracker.fetcher.fetch_funding_balance()
            for balance in funding_balances:
                asset = balance.get("asset", "")
                free = float(balance.get("free", 0.0))
                if free > 0:
                    funding_details[asset] = free
        except Exception:
            pass

    except Exception as e:
        st.info(f"Balance information unavailable. Error: {e}")
        return

    title = "🔄 Transfer Between Wallets"
    if context:
        title += f" (for {context})"

    with st.expander(title, expanded=False):
        # Display wallet balances
        st.markdown("#### 💰 Wallet Balances")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spot + Earn", f"${spot_earn:,.2f}")
        with col2:
            st.metric("Funding", f"${funding:,.2f}")
        with col3:
            st.metric("Futures", f"${futures_balance:,.2f}")

        # Show detailed funding balances if available
        if funding_details:
            st.markdown("##### Funding Wallet Details")
            funding_df = pd.DataFrame(
                [
                    {"Asset": asset, "Balance": f"{balance:,.8f}"}
                    for asset, balance in funding_details.items()
                ]
            )
            st.dataframe(funding_df, use_container_width=True, hide_index=True)

        # Transfer form
        st.markdown("#### 🔄 Transfer Funds")

        # Wallet selection (outside form so it updates immediately)
        wallet_options = {
            "Funding → Spot": ("funding", "spot"),
            "Spot → Funding": ("spot", "funding"),
            "Spot → Futures": ("spot", "futures"),
            "Futures → Spot": ("futures", "spot"),
            "Funding → Futures": ("funding", "futures"),
            "Futures → Funding": ("futures", "funding"),
        }

        transfer_type = st.selectbox(
            "Transfer Type", options=list(wallet_options.keys()), index=0
        )

        # Asset selection - default to USDT but allow others from funding
        available_assets = ["USDT"]
        if funding_details:
            available_assets = list(funding_details.keys())

        asset = st.selectbox("Asset", options=available_assets, index=0)

        # Get source wallet balance for the selected asset
        source_wallet = wallet_options[transfer_type][0]
        source_balance = 0.0

        if source_wallet == "funding":
            source_balance = funding_details.get(asset, 0.0)
        elif source_wallet == "spot":
            # For spot, we'll use the spot_earn balance for USDT
            if asset == "USDT":
                source_balance = spot_earn
            # For other assets, we would need to fetch actual spot balances
        elif source_wallet == "futures":
            if asset == "USDT":
                source_balance = futures_balance

        # Ensure consistent rounding for display and validation
        source_balance = round(source_balance, 8)

        # Amount input
        suggested = 0.0
        if required_amount is not None and required_asset == asset:
            # If a specific amount is required for this asset
            suggested = min(max(required_amount, 0.0), source_balance)

        default_amount = (
            suggested
            if suggested > 0
            else min(100.0, source_balance)
            if source_balance > 0
            else 0.01
        )

        # Use a much higher max_value to prevent Streamlit validation errors
        # Set to 1 million by default, which should be higher than any reasonable balance
        max_balance = 1000000.0
        amount = st.number_input(
            f"Amount to transfer ({asset})",
            min_value=0.01,
            max_value=max_balance,
            value=float(round(default_amount, 2))
            if source_balance >= 0.01
            else 0.01,
            step=0.01,
            help=f"Maximum available in {source_wallet}: {source_balance:,.8f} {asset}",
        )

        # Real-time warning if amount exceeds available balance
        if amount > source_balance and source_balance > 0:
            st.warning(f"⚠️ Amount exceeds available balance in {source_wallet} wallet ({source_balance:,.8f} {asset}). "
                      f"You're trying to transfer {amount:,.8f} {asset}.")

        if required_amount is not None and required_asset is not None:
            st.caption(
                f"Required: {required_amount:,.2f} {required_asset} | Available in {source_wallet}: {source_balance:,.8f} {asset}"
            )

        # Transfer button
        submitted = st.button("Transfer Funds", type="primary")

        if submitted:
                if amount <= 0:
                    st.error("❌ Amount must be greater than zero.")
                    # Re-render the form to ensure submit button is available
                    st.rerun()
                    return

                if amount > source_balance:
                    st.error(
                        f"❌ Amount exceeds available balance in {source_wallet} wallet ({source_balance:,.8f} {asset})."
                    )
                    # Re-render the form to ensure submit button is available
                    st.rerun()
                    return

                with st.spinner("⏳ Executing transfer..."):
                    # Map transfer type to appropriate method
                    if transfer_type == "Funding → Spot":
                        result = asyncio.run(
                            tracker.transfer_funding_to_spot(
                                amount=float(amount), asset=asset, is_live=is_live
                            )
                        )
                    elif transfer_type == "Spot → Funding":
                        result = asyncio.run(
                            tracker.transfer_spot_to_funding(
                                amount=float(amount), asset=asset, is_live=is_live
                            )
                        )
                    elif transfer_type == "Spot → Futures":
                        result = asyncio.run(
                            tracker.transfer_spot_to_futures(
                                amount=float(amount), asset=asset, is_live=is_live
                            )
                        )
                    elif transfer_type == "Futures → Spot":
                        result = asyncio.run(
                            tracker.transfer_futures_to_spot(
                                amount=float(amount), asset=asset, is_live=is_live
                            )
                        )
                    elif transfer_type == "Funding → Futures":
                        result = asyncio.run(
                            tracker.transfer_funding_to_futures(
                                amount=float(amount), asset=asset, is_live=is_live
                            )
                        )
                    elif transfer_type == "Futures → Funding":
                        result = asyncio.run(
                            tracker.transfer_futures_to_funding(
                                amount=float(amount), asset=asset, is_live=is_live
                            )
                        )
                    else:
                        st.error("❌ Unsupported transfer type.")
                        # Re-render the form to ensure submit button is available
                        st.rerun()
                        return

                if result.success:
                    st.success("✅ Transfer executed successfully!")
                    for msg in result.messages:
                        st.write(f"• {msg}")
                else:
                    st.error("❌ Transfer failed:")
                    for err in result.errors:
                        st.write(f"• {err}")

                # Always rerun after processing to ensure form state is clean
                st.rerun()
