import streamlit as st
import asyncio
import pandas as pd
from typing import Dict, Any


def render_redeem_widget(dashboard, context: str = "") -> None:
    """
    Widget to redeem assets from Binance Earn products.
    
    Allows users to select which assets to redeem and how much.
    """
    if dashboard is None:
        st.warning("Redeem widget requires a valid dashboard instance.")
        return
        
    tracker = dashboard.initialize_tracker()
    if not tracker:
        st.error("❌ Failed to initialize tracker.")
        return

    # Trading status
    is_live = dashboard.config_manager.is_live
    is_testnet = dashboard.config_manager.is_testnet_mode

    title = "💸 Redeem from Earn"
    if context:
        title += f" (for {context})"

    with st.expander(title, expanded=False):
        # Check if we're in testnet mode
        if is_testnet:
            st.warning("⚠️ Earn redemption is not available in TESTNET mode.")
            st.info("Switch to MAINNET mode to use this feature.")
            return
            
        try:
            # Get current spot balances to identify which assets might have Earn positions
            spot_balances_df = tracker.fetch_binance_balances()
            
            if spot_balances_df.empty:
                st.info("No spot balances found. Cannot check for Earn positions.")
                return
                
            # Get Earn balances
            earn_balances = tracker.fetcher.fetch_simple_earn_balances(spot_balances_df)
            
            if not earn_balances:
                st.info("No assets found in Binance Earn products.")
                return
                
            # Filter out zero balances
            non_zero_earn = {k: v for k, v in earn_balances.items() if v > 1e-8}
            
            if not non_zero_earn:
                st.info("No non-zero balances found in Binance Earn products.")
                return
                
            # Display current Earn positions
            st.markdown("#### Current Earn Positions")
            earn_data = []
            for asset, balance in non_zero_earn.items():
                earn_data.append({
                    "Asset": asset,
                    "Balance": f"{balance:.8f}"
                })
            
            earn_df = pd.DataFrame(earn_data)
            st.dataframe(earn_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Redemption form
            st.markdown("#### Redeem Assets")
            
            # Select asset to redeem
            asset_options = list(non_zero_earn.keys())
            selected_asset = st.selectbox(
                "Select Asset to Redeem",
                options=asset_options,
                key=f"redeem_asset_{context}"
            )
            
            if selected_asset:
                max_balance = non_zero_earn[selected_asset]
                st.caption(f"Available balance: {max_balance:.8f} {selected_asset}")
                
                # Input redemption amount
                redeem_amount = st.number_input(
                    f"Amount to Redeem ({selected_asset})",
                    min_value=0.0,
                    max_value=max_balance,
                    value=min(10.0, max_balance) if max_balance >= 10.0 else max_balance,
                    step=0.00000001,
                    format="%.8f",
                    key=f"redeem_amount_{context}"
                )
                
                # Show warning for live trading
                if is_live:
                    st.warning("⚠️ This will execute a REAL redemption from your Binance Earn account!")
                else:
                    st.info("ℹ️ This is a DRY RUN. No actual redemption will be executed.")
                
                # Redeem button
                if st.button("Redeem", key=f"redeem_button_{context}", type="primary"):
                    if redeem_amount <= 0:
                        st.error("❌ Redemption amount must be greater than zero.")
                        return
                        
                    if redeem_amount > max_balance:
                        st.error(f"❌ Amount exceeds available balance ({max_balance:.8f} {selected_asset}).")
                        return
                    
                    with st.spinner("Executing redemption..."):
                        try:
                            # Execute redemption directly using the new public method
                            result = tracker.redeem_from_earn(
                                asset=selected_asset,
                                amount=redeem_amount,
                                is_live=is_live
                            )
                            
                            # Display results
                            if result.success:
                                st.success("✅ Redemption executed successfully!")
                                for msg in result.messages:
                                    st.write(f"• {msg}")
                                    
                                # Add a note about when funds will be available
                                st.info("ℹ️ Redeemed funds will be available in your Spot wallet shortly.")
                            else:
                                st.error("❌ Redemption failed:")
                                for msg in result.messages:
                                    st.write(f"• {msg}")
                                if result.errors:
                                    st.caption("Technical details:")
                                    for err in result.errors:
                                        st.caption(f"• {err}")
                                
                        except Exception as e:
                            st.error(f"❌ Error during redemption: {str(e)}")
                            
        except Exception as e:
            st.error(f"❌ Error loading Earn positions: {str(e)}")
            st.info("Please try again or check your Binance API permissions.")