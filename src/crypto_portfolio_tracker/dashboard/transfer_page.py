import streamlit as st


def render_transfer_page(dashboard):
        """Renders the transfer funds page."""
        st.header("💵 Transfer Funds")
        st.markdown("Transfer funds from funding wallet to spot wallet for trading.")

        # Get tracker first
        tracker = dashboard.initialize_tracker()
        if not tracker:
            st.error("❌ Failed to initialize tracker.")
            return

        # --- Trading Status Banner ---
        is_live = dashboard.config_manager.is_live
        is_testnet = dashboard.config_manager.is_testnet_mode
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

        # Get current balances
        try:
            # Get USDT balances
            usdt_balances = tracker.get_available_usdt_balance()

            # Get comprehensive spot wallet information
            spot_balances_df = tracker.fetch_binance_balances()
            total_spot_value = 0.0
            spot_holdings = []

            if not spot_balances_df.empty:
                symbols = spot_balances_df["symbol"].tolist()
                current_prices = tracker._get_current_prices(symbols)

                for _, row in spot_balances_df.iterrows():
                    symbol = row["symbol"]
                    quantity = row["quantity"]
                    price = current_prices.get(symbol, 0.0)
                    asset_value = quantity * price
                    total_spot_value += asset_value

                    if asset_value > 1.0:  # Only show holdings worth more than $1
                        spot_holdings.append(
                            {
                                "symbol": symbol,
                                "quantity": quantity,
                                "price": price,
                                "value": asset_value,
                            }
                        )

            # Display comprehensive balance information
            st.markdown("#### 💰 Wallet Balance Overview")

            # Show individual holdings
            if spot_holdings:
                st.markdown("**📊 Current Spot Wallet Holdings:**")
                for holding in sorted(
                    spot_holdings, key=lambda x: x["value"], reverse=True
                ):
                    st.text(
                        f"   {holding['symbol']}: ${holding['value']:,.2f} ({holding['quantity']:.8g} @ ${holding['price']:,.2f})"
                    )

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spot Wallet Value", f"${total_spot_value:,.2f}")
            with col2:
                st.metric("USDT in Spot + Earn", f"${usdt_balances['spot_earn']:,.2f}")
            with col3:
                st.metric("USDT in Funding", f"${usdt_balances['funding']:,.2f}")
            with col4:
                st.metric("Total USDT Available", f"${usdt_balances['total']:,.2f}")

            if usdt_balances["funding"] <= 0:
                st.error("❌ No USDT available in funding wallet for transfer.")
                return

            # Transfer form
            st.markdown("#### 💵 Transfer Configuration")

            with st.form("transfer_form"):
                transfer_amount = st.number_input(
                    "Amount to Transfer (USDT)",
                    min_value=0.01,
                    max_value=float(usdt_balances["funding"]),
                    value=min(100.0, float(usdt_balances["funding"])),
                    step=0.01,
                    help=f"Maximum available: ${usdt_balances['funding']:,.2f}",
                )

                # Show transfer impact
                st.markdown("** Transfer Impact:**")
                st.text(f"   After transfer, you'll have:")
                st.text(
                    f"   - ${usdt_balances['spot_earn'] + transfer_amount:,.2f} USDT in Spot + Earn"
                )
                st.text(
                    f"   - ${usdt_balances['funding'] - transfer_amount:,.2f} USDT remaining in Funding"
                )
                st.text(
                    f"   - Total trading power: ${total_spot_value + transfer_amount:,.2f}"
                )

                submitted = st.form_submit_button("Transfer Funds")

                if submitted:
                    if transfer_amount <= 0:
                        st.error("❌ Amount must be greater than zero.")
                        return

                    if transfer_amount > usdt_balances["funding"]:
                        st.error(
                            f"❌ Amount exceeds available funding balance (${usdt_balances['funding']:,.2f})."
                        )
                        return

                    # Execute transfer
                    with st.spinner(" Executing transfer..."):
                        import asyncio

                        result = asyncio.run(
                            tracker.transfer_funding_to_spot(
                                asset="USDT", amount=transfer_amount, is_live=is_live
                            )
                        )

                    # Display results
                    st.markdown("#### 📋 Transfer Results")

                    if result.success:
                        st.success("✅ Transfer executed successfully!")
                        for msg in result.messages:
                            st.write(f"• {msg}")
                    else:
                        st.error("❌ Transfer failed:")
                        for err in result.errors:
                            st.write(f"• {err}")

                    # Refresh the page to show updated balances
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")
