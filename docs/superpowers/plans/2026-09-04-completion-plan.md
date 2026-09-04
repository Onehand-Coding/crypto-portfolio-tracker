# Completion Plan (no-sell) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only "finish to targets, no sells" completion table to the CLI DCA menu and the Streamlit DCA page.

**Architecture:** Inline ~5-line `max()` math per frontend over data already in scope (`core_holdings_df` + `target_allocation`); no shared helper, no core/execution changes, React/API deferred per spec.

**Tech Stack:** Python asyncio CLI (`__main__.py`), Streamlit + pandas (`dca_page.py`), `uv` + `ruff`.

**Files:**
- Modify: `src/crypto_portfolio_tracker/__main__.py` (new `run_completion_plan()` + sub-menu prompt in `run_dca_menu`, after line 1485)
- Modify: `src/crypto_portfolio_tracker/dashboard/dca_page.py` (new section after line 131, inside the `if not core_portfolio.empty:` block)
- No new files. No test files (no display harness exists; spec §5 — numeric + manual verification instead).

**Conventions that bite:**
- `pd.NaT`/`NaN`: a SQL NULL arrives as `NaN`, which is truthy. Guard every value with `pd.isna()` before use; unknown renders as `—`/“unknown”, never `$0.00`.
- `core_holdings_df` columns: `symbol`, `value_usd`, `core_allocation`. Target-only assets are absent rows (treat as `0.0`); non-target holdings are ignored.
- CLI option 5 is blocked in offline mode (`unavailable_offline`, `__main__.py:1843`) — the completion plan inherits that; no behavior change.
- Streamlit DCA page returns early in offline mode (`dca_page.py:24`) — unchanged.

---

### Task 1: CLI completion plan

**Files:**
- Modify: `src/crypto_portfolio_tracker/__main__.py:1485` (insert sub-menu prompt after `loop = asyncio.get_event_loop()`)
- Modify: `src/crypto_portfolio_tracker/__main__.py:1465` (new `run_completion_plan()` above `run_dca_menu`)

- [ ] **Step 1: Insert the sub-menu prompt in `run_dca_menu`**

Old (lines 1485–1487):
```python
    loop = asyncio.get_event_loop()

    try:
        # 1. Display current USDT balance
```

New:
```python
    loop = asyncio.get_event_loop()

    # 0. Completion plan vs DCA with new funds. The plan needs no USDT, so
    # it is offered before the balance gate below (a zero balance returns).
    plan_choice = await loop.run_in_executor(
        None, input,
        "\n1. 📐 Completion plan (finish targets, no sells)\n"
        "2. 💸 DCA with new funds\nSelect (1-2, Enter to return): ",
    )
    plan_choice = (plan_choice or "").strip()
    if not plan_choice:
        print("Returning to main menu...")
        return
    if plan_choice == "1":
        await run_completion_plan(tracker)
        return
    if plan_choice != "2":
        print("❌ Invalid choice. Please select 1 or 2.")
        return

    try:
        # 1. Display current USDT balance
```

- [ ] **Step 2: Add `run_completion_plan()` above `run_dca_menu` (line 1465)**

```python
async def run_completion_plan(tracker: CryptoPortfolioTracker):
    """Prints how much of each target asset is still needed to finish.

    Read-only: implied total T = max(current / weight) over anchored
    holdings, need = max(0, T * weight - current). Never suggests sells.
    """
    print("\n--- 📐 Completion Plan (finish targets, no sells) ---")

    metrics = await tracker.calculate_portfolio_metrics()
    core = metrics.get("core_holdings_df", pd.DataFrame())
    target = tracker.config.get("target_allocation", {}) or {}
    if not target:
        print("❌ No target allocation configured.")
        return

    values: Dict[str, float] = {}
    unknown: List[str] = []
    for asset in target:
        row = core[core["symbol"] == asset] if not core.empty else core
        if row.empty:
            values[asset] = 0.0
            continue
        raw = row["value_usd"].iloc[0]
        # NaN is truthy: a missing price must read as unknown, never $0.00.
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            unknown.append(asset)
            values[asset] = 0.0
        else:
            values[asset] = float(raw)

    anchors = {
        s: values[s] / float(w)
        for s, w in target.items()
        if s not in unknown and values[s] > 0 and float(w) > 0
    }
    if not anchors:
        print("No holdings to anchor from yet — nothing held, nothing to finish.")
        return

    total = max(anchors.values())
    winner = max(anchors, key=lambda s: anchors[s])
    print(f"   Implied finished size: ${total:,.2f} (anchored by {winner})")

    total_need = 0.0
    for asset, weight in target.items():
        goal = total * float(weight)
        need = max(0.0, goal - values[asset])
        total_need += need
        current_txt = "unknown" if asset in unknown else f"${values[asset]:,.2f}"
        print(
            f"   {asset}: target {float(weight) * 100:.1f}% = ${goal:,.2f} "
            f"(have {current_txt}) → buy ${need:,.2f}"
        )
    print(f"\n   Additional cash needed: ${total_need:,.2f}")
```

- [ ] **Step 3: Verify the math against the spec example (no harness needed)**

Run:
```bash
uv run python -c "
vals = {'BTC': 146.49, 'ETH': 24.49, 'SOL': 0.0, 'RENDER': 0.0, 'TAO': 0.0, 'AVAX': 0.0, 'LINK': 0.0, 'ONDO': 0.0}
t = {'BTC': .35, 'ETH': .3, 'SOL': .1, 'RENDER': .06, 'TAO': .06, 'AVAX': .05, 'LINK': .05, 'ONDO': .03}
T = max(v / t[s] for s, v in vals.items() if v > 0)
needs = {s: max(0.0, T * w - vals[s]) for s, w in t.items()}
print(f'T={T:.2f} ETH_need={needs[\"ETH\"]:.2f} total={sum(needs.values()):.2f}')
"
```
Expected: `T=418.54 ETH_need=101.07 total=247.56`

- [ ] **Step 4: Lint the touched file, run the suite**

Run: `uv run ruff check src/crypto_portfolio_tracker/__main__.py`
Expected: only pre-existing warnings, none on the new `run_completion_plan` lines.
Run: `uv run pytest -q`
Expected: pass (failures, if any, must match the pre-change baseline — stash and compare).

- [ ] **Step 5: Manual check**

Run: `uv run track-portfolio-cli`, choose 5, then 1. Expect the worked-example-shaped table (live values will differ; shape is what matters).

- [ ] **Step 6: Commit**

```bash
git add src/crypto_portfolio_tracker/__main__.py
git commit -m "feat: add no-sell completion plan to CLI DCA menu"
```

---

### Task 2: Streamlit completion section

**Files:**
- Modify: `src/crypto_portfolio_tracker/dashboard/dca_page.py:131` (insert after `st.session_state.current_portfolio_value = total_value`, still inside `if not core_portfolio.empty:`)

- [ ] **Step 1: Insert the section**

Old (lines 129–135):
```python
        # Store current portfolio value for DCA calculations
        total_value = core_portfolio["value_usd"].sum()
        st.session_state.current_portfolio_value = total_value
    else:
        st.info("No core portfolio data available")
```

New (replaces the tail — keep the two stored lines, add the section before `else:`):
```python
        # Store current portfolio value for DCA calculations
        total_value = core_portfolio["value_usd"].sum()
        st.session_state.current_portfolio_value = total_value

        st.markdown("#### 📐 Finish to targets (no sells)")
        st.caption("Implied total comes from whichever holding demands the "
                   "biggest finished size, so nothing held ever needs selling.")
        plan_values: dict[str, float] = {}
        plan_unknown: list[str] = []
        for asset in target_allocation.keys():
            asset_row = core_portfolio[core_portfolio["symbol"] == asset]
            if asset_row.empty:
                plan_values[asset] = 0.0
                continue
            raw_value = asset_row["value_usd"].iloc[0]
            # NaN is truthy: unknown must render as em dash, never $0.00.
            if raw_value is None or pd.isna(raw_value):
                plan_unknown.append(asset)
                plan_values[asset] = 0.0
            else:
                plan_values[asset] = float(raw_value)

        plan_anchors = {
            s: plan_values[s] / float(w)
            for s, w in target_allocation.items()
            if s not in plan_unknown and plan_values[s] > 0 and float(w) > 0
        }
        if not plan_anchors:
            st.info("No holdings to anchor from yet — nothing held, nothing to finish.")
        else:
            plan_total = max(plan_anchors.values())
            plan_winner = max(plan_anchors, key=lambda s: plan_anchors[s])
            st.metric("Implied finished size",
                      f"${plan_total:,.2f}",
                      help=f"Anchored by {plan_winner}")
            plan_rows = []
            plan_need_total = 0.0
            for asset, weight in target_allocation.items():
                goal = plan_total * float(weight)
                need = max(0.0, goal - plan_values[asset])
                plan_need_total += need
                plan_rows.append(
                    {
                        "Asset": asset,
                        "Target %": f"{float(weight) * 100:.2f}%",
                        "Target Value": f"${goal:,.2f}",
                        "Current Value": ("—" if asset in plan_unknown
                                          else f"${plan_values[asset]:,.2f}"),
                        "Still to Buy": f"${need:,.2f}",
                    }
                )
            st.dataframe(pd.DataFrame(plan_rows),
                         use_container_width=True, hide_index=True)
            st.metric("Additional cash needed", f"${plan_need_total:,.2f}")
    else:
        st.info("No core portfolio data available")
```

- [ ] **Step 2: Verify the math (same check as Task 1, Step 3)**

Run the identical `uv run python -c` command.
Expected: `T=418.54 ETH_need=101.07 total=247.56`

- [ ] **Step 3: Lint the touched file, run the suite**

Run: `uv run ruff check src/crypto_portfolio_tracker/dashboard/dca_page.py`
Expected: only pre-existing warnings, none on the new section.
Run: `uv run pytest -q`
Expected: pass (or identical to pre-change baseline).

- [ ] **Step 4: Manual check**

Run: `uv run track-portfolio-web`, open DCA after a sync. Expect the “Finish to targets” table under Current-vs-Target plus the two metrics; with an empty portfolio expect the info line.

- [ ] **Step 5: Commit**

```bash
git add src/crypto_portfolio_tracker/dashboard/dca_page.py
git commit -m "feat: add no-sell completion plan to Streamlit DCA page"
```

---

## Out of scope (do not build)

- FastAPI endpoint + React screen (deferred until the parity branch merges).
- Shared core helper in `dca_manager.py` (duplication is deliberate; keeps `src/` diff to additive UI).
- Execution of the plan, CSV export, hypothetical-anchor input for empty portfolios.
- CONTEXT.md changes (no durable-knowledge change: additive UI following existing rules).
