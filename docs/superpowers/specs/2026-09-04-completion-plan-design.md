# Completion Plan (no-sell) — Design

Date: 2026-09-04 | Branch: `feat/completion-plan` (off local `main`) | Status: approved for CLI + Streamlit

## 1. Goal

Answer: "given my target allocation and what I already hold, how much of each
asset do I still need to buy to finish — without selling anything?"

Money needed is the **output**, not an input. This is the inverse of DCA
("I have $X, where does it go?") and distinct from rebalance (which suggests
sells at the current total). Verified: no existing feature does this.

## 2. Semantics (locked)

- Inputs: `target_allocation` weights from config + current core-holding values.
- Anchor set `H` = holdings with known `value_usd > 0` and weight `w > 0`.
- Implied finished total `T = max(H_i / w_i)` — the smallest total at which no
  held asset is over its target. The `max()` means the anchor is whichever
  holding demands the biggest total (BTC today at $418.54; ETH would take over
  if it ever demanded more).
- Per asset: `target_i = T * w_i`, `need_i = max(0, target_i - current_i)`.
- `additional_total = sum(need_i)`. The anchor asset(s) at the max ratio show
  $0 by construction.
- Worked example (BTC $146.49, ETH $24.49, rest $0; BTC .35 ETH .30 SOL .10
  RENDER .06 TAO .06 AVAX .05 LINK .05 ONDO .03): T = $418.54, ETH needs
  $101.07, SOL $41.85, …, total new money $247.56, BTC $0.

## 3. Scope

**Now (this branch):** CLI + Streamlit, additive UI only.
- CLI: new option in the DCA menu (`run_dca_menu`) printing the completion
  table. No changes to execution, validation, or balances.
- Streamlit (`dashboard/dca_page.py`): "Finish to targets (no sells)" section
  under the existing Current-vs-Target table, reusing page-state metrics.
- Math is ~5 lines on plain dicts, duplicated inline per frontend. No shared
  core function, no changes to `dca_manager.py`, accounting, or executors.

**Deferred:** FastAPI endpoint + React screen. `main` has no DCA screen or
strategy routes; the parity branch built both from scratch. Building React here
guarantees a rebase conflict and possible rework — revisit after parity merges.

**`src/` permission:** granted by owner 2026-09-04, scoped to these two UI
files, additive display only.

## 4. Edge cases

- Assets outside `target_allocation`: ignored (same as DCA core view).
- Zero/missing weight: excluded from the `max()` (no division by zero).
- Unpriced holdings (unknown value): excluded from the `max()`, shown as
  unknown (em dash in Streamlit, "unknown" in CLI) — never $0.00, per the
  unknown-is-never-zero rule.
- Empty portfolio (no H): show "no holdings to anchor from" instead of a table.
- Over-allocated non-anchor assets: clamped to $0 need, never a sell.
- Rows: show every target asset, including $0 anchor rows (they prove the
  no-sell invariant). No dust filter — small needs are still informative here.

## 5. Verification

- Manual: CLI option + Streamlit section against the worked example above.
- `uv run pytest -q` and `uv run ruff check` on touched files (full lint scope
  per CONTEXT; `src/` has a pre-existing backlog — no new warnings).
- No new automated tests: markup/menu display code has no test harness in this
  repo; the math will be covered when the API endpoint lands with the React
  work (typed schemas + `tests/api/`).
