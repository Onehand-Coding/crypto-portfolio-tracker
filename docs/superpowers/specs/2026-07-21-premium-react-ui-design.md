# Premium React UI — Design Spec

**Date:** 2026-07-21
**Status:** Approved for planning
**Supersedes:** none

---

## 1. Problem

The project has a capable Python core (~13,500 lines across 17 modules) exposed through a
Streamlit dashboard (~6,100 lines, 15 page modules). The engine is sophisticated; the
interface is the limiting factor. Two specific complaints drive this work:

- **Interaction latency.** Streamlit re-runs the entire script on every widget change.
- **Presentation ceiling.** Streamlit cannot express the density or the interaction
  patterns the domain needs (drill-down, simulation-before-execution, event-marked charts).

A third motive is explicit and legitimate: building FastAPI + React on a real financial
domain is a deliberate skill-development goal.

## 2. Verdict on the prior design direction

An earlier ChatGPT exploration produced a 12-screen brief, rendered by Stitch into
11 screens (project `915468954340907029`, "Crypto Investment Operating System").

**Sound and adopted:**

- Workflow-shaped navigation (Portfolio / Analyze / Strategies / Execute / Data / System)
  in preference to a menu mirroring backend modules.
- `analyse → simulate → confirm → execute → verify` as a structural guarantee, not a
  convention. This matters more than usual because execution moves real money.
- Elevating **FIFO cost basis vs. net invested capital** to a first-class UI concept.
  This is the strongest idea in the exploration and becomes this product's signature.

**Rejected or corrected:**

- The brief was written without reference to the actual portfolio or codebase. It assumed
  no API layer question, was unaware a Textual TUI had been attempted and abandoned
  (`src/crypto_portfolio_tracker/tui/` retains only `__pycache__`; source was never
  committed), and specified figures ~430x the real portfolio.
- It recommended React partly because "you are learning web development" — a learning
  rationale presented as a product rationale. Both are real; they are stated separately here.

**Defects found in the generated Stitch screens** (verified against the rendered output):

| # | Defect | Evidence |
|---|---|---|
| 1 | FIFO cost basis and net invested rendered as equal | Asset Detail: both `$7,652.12`. These are distinct concepts; in real data they differ 2.6x. |
| 2 | Contradictory allocation across screens | BTC is 42%/35%, 18.5%/20%, and 45%/38% on three screens. |
| 3 | Semantic colour broken on Cockpit | `designMd` prose specifies indigo/mint/crimson; generated tokens are Material-3 purple. Gains and losses render identically. |
| 4 | Density contradicts the brief | Cockpit wastes ~25% vertical space, Rebalancing ~20%, against a stated "control room" density goal. |
| 5 | Decorative chrome | `WS_NODE_01_ACTIVE`, `SYS_ADMIN_01`, "Power User Access" convey nothing. |
| 6 | Coverage gaps | Reports & Exports absent; Wallets (Spot/Futures/Funding) and Market absent though both exist in Streamlit. |

**Decision:** Stitch screens are **reference, not source of truth**. React is hand-built
against the corrected design system. The Stitch design system is corrected first (§5).

## 3. Goals

1. A daily-use interface that is genuinely better than the Streamlit dashboard.
2. Feature parity with the existing Streamlit app — no capability removed.
3. Perceived speed: page loads read local SQLite, never block on Binance.
4. Make the two accounting models legible and impossible to confuse.
5. Build competence in FastAPI + React + TypeScript.

## 4. Non-goals

- Rewriting or refactoring the Python core. It is wrapped, not modified.
- Removing the Streamlit dashboard. It runs in parallel until parity is reached and is
  removed only by a separate, later decision.
- Multi-user support, authentication, or hosting. This is a single-operator local tool.
- Mobile-first layout. Desktop-first (1440px/1920px); smaller screens degrade, not lead.
- Porting `visualizations.py` (844 lines, Plotly). Charts are rebuilt client-side.

## 5. Design system (corrected)

The generated Stitch tokens drifted from the `designMd` prose. The prose is authoritative.
These values are the contract; the Stitch design system is updated to match before any
screen work begins.

```
SURFACES                        SEMANTIC
level-0  #0B0C0E  app bg        positive  #00C076  mint
level-1  #14161A  panels        negative  #E04E4E  crimson
level-2  #1C1F26  modals        warning   #D9A441  amber
border   #2D3139  1px structural  action  #4F46E5  indigo

TEXT                            TYPE
primary    #F9FAFB              Inter          UI, labels, prose
secondary  #9BA3AF              JetBrains Mono ALL numerics, tickers, addresses

SHAPE                           SPACING
4px  buttons, inputs            4px base unit
6px  panels, cards              12px panel padding
0px  active-state selection bars
```

**Rules:**

- Depth is tonal layering plus 1px borders. No shadows, no glow, no glassmorphism.
- All numerics use JetBrains Mono and are right-aligned in tables so decimals align.
- **Colour is never the sole carrier of meaning.** Gain/loss always pairs the colour with
  an explicit sign (`+`/`-`) and directional glyph, so the information survives
  colour-blindness and greyscale.
- Semantic colours are reserved for semantics. Mint means gain or success — never
  decoration, never an asset series colour. Categorical series use a separate palette.

**Naming:** "Alpha Terminal" is a placeholder and is replaced. Decorative status chrome
is deleted.

## 6. Architecture

```
crypto-portfolio-tracker/
├── src/crypto_portfolio_tracker/   UNTOUCHED — the engine
├── api/
│   ├── main.py       FastAPI; serves frontend/dist and /api/*
│   ├── cache.py      TTL cache over core calls
│   ├── deps.py       config, db, tracker singletons
│   ├── routes/       portfolio, capital, assets, wallets, sync,
│   │                 rebalance, dca, trading, profit, backtest,
│   │                 market, analysis, reports, system
│   └── schemas/      Pydantic — the contract React types against
└── frontend/         Vite + React + TypeScript + Tailwind
    └── dist/         built bundle, served by FastAPI
```

**Wiring.** One process, one port. FastAPI serves the built bundle in production; Vite
dev-proxies to FastAPI in development for hot reload. No CORS configuration. Entry point
is a single command. Streamlit continues to run independently on its own port.

**Caching is load-bearing.** `calculate_portfolio_metrics()` is `async` and reaches for
live prices. Wrapping it naively would reproduce Streamlit's latency behind new paint.
Therefore:

- Read endpoints serve from SQLite (`get_holdings`, `get_all_snapshots`,
  `get_all_transactions`) and return a staleness timestamp.
- Binance is contacted only on **explicit** user-initiated sync.
- The UI displays staleness rather than hiding it behind a spinner.

**Sync progress streams over SSE.** `binance_fetcher` already chunks 90-day windows into
30-day segments with per-chunk logging; that progress becomes visible in the UI instead of
being hidden behind an indeterminate spinner.

**Testnet is a first-class UI state.** `config.is_testnet_mode` already switches to a
separate database (`data/testnet_portfolio.db`) with separate backups, and
`trade_executor.py` branches on it in three places. The environment indicator must be
unmissable and globally persistent. Testnet and live data must never be presentable as
the same thing.

## 7. Verified core surface

These exist and were confirmed by reading the source. Routes bind to them directly.

| Capability | Call |
|---|---|
| Portfolio metrics | `portfolio_analyzer.calculate_portfolio_metrics()` (async) |
| Net invested capital | `portfolio_analyzer.calculate_total_invested_capital()` |
| Capital-flow source rows | `database.get_invested_capital_transactions()` |
| FIFO cost basis | `utils.calculate_fifo_cost_basis(transactions_df)` |
| FIFO realized gains | `utils.calculate_fifo_realized_gains(transactions_df)` |
| Holdings | `database.get_holdings()` |
| Transactions | `database.get_all_transactions()` |
| Historical snapshots | `database.get_all_snapshots()` |
| Historical prices | `database.get_historical_prices()` |
| Available USDT | `portfolio_tracker.get_available_usdt_balance()` |
| DCA proposals | `portfolio_tracker.calculate_proportional_dca()`, `calculate_target_weight_dca()` |
| DCA validation | `portfolio_tracker.validate_dca_amount()`, `validate_dca_execution()` |
| Earn redemption | `portfolio_tracker.redeem_from_earn()` |

**Not yet enumerated.** The public surface of `rebalancing_logic.py`,
`profit_taking_logic.py`, `rebalancing_backtester.py`, `crypto_trend_analyzer.py`, and
`trade_executor.py` has not been read in full. Routes for those screens must be grounded
by reading each module at implementation time. No endpoint may be written against an
assumed method name.

**Domain nuance to preserve:** `__main__.py:231` computes crypto-only unrealized P/L
*excluding stablecoins*, "for more intuitive performance." The UI must not silently
discard this distinction.

## 8. Screens

Fifteen screens. Full parity with Streamlit plus two additions (Capital Flow, and Reports
which Stitch omitted).

| # | Screen | Notes |
|---|---|---|
| 1 | Cockpit | Dual accounting headline; holdings; event-marked performance; review items |
| 2 | Capital Flow | **New.** Fiat → USDT → asset provenance |
| 3 | Portfolio Overview | Snapshot timeline via `get_all_snapshots()` |
| 4 | Asset Detail | Tabs: Overview / Cost Basis / Transactions / Strategy / Technical |
| 5 | Wallets | **Missing from Stitch.** Spot / Futures / Funding |
| 6 | Sync & Transactions | SSE chunk progress; per-endpoint health; provenance drawer |
| 7 | Rebalancing | Current vs target, proposal, simulation, execution |
| 8 | DCA | Proportional / target-weight / custom; resulting-allocation preview |
| 9 | Profit-Taking | Rules, simulation, history |
| 10 | Trading | Order review with portfolio-impact preview before confirm |
| 11 | Backtesting | Strategy comparison |
| 12 | Technical Analysis | RSI / MACD / SMA / momentum, multi-timeframe |
| 13 | Market | **Missing from Stitch.** Discovery, movers |
| 14 | Reports & Exports | **Missing from Stitch.** CSV / Excel / HTML |
| 15 | Settings & System Health | API status, DB health, testnet indicator |

### 8.1 Cockpit headline

The signature element. Both accounting models, side by side, each labelled with the
question it answers:

```
$57.78  portfolio value

NET INVESTED BASIS          FIFO BASIS
-$18.63  (-24.38%)          -$142.85  (-71.52%)
on $76.41 net in            on $199.75 cost basis
"did I make money?"         "are my holdings underwater?"
```

Both are correct; they answer different questions. Real data diverges 7.7x. The Stitch
mock erased this by printing them equal — that defect must not be reproduced.

### 8.2 Capital Flow

Not in the original brief. Justified because it answers "why does my portfolio show this
number," and because a real bug — a failed yfinance PHP/USD lookup silently zeroing
`price_usd` — hid $50.41 of inflow and inverted reported P/L from +122% to -24%.

Rows carry **provenance**: whether a rate was computed from yfinance or fell back to the
USDT peg. This makes that class of bug visible rather than invisible.

### 8.3 Dual-state requirement (applies to every screen)

Every screen specifies two designed states:

- **Populated** — the rich case, exercised against testnet data.
- **Constrained** — small balances, sparse history, dust positions.

The constrained state must state the constraint plainly. A rebalancing screen must render
"below minimum notional — not executable" rather than a confident action plan for a $4
trade, an empty panel, or a dead button. Dust positions collapse into an aggregate row
rather than presenting sub-$0.40 holdings as a meaningful allocation.

This is a truthfulness requirement, not a scope reduction. No feature is cut.

## 9. Sequencing

Ordered so the app is usable throughout rather than only at the end.

| Step | Delivers | Usable? |
|---|---|---|
| 1 | Design system correction (Stitch + repo tokens) | — |
| 2 | API skeleton, cache, schemas, testnet plumbing | — |
| 3 | App shell, routing, design system in code | — |
| 4 | Cockpit · Capital Flow · Sync | **daily driver** |
| 5 | Asset Detail · Wallets · Overview | + |
| 6 | Rebalancing · DCA · Profit-Taking | + |
| 7 | Trading · Backtesting · Technical Analysis | + |
| 8 | Market · Reports · Settings & Health | parity |

Steps 1–3 unblock; 4 onward add. Streamlit is retired only after step 8, by separate
decision.

## 10. Verification

- **Core untouched:** `grep -L "import streamlit" src/crypto_portfolio_tracker/*.py`
  must continue to show zero Streamlit coupling; the existing 57 tests must stay green
  throughout, since the core is not modified.
- **API:** pytest against routes, with the core mocked. Schema round-trip tests.
- **Contract:** TypeScript types generated from the OpenAPI schema, so a backend change
  that breaks the frontend fails at build rather than at runtime.
- **Dual states:** every screen has a fixture for both populated and constrained states.
- **Accounting correctness:** golden-value tests asserting FIFO and net-invested figures
  against the known real portfolio ($57.78 / $76.41 / $199.75). These are the numbers the
  UI must never conflate.
- **Parity gate:** each Streamlit page is checked off only when its React equivalent is
  confirmed to produce matching figures.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Scope: 15 screens is a long build | Sequencing (§9) makes it usable from step 4; steps are independently shippable |
| Two UIs to maintain during transition | Core is shared and untouched; only presentation is duplicated |
| Endpoints written against assumed methods | §7 forbids it — read each module before binding |
| Cache serves stale figures silently | Staleness is displayed, not hidden; sync is explicit |
| Testnet and live data confused | Separate DB already enforced; global unmissable indicator required |
| Charts regress vs Plotly | Chart parity checked per screen during the parity gate |
