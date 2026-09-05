# Crypto Portfolio Tracker

> Persistent project memory — durable knowledge only.
>
> This file is NOT a changelog, session log, TODO list, or git history.
> Git already records what changed and when. This file records what
> Git cannot: architecture, philosophy, non-goals, decisions, conventions,
> domain rules, and the current state of unfinished or partial work.
>
> Update ONLY when durable project knowledge changes (see Maintenance
> Rules at the bottom). Do not touch this file for commits, bug fixes,
> refactors, or session summaries.

---

## 1. Project Overview

**Purpose:** Personal cryptocurrency portfolio tracker and rebalancing advisor. It
syncs the full transaction history from Binance, enriches it with historical
prices, and reports portfolio value and profit/loss under two distinct accounting
models. It also generates rebalancing, DCA, and profit-taking suggestions from
multi-timeframe technical analysis, and can execute trades.

**Target Users:** A single owner-operator tracking their own money. Not a product,
not multi-tenant.

**Current Milestone:** Beta — working and in real use, still stabilising.

**Current Development Focus:** A React UI over a FastAPI layer, alongside the
existing Streamlit dashboard.

---

## 2. Technology Stack

### Frontend
| Component | Choice |
|---|---|
| Framework | React 19 (Vite 8, TypeScript 6) — plus Streamlit for the original dashboard |
| State Management | Local component state; no store library |
| Routing | react-router-dom 7 |
| Styling | Tailwind v3 over CSS custom-property design tokens |
| Local Storage | None |

### Backend
| Component | Choice |
|---|---|
| Framework | FastAPI (wraps the Python core); Streamlit calls the core directly |
| ORM | None — hand-written SQL via `DatabaseManager` |
| Validation | Pydantic v2 (`api/schemas/`) |
| Authentication | None — binds to 127.0.0.1 only |

### Database
| Component | Choice |
|---|---|
| Database | SQLite (`data/portfolio.db`, `data/testnet_portfolio.db`) |
| Migrations | Hand-rolled `_check_and_update_schema()` on init, with auto-backup |

### Infrastructure
| Component | Choice |
|---|---|
| Container | None |
| Reverse Proxy | None |
| Deployment | Runs locally; `uv` manages the environment |
| CI | None |

---

## 3. Repository Structure

```text
src/crypto_portfolio_tracker/   Python core: sync, analysis, trading, Streamlit UI
api/                            FastAPI layer wrapping the core (read-only over src/)
frontend/                       React UI (Vite, TypeScript, Tailwind)
tests/                          Core test suite; tests/api/ covers the FastAPI layer
config/                         Runtime configuration
data/                           SQLite DBs, api_cache/, exports
docs/                           Specs, plans, design notes
```

---

## 4. Architecture

**Overall:** A facade (`CryptoPortfolioTracker`) orchestrates specialised
components — fetching, price enrichment, persistence, analysis, execution. Two
front ends sit on top of that core: Streamlit calls it directly, and a FastAPI
layer wraps it for the React UI. The guiding principle throughout is that an
unknown value must never be presented as a known one; this is a tool for
deciding what to do with real money, and a confident wrong number is worse than
a visible gap.

**Folder Organization:** The core is flat and module-per-responsibility
(`binance_fetcher.py`, `price_enricher.py`, `portfolio_analyzer.py`, …). The API
layer mirrors that: `routes/` per resource, `schemas/` per response shape, with
one module each for caching, serialization, accounting, and the sync runner.

**Data Flow:**

```text
Sync (the only path that contacts Binance):
  BinanceFetcher → PriceEnricher → DatabaseManager → PortfolioAnalyzer
                                                   → MetricsCache (JSON snapshot)

Reads (never touch the network):
  MetricsCache + SQLite → ReadContext → api/routes/* → React
```

**Serving:** One process, one port. `mount_spa()` in `api/main.py` registers a
catch-all after the routers: unmatched `/api/*` 404s, real files in `frontend/dist`
are served as themselves, missing assets 404, and extensionless paths return the
SPA so client-side deep links resolve. In development Vite proxies `/api` to the
same app. Neither arrangement needs CORS.

---

## 5. Non-Goals

- **No multi-user support, accounts, or auth.** Single operator, bound to localhost.
- **No network exposure or hosting.** It reads live API credentials with trading
  permissions; it is not built to face a network.
- **No microservices.** One process per front end, one SQLite file.
- **No ORM.** Hand-written SQL is deliberate and the schema is small.
- **No rewrite of the core to serve the API.** See §6.

---

## 6. Architecture Rules

- **`src/crypto_portfolio_tracker/**` is read-only to the API layer.** The API
  wraps the core; it does not modify it. Where the core reports something
  misleadingly, the correction happens at the API boundary
  (see `api/routes/portfolio.py::_holding`). Ask before changing anything in `src/`.
- **Read endpoints never construct `CryptoPortfolioTracker`.** Its `__init__`
  pings Binance and calls `get_server_time()`, so building it is a network
  operation that fails or hangs offline. Reads depend on `get_read_context()`;
  `get_tracker()` appears in exactly one place, `api/sync_runner.py`. A test pins
  this: `test_cockpit_never_constructs_the_networked_tracker`.
- **Only sync contacts Binance.** Every other screen reads SQLite and the cache.
- **pandas types never leak past `api/serialization.py`.** DataFrames and
  Timestamps are converted at that single boundary so no route or schema imports
  pandas semantics.
- **Sync age has one source: `GET /api/sync/status` via `useSyncStatus()`.**
  The app shell's top bar is the only place metrics-cache age appears, polled
  so it cannot freeze. Screens never show it; analysis-run ages stay next to
  their run buttons labelled as runs (`verb="run"`).
- **Unknown is never rendered as zero.** A missing figure is `null` in the API and
  an em dash in the UI. An undefined percentage (zero denominator) is `null`, not
  `0.0`. This rule exists because every violation found so far read as a
  confident, wrong, money-shaped number.
- **Colour is never the sole carrier of meaning.** Signs and words are explicit,
  so a value's meaning survives a monochrome screen or a colour-blind reader.

---

## 7. Coding Conventions

**General:**
- `uv` for all Python execution and dependency management.
- Comments explain *why*, especially where a subtle failure mode motivated the
  code. Match the existing density — this codebase comments its traps.

**Frontend:**
- Hand-written TypeScript types in `src/types.ts` mirror the Pydantic schemas
  field for field, including nullability. They are the contract; keep them in sync.
- Design tokens are CSS custom properties. Tonal layering and 1px borders — no
  shadows, glow, or glassmorphism.
- JetBrains Mono with `tabular-nums` for every numeric.
- Formatting helpers in `lib/format.ts` render `null`/`undefined`/`NaN` as an em dash.

**Backend:**
- Pydantic v2 response models for every endpoint; `Optional[...]` means genuinely
  unknown, and that distinction is load-bearing.
- Type hints throughout.

**Database:**
- Additive schema changes; auto-backup before migration; never drop data without
  explicit approval.

**Testing:**
- pytest for Python, Vitest + Testing Library for React.
- Every screen needs a test that mounts it with fetch stubbed to **fail**, not
  only tests that pass it good data.
- Tests must not depend on build artifacts. `frontend/dist` is gitignored, so a
  test that reads it passes vacuously in a fresh clone — `mount_spa()` takes its
  dist directory as an argument for exactly this reason.

---

## 8. Project Decisions

### Two accounting models, always shown side by side
**Choice:** Report both FIFO cost basis and net invested capital, each labelled
with the question it answers, never merged into one number.
**Status:** Current
**Reason:** They answer different questions ("are my holdings underwater?" vs
"did I make money?") and routinely differ several-fold on real data. Presenting
either alone, or averaging them, misrepresents the portfolio.

### FastAPI wraps the core rather than replacing Streamlit's access to it
**Choice:** A new `api/` package depends on the core; the core has no knowledge
of it. Streamlit is untouched.
**Status:** Current
**Reason:** The core is load-bearing and handles real money. Wrapping is
reversible; rewriting is not.
**Alternatives Considered:** Rewriting the core to be API-first; running the
React UI against a separate service.

### The API owns a metrics cache
**Choice:** Sync writes a JSON snapshot to `data/api_cache/metrics_{live|testnet}.json`;
read endpoints serve from it.
**Status:** Current
**Reason:** `calculate_portfolio_metrics()`'s offline branch zeroes
`current_price`, and no per-asset price is persisted anywhere. Without the
snapshot, an offline read would report every holding as worth $0.00. Reads must
never block on Binance.
**Alternatives Considered:** Persisting prices in SQLite (changes the core);
fetching live on read (makes every page load network-dependent).

### The cache is keyed per environment
**Choice:** Separate cache files for live and testnet.
**Status:** Current
**Reason:** Testnet uses a physically separate database. A shared cache file
would render testnet figures under a LIVE banner.

### Frontend serving is one process, one port
**Choice:** FastAPI serves the built React bundle; Vite proxies `/api` in dev.
**Status:** Current
**Reason:** No CORS in either arrangement, and one command to run.

### React is the primary UI; Streamlit is frozen but working
**Choice:** All new work lands in the React UI first. Streamlit stays runnable
and gets breakage fixes only - no new features, no removal. Supersedes the
earlier undecided direction (see §11).
**Status:** Current
**Reason:** The React UI reached read-screen parity with Streamlit and exceeds
it in places (per-coin indicator history, disposal-kind breakdown). Splitting
new work across two front ends doubles every change for a single operator.
**Alternatives Considered:** Keeping both under active development; deleting
Streamlit outright (rejected - irreversible, and it remains a useful fallback).

---

## 9. Domain Knowledge

- **Net invested capital = P2P buys − withdrawals − P2P sells.** A P2P sell is a
  cash-out to fiat and must be subtracted. Omitting it made the tracker report a
  ~99% loss on a portfolio that had simply been partly cashed out.
- **FIFO cost basis is per-asset**; the portfolio-wide figure is the sum of
  `quantity × average_cost` across symbols (`api/accounting.py`).
- **P2P buys are USDT purchased with fiat.** When the historical fiat rate lookup
  fails, price falls back to the USDT peg of $1.00 rather than $0.00 — a zero
  there silently erases real capital inflow from the invested-capital total.
- **A price of 0.0 usually means "lookup failed", not "worthless".**
  `portfolio_analyzer` seeds its price map with `0.0` and overwrites only on
  success, so the two are indistinguishable downstream. The API treats a holding
  with quantity but no price as *unpriced* and nulls every figure derived from it.
- **Core holdings vs other holdings** are distinct: only assets in
  `target_allocation` participate in rebalancing, but all assets count toward
  portfolio value.
- **Rebalancing is regime-aware.** Bear market is defined as BTC below its SMA200,
  and buys can be suppressed in that regime. Majors (BTC/ETH) use tighter drift
  thresholds than alts.

---

## 10. Known Gotchas

### Binance API
- Simple Earn endpoints cap the query range at **30 days**; longer windows fail
  with `APIError(code=-6021)`. Requests are split into 30-day chunks, centralised
  in `_get_30day_chunks()`.
- `/sapi/v1/staking/history` is deprecated. ETH and SOL staking use their own
  endpoints; failures there are logged at WARNING, not ERROR, deliberately.
- The Simple Earn *rewards* endpoint is deprecated. With auto-earn, rewards
  compound rather than appearing as separate transactions.
- Connection pool defaults are too small; the client raises them and adds
  configurable inter-request delays plus a concurrency semaphore.

### pandas
- **`pd.NaT` is an instance of `datetime.datetime`.** A missing-value check must
  come *before* any datetime branch, or a missing timestamp serializes as the
  string `"NaT"` — a plausible-looking value where `null` belongs.
- A SQL NULL in a mixed float column arrives as `NaN`, and **`NaN` is truthy**.
  `if not price` catches `0.0` but not `NaN`.

### Frontend
- `frontend/dist` is gitignored. Anything that reads it is untested in a fresh
  clone or on CI.
- `.gitignore` has an unanchored Python `lib/` rule that silently swallowed
  `frontend/src/lib/`; it is un-ignored explicitly. Watch for similar collisions.

**Assumptions to avoid:**
- Never assume a zero is a measurement. In this codebase it is usually a failure.
- Never assume a log line at ERROR aborted the sync — the core logs failures and
  keeps going, so a run can "complete" with data missing.
- Never assume local config matches testnet: they use different databases.

---

## 11. Implementation Notes

- **Two front ends coexist, React leads.** Streamlit (`uv run track-portfolio-web`)
  remains runnable as a fallback but is frozen: breakage fixes only, no new
  features, no removal (decision in §8). The React UI (`uv run python run_ui.py`)
  is where all new work lands.
- `total_value_usd` comes from the core and omits holdings whose price could not
  be fetched. The API surfaces `unpriced_count` so the UI can caveat the figure;
  correcting the total itself would require changing the core.
- Net invested is computed two ways — `api/routes/portfolio.py` reads it from the
  sync-time cache, `api/routes/capital.py` recomputes it live from the database.
  They agree in practice, but a sync that writes the DB and then fails before the
  cache write would leave the two screens disagreeing.
- `docs/design/tokens.css` and `frontend/src/tokens.css` are byte-identical
  duplicates with no build link. Only the latter is imported.

---

## 12. Repository Map

**Important Directories:**
| Directory | Purpose |
|---|---|
| `src/crypto_portfolio_tracker/` | The core. Read-only to the API layer. |
| `src/crypto_portfolio_tracker/dashboard/` | Streamlit UI |
| `api/routes/` | FastAPI endpoints |
| `api/schemas/` | Pydantic response models — the frontend's contract |
| `frontend/src/screens/` | One component per React screen |
| `frontend/src/components/` | Shared presentational components |
| `tests/api/` | FastAPI layer tests |

**Important Files:**
| File | Purpose |
|---|---|
| `start.sh` | **React UI launcher** — build, graceful restart, `--dev` mode |
| `run_ui.py` | **React UI entry point** (uvicorn, 127.0.0.1:8000) |
| `api/main.py` | **FastAPI app** — routers plus `mount_spa()` |
| `src/crypto_portfolio_tracker/__main__.py` | **CLI entry point** |
| `src/crypto_portfolio_tracker/dashboard_launcher.py` | **Streamlit entry point** |
| `api/deps.py` | `ReadContext` (network-free) vs `get_tracker()` (networked) |
| `api/cache.py` | The metrics snapshot reads serve from |
| `api/serialization.py` | The single pandas → JSON boundary |
| `api/sync_runner.py` | Background sync and SSE progress |
| `frontend/src/types.ts` | TypeScript mirror of the Pydantic schemas |
| `frontend/src/tokens.css` | Design tokens (the imported copy) |

**Generated — never edit manually:**
| Path | Type | Regenerated by |
|---|---|---|
| `frontend/dist/` | dir | `npm --prefix frontend run build` |
| `data/api_cache/` | dir | a sync |
| `data/*.db` | files | a sync (schema migrations on init) |

---

## 13. Development Commands

```bash
# Development
uv sync                            # install Python dependencies
uv run track-portfolio-cli         # CLI
uv run track-portfolio-web         # Streamlit UI

./start.sh                         # React UI: builds, then serves on :8000
./start.sh --dev                   # hot reload: Vite on :5173 proxying to :8000
./start.sh --skip-build            # serve the existing bundle unchanged

# start.sh gracefully stops a previous run first (PID file, SIGTERM then
# SIGKILL) and aborts rather than serving a stale bundle if the build fails.
# The manual equivalent, if you need it:
npm --prefix frontend run build && uv run python run_ui.py
```

**Verification commands** — run before claiming a task is done:

```bash
uv run pytest -q                          # Python tests
uv run ruff check api tests/api run_ui.py # lint (src/ has a large pre-existing backlog)
cd frontend && npx vitest run             # React tests (no `test` script defined)
cd frontend && npx tsc -b                 # typecheck
cd frontend && npx oxlint                 # frontend lint
```

**Common debugging:**

```bash
# Inspect the database
sqlite3 data/portfolio.db "SELECT * FROM transactions LIMIT 5;"

# Application logs
tail -f logs/portfolio_tracker.log

# Force a full resync — deletes local history, resyncs from Binance
rm data/portfolio.db

# Test API connectivity
uv run track-portfolio-cli   # menu option 14
```

---

## 14. External Services

| Service | Purpose |
|---|---|
| Binance API | Balances, transaction history, trade execution |
| Binance Testnet | Safe testing against a separate database |
| CoinGecko | Current and historical crypto prices |
| yFinance | Fiat exchange rates (e.g. PHP/USD) for P2P valuation |

**Secrets Location:** `.env` (gitignored) — `MAIN_API_KEY`/`MAIN_API_SECRET`,
`TESTNET_API_KEY`/`TESTNET_API_SECRET`, optional `COINGECKO_API_KEY`.
`.mcp.json` is also gitignored: it carries an API-key header.

Failure-mode behaviour worth knowing: a failed yFinance fiat-rate lookup falls
back to the USDT peg of $1.00. A failed CoinGecko price lookup leaves the price
at `0.0`, which the API reinterprets as unpriced rather than worthless. Neither
failure aborts a sync — both are logged and the run continues, so a "complete"
sync can still be missing data.

**Trading safety:** `live_trading_enabled` gates real trades, `testnet_mode`
redirects to the testnet database, and read-only API keys are recommended unless
trading is actually needed.

**React execute layer honours the two config switches (not testnet-only).**
`api/routes/execute.py` used to hard-refuse (403) outside testnet and force
`is_live=True`. It now mirrors the CLI/Streamlit path: every route passes
`cm.is_live` (from `live_trading_enabled`) to the core, and `testnet_mode` only
selects the endpoint. So live trading off = dry run on either endpoint; on =
real orders. Both `testnet_mode` and `live_trading_enabled` are editable from the
React Settings screen (Trading mode panel) and every execution screen shows the
three-cell `TradingStatusBanner`. A `testnet_mode` flip needs a server restart to
take full effect (the running process caches the tracker/DB path). The typed
`EXECUTE` confirmation gate is unchanged.

---

## 15. AI Collaboration Notes

**Implementation Style:**
- **Never modify `src/crypto_portfolio_tracker/**` without asking first.** It is
  load-bearing and handles real money. Fix things at the API boundary by default.
- Prefer incremental changes over rewrites.
- Explain architectural changes before implementing them.
- Minimize unnecessary abstractions; match existing code style.
- If introducing a new dependency, justify why the existing stack was insufficient.

**Communication:**
- Ask before making architectural decisions.
- Present trade-offs when multiple approaches are reasonable.
- State assumptions explicitly.
- Report outcomes honestly — a failing test reported as passing is worse here
  than a bug, because the whole point of this tool is trustworthy figures.

---

## 16. Known Limitations

- **Testnet syncs spot trades only.** Most other endpoints are unavailable there,
  so a testnet portfolio is not a faithful rehearsal of a live one.
- **Single-consumer sync stream.** Two simultaneous SSE clients split the event
  stream and each sees only part of it. Correct for one operator; would need
  fan-out otherwise.
- **Backtests exclude assets with missing historical data** rather than
  interpolating.
- **CoinGecko free-tier rate limits** constrain sync frequency; mitigated by disk
  caching and batched requests.
- **No CI.** Verification is whatever gets run locally.

---

## Maintenance Rules

> Decisions are append-only (see §8). Never delete a decision — supersede or
> deprecate it instead.

Umbrella rule: never update this file for implementation work unless it changes
durable project knowledge. (A bug fix is usually just a bug fix — but if fixing
it revealed "pandas NaT is a datetime," that's a gotcha, and it belongs.)

Before touching this file, run through this checklist. If every answer is "No,"
leave CONTEXT.md untouched.

- Did the architecture or a design decision change?
- Did project philosophy or a non-goal change?
- Did an architecture rule change (added, removed, or relaxed)?
- Did a coding convention change project-wide?
- Did domain knowledge get discovered or clarified?
- Did a new gotcha appear, or a wrong assumption get exposed?
- Did an implementation note change (something now mid-flight, or now resolved)?
- Did the stack, milestone, long-term direction, or external services change?
- Did a permanent limitation get added or lifted?

**Never update this file for:** commits, completed tasks, bug fixes, refactors,
git history, branch changes, session summaries, TODO lists, or transient
blockers. Git already records all of those.
