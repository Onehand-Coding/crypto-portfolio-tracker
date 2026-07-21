# Design Tokens

Authoritative source for the React frontend's visual design system. These
values correct a defect in the generated Stitch design system, where the
Material-3 purple primary and undifferentiated semantic colours rendered
gains and losses identically, contradicting the system's own `designMd`
prose (which specified indigo/mint/crimson). `docs/design/tokens.css` is
the machine-readable copy consumed by later frontend tasks; this file
documents the same values plus the rules that govern their use.

## Token table

| Token | Value | Purpose |
|---|---|---|
| `--surface-0` | `#0B0C0E` | Level 0 — main app background |
| `--surface-1` | `#14161A` | Level 1 — panels, floating containers, table rows |
| `--surface-2` | `#1C1F26` | Level 2 — modals, popovers, active overlays |
| `--border` | `#2D3139` | 1px structural borders on all containers |
| `--positive` | `#00C076` | Gains, buy signals, successful executions |
| `--negative` | `#E04E4E` | Losses, sell signals, system warnings |
| `--warning` | `#D9A441` | Caution states, distinct from positive/negative |
| `--action` | `#4F46E5` | Primary calls to action, active selection states |
| `--text-primary` | `#F9FAFB` | Primary text |
| `--text-secondary` | `#9BA3AF` | Secondary/muted text |
| `--radius-control` | `4px` | Buttons, inputs, standard controls |
| `--radius-panel` | `6px` | Main panels/cards |
| `--space` | `4px` | Base spacing unit — all spacing is a multiple of this |
| `--panel-padding` | `12px` | Internal padding for panel elements |
| `--font-ui` | `Inter, system-ui, sans-serif` | All interface labels, headers, instructional text |
| `--font-mono` | `'JetBrains Mono', ui-monospace, monospace` | All numerical values, wallet addresses, ticker symbols |

## Rules

1. **Tonal layering only.** Depth is communicated by stepping up the
   background hex value (`--surface-0` → `--surface-1` → `--surface-2`)
   plus a 1px `--border`. No shadows, no glow, no glassmorphism.
2. **Mono numerics, right-aligned.** All digits, wallet addresses, and
   ticker symbols use `--font-mono`. Numerical columns in tables are
   right-aligned so decimal points line up vertically.
3. **Colour is never the sole carrier of meaning.** Gains/losses, status,
   and other semantic states must also be conveyed through text, icons,
   or sign/direction — not colour alone.
4. **Semantic colours are reserved for semantics.** `--positive`,
   `--negative`, `--warning`, and `--action` are never repurposed for
   categorical or decorative use. Categorical series (e.g. chart lines
   for different assets) use a separate palette, not these tokens.
