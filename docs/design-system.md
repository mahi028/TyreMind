# TyreMind Design System

**Status: living spec.** Anything built for this product — a new screen, a new
chart, a new card — follows this document. If a pattern you need is not
described here, extend this document in the same spirit before you extend the
code, so the two never drift apart.

Reference inspiration: a card-based analytics dashboard (light sidebar,
soft-shadowed white cards, rounded stat tiles with trend deltas, a single
confident brand colour). This document adapts that visual language to
TyreMind's own domain — motorsport telemetry and uncertainty — rather than
copying it literally.

---

## 1. Design principles

1. **Soft surfaces, confident data.** Chrome (cards, nav, buttons) is calm,
   rounded and quiet — soft shadows instead of hard borders, generous
   whitespace, one brand colour. Data is what gets the reader's attention:
   compound colours, uncertainty bands, deltas.

2. **One brand colour, used everywhere it means "this is the interactive or
   primary thing."** Active nav item, primary button, focus ring, positive
   trend, the recommended strategy. It does not encode tyre data — that is
   what compound colours are for (see §3.3). Confusing the two would make the
   one place brand colour matters (the CTA) compete with the one place it must
   not appear (a data encoding).

3. **Every estimate is an interval, never a bare number with a ± appended.**
   Unchanged from the previous system — this is a product principle, not a
   visual one. The `Beam` component still draws every posterior to scale.

4. **Both themes are first-class.** Same token names in light and dark; a
   component never branches on `theme`. `[data-theme]` on `<html>` swaps
   values. Charts read concrete colours via `useThemeColours()` because
   canvas cannot resolve CSS custom properties — never hard-code a hex value
   in a chart option.

---

## 2. Colour

### 2.1 Depth scale

Four surface levels, same names in both themes, different values:

| Token | Role | Light | Dark |
|---|---|---|---|
| `--color-ground` | Page background, behind everything | `#f3f5f8` | `#0b0e13` |
| `--color-surface` | Sidebar, header, top-level chrome | `#ffffff` | `#12161d` |
| `--color-card` | Card / panel background | `#ffffff` | `#161b23` |
| `--color-raised` | Inset fields, hover targets, track backgrounds | `#f1f4f7` | `#1c2129` |
| `--color-line` | Default border / divider | `#e7eaef` | `#232a34` |
| `--color-line-bright` | Hover border, active separators | `#d2d8e0` | `#333c4a` |
| `--color-line-subtle` | Faint dividers (table rows) | `#eef1f4` | `#1b2028` |

Light mode reads as **paper on a soft grey desk**: the page ground is a
whisper darker than the cards sitting on it, so a shadow has something to
fall onto. Dark mode reads as an **evening instrument cluster**: cards sit
one step brighter than the page, never inverted.

### 2.2 Type scale

| Token | Role | Light | Dark |
|---|---|---|---|
| `--color-ink` | Primary text | `#10151c` | `#eef1f5` |
| `--color-ink-dim` | Secondary text | `#57626f` | `#98a3b0` |
| `--color-ink-faint` | Tertiary text, labels | `#89939f` | `#5f6b78` |
| `--color-ink-ghost` | Placeholder, disabled | `#c2c9d1` | `#333c48` |

### 2.3 Brand colour — the only accent used for UI chrome

`--color-alert` (kept its historical name so every existing call site — active
nav pills, buttons, focus rings, `EstimateTag`, recommended-strategy numbers —
repaints automatically) is now the **brand green**, not the old brake-caliper
orange. It is the single colour used for: primary buttons, the active nav
item, links, the focus ring, positive deltas, and "this is the model's
headline answer" numbers.

| Token | Role | Light | Dark |
|---|---|---|---|
| `--color-alert` | Brand / primary accent | `#16a34a` | `#34d399` |
| `--color-alert-dim` | Tint background for badges, active pills | `#dcf5e3` | `#153823` |

It is deliberately **not** the same hue as `--color-good` (§2.5) — brand green
skews emerald/grass, `good` skews teal — so "this is clickable / primary" and
"this measurement came out well" never collide on the same screen.

### 2.4 Compound colours — the tyre's own colours, untouched

Pirelli's own bands. These encode data, never UI chrome, and do not change
between the old and new system beyond a contrast tweak for white backgrounds:

| Token | Compound | Light | Dark |
|---|---|---|---|
| `--color-soft` | Soft | `#c92020` | `#e8352e` |
| `--color-medium` | Medium | `#a07008` | `#f5c518` |
| `--color-hard` | Hard | `#556070` | `#dde0e3` |
| `--color-intermediate` | Intermediate | `#278020` | `#3ab032` |
| `--color-wet` | Wet | `#1456b8` | `#1e6fe0` |

### 2.5 Confounders and feedback

| Token | Role | Light | Dark |
|---|---|---|---|
| `--color-fuel` | Fuel burn-off | `#1a7a9c` | `#41a3c2` |
| `--color-track` | Track evolution | `#4a6070` | `#6b85a0` |
| `--color-traffic` | Traffic | `#6a3a90` | `#9e6ec8` |
| `--color-residual` | Unexplained residual | `#7a8e9c` | `#445a69` |
| `--color-good` | Positive / validated / low risk | `#0f766e` | `#2dd4bf` |
| `--color-warn` | Caution / medium risk | `#a1720a` | `#e8b93a` |

A dedicated `--color-danger` (light `#dc2626`, dark `#f87171`, with a
`--color-danger-dim` tint) exists for genuine failure states — a load error,
an impossible negative degradation value from the naive method, a "this
interval was missed" flag. Use it there, not brand green: green means
"primary / clickable / good," red means "wrong," and a dashboard where the
CTA colour and the error colour are the same hue reads as broken the moment
both appear on screen together.

### 2.6 Contrast rule

Every text/background pairing must hit **WCAG AA (4.5:1 for body text, 3:1
for large/bold text)**. The palette above is chosen to satisfy this in both
themes; if you introduce a new tint, check it before shipping — a card that
"looks fine" in isolation frequently fails against `--color-ink-faint` at
small sizes.

---

## 3. Geometry

| Token | Value | Used for |
|---|---|---|
| `--radius-sm` | `8px` | Small chips, badges, inputs, buttons |
| `--radius-md` | `10px` | Inline cards, table containers |
| `--radius-card` | `16px` | Panels — the primary content container |
| `--radius-pill` | `999px` | Nav pills, progress tracks, avatar, toggle |

Corners got rounder across the board versus the previous instrument aesthetic
(3–6px). Nothing in the product should still use a sub-8px radius on a
container; small accent marks (a 1px tick on a beam, a table underline) are
not containers and are exempt.

## 4. Elevation

Two shadow tokens, tuned per theme so dark mode doesn't get a muddy grey halo:

```css
/* light */
--shadow-card: 0 1px 2px rgba(16, 24, 32, 0.04), 0 8px 24px rgba(16, 24, 32, 0.06);
--shadow-pop:  0 4px 10px rgba(16, 24, 32, 0.08), 0 16px 40px rgba(16, 24, 32, 0.10);

/* dark */
--shadow-card: 0 1px 0 rgba(255, 255, 255, 0.02) inset, 0 8px 24px rgba(0, 0, 0, 0.35);
--shadow-pop:  0 1px 0 rgba(255, 255, 255, 0.03) inset, 0 16px 40px rgba(0, 0, 0, 0.5);
```

- `--shadow-card`: every `Panel` / card-level container, at rest.
- `--shadow-pop`: a dropdown, a modal, anything that floats above cards.

Never combine a strong shadow with a strong border — pick one. A card gets
`--shadow-card` and a hairline `--color-line` border for definition on flat
displays where shadow may be subtle; a chip or pill inside a card gets a
border only, no shadow (shadows are for things resting on the page, not for
things resting on a card).

## 5. Typography

Unchanged type stack — it already reads clean and modern:

- **Sans:** `Inter` — UI text, headings, prose.
- **Mono:** `JetBrains Mono` — every number that sits in a data column
  (`.num` utility: tabular figures, zero with a slash).

Scale (unchanged from the instrument system, still correct for a dense data
product):

| Class | Size | Use |
|---|---|---|
| `.heading-display` | 28px / 700 | Page-level headline (rare — most screens lead with a card, not an H1) |
| Card title | 14px / 600 | `Panel`'s `title` prop |
| Body | 12.5–13px / 400 | Prose, explanations |
| `.label-caps` | 10px / 600, uppercase, tracked | Field labels, table headers |
| Stat number | 22–30px / 600, mono | `Stat` value |

## 6. Components

### 6.1 Panel (the card)

The one container every screen is built from. New spec:

```
background: var(--color-card)
border: 1px solid var(--color-line)
border-radius: var(--radius-card)
box-shadow: var(--shadow-card)
padding: 20px (was 16px)
header padding: 16px 20px, border-bottom: 1px solid var(--color-line)
header title: 14px / 600
```

Same `title` / `aside` API as before — no prop changes, so every existing
call site repaints without edits.

### 6.2 Stat (KPI tile)

Same API (`label`, `value`, `unit`, `tone`, `hint`). Visual tweak: the number
is slightly heavier (font-weight 700) so a stat grid reads closer to the
reference dashboard's bold KPI row. A stat may optionally carry a **trend**:

```tsx
<Stat label="Laps analysed" value="53" trend={{ delta: '+4', direction: 'up' }} />
```

`direction: 'up'` renders in `--color-alert` (brand) with a small ▲; `'down'`
renders in `--color-warn` with a ▼. Trend is optional and most stats in this
product won't have one — a lap count has no "vs last time" to compare against
in a single-session view. Use it only where a genuine comparison exists (e.g.
a future session-over-session view).

### 6.3 Buttons

Two variants, both pill-shaped (`--radius-pill`) to match the softer system:

- **Primary** — filled brand green, white text. One per view, for the single
  most important action (`Start live replay`, `Search`).
- **Secondary** — `--color-card` background, `--color-line` border, `--color-ink-dim`
  text. Everything else: run-selector chips, mode toggles, "Rotate/Pause."

Selected/active state (a chip that's the current selection, a mode toggle
that's active): `--color-alert-dim` background, `--color-alert` text and
border — the same treatment already used for run-selector chips, just on the
new pill radius.

### 6.4 Badges & tags

`CompoundChip`, `EstimateTag`, and small status labels move to
`--radius-pill` with a soft tint background (`color-mix(in oklab, <colour>
14%, transparent)`) instead of a bare colon-dot + text. This matches the
reference dashboard's small rounded pills ("+12.95%", country tags) while
keeping every colour meaning from the old system intact.

### 6.5 Beam & progress tracks

Every horizontal track (`Beam`'s background rail, health bars, progress
bars) becomes fully rounded (`--radius-pill`) instead of the old 1–2px
radius. The beam gradient and mean-tick logic are unchanged — only the track
container's corners soften.

### 6.6 Sidebar

- Background `--color-surface`, `1px solid var(--color-line)` right border,
  no shadow (it's structural chrome, not a floating card).
- Logo mark: a small rounded-square brand-green tile with the tyre glyph,
  next to the wordmark — replaces the old bare dot.
- Nav item: icon (16px, inline SVG, stroke-based — no emoji) + label. Active
  state is a full-width `--radius-sm` pill filled with `--color-alert-dim`,
  brand-green text/icon — replacing the old thin left-border tick, which
  read as "instrument," not "app."
- A footer callout card (`--radius-card`, brand-green gradient background,
  white text) linking to the model card / docs — the one place in the UI
  that intentionally uses a filled brand-colour surface, mirroring the
  reference dashboard's "upgrade" card. Content: "Read how this works →
  model card," not a sales pitch.

### 6.7 Header

- White/`--color-surface` bar, `1px solid var(--color-line)` bottom border.
- A search-style input is out of scope for now (nothing in the product is
  searchable from the header — search lives in **Ask The Method**), so the
  header keeps its session context + status readout, restyled onto the new
  pill/badge system rather than gaining a decorative search box it can't use.
- Icon buttons (theme toggle, any future export/share action) are
  `--radius-pill`, `--color-card` background, `--color-line` border, 32px
  square — up from the old 28px flat icon button.

### 6.8 Tables

`.data-table` keeps its structure (mono numeric columns, right-aligned data,
`.label-caps` headers) but rows gain a touch more vertical padding (8px vs
6px) to breathe inside the softer system, and the header rule uses
`--color-line` at full width rather than a hairline that disappears against
the new lighter light-theme background.

---

## 7. Motion

Unchanged: 150ms ease for hover/active transitions, 200ms for panel-level
transitions (sidebar slide, progress bar width). No motion changes are part
of this redesign — the existing pulse/ring-spin/speed-sweep keyframes stay
as-is; they already fit a calmer chrome.

---

## 8. What did *not* change, on purpose

- **Compound colours** (§2.4) — they are Pirelli's own bands and the whole
  product's argument rests on the reader trusting them as data, not decor.
- **The uncertainty beam** — every estimate is still an interval.
- **Chart logic** — axis/tooltip/series construction in `charts.tsx`,
  `Circuit3D.tsx` etc. is untouched. Charts read colours from
  `useThemeColours()`, so they repaint automatically from the token changes
  above with zero code changes.
- **Copy, structure, information hierarchy** — this is a visual system
  change, not a content or IA change.

## 9. Checklist for anything new

- [ ] Uses `Panel` for any card-level container — do not hand-roll a bordered
      div with its own radius/shadow.
- [ ] Uses `--radius-pill` for any button, chip, badge, or track.
- [ ] Uses `--color-alert` only for brand/CTA meaning, never for a data
      encoding.
- [ ] Every number that is an estimate ships with its interval, drawn.
- [ ] Reads correctly in both `[data-theme="light"]` and the dark default —
      check both before calling a screen done.
- [ ] Any chart colour comes from `useThemeColours()` / `useCompoundColour()`,
      never a hard-coded hex.
