/**
 * TyreMind shared display primitives.
 *
 * These are the atoms of the design system. Every quantitative estimate is
 * drawn as an interval (Beam), never as a bare number with ±. The Panel card
 * is the container for all content. Stat is the KPI tile. Everything here
 * follows docs/design-system.md exactly — soft shadowed cards, generous
 * radius, one brand colour (--color-alert) for chrome.
 */

import type { ReactNode } from 'react'
import { compoundColour, type RunRow } from '../lib/api'

// ---------------------------------------------------------------------------
// REGIME_TONE — how a stint's fitted curve shape is coloured, everywhere
// ---------------------------------------------------------------------------

/**
 * One tone per curve regime, matching DegradationRegimes in charts.tsx.
 *
 * `cliff` is danger rather than brand: a tyre falling off its cliff is the bad
 * outcome, and brand green is reserved for chrome (see docs/design-system.md).
 */
export const REGIME_TONE: Record<string, string> = {
  linear: 'text-ink-faint',
  'warm-up': 'text-fuel',
  cliff: 'text-danger',
  recovery: 'text-good',
}

// ---------------------------------------------------------------------------
// RunChip — the stint selector, shared by explain / strategy / tyre-twin
// ---------------------------------------------------------------------------

/**
 * One stint, selectable.
 *
 * Shows the regime label only when the curve is something other than linear.
 * Over half of all stints are linear, so labelling those would be noise; the
 * ones worth a second look are exactly the ones that get a word next to them.
 */
export function RunChip({
  run,
  selected,
  onSelect,
}: {
  run: RunRow
  selected: boolean
  onSelect: () => void
}) {
  const regime = run.curve?.regime
  const notable = regime && regime !== 'linear' ? regime : null

  return (
    <button
      onClick={onSelect}
      title={run.curve?.description}
      className={`flex items-center gap-1.5 rounded-pill border px-2.5 py-1 text-[11px] transition-colors duration-150 ${
        selected
          ? 'border-alert/60 bg-alert-dim text-ink'
          : 'border-line text-ink-dim hover:border-line-bright hover:text-ink'
      }`}
    >
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ background: compoundColour(run.compound) }}
      />
      <span className="num">{run.driver}</span>
      <span className="text-ink-faint">{run.laps}L</span>
      {notable && (
        <span className={REGIME_TONE[notable] ?? 'text-ink-faint'}>{notable}</span>
      )}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Panel — the primary content container
// ---------------------------------------------------------------------------

export function Panel({
  title,
  aside,
  children,
  className = '',
}: {
  title?: string
  aside?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={`overflow-hidden rounded-card border border-line bg-card shadow-card ${className}`}
    >
      {title && (
        <header className="flex items-center justify-between gap-4 border-b border-line px-5 py-4">
          <h2 className="text-[14px] font-semibold tracking-[-0.01em] text-ink">
            {title}
          </h2>
          {aside && (
            <div className="shrink-0 text-[10.5px] text-ink-faint">{aside}</div>
          )}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Beam — posterior drawn to scale
// ---------------------------------------------------------------------------

/**
 * A posterior drawn to scale.
 *
 * @param mean   Point estimate.
 * @param sd     Posterior SD. The beam spans the 95% interval (±1.96σ).
 * @param domain Value range the track represents.
 * @param colour CSS colour for the beam and tick.
 * @param zero   Draw a reference line at zero for signed quantities.
 * @param height Track height in px.
 */
export function Beam({
  mean,
  sd,
  domain,
  colour,
  zero = false,
  height = 16,
}: {
  mean: number
  sd: number
  domain: [number, number]
  colour: string
  zero?: boolean
  height?: number
}) {
  const [lo, hi] = domain
  const span = hi - lo || 1
  const pct = (v: number) => ((v - lo) / span) * 100

  const left    = Math.max(0, pct(mean - 1.96 * sd))
  const right   = Math.min(100, pct(mean + 1.96 * sd))
  const width   = Math.max(right - left, 0.5)
  const meanPct = Math.min(Math.max(pct(mean), 0), 100)

  return (
    <div
      className="relative w-full overflow-hidden rounded-pill bg-raised"
      style={{ height }}
    >
      {zero && lo < 0 && hi > 0 && (
        <div
          className="absolute top-0 bottom-0 w-px"
          style={{ left: `${pct(0)}%`, background: 'var(--color-line-bright)' }}
        />
      )}
      {/* Gaussian-shaped beam */}
      <div
        className="beam absolute top-0 bottom-0"
        style={{
          left: `${left}%`,
          width: `${width}%`,
          ['--beam-color' as string]: colour,
        }}
      />
      {/* Hard tick at mean */}
      <div
        className="absolute top-0 bottom-0 w-[2px]"
        style={{ left: `${meanPct}%`, background: colour }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Stat — labelled KPI tile
// ---------------------------------------------------------------------------

/** A labelled figure. The label never competes with the number. */
export function Stat({
  label,
  value,
  unit,
  tone = 'default',
  hint,
}: {
  label: string
  value: string
  unit?: string
  tone?: 'default' | 'warm' | 'dim' | 'good'
  hint?: string
}) {
  const colour =
    tone === 'warm'    ? 'var(--color-alert)' :
    tone === 'good'    ? 'var(--color-good)'  :
    tone === 'dim'     ? 'var(--color-ink-dim)' :
    'var(--color-ink)'

  return (
    <div>
      <div className="label-caps mb-1">{label}</div>
      <div
        className="num text-[22px] leading-none font-semibold tracking-[-0.02em]"
        style={{ color: colour }}
      >
        {value}
        {unit && (
          <span
            className="ml-1 text-[11px] font-normal"
            style={{ color: 'var(--color-ink-faint)' }}
          >
            {unit}
          </span>
        )}
      </div>
      {hint && (
        <div className="mt-1 text-[10.5px] leading-tight text-ink-faint">{hint}</div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// CompoundChip — Pirelli compound indicator
// ---------------------------------------------------------------------------

/** Compound band in Pirelli's own colours. */
export function CompoundChip({ compound }: { compound: string }) {
  const colour =
    { SOFT: 'var(--color-soft)', MEDIUM: 'var(--color-medium)', HARD: 'var(--color-hard)' }[
      compound?.toUpperCase()
    ] ?? 'var(--color-ink-dim)'

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-pill px-2 py-0.5 text-[11px] font-medium"
      style={{
        background: `color-mix(in oklab, ${colour} 14%, transparent)`,
        color: colour,
      }}
    >
      <span className="inline-block h-2 w-2 rounded-full" style={{ background: colour }} />
      {compound}
    </span>
  )
}

// ---------------------------------------------------------------------------
// EstimateTag — marks a number as inferred, not observed
// ---------------------------------------------------------------------------

/**
 * Marks a number as inferred rather than observed.
 * Used wherever a counterfactual or projection is shown.
 */
export function EstimateTag({ children }: { children?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-pill border border-line px-2 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.06em] text-ink-faint">
      <span
        className="inline-block h-1 w-1 rounded-full"
        style={{ background: 'var(--color-ink-faint)' }}
      />
      {children ?? 'estimate'}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Empty — empty state placeholder
// ---------------------------------------------------------------------------

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="px-4 py-10 text-center text-[12.5px] text-ink-faint">
      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Loading — compound-ring spinner
// ---------------------------------------------------------------------------

export function Loading({ what }: { what: string }) {
  return (
    <div className="flex items-center gap-3 px-4 py-10 text-[12px] text-ink-faint">
      {/* Compound ring — concentric arcs echoing tyre cross-section */}
      <span
        className="ring-spin inline-block h-5 w-5 rounded-full border-2 border-line"
        style={{ borderTopColor: 'var(--color-alert)' }}
        aria-hidden
      />
      Fitting {what}…
    </div>
  )
}

// ---------------------------------------------------------------------------
// ErrorNote — error state
// ---------------------------------------------------------------------------

export function ErrorNote({ error }: { error: string }) {
  return (
    <div className="rounded-card border border-line bg-card p-4 shadow-card">
      <div
        className="mb-1.5 inline-flex items-center gap-1.5 rounded-pill px-2 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.08em]"
        style={{
          background: 'color-mix(in oklab, var(--color-danger) 14%, transparent)',
          color: 'var(--color-danger)',
        }}
      >
        <span
          className="inline-block h-1.5 w-1.5 rounded-full"
          style={{ background: 'var(--color-danger)' }}
        />
        Could not load
      </div>
      <p className="text-[12px] leading-relaxed text-ink-dim">{error}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// SectionDivider — labelled horizontal rule
// ---------------------------------------------------------------------------

export function SectionDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 py-1">
      <div className="h-px flex-1 bg-line" />
      <span className="label-caps">{label}</span>
      <div className="h-px flex-1 bg-line" />
    </div>
  )
}
