/**
 * Plain-language framing for people who do not do this for a living.
 *
 * Every screen opens with one of these. Collapsed by default after first read
 * — an explanation that cannot be dismissed becomes clutter on the second visit.
 * Persisted to localStorage per-id.
 */

import { useEffect, useState, type ReactNode } from 'react'

const STORAGE_PREFIX = 'tyremind.explainer.'

export function Explainer({
  id,
  question,
  children,
  defaultOpen = true,
}: {
  id: string
  question: string
  children: ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_PREFIX + id)
      if (stored !== null) setOpen(stored === 'open')
    } catch {
      /* private browsing — the default is fine */
    }
  }, [id])

  const toggle = () => {
    const next = !open
    setOpen(next)
    try {
      localStorage.setItem(STORAGE_PREFIX + id, next ? 'open' : 'closed')
    } catch {
      /* nothing to recover from */
    }
  }

  return (
    <div
      className="rounded-md border-l-2 bg-gradient-to-r from-alert/[0.05] to-transparent"
      style={{ borderLeftColor: 'color-mix(in oklab, var(--color-alert) 50%, transparent)' }}
    >
      <button
        onClick={toggle}
        className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left"
        aria-expanded={open}
      >
        {/* Chevron — rotates open/closed */}
        <svg
          viewBox="0 0 10 10"
          width="9"
          height="9"
          aria-hidden
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="shrink-0 transition-transform duration-150"
          style={{
            transform: open ? 'rotate(90deg)' : 'rotate(0deg)',
            color: 'var(--color-alert)',
          }}
        >
          <path d="M3 2l4 3-4 3" />
        </svg>
        <span className="text-[12px] font-semibold" style={{ color: 'var(--color-alert)' }}>
          {question}
        </span>
      </button>

      {open && (
        <div className="space-y-2 px-4 pb-3.5 pl-9 prose-data">
          {children}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Term — inline definition on hover
// ---------------------------------------------------------------------------

/** Inline definition for a term of art, revealed on hover or focus. */
export function Term({ word, meaning }: { word: string; meaning: string }) {
  return (
    <span
      tabIndex={0}
      title={meaning}
      className="cursor-help border-b border-dotted border-ink-faint text-ink focus:outline-none focus-visible:border-alert"
    >
      {word}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Glossary — single source of truth for F1 terminology
// ---------------------------------------------------------------------------

export const GLOSSARY: Record<string, string> = {
  degradation:
    'How much slower a tyre gets each lap, in seconds per lap: a rate, not a total.',
  compound:
    'The rubber recipe: softer compounds grip better and wear out faster, harder ones last longer and are slower.',
  stint: 'A continuous run on one set of tyres, between pit stops.',
  'fuel burn-off':
    'A car gets lighter as it burns fuel, gaining about 0.08 seconds a lap.',
  'track evolution':
    'The circuit gets faster during a session as cars lay down rubber, adding grip for everyone.',
  'dirty air':
    'Turbulence behind another car that removes downforce from the car following, costing lap time.',
  'the cliff':
    'The point where a tyre stops degrading gradually and starts falling apart quickly.',
  confounder:
    'Something that changes the lap time without the tyre having changed at all.',
  posterior:
    'What the model believes after seeing the data, including how sure it is.',
  filtered:
    'An estimate using only laps up to now, what the pit wall could legitimately know at the time.',
  smoothed:
    'An estimate using the whole session, including later laps, what engineers know afterwards.',
  coverage:
    "How often the true value falls inside the model's claimed interval; should be about 95%.",
  CRPS: 'A score for a whole predicted distribution, not just its centre. Lower is better.',
}
