/**
 * TyreMind dashboard shell.
 *
 * Layout: fixed header + fixed sidebar + scrollable main area.
 * Navigation order follows the argument: what, why, where, what-if, live,
 * does it work, ask, beyond.
 */

import { useCallback, useEffect, useState, type ReactNode } from 'react'
import {
  advanced,
  api,
  compoundColour,
  type Decomposition,
  type DecompositionRow,
  type RunRow,
  type SessionRef,
  type SessionSummary,
} from './lib/api'
import { ErrorNote, Loading, Panel, RunChip } from './components/primitives'
import { PeelAway } from './components/PeelAway'
import { Waterfall } from './components/Waterfall'
import { LiveMonitor } from './components/LiveMonitor'
import { SciencePanel } from './components/SciencePanel'
import { Overview } from './components/Overview'
import { StrategyView } from './components/StrategyView'
import { TyreStateView } from './components/TyreStateView'
import { BeyondRacing } from './components/BeyondRacing'
import { Explainer } from './components/Explainer'
import { StintDecomposition, TrackEvolutionChart } from './components/charts'
import { CircuitView } from './components/CircuitView'
import { RaceView } from './components/RaceView'
import { AskPanel } from './components/AskPanel'
import { ThemeToggle } from './lib/theme'

type View =
  | 'overview'
  | 'race'
  | 'explain'
  | 'circuit'
  | 'tyre'
  | 'strategy'
  | 'live'
  | 'evidence'
  | 'ask'
  | 'beyond'

// ---------------------------------------------------------------------------
// Nav icons — 16px stroke icons, no library, so the sidebar stays dependency-free
// ---------------------------------------------------------------------------

function icon(paths: ReactNode) {
  return function Icon({ className = '' }: { className?: string }) {
    return (
      <svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor"
        strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden>
        {paths}
      </svg>
    )
  }
}

const IconHome = icon(<path d="M2 7.5 8 2l6 5.5V14a1 1 0 0 1-1 1h-3v-4.5H6V15H3a1 1 0 0 1-1-1Z" />)
const IconRace = icon(<><path d="M1.5 10.5c2-3.5 4-5 6.5-5s4.5 1.5 6.5 5" /><circle cx="8" cy="12" r="1.6" /></>)
const IconLayers = icon(<path d="m8 2 6 3-6 3-6-3 6-3ZM2 8l6 3 6-3M2 11l6 3 6-3" />)
const IconTrack = icon(<path d="M3 12c0-4 2-8 5-8s5 4 5 8-2 3-5 3-5 1-5-3Z" />)
const IconGauge = icon(<><circle cx="8" cy="8.5" r="5.5" /><path d="M8 8.5 10.5 6M8 3v1.2M3 8.5h1.2M12.8 8.5H14" /></>)
const IconFlag = icon(<path d="M4 14V2m0 1 8 1.2L9.5 7 12 9.8 4 8.6" />)
const IconActivity = icon(<path d="M2 8.5h3l1.5-4L9 12l1.5-6.5L11.5 8.5H14" />)
const IconCheck = icon(<path d="M8 1.5 13.5 4v4c0 3.5-2.5 5.6-5.5 6.5-3-0.9-5.5-3-5.5-6.5V4L8 1.5ZM5.7 8l1.6 1.6L10.4 6.4" />)
const IconSearch = icon(<><circle cx="7" cy="7" r="4.5" /><path d="m13.5 13.5-3-3" /></>)
const IconGlobe = icon(<><circle cx="8" cy="8" r="5.8" /><path d="M2.2 8h11.6M8 2.2c-2 2-2 9.6 0 11.6m0-11.6c2 2 2 9.6 0 11.6" /></>)

const VIEWS: { key: View; label: string; icon: (props: { className?: string }) => ReactNode }[] = [
  { key: 'overview', label: 'Overview',    icon: IconHome },
  { key: 'race',     label: 'Race',        icon: IconRace },
  { key: 'explain',  label: 'Explain',     icon: IconLayers },
  { key: 'circuit',  label: 'Circuit',     icon: IconTrack },
  { key: 'tyre',     label: 'Tyre twin',   icon: IconGauge },
  { key: 'strategy', label: 'Strategy',    icon: IconFlag },
  { key: 'live',     label: 'Live',        icon: IconActivity },
  { key: 'evidence', label: 'Evidence',    icon: IconCheck },
  { key: 'ask',      label: 'Ask',         icon: IconSearch },
  { key: 'beyond',   label: 'Beyond',      icon: IconGlobe },
]

const VIEW_KEYS = new Set<string>(VIEWS.map((v) => v.key))

function viewFromHash(): View {
  const key = window.location.hash.replace(/^#\/?/, '')
  return VIEW_KEYS.has(key) ? (key as View) : 'overview'
}

/** Session type badge: Race, FP2, Q, etc. */
function SessionBadge({ session }: { session: string }) {
  const isRace = session === 'R'
  const label = isRace ? 'RACE' : session
  return (
    <span
      className="inline-flex items-center rounded-pill px-1.5 py-0.5 text-[9px] font-semibold tracking-[0.08em]"
      style={{
        background: isRace
          ? 'color-mix(in oklab, var(--color-alert) 15%, transparent)'
          : 'color-mix(in oklab, var(--color-fuel) 15%, transparent)',
        color: isRace ? 'var(--color-alert)' : 'var(--color-fuel)',
      }}
    >
      {label}
    </span>
  )
}

/** Small brand mark: a rounded tile with a tyre glyph. */
function Logo() {
  return (
    <span
      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-sm text-[color:var(--color-ground)]"
      style={{ background: 'var(--color-alert)' }}
    >
      <svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
        <circle cx="8" cy="8" r="5.6" />
        <circle cx="8" cy="8" r="1.8" fill="currentColor" stroke="none" />
      </svg>
    </span>
  )
}

/** Status pill: offline-ready (good) or needs network (warn). */
function StatusPill({ offline, compact = false }: { offline: boolean; compact?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-[10px] text-ink-faint">
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: offline ? 'var(--color-good)' : 'var(--color-warn)' }}
      />
      {compact ? (offline ? 'Offline' : 'Online') : offline ? 'Offline ready' : 'Needs network'}
    </span>
  )
}

export default function App() {
  const [sessions, setSessions] = useState<SessionRef[]>([])
  const [sessionId, setSessionId] = useState('')
  const [view, setViewState] = useState<View>(viewFromHash)
  const [offline, setOffline] = useState<boolean | null>(null)
  const [bootError, setBootError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const setView = useCallback((next: View) => {
    setViewState(next)
    setSidebarOpen(false)
    if (viewFromHash() !== next) window.location.hash = `/${next}`
  }, [])

  useEffect(() => {
    const onHash = () => setViewState(viewFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    Promise.all([api.sessions(), api.health()])
      .then(([list, health]) => {
        setSessions(list)
        setOffline(health.offline_ready)
        const race = list.find((s) => s.session === 'R') ?? list[0]
        if (race) setSessionId(race.session_id)
      })
      .catch((e) => setBootError(String(e.message ?? e)))
  }, [])

  if (bootError) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-ground p-10">
        <div className="w-full max-w-md">
          <div className="mb-6 flex items-center gap-2.5">
            <Logo />
            <span className="text-[15px] font-bold tracking-[-0.02em] text-ink">TyreMind</span>
          </div>
          <ErrorNote error={`${bootError}. Start the server with: python -m tyremind.serve`} />
        </div>
      </div>
    )
  }

  const current = sessions.find((s) => s.session_id === sessionId)

  return (
    <div className="flex h-dvh flex-col bg-ground lg:flex-row lg:overflow-hidden">

      {/* Mobile header, folds into the sidebar on desktop */}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-line bg-surface px-4 lg:hidden">
        <button
          onClick={() => setSidebarOpen((o) => !o)}
          className="flex h-8 w-8 flex-col items-center justify-center gap-1.5 rounded-pill border border-line text-ink-dim hover:border-line-bright hover:text-ink"
          aria-label="Toggle navigation"
        >
          <span className="block h-px w-4 bg-current" />
          <span className="block h-px w-4 bg-current" />
          <span className="block h-px w-3 bg-current" />
        </button>
        <Logo />
        <span className="text-[14px] font-bold tracking-[-0.02em] text-ink">TyreMind</span>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </header>

      {/* ── Sidebar overlay (mobile) ──────────────────────────────────────── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-ground/70 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-40 flex w-60 shrink-0 flex-col
          border-r border-line bg-surface
          transition-transform duration-200 ease-out
          lg:relative lg:translate-x-0 lg:transition-none
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        <div className="flex h-14 shrink-0 items-center gap-2.5 border-b border-line px-4">
          <Logo />
          <span className="text-[14px] font-bold tracking-[-0.02em] text-ink">TyreMind</span>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-4">
          <div className="mb-1 px-1 label-caps">Sessions</div>
          <div className="mb-4 space-y-0.5">
            {sessions.map((s) => (
              <button
                key={s.session_id}
                onClick={() => setSessionId(s.session_id)}
                className={`
                  flex w-full items-center gap-2 rounded-sm px-2.5 py-1.5
                  text-left transition-colors duration-150
                  ${s.session_id === sessionId
                    ? 'bg-alert-dim text-ink'
                    : 'text-ink-dim hover:bg-raised hover:text-ink'
                  }
                `}
              >
                <span className="min-w-0 flex-1 truncate text-[11.5px] font-medium">
                  {s.grand_prix}
                </span>
                <SessionBadge session={s.session} />
              </button>
            ))}
          </div>

          <div className="mb-1 px-1 label-caps">Views</div>
          <div className="space-y-0.5">
            {VIEWS.map((v) => {
              const Icon = v.icon
              const active = v.key === view
              return (
                <button
                  key={v.key}
                  onClick={() => setView(v.key)}
                  className={`
                    flex w-full items-center gap-2.5 rounded-sm px-2.5 py-2
                    text-left text-[12.5px] font-medium transition-colors duration-150
                    ${active
                      ? 'bg-alert-dim text-alert'
                      : 'text-ink-dim hover:bg-raised hover:text-ink'
                    }
                  `}
                >
                  <Icon className="shrink-0" />
                  {v.label}
                </button>
              )
            })}
          </div>
        </div>

        {/* Footer callout, the one place brand colour fills a whole surface */}
        <div className="shrink-0 p-3">
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="block rounded-card p-3.5 text-[color:var(--color-ground)] shadow-card"
            style={{ background: 'linear-gradient(155deg, var(--color-alert), color-mix(in oklab, var(--color-alert) 65%, var(--color-fuel)))' }}
          >
            <div className="text-[12px] font-semibold">API docs</div>
            <div className="mt-0.5 text-[10.5px] opacity-90">Every endpoint, live</div>
          </a>
          {offline != null && (
            <div className="mt-3 flex items-center justify-between px-1">
              <StatusPill offline={offline} />
              <ThemeToggle />
            </div>
          )}
        </div>
      </aside>

      {/* ── Main content area ─────────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col overflow-hidden">

        {/* Desktop header strip */}
        <header className="hidden h-14 shrink-0 items-center justify-between border-b border-line bg-surface px-5 lg:flex">
          <div className="flex items-center gap-3">
            {current && (
              <span className="num text-[12px] font-medium text-ink">
                {current.year} {current.grand_prix}
              </span>
            )}
            {current && <SessionBadge session={current.session} />}
          </div>

          <span className="text-[10.5px] tracking-[0.02em] text-ink-faint">
            Observed pace is not tyre degradation
          </span>

          <div className="flex items-center gap-3">
            {offline != null && <StatusPill offline={offline} compact />}
            <ThemeToggle />
          </div>
        </header>

        {/* View area */}
        <main className="flex-1 overflow-y-auto p-4">
          {!sessionId ? (
            <Loading what="the session catalogue" />
          ) : view === 'beyond' ? (
            <BeyondRacing />
          ) : view === 'ask' ? (
            <AskPanel />
          ) : view === 'circuit' ? (
            <CircuitView circuit={current?.grand_prix ?? ''} />
          ) : view === 'live' ? (
            <LiveMonitor sessionId={sessionId} />
          ) : view === 'evidence' ? (
            <SciencePanel sessionId={sessionId} />
          ) : view === 'overview' ? (
            <Overview sessionId={sessionId} onOpenExplain={() => setView('explain')} />
          ) : (
            <RunScopedView
              key={sessionId}
              view={view}
              sessionId={sessionId}
              circuit={current?.grand_prix ?? ''}
            />
          )}
        </main>
      </div>
    </div>
  )
}

/**
 * Views that share a run selection (explain / tyre / strategy).
 * Switching between them keeps the same car in front of you.
 */
function RunScopedView({
  view,
  sessionId,
  circuit,
}: {
  view: View
  sessionId: string
  circuit: string
}) {
  const [runs, setRuns] = useState<RunRow[]>([])
  const [selected, setSelected] = useState<RunRow | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .runs(sessionId)
      .then((rows) => {
        const usable = rows.filter((r) => r.laps >= 5)
        setRuns(usable)
        setSelected((current) =>
          current && usable.some((r) => r.run_id === current.run_id)
            ? current
            : usable[0] ?? null,
        )
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [sessionId])

  if (error) return <ErrorNote error={error} />
  if (!selected) return <Loading what="the session" />

  if (view === 'strategy') {
    return (
      <StrategyView
        sessionId={sessionId}
        runs={runs}
        selected={selected}
        onSelect={setSelected}
      />
    )
  }
  if (view === 'tyre') {
    return (
      <TyreStateView
        sessionId={sessionId}
        circuit={circuit}
        runs={runs}
        selected={selected}
        onSelect={setSelected}
      />
    )
  }
  if (view === 'race') {
    return (
      <RaceView
        sessionId={sessionId}
        circuit={circuit}
        runs={runs}
        selected={selected}
        onSelect={setSelected}
      />
    )
  }
  return (
    <ExplainView
      sessionId={sessionId}
      runs={runs}
      selected={selected}
      onSelect={setSelected}
    />
  )
}

function ExplainView({
  sessionId,
  runs,
  selected,
  onSelect,
}: {
  sessionId: string
  runs: RunRow[]
  selected: RunRow
  onSelect: (r: RunRow) => void
}) {
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [rows, setRows] = useState<DecompositionRow[]>([])
  const [decomposition, setDecomposition] = useState<Decomposition | null>(null)
  const [narration, setNarration] = useState('')
  const [track, setTrack] = useState<
    { session_lap: number; track_effect: number; track_effect_sd: number }[]
  >([])
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.summary(sessionId).then(setSummary).catch(() => undefined)
    api
      .track(sessionId)
      .then((t) => setTrack(t.rows))
      .catch(() => setTrack([]))
  }, [sessionId])

  useEffect(() => {
    setBusy(true)
    Promise.all([
      api.decomposeRun(sessionId, selected.driver, selected.run_id),
      api.decompose(sessionId, selected.driver, selected.last_lap),
    ])
      .then(([run, lap]) => {
        setRows(run.rows)
        setDecomposition(lap)
      })
      .catch(() => undefined)
      .finally(() => setBusy(false))
  }, [sessionId, selected])

  useEffect(() => {
    advanced
      .narrate(sessionId, selected.driver, selected.last_lap)
      .then((n) => setNarration(n.decomposition.text))
      .catch(() => setNarration(''))
  }, [sessionId, selected])

  return (
    <div className="space-y-4">
      <Explainer id="explain" question="What am I looking at?">
        <p>
          A stint is a run on one set of tyres. This screen splits the lap-time
          trend into how much was the tyre, and how much was everything else.
        </p>
      </Explainer>

      {/* Run selector + peel chart */}
      <Panel
        title="Peel the confounders away"
        aside={`${selected.driver} · ${selected.laps} laps on ${selected.compound}`}
      >
        {/* Run selector chips */}
        <div className="mb-4 flex flex-wrap gap-1.5">
          {runs.slice(0, 14).map((run) => (
            <RunChip
              key={run.run_id}
              run={run}
              selected={selected.run_id === run.run_id}
              onSelect={() => onSelect(run)}
            />
          ))}
        </div>

        {busy || rows.length === 0 ? (
          <Loading what="this run" />
        ) : (
          <PeelAway rows={rows} compound={selected.compound} />
        )}
      </Panel>

      {/* LLM narration */}
      {narration && (
        <Panel title="In plain language">
          <p className="prose-data max-w-[76ch]">{narration}</p>
          <p className="mt-2 text-[10.5px] text-ink-faint">
            Generated from the model&rsquo;s own output. Every number here was computed,
            not written.
          </p>
        </Panel>
      )}

      {/* Stint decomposition */}
      {rows.length > 0 && (
        <Panel
          title="Where the time went, lap by lap"
          aside="the whole stint, not one lap"
        >
          <StintDecomposition rows={rows} />
          <p className="mt-3 max-w-[86ch] text-[11.5px] leading-relaxed text-ink-dim">
            Bars above the line cost time, bars below gain it. Watch the orange
            tyre bar grow while the blue fuel bar sinks: the lap where they cross
            is the moment the stint turns from getting faster to getting slower.
            The dashed line is what the stopwatch actually recorded.
          </p>
        </Panel>
      )}

      {/* Waterfall + compound summary */}
      <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
        <Panel title="Where the lap time went" aside="one lap, against the stint start">
          {decomposition ? (
            <Waterfall decomposition={decomposition} />
          ) : (
            <Loading what="the lap" />
          )}
        </Panel>

        {summary && <CompoundSummary summary={summary} />}
      </div>

      {/* Track evolution */}
      {track.length > 0 && (
        <Panel title="The circuit getting faster" aside="shared by every car">
          <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
            <TrackEvolutionChart rows={track} />
            <div className="space-y-2 prose-data">
              <p>
                As cars lay down rubber, the whole circuit gains grip and everyone
                speeds up, with no change to any individual tyre.
              </p>
              <p className="text-[11.5px] text-ink-faint">
                The curve flattens because rubber build-up saturates: the most
                assumption-heavy part of the model, so the band is shown wide.
              </p>
            </div>
          </div>
        </Panel>
      )}
    </div>
  )
}

function CompoundSummary({ summary }: { summary: SessionSummary }) {
  return (
    <Panel title="Degradation by compound" aside="seconds lost per lap">
      <div className="space-y-4">
        {Object.entries(summary.compounds).map(([compound, estimate]) => {
          const colour = compoundColour(compound)
          return (
            <div key={compound}>
              {/* Row header */}
              <div className="mb-1.5 flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-[12px] text-ink-dim">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full border"
                    style={{
                      borderColor: colour,
                      background: `color-mix(in oklab, ${colour} 30%, transparent)`,
                    }}
                  />
                  {compound}
                </span>
                <span className="num text-[13px] text-ink">
                  {estimate.degradation_rate.toFixed(3)}
                  <span className="ml-1 text-[10px] text-ink-faint">
                    ± {estimate.degradation_rate_sd.toFixed(3)}
                  </span>
                </span>
              </div>

              {/* Beam track */}
              <div className="relative h-3 overflow-hidden rounded-pill bg-raised">
                {/* Zero reference */}
                <div
                  className="absolute top-0 bottom-0 w-px bg-line-bright"
                  style={{ left: `${((0.08) / 0.36) * 100}%` }}
                />
                {/* Beam */}
                <div
                  className="beam absolute top-0 bottom-0"
                  style={{
                    left: `${Math.max(0, ((estimate.ci95[0] + 0.08) / 0.36) * 100)}%`,
                    width: `${Math.max(1, ((estimate.ci95[1] - estimate.ci95[0]) / 0.36) * 100)}%`,
                    ['--beam-color' as string]: colour,
                  }}
                />
              </div>

              <div className="mt-1 text-[10.5px] text-ink-faint">
                Over 25 laps:{' '}
                <span className="num text-ink-dim">
                  {(estimate.degradation_rate * 25).toFixed(1)} s
                </span>{' '}
                accumulated pace loss
              </div>
            </div>
          )
        })}
      </div>
    </Panel>
  )
}
