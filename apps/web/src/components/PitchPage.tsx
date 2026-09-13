/**
 * The pitch: one scroll, one argument, a picture at every beat.
 *
 * The summary page answers "what is this and can I trust it". This one answers
 * a different question, the one a panel actually decides on: **why does this
 * need to exist, and why has nobody done it.** So it is built as a narrative
 * rather than a dashboard, and it obeys three rules the rest of the application
 * does not.
 *
 * **A chart per beat, a sentence per chart.** Numbers persuade a reader who has
 * already decided to concentrate. A panel deciding whether to concentrate is
 * persuaded by a shape -- a line that bends the wrong way, a band that narrows,
 * a diagonal that a model sits off. Every figure on this page is the live chart
 * from the screen that owns it, not a picture of one, so a sceptic can open that
 * screen and find the same thing.
 *
 * **The problem before the product.** Beat 1 is the four causes tangled inside
 * one lap time, drawn from the race in the rail, and nothing describes what we
 * built until the reader has seen that.
 *
 * **Where we are weak, we still say so**, in one line at the end rather than in
 * a footnote nobody reaches. A panel that later finds an unmentioned weakness
 * discards everything else on the page.
 */

import { useEffect, useMemo, useState } from 'react'

import {
  advanced,
  api,
  type DecompositionRow,
  type DegradationRow,
  type ProjectionResult,
  type PitWindow,
  type RunRow,
  type SessionRef,
  type TrustResult,
} from '../lib/api'
import { DegradationRegimes, PitWindowChart, ReliabilityCurve, RulScatter } from './charts'
import { cornerEnergy, type CornerEnergy } from '../lib/api'
import {
  ConsensusSpread,
  ContributionStack,
  ProjectionFan,
  RateRibbon,
} from './briefing/charts'

/** A beat of the argument: a claim, the picture that proves it, one line under. */
function Beat({
  n,
  kicker,
  claim,
  line,
  read,
  children,
  tone = 'var(--color-alert)',
}: {
  n: string
  kicker: string
  claim: string
  line: string
  /** How to read the chart, in one plain sentence. Sits under it, not beside. */
  read?: string
  children: React.ReactNode
  tone?: string
}) {
  return (
    <section className="border border-line bg-surface" style={{ borderTop: `3px solid ${tone}` }}>
      <div className="grid gap-x-8 gap-y-4 px-6 py-6 xl:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        <div>
          <div className="flex items-baseline gap-2.5">
            <span className="num text-[12px]" style={{ color: tone }}>
              {n}
            </span>
            <span className="text-[10.5px] tracking-[0.18em] text-ink-faint uppercase">
              {kicker}
            </span>
          </div>
          <h2 className="mt-2.5 text-[23px] leading-[1.15] font-bold tracking-[-0.02em] text-ink">
            {claim}
          </h2>
          <p className="mt-3 max-w-[44ch] text-[13.5px] leading-relaxed text-ink-dim">{line}</p>
        </div>
        <div className="min-w-0">
          {children}
          {read && (
            <div
              className="mt-3 flex items-start gap-2.5 border-l-2 py-1 pl-3"
              style={{ borderColor: tone }}
            >
              <span
                className="mt-[1px] shrink-0 text-[9.5px] tracking-[0.16em] uppercase"
                style={{ color: tone }}
              >
                Read
              </span>
              <span className="text-[12.5px] leading-snug text-ink-dim">{read}</span>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function Waiting({ what }: { what: string }) {
  return (
    <div className="flex h-[240px] items-center justify-center border border-line bg-raised/30 text-[12.5px] text-ink-faint">
      {what}
    </div>
  )
}

export function PitchPage({
  sessionId,
  session,
}: {
  sessionId: string
  session: SessionRef | undefined
}) {
  const [runs, setRuns] = useState<RunRow[]>([])
  const [rates, setRates] = useState<DegradationRow[]>([])
  const [decomp, setDecomp] = useState<DecompositionRow[]>([])
  const [projection, setProjection] = useState<ProjectionResult | null>(null)
  const [pit, setPit] = useState<PitWindow | null>(null)
  const [trust, setTrust] = useState<TrustResult | null>(null)
  const [corners, setCorners] = useState<CornerEnergy | null>(null)
  const [experiments, setExperiments] = useState<Record<string, any>>({})

  useEffect(() => {
    api.experiments().then(setExperiments).catch(() => undefined)
  }, [])

  useEffect(() => {
    const circuit = session?.grand_prix
    if (!circuit) return
    setCorners(null)
    cornerEnergy(circuit).then(setCorners).catch(() => undefined)
  }, [session?.grand_prix])

  useEffect(() => {
    if (!sessionId) return
    setRuns([]); setRates([]); setDecomp([])
    setProjection(null); setPit(null); setTrust(null); setCorners(null)
    api.runs(sessionId).then(setRuns).catch(() => undefined)
    api.degradation(sessionId, false).then((d) => setRates(d.rows)).catch(() => undefined)
  }, [sessionId])

  // The longest stint in the session is the one worth showing: it has the most
  // laps behind its estimate, so the band visibly narrows rather than staying
  // wide for want of evidence.
  const hero = useMemo(
    () => (runs.length ? [...runs].sort((a, b) => b.laps - a.laps)[0] : null),
    [runs],
  )

  useEffect(() => {
    if (!sessionId || !hero) return
    api.decomposeRun(sessionId, hero.driver, hero.run_id)
      .then((d) => setDecomp(d.rows ?? []))
      .catch(() => undefined)
    const lap = Math.round(hero.first_lap + (hero.last_lap - hero.first_lap) * 0.7)
    api.projection(sessionId, hero.driver, lap, 20).then(setProjection).catch(() => undefined)
    // A third of the way in: far enough that the model has evidence, early
    // enough that the stop is still a live decision rather than a formality.
    const decide = Math.round(hero.first_lap + (hero.last_lap - hero.first_lap) * 0.34)
    advanced.pitWindow(sessionId, hero.driver, decide).then(setPit).catch(() => undefined)
    advanced.trust(sessionId, hero.compound, 20).then(setTrust).catch(() => undefined)
  }, [sessionId, hero])

  const heroRates = useMemo(
    () => (hero ? rates.filter((r) => r.driver === hero.driver && r.run_id === hero.run_id) : []),
    [rates, hero],
  )

  const cliff = experiments['exp17_degradation_cliff']
  const regimes = useMemo(() => {
    if (!cliff?.regimes) return []
    const total: number = Object.values<number>(cliff.regimes).reduce((a, b) => a + b, 0)
    return Object.entries<number>(cliff.regimes).map(([regime, n]) => ({
      regime,
      n,
      share: total ? n / total : 0,
      slopeBefore: null,
      delta: null,
      position: null,
    }))
  }, [cliff])

  const nasa = experiments['exp07_cross_domain']

  const shape = experiments['exp16_calibration_shape']
  const reliability = useMemo(() => {
    if (!shape?.per_model) return []
    return Object.entries<any>(shape.per_model).map(([model, m]) => ({
      model,
      levels: m.reliability.levels,
      gaussian: m.reliability.gaussian,
      adaptive: m.reliability.adaptive,
    }))
  }, [shape])

  const consensus = useMemo(() => {
    const block = trust?.consensus
    if (!block) return null
    const key = hero?.compound && block[hero.compound] ? hero.compound : Object.keys(block)[0]
    return key ? block[key] : null
  }, [trust, hero])

  return (
    <div className="space-y-3">
      {/* ── the hook ─────────────────────────────────────────────────────── */}
      <section
        className="border border-line bg-surface"
        style={{ borderTop: '3px solid var(--color-alert)' }}
      >
        <div className="px-6 py-8">
          <div className="text-[11px] tracking-[0.22em] text-alert uppercase">
            The pitch · {session ? `${session.year} ${session.grand_prix}` : 'live'}
          </div>
          <h1 className="mt-3 max-w-[19ch] text-[40px] leading-[1.02] font-bold tracking-[-0.035em] text-ink sm:text-[54px]">
            A race is decided by one number nobody can measure.
          </h1>
          <p className="mt-5 max-w-[74ch] text-[16px] leading-relaxed text-ink-dim">
            How fast the tyre is going away. Get it right and you win the stop. Get it wrong and you
            lose the race in the pit lane.{' '}
            <strong className="text-ink">
              Nobody outside a handful of Formula 1 teams can measure it, and the method everyone
              else uses gets it backwards.
            </strong>
          </p>
          <p className="mt-4 max-w-[74ch] text-[13.5px] leading-relaxed text-ink-faint">
            Everything below is drawn live from the race selected in the sidebar. Change the race
            and every chart on this page changes with it.
          </p>
        </div>
      </section>

      {/* ── 01 the tangle ────────────────────────────────────────────────── */}
      <Beat
        n="01"
        kicker="The problem"
        claim="Four things move a lap time. Only one of them is the tyre."
        read="The orange band is the tyre and it grows every lap. The blue below the line is fuel burn pulling the other way, which is why the dashed total barely moves."
        line="The tyre is wearing out and slowing the car down. Meanwhile the fuel is burning off and speeding it up by about the same amount, the track is gaining grip, and traffic comes and goes. They arrive added together, as a single number on a timing screen."
      >
        {decomp.length ? (
          <ContributionStack rows={decomp} />
        ) : (
          <Waiting what="Separating this stint…" />
        )}
      </Beat>

      {/* ── 02 the physics check ─────────────────────────────────────────── */}
      <Beat
        n="02"
        kicker="How we know the physics is right"
        claim="We can tell which way a circuit turns, just from how the tyres worked."
        read="The two dark tyres did the most work. Lean on right-hand corners all lap and the left-hand tyres carry the load — so the heavier side tells you which way the track goes."
        line="A sanity check nobody can fudge. We never tell the model which direction the circuit runs. It reads the load through each of the four tyres and infers it, and it gets this right on seven of the eight circuits we tested. If the physics underneath were wrong, it would not be able to."
        tone="var(--color-traffic)"
      >
        {corners?.measured && corners.corner_share ? (
          <CornerLoadCar data={corners} />
        ) : (
          <Waiting what="Reading the load through each tyre…" />
        )}
      </Beat>

      {/* ── 02 the solution ──────────────────────────────────────────────── */}
      <Beat
        n="03"
        kicker="What we built"
        claim="We pull the tyre out, and the answer sharpens as the stint runs."
        read="Follow the grey band from left to right. It starts wide because the model has seen almost nothing, and closes as laps accumulate."
        line="The dark line is what the pit wall knew at that moment, using only the laps already driven. Watch the shaded band close in. That is the model learning from evidence rather than asserting a number — and it is the difference between a guess and a decision."
        tone="var(--color-good)"
      >
        {heroRates.length ? (
          <RateRibbon rows={heroRates as never} which="rate" colour="var(--color-medium)" />
        ) : (
          <Waiting what="Following the longest stint…" />
        )}
      </Beat>

      {/* ── 03 the decision ──────────────────────────────────────────────── */}
      <Beat
        n="04"
        kicker="What a strategist gets"
        claim="Then it tells you where the car will be, and how sure it is."
        read="The solid line is the expected loss and the shaded band is the honest range. The dotted line marks the point where the tyre stops being competitive."
        line="The band widens the further out it looks, and it should. A tool that drew a confident line twenty laps into the future would be lying, and a strategist would find out the hard way. Ours widens honestly, which is what makes the near-term call worth acting on."
        tone="var(--color-good)"
      >
        {projection ? (
          <ProjectionFan projection={projection} colour="var(--color-medium)" marks={[3, 15]} />
        ) : (
          <Waiting what="Projecting ahead…" />
        )}
      </Beat>

      {/* ── 04 the decision ──────────────────────────────────────────────── */}
      <Beat
        n="05"
        kicker="The decision it exists to make"
        claim="Every candidate lap, raced a thousand times over."
        read="Lowest point on the solid curve is the best lap to stop. The shaded block is every lap within a second of it, which is the window a strategist keeps open."
        line="This is not a rule of thumb. For each lap he could stop on, we simulate the rest of the race repeatedly and count how often that lap turns out to have been the right one. The dip is the answer, and the shaded band is the range a strategist should actually keep open."
        tone="var(--color-soft)"
      >
        {/* The optimiser declines when degradation is too flat to separate the
            candidate laps, and says so rather than inventing a lap. Both fields
            are nullable for that reason, so the chart only renders when there is
            a real decision behind it. */}
        {pit?.sweep?.length && pit.optimum_lap != null && pit.stay_out_expected_time != null ? (
          <PitWindowChart
            sweep={pit.sweep}
            optimum={pit.optimum_lap}
            window={pit.window_within_1s as [number, number] | null}
            stayOut={pit.stay_out_expected_time}
          />
        ) : pit?.declined ? (
          <Waiting what="This stint is too flat to separate one lap from another — so we decline rather than guess." />
        ) : (
          <Waiting what="Simulating the rest of the race…" />
        )}
      </Beat>

      {/* ── 05 why the simple fix fails ──────────────────────────────────── */}
      <Beat
        n="06"
        kicker="Why a simpler tool cannot do this"
        claim="Tyres do not wear out in straight lines."
        read="Only the grey bar fades evenly. Everything else warms up, recovers, or falls off a cliff, and a straight-line model is the wrong shape for all of them."
        line="We measured the shape of nearly three thousand real stints. Barely half fade evenly. The rest warm up, recover, or fall off a cliff — and the cliff is the one that ends races. Every competing model assumes a straight line, so it is the wrong shape for almost half of real racing. Ours was never told what shape to expect."
        tone="var(--color-medium)"
      >
        {regimes.length ? (
          <DegradationRegimes rows={regimes} />
        ) : (
          <Waiting what="Loading measured stint shapes…" />
        )}
      </Beat>

      {/* ── 06 four methods, one answer ──────────────────────────────────── */}
      <Beat
        n="07"
        kicker="Why you can trust the number"
        claim="Four independent methods, asked the same question separately."
        read="Each row is the same question under a different assumption. The bars overlap heavily, so the answer is not an artefact of any one choice."
        line="We do not ask the model once. We re-run it under deliberately different assumptions about the things we cannot measure, and see whether the answer moves. When four disagreeing starting points land on the same figure, that figure is a property of the race rather than of our choices. When they scatter, we say so instead of picking one."
        tone="var(--color-fuel)"
      >
        {consensus ? (
          <ConsensusSpread
            estimates={consensus.estimates}
            consensus={consensus.consensus}
            consensusSd={consensus.consensus_sd}
            colour="var(--color-medium)"
          />
        ) : (
          <Waiting what="Re-running under alternative assumptions…" />
        )}
      </Beat>

      {/* ── 07 calibration ───────────────────────────────────────────────── */}
      <Beat
        n="08"
        kicker="And why you can trust the confidence"
        claim="Most tools claim a confidence. We measured ours, and it was wrong."
        read="The dashed diagonal is perfect honesty. Green tracks it. Orange sags below, which is a model claiming more certainty than it has earned."
        line="The straight diagonal is a tool being exactly as sure as it should be. Every competing model sags below it — claiming more certainty than it earns, which is the failure that gets a strategist hurt. Ours sagged too until we rebuilt the confidence from measured outcomes rather than theory. Almost nobody runs this test at all."
        tone="var(--color-fuel)"
      >
        {reliability.length ? (
          <ReliabilityCurve rows={reliability} />
        ) : (
          <Waiting what="Loading the calibration test…" />
        )}
      </Beat>

      {/* ── 07 not a racing trick ────────────────────────────────────────── */}
      <Beat
        n="09"
        kicker="It is bigger than racing"
        claim="The same engine, unchanged, predicts when a jet engine will fail."
        read="Each dot is one engine. The dashed line is a perfect prediction, and dots below it mean we expected failure sooner than it came, which is the safe way to be wrong."
        line="Strip the motorsport words away and this is a general problem: something wears out while it works, you cannot measure it directly, and other things move the only signal you can see. We pointed the identical code at NASA's turbofan benchmark without changing a line. Each dot is an engine. Below the line is the safe direction to be wrong, and most of ours are."
        tone="var(--color-good)"
      >
        {nasa?.predictions ? (
          <RulScatter predictions={nasa.predictions} truths={nasa.truths} />
        ) : (
          <Waiting what="Loading the cross-industry test…" />
        )}
      </Beat>

      {/* ── 08 the business ──────────────────────────────────────────────── */}
      <section
        className="border border-line bg-surface"
        style={{ borderTop: '3px solid var(--color-alert)' }}
      >
        <div className="px-6 py-6">
          <div className="flex items-baseline gap-2.5">
            <span className="num text-[12px] text-alert">10</span>
            <span className="text-[10.5px] tracking-[0.18em] text-ink-faint uppercase">
              Why the industry can actually take this up
            </span>
          </div>
          <h2 className="mt-2.5 max-w-[34ch] text-[26px] leading-[1.12] font-bold tracking-[-0.02em] text-ink sm:text-[30px]">
            No hardware to buy. No data to license. Nothing leaves the garage.
          </h2>
        </div>
        <div className="grid gap-px border-t border-line bg-line sm:grid-cols-2 xl:grid-cols-4">
          <Pillar
            head="Affordable"
            body="It runs on the laptop a team already owns. No GPU, no cloud bill, and the timing data it needs is public."
          />
          <Pillar
            head="Implementable"
            body="Four fields, all of which any series already publishes. There is no integration project and nothing to migrate."
          />
          <Pillar
            head="Scalable"
            body="Each session is independent, so a whole championship weekend runs in parallel. During a session it updates in milliseconds."
          />
          <Pillar
            head="Adoptable"
            body="A new circuit works immediately and new car regulations work the same day, because nothing is trained and there is nothing to retrain."
          />
        </div>
        <div className="border-t border-line px-6 py-5">
          <p className="max-w-[82ch] text-[15.5px] leading-snug font-semibold text-ink">
            Roughly ten teams in the world can model this properly today. Every series below them
            makes the same pit calls, on the same compounds, from the same public feed, with
            nothing.
          </p>
          <p className="mt-2.5 max-w-[82ch] text-[12.5px] leading-relaxed text-ink-dim">
            That is the opportunity. The capability is not new — it has simply never been cheap
            enough to reach anyone outside the top of one championship.
          </p>
        </div>
      </section>

      {/* ── the honest line ──────────────────────────────────────────────── */}
      <section className="border border-line bg-surface px-6 py-5">
        <div className="text-[10.5px] tracking-[0.18em] text-ink-faint uppercase">
          Before you ask us
        </div>
        <p className="mt-2.5 max-w-[84ch] text-[13.5px] leading-relaxed text-ink-dim">
          We are not best at everything and we say so on the evidence screen. A simpler model
          forecasts raw lap times better than we do, and we send that question to it. On real pit
          stops we tie with the closest published research rather than beating it. Thirteen of our
          own ideas failed and we published all of them.{' '}
          <strong className="text-ink">
            We would rather you checked us than believed us.
          </strong>
        </p>
      </section>
    </div>
  )
}


/**
 * The four-corner load check, drawn as a car seen from above.
 *
 * A bar chart of four numbers would be technically identical and would land on
 * nobody. Drawn as tyres in their real positions, the asymmetry is immediate --
 * the loaded side is simply darker and fatter, and a reader with no motorsport
 * background can see which way the car has been leaning.
 *
 * The claim underneath is the part that cannot be fudged. The model is never
 * told which way a circuit runs; it reads the energy through each tyre and
 * infers it, and exp06 got that right on seven of eight circuits. The published
 * direction is shown beside the inferred one so a reader can see the check
 * passing rather than take our word for it.
 */
function CornerLoadCar({ data }: { data: CornerEnergy }) {
  const share = data.corner_share!
  const values = [share.FL, share.FR, share.RL, share.RR]
  const peak = Math.max(...values)
  const quiet = Math.min(...values)
  const span = Math.max(peak - quiet, 1e-6)

  const tyre = (key: 'FL' | 'FR' | 'RL' | 'RR', label: string) => {
    const v = share[key]
    // Normalised within this circuit, because the interesting quantity is which
    // corner worked hardest here, not how this circuit compares with another.
    const heat = (v - quiet) / span
    const width = 26 + heat * 12
    const height = 54 + heat * 14
    return (
      <div className="flex flex-col items-center gap-1.5">
        <div
          className="rounded-[3px]"
          style={{
            width,
            height,
            background: `color-mix(in oklab, var(--color-alert) ${18 + heat * 72}%, var(--color-raised))`,
            border: '1px solid var(--color-line-bright)',
          }}
        />
        <span className="text-[10.5px] text-ink-faint">{label}</span>
        <span className="num text-[12px] font-medium text-ink">{(v * 100).toFixed(1)}%</span>
      </div>
    )
  }

  const agrees =
    data.predicted_direction != null &&
    data.published_direction != null &&
    data.predicted_direction === data.published_direction

  return (
    <div className="grid items-center gap-6 sm:grid-cols-[auto_1fr]">
      <div className="border border-line bg-raised/30 px-7 py-5">
        <div className="mb-3 text-center text-[10px] tracking-[0.16em] text-ink-faint uppercase">
          Front
        </div>
        <div className="flex flex-col gap-3">
          <div className="flex items-end gap-9">
            {tyre('FL', 'Front left')}
            {tyre('FR', 'Front right')}
          </div>
          {/* The chassis, drawn only so the four blocks read as a car. */}
          <div
            className="mx-auto w-[86px] rounded-[2px]"
            style={{ height: 4, background: 'var(--color-line-bright)' }}
          />
          <div className="flex items-start gap-9">
            {tyre('RL', 'Rear left')}
            {tyre('RR', 'Rear right')}
          </div>
        </div>
        <div className="mt-3 text-center text-[10px] tracking-[0.16em] text-ink-faint uppercase">
          Rear
        </div>
      </div>

      <div className="space-y-3">
        <div className="border border-line bg-raised/30 px-4 py-3.5">
          <div className="text-[11px] text-ink-faint">Which side worked harder</div>
          <div className="num mt-1.5 text-[24px] leading-none font-semibold text-ink">
            {((data.left_side_energy_share ?? 0.5) * 100).toFixed(1)}%
            <span className="ml-2 text-[12px] font-normal text-ink-faint">
              of the load went through the left
            </span>
          </div>
        </div>

        <div
          className="border px-4 py-3.5"
          style={{
            borderColor: agrees ? 'var(--color-good)' : 'var(--color-line)',
            background: agrees
              ? 'color-mix(in oklab, var(--color-good) 8%, transparent)'
              : 'transparent',
          }}
        >
          <div className="text-[11px] text-ink-faint">
            Which way the circuit runs
          </div>
          <div className="mt-2 flex flex-wrap items-baseline gap-x-6 gap-y-1">
            <span>
              <span className="text-[11px] text-ink-faint">we inferred </span>
              <span
                className="text-[15px] font-semibold"
                style={{ color: agrees ? 'var(--color-good)' : 'var(--color-ink)' }}
              >
                {data.predicted_direction ?? '—'}
              </span>
            </span>
            <span>
              <span className="text-[11px] text-ink-faint">it is actually </span>
              <span className="text-[15px] font-semibold text-ink">
                {data.published_direction ?? '—'}
              </span>
            </span>
          </div>
          {agrees && (
            <div className="mt-1.5 text-[11.5px]" style={{ color: 'var(--color-good)' }}>
              Correct — and nothing about the circuit was given to the model.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function Pillar({ head, body }: { head: string; body: string }) {
  return (
    <div className="bg-surface px-5 py-5">
      <div className="text-[15px] font-semibold text-ink">{head}</div>
      <p className="mt-2 text-[12.5px] leading-relaxed text-ink-dim">{body}</p>
    </div>
  )
}
