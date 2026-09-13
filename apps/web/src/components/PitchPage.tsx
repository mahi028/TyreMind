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
 * **The problem before the product.** Nothing here describes what we built until
 * the reader has seen the thing that is broken. Beat 2 is the standard method
 * failing, drawn from the race in the rail, and it is the whole pitch: if that
 * chart does not land, no amount of method afterwards will.
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
  type SessionSummary,
  type TrustResult,
} from '../lib/api'
import { DegradationRegimes, PitWindowChart, ReliabilityCurve, RulScatter } from './charts'
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
  children,
  tone = 'var(--color-alert)',
}: {
  n: string
  kicker: string
  claim: string
  line: string
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
        <div className="min-w-0">{children}</div>
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
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [runs, setRuns] = useState<RunRow[]>([])
  const [rates, setRates] = useState<DegradationRow[]>([])
  const [decomp, setDecomp] = useState<DecompositionRow[]>([])
  const [projection, setProjection] = useState<ProjectionResult | null>(null)
  const [pit, setPit] = useState<PitWindow | null>(null)
  const [trust, setTrust] = useState<TrustResult | null>(null)
  const [experiments, setExperiments] = useState<Record<string, any>>({})

  useEffect(() => {
    api.experiments().then(setExperiments).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!sessionId) return
    setSummary(null); setRuns([]); setRates([]); setDecomp([])
    setProjection(null); setPit(null); setTrust(null)
    api.summary(sessionId).then(setSummary).catch(() => undefined)
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

  const backwards = useMemo(
    () =>
      Object.entries(summary?.compounds ?? {}).filter(
        ([, c]) => c.naive_estimate != null && c.naive_estimate < 0,
      ),
    [summary],
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
        line="The tyre is wearing out and slowing the car down. Meanwhile the fuel is burning off and speeding it up by about the same amount, the track is gaining grip, and traffic comes and goes. They arrive added together, as a single number on a timing screen."
      >
        {decomp.length ? (
          <ContributionStack rows={decomp} />
        ) : (
          <Waiting what="Separating this stint…" />
        )}
      </Beat>

      {/* ── 02 the failure ───────────────────────────────────────────────── */}
      <Beat
        n="02"
        kicker="Why it is still unsolved"
        claim="So the obvious method says tyres get faster as they wear out."
        line={
          backwards.length
            ? `Fit a line through lap time against tyre age — the textbook approach — and on this race it reports tyres improving with age on ${backwards.length} of ${Object.keys(summary?.compounds ?? {}).length} compounds. Fuel is hiding the tyre, and it is bigger than the tyre. This is not a corner case. It happens in roughly three races out of four.`
            : 'Fit a line through lap time against tyre age — the textbook approach — and fuel burn cancels the tyre out. On most races it reports tyres improving with age, which is impossible. Try a longer race in the sidebar to see it.'
        }
      >
        {summary ? (
          <NaiveVersusOurs summary={summary} />
        ) : (
          <Waiting what="Fitting the race…" />
        )}
      </Beat>

      {/* ── 03 the solution ──────────────────────────────────────────────── */}
      <Beat
        n="03"
        kicker="What we built"
        claim="We pull the tyre out, and the answer sharpens as the stint runs."
        line="The dark line is what the pit wall knew at that moment, using only the laps already driven. Watch the shaded band close in. That is the model learning from evidence rather than asserting a number — and it is the difference between a guess and a decision."
        tone="var(--color-good)"
      >
        {heroRates.length ? (
          <RateRibbon rows={heroRates as never} which="rate" colour="var(--color-medium)" />
        ) : (
          <Waiting what="Following the longest stint…" />
        )}
      </Beat>

      {/* ── 04 the decision ──────────────────────────────────────────────── */}
      <Beat
        n="04"
        kicker="What a strategist gets"
        claim="Then it tells you where the car will be, and how sure it is."
        line="The band widens the further out it looks, and it should. A tool that drew a confident line twenty laps into the future would be lying, and a strategist would find out the hard way. Ours widens honestly, which is what makes the near-term call worth acting on."
        tone="var(--color-good)"
      >
        {projection ? (
          <ProjectionFan projection={projection} colour="var(--color-medium)" marks={[3, 15]} />
        ) : (
          <Waiting what="Projecting ahead…" />
        )}
      </Beat>

      {/* ── 05 the decision ──────────────────────────────────────────────── */}
      <Beat
        n="05"
        kicker="The decision it exists to make"
        claim="Every candidate lap, raced a thousand times over."
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

      {/* ── 06 why the simple fix fails ──────────────────────────────────── */}
      <Beat
        n="06"
        kicker="Why a simpler tool cannot do this"
        claim="Tyres do not wear out in straight lines."
        line="We measured the shape of nearly three thousand real stints. Barely half fade evenly. The rest warm up, recover, or fall off a cliff — and the cliff is the one that ends races. Every competing model assumes a straight line, so it is the wrong shape for almost half of real racing. Ours was never told what shape to expect."
        tone="var(--color-medium)"
      >
        {regimes.length ? (
          <DegradationRegimes rows={regimes} />
        ) : (
          <Waiting what="Loading measured stint shapes…" />
        )}
      </Beat>

      {/* ── 07 four methods, one answer ──────────────────────────────────── */}
      <Beat
        n="07"
        kicker="Why you can trust the number"
        claim="Four independent methods, asked the same question separately."
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

      {/* ── 08 calibration ───────────────────────────────────────────────── */}
      <Beat
        n="08"
        kicker="And why you can trust the confidence"
        claim="Most tools claim a confidence. We measured ours, and it was wrong."
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
 * The one chart the whole pitch rests on: what the textbook method reports for
 * this race, against what we report, on one axis through zero.
 *
 * Built by hand rather than reached for from `charts.tsx` because none of those
 * answers this question. Beat 2 originally showed `DegradationCurves`, which
 * draws *our* fitted curves rising correctly -- a chart that quietly contradicted
 * the headline above it. A panel reads the claim, looks at the picture, and finds
 * the picture showing tyres wearing normally. That is worse than no chart.
 *
 * Everything left of the zero line is a physical impossibility: a tyre reported
 * as getting faster the longer it runs. Drawing both on one axis is what makes
 * the point without a sentence of explanation.
 */
function NaiveVersusOurs({ summary }: { summary: SessionSummary }) {
  const rows = Object.entries(summary.compounds)
    .filter(([, c]) => c.naive_estimate != null && Number.isFinite(c.naive_estimate))
    .map(([compound, c]) => ({
      compound,
      naive: c.naive_estimate as number,
      ours: c.degradation_rate,
    }))

  if (!rows.length) {
    return <Waiting what="This race reports no comparison." />
  }

  // A symmetric axis so the zero line sits in the middle and "left of zero"
  // reads as wrong without needing a caption to say so.
  const extent = Math.max(
    0.05,
    ...rows.flatMap((r) => [Math.abs(r.naive), Math.abs(r.ours)]),
  ) * 1.15
  const x = (v: number) => ((v + extent) / (2 * extent)) * 100

  return (
    <div>
      <div className="flex items-baseline justify-between text-[11px] text-ink-faint">
        <span className="text-alert">← impossible: tyre getting faster</span>
        <span>seconds lost per lap</span>
        <span style={{ color: 'var(--color-good)' }}>real wear →</span>
      </div>

      <div className="mt-3 space-y-4">
        {rows.map((r) => (
          <div key={r.compound}>
            <div className="mb-1.5 text-[12px] font-medium text-ink">{r.compound}</div>
            {[
              { name: 'Textbook method', v: r.naive, bad: r.naive < 0 },
              { name: 'TyreMind', v: r.ours, bad: false },
            ].map((bar) => {
              const zero = x(0)
              const here = x(bar.v)
              const left = Math.min(zero, here)
              const width = Math.max(Math.abs(here - zero), 0.4)
              const colour = bar.bad ? 'var(--color-alert)' : 'var(--color-good)'
              return (
                <div
                  key={bar.name}
                  className="grid grid-cols-[104px_1fr_66px] items-center gap-3 py-[3px]"
                >
                  <span className="text-[11.5px] text-ink-dim">{bar.name}</span>
                  <div className="relative h-5 bg-raised">
                    <div
                      className="absolute top-0 bottom-0 w-px"
                      style={{ left: `${zero}%`, background: 'var(--color-line-bright)' }}
                    />
                    <div
                      className="absolute top-[3px] bottom-[3px]"
                      style={{ left: `${left}%`, width: `${width}%`, background: colour }}
                    />
                  </div>
                  <span
                    className="num text-right text-[11.5px]"
                    style={{ color: bar.bad ? 'var(--color-alert)' : 'var(--color-ink)' }}
                  >
                    {bar.v >= 0 ? '+' : '−'}
                    {Math.abs(bar.v).toFixed(3)}
                  </span>
                </div>
              )
            })}
          </div>
        ))}
      </div>

      <p className="mt-3 text-[11.5px] leading-relaxed text-ink-faint">
        Both are fitted on exactly the same laps of this race. The only difference is that ours
        accounts for the fuel burning off underneath.
      </p>
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
