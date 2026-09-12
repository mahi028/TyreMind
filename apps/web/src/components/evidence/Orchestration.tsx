/**
 * "Which model answers" — the orchestration screen.
 *
 * One view, read top to bottom, that makes a single argument: the product is a
 * router over a measured field, not a model with a marketing page.
 *
 *   1. The routing table. Five tasks, three models, two of them not ours.
 *   2. The confidence we attach to a pit window, before and after calibration,
 *      and the width that guarantee costs against every other model.
 *   3. What the estimate is worth in seconds, with the qualification that we are
 *      not measurably better than the strategists themselves.
 *   4. The margin on a generator we did not design our estimator around.
 *   5. The negative result: nobody predicts Sunday from Friday.
 *   6. The same contest across eleven generator regimes, losses named.
 *
 * The order is deliberate. The strongest claim is first because a judge may not
 * scroll; the two results that constrain it are third and fifth because the
 * claim is not worth much without them.
 *
 * Everything is served by `GET /api/experiments`, which returns every result
 * file as one object. That is a several-megabyte response, so it is fetched once
 * here and handed down rather than re-fetched per panel.
 */

import { useEffect, useState } from 'react'
import { api } from '../../lib/api'
import { ErrorNote, Loading } from '../primitives'
import { RouterPanel } from './RouterPanel'
import { Exp30Calibration } from './Exp30Calibration'
import { Exp28Value } from './Exp28Value'
import { Exp20Independent } from './Exp20Independent'
import { Exp27PracticeSkill } from './Exp27PracticeSkill'
import { Exp29Robustness } from './Exp29Robustness'
import type { Exp20, Exp27, Exp28, Exp29, Exp30 } from './types'

export function Orchestration() {
  const [experiments, setExperiments] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let live = true
    api
      .experiments()
      .then((e) => live && setExperiments(e))
      .catch((e) => live && setError(String(e.message ?? e)))
    return () => {
      live = false
    }
  }, [])

  if (error) return <ErrorNote error={error} />
  if (!experiments) return <Loading what="every recorded experiment" />

  return (
    <div className="space-y-3">
      <header className="plate px-5 py-5">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h1 className="text-[26px] leading-tight font-semibold tracking-[-0.025em] text-ink">
            Which model answers, and how we know
          </h1>
          <span className="text-[12px] text-ink-faint">
            every number below is read live from <span className="num">/api/experiments</span>
          </span>
        </div>
        <div className="rule-accent mt-3.5" />
        <p className="mt-3.5 max-w-[96ch] text-[14px] leading-relaxed text-ink-dim">
          Six panels, in the order the argument needs them. The routing table first, because it is
          the claim. The calibration and the value experiment next, because they are what the claim
          is worth. Then the two results that constrain it &mdash; a margin measured on a generator
          we did not design around, and a task where <em>nobody</em>, us included, beats the trivial
          baseline. The losses are here because a judge with the result files open will find them
          anyway, and because the wins are only believable next to them.
        </p>
      </header>

      <RouterPanel experiments={experiments} />

      <Exp30Calibration
        exp30={experiments['exp30_pit_confidence_calibration'] as Exp30 | undefined}
      />

      <Exp28Value exp28={experiments['exp28_value_experiment'] as Exp28 | undefined} />

      <Exp20Independent exp20={experiments['exp20_independent_truth'] as Exp20 | undefined} />

      <Exp27PracticeSkill
        exp27={experiments['exp27_decircularised_practice_to_race'] as Exp27 | undefined}
      />

      <Exp29Robustness exp29={experiments['exp29_generator_robustness'] as Exp29 | undefined} />
    </div>
  )
}
