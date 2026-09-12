/**
 * Shapes of the experiment payloads served by `/api/experiments`.
 *
 * Typed against the JSON that is actually on the wire, verified by calling the
 * running API rather than by reading `docs/FRONTEND_HANDOFF.md`, which is stale
 * in places. Anything optional here is optional on the wire: an experiment that
 * has not been re-run leaves the key out, and the panels say so rather than
 * rendering a plausible zero.
 */

/** One model's row in a scoring table. The standard error is never optional. */
export interface ScoredModel {
  model: string
  n: number
  rate_mae: number
  rate_mae_se: number
  rate_bias: number
  coverage_95: number
  n_failed?: number
}

/** A paired margin between two models, and whether it survives its own noise. */
export interface PairedMargin {
  better?: string
  worse?: string
  n_paired?: number
  margin?: number
  margin_se?: number
  decisive?: boolean
}

// --- exp20 -----------------------------------------------------------------

export interface Exp20TargetReading {
  target: string
  winner: string
  best_model: string
  best_mae: number
  runner_up: string
  paired_margin_best_over_runner_up: PairedMargin
  tyremind_mae: number
  tyremind_rank: number
  best_non_tyremind: string
  paired_margin_tyremind_over_best_other: PairedMargin
  table: ScoredModel[]
}

export interface Exp20 {
  n_seeds: number
  arms: string[]
  targets: Record<string, string>
  primary_target: string
  honest_framing: string
  verdict: {
    primary_target: string
    tyremind_first_on_physics: boolean
    tyremind_decisive_on_physics: boolean
    physics_margin: number
    physics_margin_se: number
    physics_best_by_target: Record<string, string>
    physics_ranking_depends_on_target: boolean
    synthetic_best_by_target: Record<string, string>
    synthetic_ranking_depends_on_target: boolean
    synthetic_margin: number
    survival_ratio: number
  }
  readings: Record<'physics' | 'synthetic', Record<string, Exp20TargetReading>>
}

// --- exp27 -----------------------------------------------------------------

export interface Exp27Row {
  model: string
  n: number
  skill_vs_own_climatology: number
  skill_se: number
  self_mae: number
  self_mae_se: number
  self_bias: number
  coverage_95: number
  climatology_mae: number
  spearman_practice_vs_race: number
}

export interface Exp27 {
  practice_session: string
  n_events: number
  n_comparisons: number
  summary: Exp27Row[]
  verdict: {
    tyremind_skill: number
    tyremind_rank_decircularised: number
    best_model: string
    win_claimed: boolean
    practice_uninformative: boolean
  }
  excluded_not_dry: Record<string, string>
}

// --- exp28 -----------------------------------------------------------------

export interface Exp28ValueRow {
  model: string
  n_stops: number
  value_s_per_stop: number
  value_se_per_stop: number
  value_s_per_car_race: number
  mean_abs_error_laps: number
  answer_rate: number
}

export interface Exp28 {
  n_sessions: number
  n_sims: number
  pit_loss_s: number
  assumed_seconds_per_position: number
  headline_sentence: string
  n_final_stops: number
  verdict: {
    gap_seconds_per_car_race: number
    gap_positions_per_car_race: number
    seconds_per_position_measured_median: number
    tyremind_value_s_per_car_race: number
    naive_value_s_per_car_race: number
    paired_referee_a: {
      model_a: string
      model_b: string
      n_paired_stops: number
      mean_difference_s_per_car_race: number
      se_s_per_car_race: number
      significant_at_1_se: boolean
    }
    paired_referee_b: {
      mean_difference_s_per_car_race: number
      se_s_per_car_race: number
      significant_at_1_se: boolean
    }
    claim_stands: boolean
    underpowered: boolean
    referees_disagree: boolean
    sentence: string
  }
  referee_a_theil_sen: {
    all_answered_final_stops: Exp28ValueRow[]
  }
  deviations: string[]
}

// --- exp29 -----------------------------------------------------------------

export interface Exp29Regime {
  regime: string
  overrides: Record<string, number>
  winner: string
  best_model: string
  best_mae: number
  runner_up: string
  runner_up_mae: number
  paired_margin_best_over_runner_up: PairedMargin
  tyremind_mae: number
  tyremind_rank: number
  table: ScoredModel[]
}

export interface Exp29 {
  n_seeds: number
  regimes: Record<string, Record<string, number>>
  verdict: {
    wins: string[]
    ties: string[]
    losses: string[]
    n_regimes: number
    ranking_survives: boolean
  }
  per_regime: Exp29Regime[]
}

// --- exp30 -----------------------------------------------------------------

export interface Exp30SummaryRow {
  model: string
  n: number
  mean_prob_on_actual: number
  uniform_chance: number
  lift: number
  /** What the pre-calibration window claimed. */
  mean_window_mass: number
  /** What it actually delivered. The pair is the finding. */
  observed_window_hit: number
  calibration_gap: number
  top_3_hit: number
  top_3_chance: number
  top_5_hit: number
  top_5_chance: number
}

export interface Exp30Row {
  model: string
  driver: string
  actual: number
  recommended: number
  session: string
}

export interface Exp30 {
  n_sessions: number
  window_half_width_laps: number
  summary: Exp30SummaryRow[]
  reliability_tyremind: {
    claimed_low: number
    claimed_high: number
    mean_claimed: number
    observed: number
    n: number
  }[]
  rows: Exp30Row[]
}

// --- exp19 and exp22, already on screen elsewhere but needed by the router ---

export interface Exp19 {
  lap_time_prediction: { model: string; crps: number; mae: number; coverage_95: number }[]
  degradation_recovery: { model: string; rate_mae: number; coverage: number; n: number }[]
}

export interface Exp22 {
  n_common_stops: number
  like_for_like: {
    model: string
    n_common: number
    mae_laps: number
    mae_se: number
    hit_rate_within_2: number
    answered: number
    answer_rate: number
  }[]
}

export const OURS = 'TyreMind state-space'
