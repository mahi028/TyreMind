"""exp29 -- does our synthetic win survive when the generator stops agreeing with us?

Pre-registered in `experiments/PREREGISTRATION_exp29.md`. Read that first; it
fixes the regimes, the metrics and the losing conditions.

The strongest objection to exp19 Leg C is the obvious one: the ground truth is
synthetic and we wrote the generator. `PREREGISTRATION_exp19.md` section 6.1
concedes it. This measures how much of the win survives when the generator is
swept away from our estimator's assumptions -- heavier tails than Gaussian, three
times the noise, a cliff five times more severe or absent entirely, tyre age fully
decoupled from laps-in-run or fully collinear with it, heavy traffic, and -- the
one we expect to lose -- no track evolution at all, so the track machinery we
carry becomes dead weight.

Every comparison is **paired**: the same generated session goes to all six models,
and the statistic is the paired difference in absolute rate error with its
standard error. A pilot at four seeds ranked Pooled above us in the default regime
while exp19 at eight seeds ranked us above Pooled; a ranking that flips with seed
count is noise, so this runs 32 seeds and reports the standard error next to
every margin.

    python experiments/exp29_generator_robustness.py
    python experiments/exp29_generator_robustness.py --seeds 8 --regimes as_shipped flat_track

Writes experiments/results/exp29_generator_robustness.json.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.synthetic import SessionConfig, generate_session
from tyremind.models.baselines import DegradationModel
from tyremind.models.literature import extended_ladder

RESULTS = Path(__file__).parent / "results" / "exp29_generator_robustness.json"

#: Seeds, identical across every regime and every model so that each comparison is
#: paired on the generated session rather than on the regime average.
BASE_SEED = 20260901

#: Each regime moves ONE axis of the generator. The comment on each says what it
#: attacks; `flat_track` is pre-declared as an expected loss and `gaussian_noise`
#: as an expected win, so neither can be quoted later as if it were neutral.
REGIMES: dict[str, dict] = {
    "as_shipped":      {},
    # Our estimator assumes Gaussian observation noise. t(2.5) has no finite variance.
    "heavy_tails":     {"observation_noise_df": 2.5},
    # Signal-to-noise: degradation is ~0.07 s/lap buried under 0.48 s of scatter.
    "triple_noise":    {"observation_noise_sd": 0.48},
    # The opposite direction: noise exactly as our estimator assumes it.
    "gaussian_noise":  {"observation_noise_df": 1e6},
    # Strong non-linearity in age. A linear rate cannot represent it.
    "severe_cliff":    {"cliff_severity": 0.020},
    # Perfectly linear degradation -- the shape the competitors assume.
    "no_cliff":        {"cliff_severity": 0.0},
    # Tyre age fully decoupled from laps completed this run.
    "all_scrubbed":    {"scrubbed_set_probability": 1.0},
    # Tyre age and laps-in-run collinear: the exp18 worst case.
    "no_scrubbed":     {"scrubbed_set_probability": 0.0},
    # A confounder Cappello & Hoegh absorb into i.i.d. noise and we model.
    "heavy_traffic":   {"traffic_probability": 0.55, "traffic_coefficient": 2.2},
    # A confounder WE model and the others do not. Our machinery becomes dead weight.
    "flat_track":      {"track_evolution_total": 0.0},
    "strong_track":    {"track_evolution_total": 2.5},
}

#: Declared in the pre-registration, before any number existed.
EXPECTED = {"flat_track": "loss", "gaussian_noise": "win"}

OUR_MODEL = "TyreMind state-space"


def rate_bearing_models() -> list:
    """The rungs that have a degradation parameter at all.

    ARIMA, LightGBM and the MLP never override `compound_rates`, so they inherit
    the base class's empty dict. That is the exp19 finding, detected here rather
    than hard-coded as a list of names -- a hard-coded list would silently go
    stale if a rung gained a rate. They are not fitted, because fitting a model
    that cannot produce the quantity being scored only costs time; the omission is
    pre-registered.
    """
    return [
        m for m in extended_ladder()
        if type(m).compound_rates is not DegradationModel.compound_rates
    ]


def score_session(session, models: list) -> list[dict]:
    """Fit every rate-bearing model to one generated session and score its rates."""
    truth = session.truth
    baseline = dict(truth.compound_rates)
    # Lap-weighted mean of the TRUE instantaneous rate over the laps actually run
    # on each compound. Differs from the baseline wherever the cliff bites, which
    # is why both targets are reported.
    instantaneous = truth.lap_truth.groupby("compound")["true_rate"].mean().to_dict()

    rows: list[dict] = []
    for model in models:
        try:
            model.fit(session.lap_table)
            rates = model.compound_rates()
        except Exception as exc:  # noqa: BLE001 - one rung failing must not stop the sweep
            rows.append({"model": model.name, "compound": None, "failed": str(exc)})
            continue
        for compound, (mean, sd) in rates.items():
            compound = str(compound)
            if compound not in baseline or not np.isfinite(mean):
                continue
            sd = float(sd) if np.isfinite(sd) else float("nan")
            error = float(mean) - baseline[compound]
            rows.append({
                "model": model.name,
                "compound": compound,
                "estimate": float(mean),
                "estimate_sd": sd,
                "true_baseline": float(baseline[compound]),
                "true_mean_instantaneous": float(instantaneous.get(compound, np.nan)),
                "error": error,
                "abs_error": abs(error),
                "error_vs_instantaneous": float(mean) - float(
                    instantaneous.get(compound, np.nan)
                ),
                "covered_95": bool(abs(error) <= 1.96 * sd) if np.isfinite(sd) and sd > 0 else False,
                "failed": None,
            })
    return rows


def run_regime(name: str, overrides: dict, seeds: list[int]) -> list[dict]:
    """One regime, every seed. The same session is handed to every model."""
    rows: list[dict] = []
    for seed in seeds:
        config = replace(SessionConfig(), seed=seed, **overrides)
        try:
            session = generate_session(config)
        except Exception as exc:  # noqa: BLE001
            rows.append({"regime": name, "seed": seed, "model": None,
                         "failed": f"generator: {exc}"})
            continue
        for row in score_session(session, rate_bearing_models()):
            rows.append({"regime": name, "seed": seed, **row})
    return rows


def summarise_regime(block: pd.DataFrame) -> pd.DataFrame:
    """Per-model aggregate inside one regime."""
    out = []
    for model, rows in block.groupby("model"):
        scored = rows[rows["abs_error"].notna()]
        if scored.empty:
            out.append({"model": model, "n": 0, "rate_mae": float("nan")})
            continue
        errors = scored["error"].to_numpy(dtype=float)
        absolute = np.abs(errors)
        n = absolute.size
        out.append({
            "model": model,
            "n": int(n),
            "rate_mae": float(absolute.mean()),
            "rate_mae_se": float(absolute.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "rate_bias": float(errors.mean()),
            "coverage_95": float(scored["covered_95"].mean()),
            "rate_mae_vs_instantaneous": float(
                np.abs(scored["error_vs_instantaneous"].to_numpy(dtype=float)).mean()
            ),
            "n_failed": int((rows["failed"].notna()).sum()),
        })
    return pd.DataFrame(out).sort_values("rate_mae")


def paired_margin(block: pd.DataFrame, a: str, b: str) -> dict:
    """Paired difference in absolute rate error between two models.

    Paired on (seed, compound): the same generated session, the same compound.
    Between-seed variance dwarfs the between-model difference, so an unpaired
    comparison would drown a real effect -- which is how a four-seed pilot managed
    to rank Pooled above us in the very regime exp19 ranks us above Pooled.

    Returns the margin `MAE(b) - MAE(a)`, positive when `a` is better.
    """
    left = block[block["model"] == a].set_index(["seed", "compound"])["abs_error"]
    right = block[block["model"] == b].set_index(["seed", "compound"])["abs_error"]
    shared = left.index.intersection(right.index)
    if len(shared) < 2:
        return {"n_paired": int(len(shared))}
    diff = (right.loc[shared] - left.loc[shared]).to_numpy(dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "better": a, "worse": b, "n_paired": int(n),
        "margin": float(diff.mean()), "margin_se": se,
        "decisive": bool(np.isfinite(se) and diff.mean() > se),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--regimes", nargs="*", default=sorted(REGIMES))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    unknown = [r for r in args.regimes if r not in REGIMES]
    if unknown:
        raise SystemExit(f"unknown regime(s): {unknown}. Known: {sorted(REGIMES)}")

    seeds = [BASE_SEED + i for i in range(args.seeds)]
    no_rate_models = sorted(
        {m.name for m in extended_ladder()} - {m.name for m in rate_bearing_models()}
    )

    all_rows: list[dict] = []
    per_regime: list[dict] = []

    for name in args.regimes:
        started = time.perf_counter()
        rows = run_regime(name, REGIMES[name], seeds)
        all_rows.extend(rows)
        block = pd.DataFrame(rows)
        table = summarise_regime(block[block["model"].notna()])
        if table.empty:
            print(f"  {name:<16} FAILED -- nothing scored")
            continue

        ranked = table.reset_index(drop=True)
        best_name = str(ranked.loc[0, "model"])
        runner_up_name = str(ranked.loc[1, "model"]) if len(ranked) > 1 else best_name
        best, runner_up = ranked.loc[0], ranked.loc[min(1, len(ranked) - 1)]

        margin = (
            paired_margin(block, best_name, runner_up_name)
            if best_name != runner_up_name else {}
        )
        # A regime is only "won" when the best beats the runner-up by more than
        # one standard error of the PAIRED difference. Anything else is a tie, and
        # a tie is a tie -- the four-seed pilot's rankings were mostly ties.
        winner = best_name if bool(margin.get("decisive", False)) else "TIE"

        our_rows = ranked.index[ranked["model"] == OUR_MODEL]
        our_rank = int(our_rows[0]) + 1 if len(our_rows) else None
        our_mae = float(ranked.loc[our_rows[0], "rate_mae"]) if len(our_rows) else float("nan")
        our_margin = (
            paired_margin(block, OUR_MODEL, best_name)
            if our_rank is not None and best_name != OUR_MODEL else {}
        )

        per_regime.append({
            "regime": name,
            "overrides": REGIMES[name],
            "pre_registered_expectation": EXPECTED.get(name),
            "winner": winner,
            "best_model": best_name,
            "best_mae": float(best["rate_mae"]),
            "runner_up": runner_up_name,
            "runner_up_mae": float(runner_up["rate_mae"]),
            "paired_margin_best_over_runner_up": margin,
            "tyremind_mae": our_mae,
            "tyremind_rank": our_rank,
            "paired_margin_tyremind_vs_best": our_margin,
            "table": ranked.to_dict(orient="records"),
        })
        print(f"  {name:<16} best {best_name[:28]:<28} "
              f"{best['rate_mae']:.4f}  ours {our_mae:.4f}  "
              f"winner {winner[:20]:<20} {time.perf_counter() - started:>6.1f}s", flush=True)

    if not per_regime:
        raise SystemExit("no regime produced a result")

    # --- the pre-registered reading -------------------------------------
    wins = [r["regime"] for r in per_regime if r["winner"] == OUR_MODEL]
    ties = [r["regime"] for r in per_regime if r["winner"] == "TIE"]
    losses = [r["regime"] for r in per_regime
              if r["winner"] not in (OUR_MODEL, "TIE")]
    survives = len(wins) > len(per_regime) / 2.0

    print("\n" + "=" * 112)
    print(f"GENERATOR ROBUSTNESS -- {len(per_regime)} regimes x {args.seeds} seeds x 3 compounds, "
          "paired on the generated session")
    print("=" * 112)
    print(f"{'regime':<16}{'winner':<32}{'ours':>9}{'best':>9}{'margin':>9}{'+-SE':>8}"
          f"{'rank':>6}  expected")
    for r in per_regime:
        m = r["paired_margin_best_over_runner_up"]
        print(f"{r['regime']:<16}{r['winner'][:30]:<32}{r['tyremind_mae']:>9.4f}"
              f"{r['best_mae']:>9.4f}{m.get('margin', float('nan')):>9.4f}"
              f"{m.get('margin_se', float('nan')):>8.4f}{str(r['tyremind_rank']):>6}"
              f"  {EXPECTED.get(r['regime'], '')}")
    print("=" * 112)
    print(f"TyreMind wins  : {len(wins)} of {len(per_regime)}  "
          + (", ".join(wins) if wins else "none"))
    print(f"Ties           : {len(ties)}  " + (", ".join(ties) if ties else "none"))
    print(f"TyreMind loses : {len(losses)}  " + (", ".join(losses) if losses else "none"))
    print()
    if survives:
        print("The exp19 Leg C ranking SURVIVES the sweep: TyreMind is best in a majority "
              "of regimes.")
    else:
        print("THE exp19 LEG C RANKING DOES NOT SURVIVE. Our synthetic advantage depends on "
              "the generator's default settings, and exp19 Leg C must be presented with "
              "that caveat attached.")

    for regime in losses:
        row = next(r for r in per_regime if r["regime"] == regime)
        margin = row["paired_margin_tyremind_vs_best"]
        note = " (pre-declared as an expected loss)" if EXPECTED.get(regime) == "loss" else ""
        print(f"  LOST {regime}{note}: {row['best_model']} at {row['best_mae']:.4f} against "
              f"our {row['tyremind_mae']:.4f}; paired margin "
              f"{margin.get('margin', float('nan')):+.4f} "
              f"+- {margin.get('margin_se', float('nan')):.4f} s/lap against us")

    flat = next((r for r in per_regime if r["regime"] == "flat_track"), None)
    if flat is not None:
        print()
        print("On flat_track specifically -- the cost of modelling a confounder that is not "
              "there:")
        print(f"  With no track evolution to explain, our track-evolution machinery is free "
              f"parameters fitted to noise. Ours {flat['tyremind_mae']:.4f}, best "
              f"{flat['best_model']} {flat['best_mae']:.4f}. "
              + ("We lost it, as pre-declared." if flat["winner"] not in (OUR_MODEL, "TIE")
                 else f"Outcome: {flat['winner']}, against the pre-declared expectation of a loss."))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "experiment": "exp29_generator_robustness",
        "preregistration": "experiments/PREREGISTRATION_exp29.md",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_seeds": args.seeds,
        "base_seed": BASE_SEED,
        "regimes": {k: REGIMES[k] for k in args.regimes},
        "pre_registered_expectations": EXPECTED,
        "models_without_degradation_parameter": no_rate_models,
        "verdict": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "n_regimes": len(per_regime),
            "ranking_survives": survives,
        },
        "per_regime": per_regime,
        "rows": all_rows,
    }, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
