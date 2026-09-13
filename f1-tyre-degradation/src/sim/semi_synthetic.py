"""Semi-synthetic validation: real, as-occurred confounders + known synthetic
tyre wear injected on top.

`src/sim/simulator.py`'s pure synthetic generator is a simplified approximation
of real confounding (its own stint-length distribution, driver pace, traffic --
all hand-picked). This module is a stronger test: take the REAL covariates from
real races (`src/data/clean.py`'s output) -- real stint lengths and pit
strategy, real weather, real traffic, real driver/circuit identities -- and
inject only the tyre-wear EFFECT synthetically, since that's the one thing
reality never labels. Fuel load, track-evolution amount and traffic's time-cost
are equally unmeasured in real data, so their effect FUNCTIONS are synthesised
too (same as the model itself assumes) while their INPUTS (`lap_in_stint`,
`session_progress`, `gap_ahead_s`) are real, not invented.

Emits the same schema `src/models/dataset.py` and `NAM` already consume, plus
`true_*` ground-truth columns, so `Vocabs`, `NAM`, `train.fit` and
`src/evaluate.py` are reused completely unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sim.simulator import SimulatorConfig, _true_cond_effect, _true_traffic_effect, true_degradation_curve


def build_semi_synthetic_dataset(
    real_covariates: pd.DataFrame, cfg: SimulatorConfig, *, seed: int = 42,
    zero_degradation: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Injects known synthetic tyre wear (plus fuel/evolution/conditions/traffic
    effect functions) onto real covariates from `src/data/clean.py`.

    Args:
        real_covariates: Output of `clean.clean_corpus` -- must carry `driver`,
            `team_id`, `circuit_id`, `event_id`, `stint_id`, `tyre_age`,
            `lap_in_stint`, `compound`, `session_progress`, `track_temp`,
            `air_temp`, `wind_speed`, `humidity`, `gap_ahead_s`, `real_lap_time`.
        cfg: Simulator config -- reuses the SAME per-compound slopes/cliffs as
            the pure-synthetic test, for direct comparability.
        seed: RNG seed for the synthetic pieces (driver offsets, per-circuit
            evolution amplitude/tau, noise).
        zero_degradation: Negative control -- true slopes/cliffs forced to zero.

    Returns:
        `(dataset, truth)` -- `dataset` matches the real schema plus `lap_time`
        (the new semi-synthetic target) and `event_id`/`driver_id`/`circuit_id`
        columns encoded as dense integers (`Vocabs.fit` handles arbitrary
        values fine, but the rest of the pipeline expects integer-like ids
        consistent with the pure-synthetic path). `truth` carries the
        per-compound curves and per-circuit evolution parameters, mirroring
        `generate_synthetic_dataset`'s `return_truth` output.
    """
    rng = np.random.default_rng(seed)
    df = real_covariates.copy()

    curves = {c: true_degradation_curve(c, cfg) for c in cfg.compounds}
    if zero_degradation:
        curves = {c: np.zeros_like(v) for c, v in curves.items()}

    # Dense integer ids -- circuit/driver/event are currently free-form strings
    # from clean.py; encode them here so downstream code (which expects
    # small-integer categoricals, same as the pure-synthetic path) is unaffected.
    circuit_categories = sorted(df["circuit_id"].unique())
    circuit_index = {c: i for i, c in enumerate(circuit_categories)}
    df["circuit_id"] = df["circuit_id"].map(circuit_index)

    driver_categories = sorted(df["driver"].unique())
    driver_index = {d: i for i, d in enumerate(driver_categories)}
    df["driver_id"] = df["driver"].map(driver_index)

    event_categories = sorted(df["event_id"].unique())
    event_index = {e: i for i, e in enumerate(event_categories)}
    df["event_id"] = df["event_id"].map(event_index)

    n_circuits = len(circuit_categories)
    n_drivers = len(driver_categories)

    # REAL circuit base pace -- the median of the actual lap times at that
    # circuit, not invented. Dominated by track length/layout, same role
    # `CircuitHead` plays in the model.
    real_circuit_pace = df.groupby("circuit_id")["real_lap_time"].median().to_dict()

    # Synthetic driver offsets: small, not the thing under test.
    driver_pace = rng.normal(0.0, 0.25, size=n_drivers)

    # Per-circuit evolution amplitude/tau -- synthetic (real evolution amount is
    # unmeasured), same functional form as the pure-synthetic generator.
    circuit_evo_amplitude = rng.uniform(0.3, 0.9, size=n_circuits)
    circuit_evo_tau = rng.uniform(0.15, 0.40, size=n_circuits)

    circuit_ids = df["circuit_id"].to_numpy()
    driver_ids = df["driver_id"].to_numpy()
    tyre_age = df["tyre_age"].to_numpy()
    lap_in_stint = df["lap_in_stint"].to_numpy()
    session_progress = df["session_progress"].to_numpy()
    compound = df["compound"].to_numpy()

    true_tyre_effect = np.array(
        [curves[c][min(int(age), cfg.max_age)] for c, age in zip(compound, tyre_age)]
    )
    true_fuel_effect = cfg.fuel_time_per_kg_s * (-cfg.fuel_burn_per_lap_kg * lap_in_stint)

    amp = circuit_evo_amplitude[circuit_ids]
    tau = circuit_evo_tau[circuit_ids]
    true_evo_effect = -amp * (1.0 - np.exp(-session_progress / tau))

    true_cond_effect = _true_cond_effect(
        df["track_temp"].to_numpy(), df["humidity"].to_numpy(), df["wind_speed"].to_numpy()
    )
    true_traffic_effect = _true_traffic_effect(df["gap_ahead_s"].to_numpy())

    true_circuit_base = np.array([real_circuit_pace[c] for c in circuit_ids])
    true_driver_base = driver_pace[driver_ids]

    noise = rng.normal(0.0, 0.06, size=len(df))
    mistake = rng.random(len(df)) < 0.05
    noise = noise + np.where(mistake, rng.exponential(0.45, size=len(df)), 0.0)

    df["fuel_kg_rel"] = -cfg.fuel_burn_per_lap_kg * lap_in_stint
    df["lap_time"] = (
        true_circuit_base + true_driver_base + true_fuel_effect + true_tyre_effect
        + true_evo_effect + true_cond_effect + true_traffic_effect + noise
    )
    df["true_tyre_effect"] = true_tyre_effect
    df["true_fuel_effect"] = true_fuel_effect
    df["true_evo_effect"] = true_evo_effect
    df["true_cond_effect"] = true_cond_effect
    df["true_traffic_effect"] = true_traffic_effect
    df["true_driver_base"] = true_driver_base
    df["true_circuit_base"] = true_circuit_base
    df["true_noise"] = noise
    df["true_total"] = df["lap_time"]

    truth = {
        "curves": curves,
        "circuit_evo_amplitude": circuit_evo_amplitude,
        "circuit_evo_tau": circuit_evo_tau,
        "circuit_base_pace": np.array([real_circuit_pace.get(i, np.nan) for i in range(n_circuits)]),
        "circuit_categories": circuit_categories,
        "driver_categories": driver_categories,
    }
    return df, truth
