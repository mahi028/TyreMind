"""Phase 3 — synthetic ground-truth simulator.

Generates fake sessions with a KNOWN hidden tyre-degradation curve, buried under
the same confounding structure as a real session: stints start at high fuel and
fresh tyres, cars run at different session times, drivers do 2-4 runs per
session. The emitted dataframe has the exact schema `features.py` produces for
real data, plus hidden ground-truth columns prefixed `true_`, so the same model
code path used on real data can be pointed at this and scored against a known
answer.

Units: lap time and all effects in seconds, fuel in kg, temperature in Celsius,
wind speed in m/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_CIRCUITS = 10
N_TEAMS = 10
N_DRIVERS = N_TEAMS * 2

# Reference conditions the condition-head effect is centered on -- track_temp of
# 35C / humidity of 50% / zero wind is "typical", not a cause of extra lap time.
_REF_TRACK_TEMP_C = 35.0
_REF_HUMIDITY_PCT = 50.0

# True traffic effect saturates to (about) zero beyond this gap, mirroring the
# `zero_penalty_gap_s` used to regularise the real traffic head.
_TRAFFIC_FULL_EFFECT_GAP_S = 0.4
_TRAFFIC_ZERO_EFFECT_GAP_S = 3.0
_TRAFFIC_MAX_EFFECT_S = 0.5


@dataclass
class SimulatorConfig:
    compounds: list[str]
    true_linear_slope_s_per_lap: dict[str, float]
    true_cliff_age: dict[str, int]
    true_post_cliff_slope_multiplier: float
    n_sessions: int
    n_drivers_per_session: int
    n_runs_per_driver: list[int]  # [min, max] inclusive
    max_age: int
    fuel_time_per_kg_s: float
    fuel_burn_per_lap_kg: float

    @classmethod
    def from_yaml(cls, path: str | Path = "config/default.yaml") -> "SimulatorConfig":
        raw = yaml.safe_load(Path(path).read_text())
        sim = raw["simulator"]
        return cls(
            compounds=sim["compounds"],
            true_linear_slope_s_per_lap=sim["true_linear_slope_s_per_lap"],
            true_cliff_age=sim["true_cliff_age"],
            true_post_cliff_slope_multiplier=sim["true_post_cliff_slope_multiplier"],
            n_sessions=sim["n_sessions"],
            n_drivers_per_session=sim["n_drivers_per_session"],
            n_runs_per_driver=sim["n_runs_per_driver"],
            max_age=raw["tyre"]["max_age"],
            fuel_time_per_kg_s=raw["fuel"]["time_per_kg_s"],
            fuel_burn_per_lap_kg=raw["fuel"]["burn_per_lap_kg"],
        )


def true_degradation_curve(compound: str, cfg: SimulatorConfig) -> np.ndarray:
    """The ground-truth cumulative tyre effect, seconds lost vs. a fresh tyre.

    Shape `(cfg.max_age + 1,)`, index 0 is a fresh tyre (`== 0` by construction),
    matching exactly what the model's `g_tyre` head is built to emit.

    Args:
        compound: One of `cfg.compounds`.
        cfg: Simulator configuration carrying the per-compound truth.

    Returns:
        Cumulative seconds lost at each integer tyre age 0..max_age.
    """
    slope = cfg.true_linear_slope_s_per_lap[compound]
    cliff = cfg.true_cliff_age[compound]
    post_slope = slope * cfg.true_post_cliff_slope_multiplier

    increments = np.zeros(cfg.max_age + 1)
    for age in range(1, cfg.max_age + 1):
        increments[age] = slope if age <= cliff else post_slope
    return np.cumsum(increments)


def _true_cond_effect(track_temp_c: np.ndarray, humidity_pct: np.ndarray,
                       wind_speed_ms: np.ndarray) -> np.ndarray:
    """Smooth, small, centered-at-reference condition effect. Seconds."""
    return (
        0.012 * (track_temp_c - _REF_TRACK_TEMP_C)
        - 0.004 * (humidity_pct - _REF_HUMIDITY_PCT)
        + 0.010 * wind_speed_ms
    )


def _true_traffic_effect(gap_ahead_s: np.ndarray) -> np.ndarray:
    """Monotone decreasing in gap, saturating to ~0 beyond `_TRAFFIC_ZERO_EFFECT_GAP_S`."""
    span = _TRAFFIC_ZERO_EFFECT_GAP_S - _TRAFFIC_FULL_EFFECT_GAP_S
    scaled = np.clip((_TRAFFIC_ZERO_EFFECT_GAP_S - gap_ahead_s) / span, 0.0, 1.0)
    return _TRAFFIC_MAX_EFFECT_S * scaled


def generate_synthetic_dataset(
    cfg: SimulatorConfig | None = None,
    *,
    seed: int = 42,
    zero_degradation: bool = False,
    return_truth: bool = False,
) -> pd.DataFrame:
    """Generate the full synthetic corpus.

    Args:
        cfg: Simulator configuration. Loaded from `config/default.yaml` if omitted.
        seed: RNG seed, for reproducibility.
        zero_degradation: If True, every compound's true slope and post-cliff
            slope are forced to zero -- the negative control. A method that
            reports nonzero degradation here is finding noise, not tyre wear.
        return_truth: If True, also return a dict of the exact ground-truth
            parameters (per-compound curves, per-circuit evolution amplitude/tau,
            circuit base pace) for the recovery test to score against directly,
            rather than reverse-engineering them from noisy `true_*` columns.

    Returns:
        One row per driver per lap, columns matching the real feature schema
        (see `src/data/features.py`) plus hidden `true_*` columns:
        `true_tyre_effect`, `true_fuel_effect`, `true_evo_effect`,
        `true_cond_effect`, `true_traffic_effect`, `true_driver_base`,
        `true_circuit_base`, `true_noise`, `true_total` (== `lap_time`).
        If `return_truth`, a `(df, truth)` tuple instead.
    """
    cfg = cfg or SimulatorConfig.from_yaml()
    rng = np.random.default_rng(seed)

    curves = {c: true_degradation_curve(c, cfg) for c in cfg.compounds}
    if zero_degradation:
        curves = {c: np.zeros_like(v) for c, v in curves.items()}

    # Fixed per-entity ground truth, drawn once.
    circuit_base_pace = rng.uniform(75.0, 108.0, size=N_CIRCUITS)
    team_pace = rng.normal(0.0, 0.35, size=N_TEAMS)
    driver_pace = rng.normal(0.0, 0.25, size=N_DRIVERS)
    # Track evolution is drawn per CIRCUIT, not per event/session: the model's
    # EvoHead is restricted to (session_progress, circuit_id) -- it has no way
    # to represent a different evolution shape at every visit to the same
    # circuit, so the ground truth must not have one either. Physically this
    # matches "the same track surface rubbers in the same way each time it's
    # raced on" -- a modelling assumption, not a proven fact, and worth noting
    # as scope in the final report.
    circuit_evo_amplitude = rng.uniform(0.3, 0.9, size=N_CIRCUITS)
    circuit_evo_tau = rng.uniform(0.15, 0.40, size=N_CIRCUITS)

    rows: list[dict] = []

    for event_id in range(cfg.n_sessions):
        circuit_id = event_id % N_CIRCUITS
        season = 2022 + (event_id // (cfg.n_sessions // 4 + 1))

        track_temp = float(rng.normal(_REF_TRACK_TEMP_C, 6.0))
        air_temp = track_temp - float(rng.uniform(8.0, 15.0))
        humidity = float(np.clip(rng.uniform(25.0, 75.0), 5.0, 95.0))
        wind_speed = float(np.clip(rng.exponential(1.5), 0.0, 12.0))
        pressure = float(rng.normal(1000.0, 5.0))

        session_total_laps = int(rng.integers(45, 71))
        # True fuel process is PHYSICALLY correct: cumulative across the WHOLE
        # session, decreasing every lap regardless of pit stops (no refuelling).
        # This is deliberately NOT what the model's fixed fuel term sees -- the
        # model estimates fuel from `lap_in_stint`, which resets at every pit
        # stop. That mismatch is the real, unresolved confound Section 1.2
        # discusses, reproduced here rather than defined away.
        true_burn_per_lap = cfg.fuel_burn_per_lap_kg
        start_fuel_kg = session_total_laps * true_burn_per_lap + float(rng.uniform(-4.0, 4.0))

        evo_amplitude = float(circuit_evo_amplitude[circuit_id])
        evo_tau = float(circuit_evo_tau[circuit_id])

        for driver_id in range(cfg.n_drivers_per_session):
            team_id = driver_id % N_TEAMS
            true_driver_base = team_pace[team_id] + driver_pace[driver_id] + float(rng.normal(0, 0.08))

            n_runs = int(rng.integers(cfg.n_runs_per_driver[0], cfg.n_runs_per_driver[1] + 1))
            # Partition this driver's laps across the session into n_runs stints.
            remaining = session_total_laps
            stint_lengths = []
            for i in range(n_runs):
                laps_left_after = n_runs - i - 1
                max_len = max(4, remaining - 4 * laps_left_after)
                length = int(rng.integers(4, min(max_len, cfg.max_age) + 1)) if max_len > 4 else 4
                stint_lengths.append(min(length, remaining))
                remaining -= stint_lengths[-1]
                if remaining <= 0:
                    break

            session_lap_cursor = 1
            for stint_id, stint_len in enumerate(stint_lengths):
                compound = cfg.compounds[rng.integers(0, len(cfg.compounds))]
                curve = curves[compound]

                for lap_in_stint in range(stint_len):
                    session_lap = session_lap_cursor
                    session_lap_cursor += 1

                    tyre_age = lap_in_stint  # 0 == fresh tyre, anchors g_tyre(0) == 0
                    true_tyre_effect = float(curve[min(tyre_age, cfg.max_age)])

                    true_fuel_kg = max(start_fuel_kg - true_burn_per_lap * session_lap, 0.0)
                    true_fuel_effect = cfg.fuel_time_per_kg_s * true_fuel_kg
                    # Model-visible (approximate) feature: resets every stint.
                    fuel_kg_rel = -cfg.fuel_burn_per_lap_kg * lap_in_stint

                    session_progress = min(session_lap / session_total_laps, 1.0)
                    true_evo_effect = -evo_amplitude * (1.0 - np.exp(-session_progress / evo_tau))

                    true_cond_effect = float(
                        _true_cond_effect(
                            np.array([track_temp]), np.array([humidity]), np.array([wind_speed])
                        )[0]
                    )

                    gap_ahead_s = float(np.clip(rng.exponential(4.0), 0.0, 10.0))
                    true_traffic_effect = float(_true_traffic_effect(np.array([gap_ahead_s]))[0])

                    # Asymmetric noise: small symmetric jitter, plus an occasional
                    # positive-only "mistake" spike. A driver loses time to an
                    # error; they do not gain time by an equivalent margin.
                    noise = float(rng.normal(0.0, 0.06))
                    if rng.random() < 0.05:
                        noise += float(rng.exponential(0.45))
                    true_noise = noise

                    lap_time = (
                        circuit_base_pace[circuit_id]
                        + true_driver_base
                        + true_fuel_effect
                        + true_tyre_effect
                        + true_evo_effect
                        + true_cond_effect
                        + true_traffic_effect
                        + true_noise
                    )

                    rows.append(
                        {
                            "event_id": event_id,
                            "season": season,
                            "circuit_id": circuit_id,
                            "driver_id": driver_id,
                            "team_id": team_id,
                            "stint_id": stint_id,
                            "session_lap": session_lap,
                            "lap_in_stint": lap_in_stint,
                            "tyre_age": tyre_age,
                            "compound": compound,
                            "is_fresh_tyre": True,
                            "fuel_kg_rel": fuel_kg_rel,
                            "session_progress": session_progress,
                            "track_temp": track_temp,
                            "air_temp": air_temp,
                            "wind_speed": wind_speed,
                            "humidity": humidity,
                            "pressure": pressure,
                            "gap_ahead_s": gap_ahead_s,
                            "lap_time": lap_time,
                            "true_tyre_effect": true_tyre_effect,
                            "true_fuel_effect": true_fuel_effect,
                            "true_evo_effect": true_evo_effect,
                            "true_cond_effect": true_cond_effect,
                            "true_traffic_effect": true_traffic_effect,
                            "true_driver_base": true_driver_base,
                            "true_circuit_base": circuit_base_pace[circuit_id],
                            "true_noise": true_noise,
                            "true_total": lap_time,
                        }
                    )

    df = pd.DataFrame(rows)
    if not return_truth:
        return df

    truth = {
        "curves": curves,  # dict[compound] -> (max_age+1,) ground-truth cumulative curve
        "circuit_evo_amplitude": circuit_evo_amplitude,
        "circuit_evo_tau": circuit_evo_tau,
        "circuit_base_pace": circuit_base_pace,
    }
    return df, truth


if __name__ == "__main__":
    dataset = generate_synthetic_dataset()
    print(f"rows: {len(dataset)}")
    print(f"events: {dataset['event_id'].nunique()}")
    print(f"drivers: {dataset['driver_id'].nunique()}")
    print(dataset.groupby("compound")["lap_time"].describe())
    Path("data/synthetic").mkdir(parents=True, exist_ok=True)
    dataset.to_parquet("data/synthetic/synthetic_laps.parquet")
    print("saved to data/synthetic/synthetic_laps.parquet")
