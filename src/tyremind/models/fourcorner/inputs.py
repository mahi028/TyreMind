"""Turning the telemetry corpus into the per-corner inputs the SDE consumes.

The four-corner model only earns its extra eight states if the corners are driven
by something measured. This module is where that happens, and it is deliberately
the only place in the package that touches a dataframe.

**What the energy column is, precisely.** `energy_mj_{fl,fr,rl,rr}` is not
physical megajoules. `physics.dynamics.frictional_power_proxy` builds it as
`F_z * (a_demand / g) * v` integrated over the lap, and its own docstring is
explicit that absolute values in watts are not meaningful and are never reported
as such. The slip factor is demanded acceleration in g, a surrogate, because
public telemetry carries no wheel-speed sensor.

An earlier version of this module inverted that column into a slip velocity in
m/s and handed it to the wear law. That was wrong: it dressed an arbitrary scale
as a physical quantity, and the plausibility bounds it then applied were
rejecting laps against a unit that does not exist. The proxy is now passed
through as a per-corner power, `energy / duration`, and the fitted wear
coefficient absorbs its unknown scale.

What survives the proxy is the **corner-to-corner ratio**, which is the only
thing four separate wear states can be identified from, and exp06 already showed
that ratio is real enough to recover circuit rotation direction on 7 of 8
circuits.

**When telemetry is missing** the model falls back to a symmetric split of an
energy estimate from lateral g. That fallback is recorded per lap in
`LapInputBundle.n_synthesised`, because a four-corner model fed symmetric inputs
is a one-corner model with nine redundant states, and an experiment that did not
notice would be measuring nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dynamics import LapInput
from .state import CORNERS

#: Telemetry column names, in CORNERS order.
ENERGY_COLUMNS = ("energy_mj_fl", "energy_mj_fr", "energy_mj_rl", "energy_mj_rr")
LOAD_COLUMNS = ("mean_load_n_fl", "mean_load_n_fr", "mean_load_n_rl", "mean_load_n_rr")

#: Floor on corner load, newtons. A corner genuinely unloads under heavy
#: transfer and a zero there is a divide-by-zero in the asymmetry report.
MIN_LOAD_N = 400.0

#: Median per-corner power proxy across the telemetry corpus, used only to keep
#: the synthesised fallback on the same scale as the measured laps. Measured at
#: 1.81e5 from `data/telemetry`; `state.REFERENCE_CORNER_POWER_W` is the same
#: quantity in its role as the wear law normaliser.
TYPICAL_CORNER_POWER_W = 1.81e5


@dataclass
class LapInputBundle:
    """Per-lap inputs for one driver's stint, plus how they were obtained.

    Attributes:
        inputs: The lap inputs, in order.
        observations: Observed lap times, same length and order.
        n_measured: Laps whose corner split came from telemetry.
        n_synthesised: Laps whose corner split was assumed symmetric.
        mean_left_right_ratio: Mean of (left energy) / (total energy) across the
            measured laps. 0.5 is a perfectly symmetric circuit, and a value that
            close means the four corner states cannot be separated on this data
            however good the filter is.
    """

    inputs: list[LapInput]
    observations: np.ndarray
    n_measured: int
    n_synthesised: int
    mean_left_right_ratio: float

    @property
    def measured_fraction(self) -> float:
        total = self.n_measured + self.n_synthesised
        return self.n_measured / total if total else 0.0


def _power_from_energy(energy_mj: np.ndarray, duration_s: float) -> np.ndarray | None:
    """Per-corner power proxy over the lap, or None if the row is unusable.

    A straight division. The only judgement is rejecting non-positive or
    non-finite totals, because a lap the telemetry could not reduce is better
    counted as synthesised than passed through as a zero, which would read to the
    filter as a lap the car spent stationary.
    """
    energy = np.asarray(energy_mj, dtype=float)
    if not np.isfinite(energy).all() or energy.sum() <= 0 or (energy < 0).any():
        return None
    return energy * 1e6 / duration_s


def _synthesise(lap: pd.Series, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """A symmetric corner split, for laps with no usable telemetry.

    Uses lateral g where it exists to scale the overall effort, and splits it
    evenly. The evenness is the point: this lap contributes nothing to separating
    the corners and is counted so the experiment can say how many such laps it
    was fed.
    """
    lateral = float(lap.get("mean_abs_lateral_g") or 1.2)
    load = np.full(4, 3_000.0 + 900.0 * lateral, dtype=float)
    power = np.full(4, TYPICAL_CORNER_POWER_W * (lateral / 1.2), dtype=float)
    return load, power


def build_inputs(
    lap_table: pd.DataFrame,
    telemetry: pd.DataFrame | None,
    driver: str,
    run_id: object,
) -> LapInputBundle:
    """Assemble one driver's stint into filter inputs.

    Args:
        lap_table: Cleaned lap table, as `data.corpus.read_lap_table` returns.
        telemetry: Telemetry table for the same session, or None.
        driver: Driver code.
        run_id: The run to extract.

    Returns:
        A `LapInputBundle`.

    Raises:
        ValueError: If the stint has no usable laps. Raising is deliberate --
            every model on the ladder raises on unusable input so that none of
            them is quietly flattered by returning NaN where others fail. That
            fairness rule is enforced by `tests/unit/test_model_fairness.py` and
            this model is held to it too.
    """
    block = lap_table[(lap_table["driver"] == driver) & (lap_table["run_id"] == run_id)]
    block = block.sort_values("session_lap")
    if block.empty:
        raise ValueError(f"no laps for {driver} run {run_id}")

    merged = block
    if telemetry is not None and not telemetry.empty:
        merged = block.merge(telemetry, on=["driver", "session_lap"], how="left")

    inputs: list[LapInput] = []
    observations: list[float] = []
    n_measured = n_synthesised = 0
    left_ratios: list[float] = []

    for _, lap in merged.iterrows():
        duration = lap.get("lap_time")
        if duration is None or not np.isfinite(duration) or duration <= 0:
            continue

        load = power = None
        if all(c in merged.columns for c in ENERGY_COLUMNS + LOAD_COLUMNS):
            energy = np.array([lap.get(c) for c in ENERGY_COLUMNS], dtype=float)
            loads = np.array([lap.get(c) for c in LOAD_COLUMNS], dtype=float)
            if np.isfinite(loads).all():
                candidate = _power_from_energy(energy, float(duration))
                if candidate is not None:
                    load, power = np.maximum(loads, MIN_LOAD_N), candidate
                    left_ratios.append(float((energy[0] + energy[2]) / energy.sum()))
                    n_measured += 1

        if load is None:
            load, power = _synthesise(lap, float(duration))
            n_synthesised += 1

        speed_kmh = lap.get("mean_speed_kmh")
        airspeed = float(speed_kmh) / 3.6 if speed_kmh and np.isfinite(speed_kmh) else 55.0

        inputs.append(
            LapInput(
                duration_s=float(duration),
                load_n=load,
                power_proxy_w=power,
                airspeed_ms=airspeed,
                laps_completed=float(lap.get("lap_in_run") or 0.0),
                traffic_index=float(lap.get("traffic_index") or 0.0),
                session_lap=int(lap["session_lap"]),
            )
        )
        observations.append(float(duration))

    if not inputs:
        raise ValueError(f"{driver} run {run_id} has no usable laps")

    return LapInputBundle(
        inputs=inputs,
        observations=np.asarray(observations, dtype=float),
        n_measured=n_measured,
        n_synthesised=n_synthesised,
        mean_left_right_ratio=float(np.mean(left_ratios)) if left_ratios else 0.5,
    )


def asymmetry_report(bundle: LapInputBundle) -> dict[str, float]:
    """How much corner-to-corner separation this stint actually offers.

    The pre-registration predicts the four corner states will not be separately
    identifiable and that only a left/right aggregate will survive. This is the
    measurement that prediction is scored against, and it is computed from the
    inputs alone -- before any filtering -- so it cannot be contaminated by how
    well or badly the filter happens to run.
    """
    power = np.array([u.power_proxy_w for u in bundle.inputs], dtype=float)
    mean_power = power.mean(axis=0)
    total = mean_power.sum()
    shares = mean_power / total if total > 0 else np.full(4, 0.25)

    left = shares[0] + shares[2]
    front = shares[0] + shares[1]
    return {
        "n_laps": float(len(bundle.inputs)),
        "measured_fraction": bundle.measured_fraction,
        **{f"power_share_{name}": float(shares[i]) for i, name in enumerate(CORNERS)},
        "left_share": float(left),
        "front_share": float(front),
        # Distance from a perfectly symmetric car. Zero means the four corner
        # states are exchangeable and the model is over-parameterised by three.
        "asymmetry": float(np.abs(shares - 0.25).sum()),
    }
