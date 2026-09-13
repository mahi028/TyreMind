"""Estimate degradation from sector times, where fuel and tyre stop being collinear.

exp18 proved that within a stint, tyre age and laps-of-fuel-burned advance
together exactly. Writing `a = a0 + f`:

    y = c + beta*a - phi*f  =  (c + beta*a0) + (beta - phi)*f

The run intercept absorbs `beta*a0`, and only the difference `beta - phi` is
recoverable. No amount of data fixes that; it is algebra. TyreMind resolves it
with an informative physical prior on `phi`, and exp18 measured what that costs:
only 6.0% of the separation comes from data.

**That is true of one observation per lap. Timing gives three.**

Fuel mass costs lap time mostly where the car accelerates and brakes. Tyre
degradation costs it mostly where the car is grip-limited. The two effects load
onto the three sectors differently, so per sector `k`:

    slope_k  =  w_k^tyre * beta  -  w_k^fuel * phi

Three equations, two unknowns. exp26 measured the two loading directions at a
median of 103 degrees apart across 40 races, and found the resulting system well
posed in 40 of 40 where the whole-lap system was singular in 40 of 40.

The remaining problem is that `w^tyre` is not known a priori, and estimating it
from the same slopes it is meant to explain would be circular. It is resolved by
pooling: **the loading is a property of the circuit and is shared across every
stint there, while the degradation rate varies from stint to stint.** With J
stints there are 3J observations and J + 4 unknowns, so for J of any size the
system is over-determined. That is the same identification trick the whole-lap
model already uses with run stagger, applied to sectors instead of to laps.

Fitted by alternating least squares, which for this bilinear structure is exact
in each half-step and converges quickly from a time-share start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Physical fuel coefficient, s/lap, matching the rest of the repository.
FUEL_SLOPE_S_PER_LAP = 0.081

SECTORS = (1, 2, 3)
SECTOR_COLUMNS = tuple(f"sector_{s}" for s in SECTORS)


@dataclass(frozen=True)
class SectorFit:
    """What the sector model recovered.

    Attributes:
        rate: Degradation rate per stint key, s/lap. This is `beta` -- the real
            tyre effect, not the `beta - phi` difference a whole-lap fit returns.
        rate_sd: Standard error per stint key.
        fuel_coefficient: Estimated `phi`, s/lap. **Estimated, not assumed.** The
            whole-lap model has to be told this number; the sector model can read
            it off the data, which is the entire point.
        fuel_sd: Standard error on `phi`.
        tyre_loading: Sector weights for the tyre effect, summing to 1.
        fuel_loading: Sector weights for the fuel effect, summing to 1.
        separation_deg: Angle between the two loading vectors. At 0 degrees they
            are parallel, the system collapses back to singular, and the estimate
            is no better than a whole-lap fit. Report it alongside every rate.
        n_stints: Stints pooled.
        converged: Whether alternating least squares reached tolerance.
    """

    rate: dict[str, float]
    rate_sd: dict[str, float]
    fuel_coefficient: float
    fuel_sd: float
    tyre_loading: np.ndarray
    fuel_loading: np.ndarray
    separation_deg: float
    n_stints: int
    converged: bool


def stint_slopes(lap_table: pd.DataFrame) -> pd.DataFrame:
    """Per-stint, per-sector slope against tyre age.

    The slope is `beta_k - phi_k` for that sector -- the confounded quantity. The
    fit's job is to take the three confounded numbers and return the two clean
    ones.
    """
    required = {"driver", "run_id", "tyre_age", *SECTOR_COLUMNS}
    missing = required - set(lap_table.columns)
    if missing:
        raise ValueError(f"sector model needs columns {sorted(missing)}")

    rows = []
    for (driver, run_id), stint in lap_table.groupby(["driver", "run_id"]):
        stint = stint.sort_values("tyre_age")
        age = stint["tyre_age"].to_numpy(dtype=float)
        if len(stint) < 6 or np.ptp(age) < 4:
            continue
        values = {}
        usable = True
        for index, column in enumerate(SECTOR_COLUMNS, start=1):
            y = stint[column].to_numpy(dtype=float)
            if not np.isfinite(y).all():
                usable = False
                break
            values[f"slope_{index}"] = float(np.polyfit(age, y, 1)[0])
            values[f"mean_{index}"] = float(y.mean())
        if not usable:
            continue
        rows.append({
            "key": f"{driver}|{run_id}",
            "driver": str(driver),
            "compound": str(stint["compound"].iloc[0]) if "compound" in stint else "UNKNOWN",
            "n_laps": int(len(stint)),
            **values,
        })
    return pd.DataFrame(rows)


def fit_sector_model(
    lap_table: pd.DataFrame,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-9,
    fuel_prior: float | None = FUEL_SLOPE_S_PER_LAP,
    fuel_prior_sd: float = 0.016,
) -> SectorFit:
    """Recover `beta` and `phi` separately by pooling sector slopes across stints.

    Args:
        lap_table: Laps carrying `sector_1..3` alongside the usual columns.
        max_iterations: Alternating least squares cap.
        tolerance: Convergence tolerance on the loading vector.
        fuel_prior: Physical prior on the fuel coefficient. Pass `None` to
            estimate it purely from data -- which is the experiment worth running,
            because a sector model that needs the prior as badly as the whole-lap
            model has not bought anything.
        fuel_prior_sd: Width of that prior.

    Returns:
        A `SectorFit`.

    Raises:
        ValueError: If fewer than two usable stints remain, which is the point
            below which the pooled loading is not identified.
    """
    slopes = stint_slopes(lap_table)
    if len(slopes) < 2:
        raise ValueError(f"need at least 2 usable stints, found {len(slopes)}")

    observed = slopes[[f"slope_{s}" for s in SECTORS]].to_numpy(dtype=float)
    means = slopes[[f"mean_{s}" for s in SECTORS]].to_numpy(dtype=float)

    # Fuel loads by where time is spent, to first order. This is the anchor that
    # makes the system identified: it is fixed, so the tyre loading is free to be
    # whatever the data says without the two collapsing onto each other.
    fuel_loading = means.mean(axis=0)
    fuel_loading = fuel_loading / fuel_loading.sum()

    # Start the tyre loading at time share too. If the data has nothing to say,
    # it stays there, the vectors are parallel, and `separation_deg` reports zero
    # rather than the fit quietly pretending to have identified something.
    tyre_loading = fuel_loading.copy()
    beta = observed.mean(axis=1) / max(tyre_loading.sum(), 1e-9)
    phi = float(fuel_prior) if fuel_prior is not None else 0.0
    converged = False

    for _ in range(max_iterations):
        previous = tyre_loading.copy()

        # Step 1: loadings fixed, solve for the per-stint rates and phi.
        # Stack every stint's three sector equations into one system.
        n = len(slopes)
        design = np.zeros((n * 3, n + 1))
        target = observed.reshape(-1)
        for j in range(n):
            design[j * 3:(j + 1) * 3, j] = tyre_loading
            design[j * 3:(j + 1) * 3, n] = -fuel_loading
        if fuel_prior is not None:
            # Prior as an extra observation, so its weight is explicit and its
            # influence shows up in the standard errors rather than being hidden.
            prior_row = np.zeros((1, n + 1))
            prior_row[0, n] = 1.0 / fuel_prior_sd
            design = np.vstack([design, prior_row])
            target = np.concatenate([target, [fuel_prior / fuel_prior_sd]])

        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        beta, phi = solution[:n], float(solution[n])

        # Step 2: rates fixed, solve for the tyre loading.
        # slope_jk + phi*fuel_k = beta_j * w_k, one independent equation per sector.
        adjusted = observed + phi * fuel_loading[None, :]
        denominator = float(beta @ beta)
        if denominator < 1e-18:
            break
        tyre_loading = (beta @ adjusted) / denominator
        total = tyre_loading.sum()
        if abs(total) < 1e-12:
            break
        # Normalise to sum to one, folding the scale into beta so the split
        # between "how fast" and "where" stays interpretable.
        tyre_loading = tyre_loading / total
        beta = beta * total

        if np.max(np.abs(tyre_loading - previous)) < tolerance:
            converged = True
            break

    # Residual standard errors from the final linear system.
    n = len(slopes)
    design = np.zeros((n * 3, n + 1))
    for j in range(n):
        design[j * 3:(j + 1) * 3, j] = tyre_loading
        design[j * 3:(j + 1) * 3, n] = -fuel_loading
    residual = observed.reshape(-1) - design @ np.concatenate([beta, [phi]])
    dof = max(n * 3 - (n + 1), 1)
    sigma2 = float(residual @ residual) / dof
    try:
        covariance = sigma2 * np.linalg.pinv(design.T @ design)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    except np.linalg.LinAlgError:
        standard_errors = np.full(n + 1, np.nan)

    cosine = float(tyre_loading @ fuel_loading /
                   (np.linalg.norm(tyre_loading) * np.linalg.norm(fuel_loading) + 1e-18))
    separation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    keys = slopes["key"].tolist()
    return SectorFit(
        rate=dict(zip(keys, beta.tolist())),
        rate_sd=dict(zip(keys, standard_errors[:n].tolist())),
        fuel_coefficient=phi,
        fuel_sd=float(standard_errors[n]),
        tyre_loading=tyre_loading,
        fuel_loading=fuel_loading,
        separation_deg=separation,
        n_stints=n,
        converged=converged,
    )


def compound_rates_from_fit(fit: SectorFit, lap_table: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Average the per-stint rates into a rate per compound.

    Weighted by nothing: every stint counts once. A length weighting would let a
    single long stint speak for a compound, and the point of pooling is that many
    stints disagree slightly and the disagreement is the standard error.
    """
    slopes = stint_slopes(lap_table)
    if slopes.empty:
        return {}
    slopes["rate"] = slopes["key"].map(fit.rate)

    out = {}
    for compound, block in slopes.groupby("compound"):
        values = block["rate"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        sd = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
        out[str(compound)] = (float(values.mean()), sd)
    return out
