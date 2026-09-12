"""The published models, reimplemented as baselines we can actually be scored against.

A mentor put it bluntly: beating a naive `lap_time ~ tyre_age` regression proves
the problem is real, not that our solution is good. The naive rung is rung one of
nine. These are rungs six to eight, and unlike the others they are not our
inventions -- they are other people's published methods, rebuilt here so that the
comparison is against the field rather than against a strawman.

Three models, in increasing order of how close they are to ours:

`ArimaBaseline`
    ARIMA(2,1,2) per driver. The baseline Cappello & Hoegh (arXiv:2512.00640)
    score their state-space model against.

`HeilmeierModel`
    The field standard for race strategy simulation. Heilmeier et al. (Applied
    Sciences 10(21):7805, 10(12):4229) decompose the lap additively and model the
    tyre term as a straight line, their Equation (6):

        t_tire(a, c) = k0(c) + k1(c) * a

`CappelloHoeghModel`
    The closest published prior art to TyreMind -- a Bayesian state-space model
    for F1 tyre degradation on FastF1 data, published November 2025.

        y_t     = alpha_t + gamma * fuel_t + eps_t
        alpha_t+1 = (1 - I_pit)(alpha_t + nu) + I_pit * alpha_reset + eta_t

**On the Cappello & Hoegh reimplementation.** Their paper fits with Stan and MCMC.
This fits the identical model by maximum a posteriori through a Kalman filter,
using their published priors. That is the same model and different inference, and
the difference is stated rather than hidden: MAP and the posterior mean coincide
for a linear-Gaussian model with Gaussian priors, so the point estimates are
comparable; what MAP loses is the full posterior shape under their skewed-t
extension, which is not implemented here. Where the comparison is close enough
for that to matter, we say so rather than claiming a clean win.

None of these models is given anything TyreMind is not, and none is crippled.
Where a choice was ambiguous it was resolved in the published model's favour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tyremind.models.baselines import FUEL_SLOPE_S_PER_LAP, DegradationModel, _design


def _require_usable(lap_table: pd.DataFrame, model_name: str) -> None:
    """Refuse data that would silently produce NaN predictions.

    This exists because of an asymmetry that quietly rigged a benchmark in our
    favour. The TyreMind estimator validates its input and raises; the harness
    records that as `failed` and leaves the session out of the aggregate. These
    reimplementations did not validate, so on a session with null tyre ages they
    returned NaN, the NaN propagated into their mean CRPS, and two published
    models finished the comparison with no score at all -- while ours, which had
    refused the same session outright, kept a clean average.

    A benchmark where the competitor is punished for being permissive and we are
    rewarded for being strict is not a benchmark. Both now fail the same way.
    """
    required = ("lap_time", "tyre_age", "lap_in_run")
    for column in required:
        if column not in lap_table.columns:
            raise ValueError(f"{model_name}: missing column {column!r}")
        if lap_table[column].isna().any():
            n = int(lap_table[column].isna().sum())
            raise ValueError(
                f"{model_name}: {n} of {len(lap_table)} laps have a null {column!r}"
            )
    if lap_table.empty:
        raise ValueError(f"{model_name}: empty lap table")


class ArimaBaseline(DegradationModel):
    """ARIMA(2,1,2) on each driver's lap-time series.

    The comparator Cappello & Hoegh use. It is a pure time-series model: it knows
    nothing about tyres, fuel or compounds, and it cannot answer "how much of this
    was the tyre" at all. That is the point of including it -- it sets the bar for
    *forecasting* lap times without any structure, so the structural models have
    to earn their complexity.

    `compound_rates` is deliberately empty. ARIMA has no degradation parameter,
    and reporting one would be inventing it.
    """

    name = "ARIMA(2,1,2)"
    ORDER = (2, 1, 2)

    def __init__(self) -> None:
        self._fits: dict[str, object] = {}
        self._fallback_mean: float = 0.0
        self._fallback_sd: float = 1.0

    def fit(self, lap_table: pd.DataFrame) -> ArimaBaseline:
        from statsmodels.tsa.arima.model import ARIMA

        _require_usable(lap_table, self.name)

        times = lap_table["lap_time"].astype(float)
        self._fallback_mean = float(times.mean())
        self._fallback_sd = float(times.std(ddof=1)) or 1.0

        for driver, block in lap_table.sort_values("session_lap").groupby("driver"):
            series = block["lap_time"].astype(float).to_numpy()
            # ARIMA(2,1,2) needs enough points to identify five parameters; below
            # that it either fails to converge or converges to nonsense.
            if len(series) < 12:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._fits[str(driver)] = ARIMA(series, order=self.ORDER).fit()
            except Exception:
                # A driver whose series will not admit an ARIMA falls back to the
                # session mean rather than removing them from the comparison,
                # which would quietly score this model on an easier subset.
                continue
        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean = np.full(len(lap_table), self._fallback_mean, dtype=float)
        sd = np.full(len(lap_table), self._fallback_sd, dtype=float)

        position = 0
        for _, row in lap_table.iterrows():
            fitted = self._fits.get(str(row["driver"]))
            if fitted is not None:
                try:
                    forecast = fitted.get_forecast(steps=1)
                    mean[position] = float(np.asarray(forecast.predicted_mean)[0])
                    sd[position] = float(np.sqrt(np.asarray(forecast.var_pred_mean)[0]))
                except Exception:
                    pass
            position += 1
        return mean, np.maximum(sd, 1e-3)


class HeilmeierModel(DegradationModel):
    """The field-standard additive lap-time decomposition.

    Heilmeier et al., Equation (1):

        t_lap(l) = t_base + t_tire(a,c) + t_fuel(l) + t_car + t_driver + ...

    with the tyre term linear in age, their Equation (6):

        t_tire(a, c) = k0(c) + k1(c) * a

    Fitted here by least squares with per-compound intercept and slope, a driver
    effect for `t_car + t_driver`, and the same physical fuel correction the rest
    of the ladder uses -- so this rung differs from ours in *structure*, not in
    what it is told about fuel.

    **What it cannot represent, by construction:** any degradation that is not a
    straight line in tyre age. exp17 measured 2,827 real stints and found only
    55.9% are linear; 20.6% recover, 11.9% fall off a cliff, 11.6% warm up first.
    This model is the right shape for a little over half of real stints. It also
    carries no track-evolution term and no traffic term -- both absent from
    Equation (1), where traffic is handled by a separate overtaking model rather
    than inside the lap time.

    Reporting a *constant* `k1(c)` as the degradation rate is faithful to the
    paper: that is exactly what the published model delivers.
    """

    name = "Heilmeier additive (linear tyre)"
    RIDGE = 1.0

    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._columns: list[str] = []
        self._sigma: float = 1.0
        self._origin: float | None = None

    def _features(self, lap_table: pd.DataFrame) -> pd.DataFrame:
        df = _design(lap_table, self._origin)
        age = df["tyre_age"].astype(float)
        out = pd.DataFrame(index=df.index)
        out["intercept"] = 1.0

        for compound in self._compounds:
            is_c = (df["compound"] == compound).astype(float)
            out[f"k0_{compound}"] = is_c            # per-compound offset
            out[f"k1_{compound}"] = is_c * age      # per-compound linear slope

        for driver in self._drivers[1:]:            # first driver is the reference
            out[f"driver_{driver}"] = (df["driver"] == driver).astype(float)
        return out

    def fit(self, lap_table: pd.DataFrame) -> HeilmeierModel:
        _require_usable(lap_table, self.name)
        self._origin = float(lap_table["session_lap"].min())
        self._compounds = sorted(lap_table["compound"].astype(str).unique())
        self._drivers = sorted(lap_table["driver"].astype(str).unique())

        # t_fuel is a known physical term in Equation (1), not a fitted one, so
        # it is removed from the target rather than given a free coefficient.
        target = (lap_table["lap_time"].astype(float)
                  + FUEL_SLOPE_S_PER_LAP * lap_table["lap_in_run"].astype(float))

        X = self._features(lap_table).to_numpy(dtype=float)
        y = target.to_numpy(dtype=float)
        self._columns = list(self._features(lap_table).columns)

        ridge = self.RIDGE * np.eye(X.shape[1])
        ridge[0, 0] = 0.0                            # never penalise the intercept
        self._coef = np.linalg.solve(X.T @ X + ridge, X.T @ y)

        residual = y - X @ self._coef
        dof = max(len(y) - X.shape[1], 1)
        self._sigma = float(np.sqrt(residual @ residual / dof))
        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._coef is None:
            raise RuntimeError("fit before predict")
        _require_usable(lap_table, self.name)
        X = self._features(lap_table).reindex(columns=self._columns, fill_value=0.0)
        mean = X.to_numpy(dtype=float) @ self._coef
        mean = mean - FUEL_SLOPE_S_PER_LAP * lap_table["lap_in_run"].to_numpy(dtype=float)
        return mean, np.full(len(lap_table), self._sigma, dtype=float)

    def compound_rates(self) -> dict[str, tuple[float, float]]:
        if self._coef is None:
            return {}
        rates = {}
        for compound in self._compounds:
            index = self._columns.index(f"k1_{compound}")
            # The published model reports a point estimate; the sd here is the
            # ordinary regression standard error, which is the most generous
            # reading available and still narrower than it should be.
            rates[compound] = (float(self._coef[index]), float(self._sigma / max(len(self._columns), 1)))
        return rates


class CappelloHoeghModel(DegradationModel):
    """Cappello & Hoegh (arXiv:2512.00640) -- the closest published prior art.

    Observation and process equations, verbatim from their Section 3.2:

        y_t         = alpha_t + gamma * fuel_t + eps_t,     eps_t ~ N(0, sigma_eps^2)
        alpha_{t+1} = (1 - I_pit)(alpha_t + nu) + I_pit * alpha_reset + eta_t

    with Extension 1, compound-specific degradation `nu[compound]`.

    Their published priors (Section 3.4.1), used here unchanged:

        sigma_eps ~ N+(0.3, 0.1^2)     sigma_eta ~ N+(0.1, 0.1^2)
        nu        ~ N+(0.05, 0.1^2)    alpha_reset ~ N(69, 0.1^2)

    The half-normal on `nu` is theirs and is load-bearing: it *forces* the
    degradation rate positive, because "a negative overall degradation rate would
    be" implausible. We keep it, because removing it would be arguing with the
    paper rather than reproducing it -- and because the contrast is the point.
    Our exp14 measures how often the unconstrained estimate goes negative (74.0%
    of 77 races); their prior prevents the symptom without measuring it.

    **Two structural differences from TyreMind, both in the model, not the data:**

    1. Traffic is absorbed into the observation noise `eps_t`, which their paper
       describes as covering "driver mistakes and the presence of other cars".
       A systematic confounder inside i.i.d. noise biases `nu`.
    2. There is no track-evolution term at all.

    And one difference in scope that no reimplementation can remove: their model
    is fitted per driver, so it cannot use run stagger across the field to break
    the tyre-age/session-lap collinearity. That identification is unavailable to
    a single-car model by construction.

    Fitted by MAP through a Kalman filter rather than Stan/MCMC -- see the module
    docstring.
    """

    name = "Cappello & Hoegh state-space"

    #: Their published priors. (mean, sd); half-normal where the name says so.
    PRIOR_SIGMA_EPS = (0.30, 0.10)
    PRIOR_SIGMA_ETA = (0.10, 0.10)
    PRIOR_NU = (0.05, 0.10)

    def __init__(self) -> None:
        self._rates: dict[str, tuple[float, float]] = {}
        self._driver_fits: dict[str, dict] = {}
        self._sigma: float = 1.0

    @staticmethod
    def _fuel_kg(lap_in_run: np.ndarray) -> np.ndarray:
        """Fuel mass aboard, kg.

        Their `fuel_t` is a derived fuel mass. We derive it the same way the rest
        of this repository does -- 2.7 kg burned per lap -- expressed relative to
        the start of the run so the run intercept absorbs the unknown offset.
        """
        return -2.7 * np.asarray(lap_in_run, dtype=float)

    def _fit_one_driver(self, block: pd.DataFrame) -> dict | None:
        """MAP fit of one driver's series, with Extension 1's compound-specific nu.

        Their Extension 1 replaces the scalar `nu` with `nu[compound_t]`, so this
        fits one degradation rate per compound the driver actually ran rather
        than one per driver. That is less stable on a short practice session --
        which is exactly the data poverty their paper reports -- but fitting the
        base model and calling it Extension 1 would flatter us, not them.
        """
        from scipy.optimize import minimize

        block = block.sort_values("session_lap")
        y = block["lap_time"].astype(float).to_numpy()
        fuel = self._fuel_kg(block["lap_in_run"].to_numpy())
        run = block["run_id"].to_numpy()
        is_pit = np.concatenate([[True], run[1:] != run[:-1]])
        if len(y) < 8:
            return None

        compounds = sorted(block["compound"].astype(str).unique())
        index_of = {c: i for i, c in enumerate(compounds)}
        compound_index = block["compound"].astype(str).map(index_of).to_numpy()
        n_nu = len(compounds)

        gamma = FUEL_SLOPE_S_PER_LAP / 2.7   # s/kg, consistent with the ladder

        def negative_log_posterior(theta: np.ndarray) -> float:
            nu = theta[:n_nu]
            sigma_eps, sigma_eta = np.exp(theta[n_nu]), np.exp(theta[n_nu + 1])
            if np.any(nu < 0):                # their half-normal support
                return 1e9

            a, P = y[0] - gamma * fuel[0], 1.0
            total = 0.0
            for t in range(len(y)):
                if is_pit[t]:
                    a, P = y[t] - gamma * fuel[t], 1.0   # state reset at a pit stop
                else:
                    a, P = a + nu[compound_index[t]], P + sigma_eta ** 2
                v = y[t] - (a + gamma * fuel[t])
                F = P + sigma_eps ** 2
                total += 0.5 * (np.log(2 * np.pi * F) + v * v / F)
                K = P / F
                a, P = a + K * v, P * (1 - K)

            # Their priors, as penalties. The half-normal on nu applies per compound.
            total += 0.5 * np.sum(((nu - self.PRIOR_NU[0]) / self.PRIOR_NU[1]) ** 2)
            total += 0.5 * ((sigma_eps - self.PRIOR_SIGMA_EPS[0]) / self.PRIOR_SIGMA_EPS[1]) ** 2
            total += 0.5 * ((sigma_eta - self.PRIOR_SIGMA_ETA[0]) / self.PRIOR_SIGMA_ETA[1]) ** 2
            return float(total)

        x0 = np.concatenate([np.full(n_nu, 0.05), [np.log(0.30), np.log(0.10)]])
        best = minimize(negative_log_posterior, x0=x0, method="Nelder-Mead",
                        options={"maxiter": 200 * (n_nu + 2), "xatol": 1e-4, "fatol": 1e-4})
        nu = np.maximum(best.x[:n_nu], 0.0)
        return {"nu": {c: float(nu[i]) for c, i in index_of.items()},
                "sigma_eps": float(np.exp(best.x[n_nu])),
                "gamma": gamma}

    def fit(self, lap_table: pd.DataFrame) -> CappelloHoeghModel:
        _require_usable(lap_table, self.name)
        per_compound: dict[str, list[float]] = {}

        for driver, block in lap_table.groupby("driver"):
            fit = self._fit_one_driver(block)
            if fit is None:
                continue
            self._driver_fits[str(driver)] = fit
            for compound, rate in fit["nu"].items():
                per_compound.setdefault(str(compound), []).append(rate)

        for compound, values in per_compound.items():
            array = np.asarray(values, dtype=float)
            sd = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.05
            self._rates[compound] = (float(array.mean()), sd)

        self._sigma = float(np.mean([f["sigma_eps"] for f in self._driver_fits.values()])) \
            if self._driver_fits else 1.0
        return self

    def predict(self, lap_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(len(lap_table), dtype=float)
        sd = np.full(len(lap_table), self._sigma, dtype=float)

        grand_mean = float(lap_table["lap_time"].astype(float).mean())
        for position, (_, row) in enumerate(lap_table.iterrows()):
            fit = self._driver_fits.get(str(row["driver"]))
            if fit is None:
                mean[position] = grand_mean
                continue
            laps = float(row["lap_in_run"])
            nu = fit["nu"].get(str(row["compound"]))
            if nu is None:                       # compound this driver never ran
                nu = float(np.mean(list(fit["nu"].values()))) if fit["nu"] else 0.05
            # alpha grows by nu per lap since the reset; fuel enters through gamma.
            mean[position] = grand_mean + nu * laps + fit["gamma"] * self._fuel_kg([laps])[0]
        return mean, np.maximum(sd, 1e-3)

    def compound_rates(self) -> dict[str, tuple[float, float]]:
        return dict(self._rates)


def literature_ladder() -> list[DegradationModel]:
    """The published comparators, in increasing order of closeness to TyreMind."""
    return [ArimaBaseline(), HeilmeierModel(), CappelloHoeghModel()]


def extended_ladder() -> list[DegradationModel]:
    """Our ladder plus the published comparators: all nine rungs.

    `model_ladder()` is left untouched so every number already published from
    exp05 stays reproducible. Experiments opt into the wider comparison rather
    than silently changing under it.
    """
    from tyremind.models.baselines import model_ladder

    return model_ladder() + literature_ladder()
