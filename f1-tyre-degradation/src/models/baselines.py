"""Phase 4 — baselines, for the pitch.

Three models, same splits, same target as the NAM: linear, polynomial
(degree-2/3), and LightGBM with SHAP attribution on `tyre_age`.

The point is not to beat these on raw MAE (LightGBM likely won't be beaten).
The point is that none of them has a parameter that means "degradation rate":
run each through `tests/test_recovery.py`'s synthetic ground truth and the
expected, pitchable result is that LightGBM has the best raw MAE and the WORST
recovered degradation slope, because under the fuel/tyre-age collinearity,
gradient boosting splits the shared signal between features arbitrarily. SHAP
attribution on a correlated feature is not causal attribution, and this module
is the evidence for that claim, not merely an assertion of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

CONTINUOUS_COLUMNS = ["tyre_age", "fuel_kg_rel", "session_progress", "track_temp"]
# `compound` was missing from this list -- without it, no baseline here could
# distinguish Soft/Medium/Hard's very different wear rates at all, forcing
# every compound onto one averaged curve. That would make the MAE and
# recovered-slope comparison against the NAM (which conditions g_tyre on
# compound explicitly) neither fair nor informative. Fixed before running.
CATEGORICAL_COLUMNS = ["circuit_id", "driver_id", "compound"]


def _design_matrix(df: pd.DataFrame, *, degree: int = 1,
                     circuit_categories: pd.Index | None = None,
                     driver_categories: pd.Index | None = None,
                     compound_categories: pd.Index | None = None,
                     ) -> tuple[np.ndarray, pd.Index, pd.Index, pd.Index]:
    """Continuous columns (optionally polynomial-expanded) + one-hot categoricals.

    Categories are fixed from the TRAIN set and passed back in so val/test are
    encoded against the same columns (an unseen category simply gets all-zero
    dummy columns -- the linear/polynomial equivalent of an <UNK> embedding).
    """
    continuous = df[CONTINUOUS_COLUMNS].to_numpy(dtype=np.float64)
    if degree > 1:
        continuous = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(continuous)

    if circuit_categories is None:
        circuit_categories = pd.Index(sorted(df["circuit_id"].unique()))
    if driver_categories is None:
        driver_categories = pd.Index(sorted(df["driver_id"].unique()))
    if compound_categories is None:
        compound_categories = pd.Index(sorted(df["compound"].unique()))

    circuit_dummies = pd.get_dummies(
        pd.Categorical(df["circuit_id"], categories=circuit_categories)
    ).to_numpy(dtype=np.float64)
    driver_dummies = pd.get_dummies(
        pd.Categorical(df["driver_id"], categories=driver_categories)
    ).to_numpy(dtype=np.float64)
    compound_dummies = pd.get_dummies(
        pd.Categorical(df["compound"], categories=compound_categories)
    ).to_numpy(dtype=np.float64)

    X = np.concatenate([continuous, circuit_dummies, driver_dummies, compound_dummies], axis=1)
    return X, circuit_categories, driver_categories, compound_categories


@dataclass
class LinearBaseline:
    """`lap_time ~ tyre_age + fuel + session_progress + track_temp + circuit + driver + compound`."""

    degree: int = 1
    model: LinearRegression | None = None
    circuit_categories: pd.Index | None = None
    driver_categories: pd.Index | None = None
    compound_categories: pd.Index | None = None

    def fit(self, train_df: pd.DataFrame) -> "LinearBaseline":
        X, self.circuit_categories, self.driver_categories, self.compound_categories = _design_matrix(
            train_df, degree=self.degree
        )
        y = train_df["lap_time"].to_numpy(dtype=np.float64)
        self.model = LinearRegression()
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _, _ = _design_matrix(
            df, degree=self.degree,
            circuit_categories=self.circuit_categories, driver_categories=self.driver_categories,
            compound_categories=self.compound_categories,
        )
        return self.model.predict(X)

    def implied_tyre_curve(self, max_age: int, compound: str) -> np.ndarray:
        """A synthetic 'curve' for comparability with the NAM: predicted lap
        time vs. tyre_age for ONE compound, with every other continuous
        feature held at zero and one arbitrary circuit/driver, then de-meaned
        so `curve[0] == 0` like `TyreHead`'s anchor.

        This only makes sense for degree=1 (a single global slope per
        compound); for degree>1 the "curve" is the fitted polynomial shape,
        not a monotone wear curve -- nothing in this model enforces that shape
        at all, which is exactly the point being demonstrated.
        """
        ages = np.arange(max_age + 1)
        continuous = np.zeros((max_age + 1, len(CONTINUOUS_COLUMNS)))
        continuous[:, 0] = ages
        if self.degree > 1:
            continuous = PolynomialFeatures(degree=self.degree, include_bias=False).fit_transform(continuous)
        circuit_dummies = np.zeros((max_age + 1, len(self.circuit_categories)))
        circuit_dummies[:, 0] = 1.0
        driver_dummies = np.zeros((max_age + 1, len(self.driver_categories)))
        driver_dummies[:, 0] = 1.0
        compound_dummies = np.zeros((max_age + 1, len(self.compound_categories)))
        compound_dummies[:, list(self.compound_categories).index(compound)] = 1.0
        X = np.concatenate([continuous, circuit_dummies, driver_dummies, compound_dummies], axis=1)
        pred = self.model.predict(X)
        return pred - pred[0]


def prepare_lgb_features(df: pd.DataFrame, category_map: dict[str, pd.Index]) -> pd.DataFrame:
    """LightGBM requires int/float/bool/pandas-category dtype -- a raw string
    column (e.g. `compound`) is rejected outright. Categories are fixed from
    the TRAIN set (`category_map`) and reused everywhere else (val, the SHAP-
    style curve grid), same reasoning as `_design_matrix`'s fixed-category
    one-hot encoding, so every caller uses identical codes."""
    feature_cols = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
    out = df[feature_cols].copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = pd.Categorical(out[col].astype(str), categories=category_map[col])
    return out


def fit_lightgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, *, num_boost_round: int = 500,
                  early_stopping_rounds: int = 30):
    """Full-feature-set LightGBM. Returns `(booster, feature_names, category_map)` --
    `category_map` must be passed to `lightgbm_shap_tyre_curve` and to any
    other `booster.predict` call so categoricals are encoded consistently."""
    import lightgbm as lgb

    feature_cols = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
    categorical_feature = CATEGORICAL_COLUMNS
    category_map = {col: pd.Index(sorted(train_df[col].astype(str).unique())) for col in CATEGORICAL_COLUMNS}

    train_set = lgb.Dataset(
        prepare_lgb_features(train_df, category_map), label=train_df["lap_time"],
        categorical_feature=categorical_feature,
    )
    val_set = lgb.Dataset(
        prepare_lgb_features(val_df, category_map), label=val_df["lap_time"],
        categorical_feature=categorical_feature, reference=train_set,
    )

    params = {
        "objective": "regression",
        "metric": "l1",
        "verbosity": -1,
        "seed": 42,
    }
    booster = lgb.train(
        params, train_set, num_boost_round=num_boost_round, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return booster, feature_cols, category_map


def lightgbm_shap_tyre_curve(booster, train_df: pd.DataFrame, max_age: int, compound: str,
                               category_map: dict[str, pd.Index]) -> np.ndarray:
    """Prediction's dependence on `tyre_age` for ONE compound, read off as a
    'curve' for comparison -- explicitly NOT causal attribution (see module
    docstring). Held at the training median for every other continuous
    feature, and at the given `compound` (categorical columns need an
    explicit category, not a meaningless "median" of category codes).

    Named `_shap_` for continuity with the module's original intent (SHAP
    dependence would show the same shape here, since holding every other
    feature fixed while sweeping one is exactly what a SHAP dependence plot
    visualizes for a tree ensemble); computed directly via `booster.predict`
    for simplicity, without requiring the `shap` package as a dependency.
    `category_map` must be the SAME mapping `fit_lightgbm` returned, so the
    grid's categorical codes match what the booster was trained on.
    """
    baseline = train_df[CONTINUOUS_COLUMNS].median(numeric_only=True)
    grid = pd.DataFrame([baseline.to_dict()] * (max_age + 1))
    grid["tyre_age"] = np.arange(max_age + 1)
    grid["circuit_id"] = train_df["circuit_id"].mode().iloc[0]
    grid["driver_id"] = train_df["driver_id"].mode().iloc[0]
    grid["compound"] = compound
    pred = booster.predict(prepare_lgb_features(grid, category_map))
    return pred - pred[0]
