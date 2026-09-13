"""Semi-synthetic recovery test: real, as-occurred confounders from actual
2022-2023 races (`src/data/clean.py`) with known synthetic tyre wear injected
on top (`src/sim/semi_synthetic.py`). Stronger than `tests/test_recovery.py`'s
pure-simulator version -- it tests the architecture against the real
complexity of confounding rather than a hand-built approximation of it, while
still being gradable against a known answer (real races have no true-wear
label to check against on their own).

Same four checks as `test_recovery.py`, same acceptance thresholds from
`config/default.yaml` -- this is additive to, not a replacement for, the
pure-synthetic gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from src.data.clean import clean_corpus
from src.data.features import split_by_event
from src.evaluate import detect_cliff, recovered_slope
from src.models.dataset import Vocabs
from src.models.nam import NAM, NAMConfig
from src.sim.semi_synthetic import build_semi_synthetic_dataset
from src.sim.simulator import SimulatorConfig
from src.train import TrainConfig, fit, pick_device


@pytest.fixture(scope="module")
def raw_config() -> dict:
    return yaml.safe_load(open("config/default.yaml"))


@pytest.fixture(scope="module")
def sim_cfg(raw_config) -> SimulatorConfig:
    return SimulatorConfig.from_yaml()


@pytest.fixture(scope="module")
def real_covariates():
    """Real race-session covariates, cleaned once for the whole module --
    the expensive part (reading and filtering every session) shouldn't repeat
    per test."""
    table, cascades = clean_corpus_cached()
    assert len(table) > 1000, (
        f"only {len(table)} real laps survived cleaning -- too few for a "
        "meaningful semi-synthetic test; check data/raw/ has real race sessions"
    )
    return table


def clean_corpus_cached():
    from pathlib import Path

    return clean_corpus(Path("data/raw"))


def _fit_on(df, sim_cfg, raw_config, *, seed: int = 42):
    torch.manual_seed(seed)
    train_df, val_df = split_by_event(df, val_fraction=0.15, seed=seed)
    vocabs = Vocabs.fit(train_df)
    nam_cfg = NAMConfig(
        n_circuits=vocabs.circuit.size, n_entries=vocabs.entry.size,
        n_compounds=vocabs.compound.size, max_age=sim_cfg.max_age,
        fuel_time_per_kg_s=sim_cfg.fuel_time_per_kg_s,
    )
    model = NAM(nam_cfg)
    train_cfg = TrainConfig.from_yaml_dict(raw_config)
    history = fit(model, train_df, val_df, vocabs, train_cfg, device=pick_device(), verbose=False)
    return model, vocabs, val_df, history


@pytest.fixture(scope="module")
def fitted(real_covariates, sim_cfg, raw_config):
    df, truth = build_semi_synthetic_dataset(real_covariates, sim_cfg, seed=42)
    model, vocabs, val_df, history = _fit_on(df, sim_cfg, raw_config)
    return model, vocabs, truth, history


def test_recovers_known_slope_per_compound_on_real_covariates(fitted, sim_cfg, raw_config):
    model, vocabs, truth, history = fitted
    acceptance = raw_config["acceptance"]
    device = pick_device()

    max_diff = 0.0
    for compound in sim_cfg.compounds:
        if compound not in vocabs.compound.mapping:
            continue  # this compound didn't appear in the real training sessions
        compound_id = vocabs.compound.mapping[compound]
        circuit_id = next(iter(vocabs.circuit.mapping.values()))
        curve = model.degradation_curve(
            compound_id=compound_id, circuit_id=circuit_id, track_temp_c=35.0, device=device
        ).cpu().numpy()

        rec_slope = recovered_slope(curve)
        true_slope = sim_cfg.true_linear_slope_s_per_lap[compound]
        diff = abs(rec_slope - true_slope)
        max_diff = max(max_diff, diff)
        print(f"{compound}: recovered={rec_slope:.4f} true={true_slope:.4f} diff={diff:.4f}")

    print(f"max diff across compounds: {max_diff:.4f} (tolerance {acceptance['slope_tolerance_s_per_lap']})")
    assert max_diff <= acceptance["slope_tolerance_s_per_lap"], (
        f"worst-case slope error {max_diff:.4f} s/lap exceeds "
        f"{acceptance['slope_tolerance_s_per_lap']} s/lap tolerance (real covariates)"
    )


def test_recovers_cliff_age_per_compound_on_real_covariates(fitted, sim_cfg, raw_config):
    model, vocabs, truth, history = fitted
    acceptance = raw_config["acceptance"]
    device = pick_device()

    for compound in sim_cfg.compounds:
        if compound not in vocabs.compound.mapping:
            continue
        compound_id = vocabs.compound.mapping[compound]
        circuit_id = next(iter(vocabs.circuit.mapping.values()))
        curve = model.degradation_curve(
            compound_id=compound_id, circuit_id=circuit_id, track_temp_c=35.0, device=device
        ).cpu().numpy()

        detected = detect_cliff(curve)
        true_cliff = sim_cfg.true_cliff_age[compound]
        print(f"{compound}: detected_cliff={detected} true_cliff={true_cliff}")
        assert detected is not None, f"{compound}: no cliff detected (true cliff at {true_cliff})"
        diff = abs(detected - true_cliff)
        assert diff <= acceptance["cliff_tolerance_laps"], (
            f"{compound}: detected cliff at {detected}, true is {true_cliff} "
            f"(tolerance {acceptance['cliff_tolerance_laps']} laps, real covariates)"
        )


def test_recovers_track_evolution_shape_on_real_covariates(fitted, raw_config):
    model, vocabs, truth, history = fitted
    acceptance = raw_config["acceptance"]
    device = pick_device()

    grid = np.linspace(0.0, 1.0, 50)
    correlations = []
    with torch.no_grad():
        for raw_circuit_id in list(vocabs.circuit.mapping.keys())[:4]:
            dense = vocabs.circuit.mapping[raw_circuit_id]
            session_progress = torch.tensor(grid, dtype=torch.float32, device=device)
            circuit_ids = torch.full((50,), dense, dtype=torch.long, device=device)
            recovered = model.heads["evo"](
                session_progress=session_progress, circuit_id=circuit_ids
            ).cpu().numpy()

            amplitude = truth["circuit_evo_amplitude"][raw_circuit_id]
            tau = truth["circuit_evo_tau"][raw_circuit_id]
            true_evo = -amplitude * (1 - np.exp(-grid / tau))

            corr = float(np.corrcoef(recovered, true_evo)[0, 1])
            correlations.append(corr)
            print(f"circuit {raw_circuit_id}: evo correlation = {corr:.4f}")

    assert min(correlations) >= acceptance["evolution_correlation_min"], (
        f"worst-case evolution correlation {min(correlations):.3f} below "
        f"{acceptance['evolution_correlation_min']} (real covariates)"
    )


def test_negative_control_zero_degradation_on_real_covariates(real_covariates, sim_cfg, raw_config):
    """A model that finds degradation where there is none is worse than useless."""
    acceptance = raw_config["acceptance"]
    df, truth = build_semi_synthetic_dataset(real_covariates, sim_cfg, seed=7, zero_degradation=True)
    model, vocabs, val_df, history = _fit_on(df, sim_cfg, raw_config, seed=7)
    device = pick_device()

    max_slope = 0.0
    for compound in sim_cfg.compounds:
        if compound not in vocabs.compound.mapping:
            continue
        compound_id = vocabs.compound.mapping[compound]
        circuit_id = next(iter(vocabs.circuit.mapping.values()))
        curve = model.degradation_curve(
            compound_id=compound_id, circuit_id=circuit_id, track_temp_c=35.0, device=device
        ).cpu().numpy()
        slope = recovered_slope(curve)
        max_slope = max(max_slope, slope)
        print(f"{compound} (zero-degradation truth): recovered slope = {slope:.4f}")

    assert max_slope <= acceptance["negative_control_slope_max_s_per_lap"], (
        f"model found {max_slope:.4f} s/lap of degradation where the truth is zero "
        f"(limit {acceptance['negative_control_slope_max_s_per_lap']}, real covariates)"
    )
