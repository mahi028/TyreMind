"""Training loop, shared by the synthetic recovery test and real-data fitting.

AdamW, cosine LR schedule, batch 1024, early stopping on val Huber (plain, not
asymmetric -- the asymmetric weighting is a training-time choice, not the
criterion we judge convergence by). Traffic's zero-in-clean-air penalty is
added to the training objective; `CondHead` is centered once training finishes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.dataset import Vocabs, make_batch, make_target
from src.models.losses import asymmetric_huber_loss, huber_loss
from src.models.nam import NAM


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 300
    early_stop_patience: int = 15
    weight_decay: float = 1e-4
    huber_delta_s: float = 0.25
    asymmetric_ratio: float = 1.5
    traffic_zero_penalty_gap_s: float = 3.0
    traffic_zero_penalty_weight: float = 0.05
    evo_shrinkage_weight: float = 0.05
    tyre_smoothness_weight: float = 1.0
    tyre_l1_weight: float = 0.0
    tyre_warmup_epochs: int = 0
    seed: int = 42

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "TrainConfig":
        t, l, tr, ev, ty = raw["train"], raw["loss"], raw["traffic"], raw["evo"], raw["tyre"]
        return cls(
            lr=t["lr"], batch_size=t["batch_size"], max_epochs=t["max_epochs"],
            early_stop_patience=t["early_stop_patience"], weight_decay=t["weight_decay"],
            tyre_warmup_epochs=t["tyre_warmup_epochs"],
            huber_delta_s=l["huber_delta_s"], asymmetric_ratio=l["asymmetric_ratio"],
            traffic_zero_penalty_gap_s=tr["zero_penalty_gap_s"],
            traffic_zero_penalty_weight=tr["zero_penalty_weight"],
            evo_shrinkage_weight=ev["shrinkage_weight"],
            tyre_smoothness_weight=ty["smoothness_weight"],
            tyre_l1_weight=ty["l1_weight"],
            seed=raw["seed"],
        )


def fit(model: NAM, train_df: pd.DataFrame, val_df: pd.DataFrame, vocabs: Vocabs,
        cfg: TrainConfig, *, device: torch.device | None = None, verbose: bool = True) -> dict:
    """Fits `model` in place. Returns a history dict with per-epoch losses and
    the epoch/state at which early stopping fired."""
    device = device or pick_device()
    model.to(device)

    torch.manual_seed(cfg.seed)

    train_batch = make_batch(train_df, vocabs, device)
    train_target = make_target(train_df, device)
    val_batch = make_batch(val_df, vocabs, device)
    val_target = make_target(val_df, device)

    # Warm-start the circuit bias from data (see NAM.warm_start_circuit_bias) --
    # without this, a 75-110s constant takes tens of thousands of Adam steps to
    # learn from a zero init, which no realistic epoch budget reaches.
    circuit_means = (
        train_df.assign(_circuit_idx=train_batch["circuit_id"].cpu().numpy())
        .groupby("_circuit_idx")["lap_time"].mean().to_dict()
    )
    model.warm_start_circuit_bias(circuit_means, float(train_df["lap_time"].mean()))

    # Center CondHead on train-only conditions before training starts.
    with torch.no_grad():
        model.heads["cond"].calibrate(
            train_batch["track_temp"], train_batch["air_temp"],
            train_batch["wind_speed"], train_batch["humidity"],
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    n = len(train_df)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_best = 0
    history = {"train_loss": [], "val_huber": []}

    rng = np.random.default_rng(cfg.seed)

    tyre_params = list(model.heads["tyre"].parameters())
    if cfg.tyre_warmup_epochs > 0:
        for p in tyre_params:
            p.requires_grad_(False)

    for epoch in range(cfg.max_epochs):
        if cfg.tyre_warmup_epochs > 0 and epoch == cfg.tyre_warmup_epochs:
            for p in tyre_params:
                p.requires_grad_(True)

        model.train()
        order = rng.permutation(n)
        epoch_losses = []

        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            idx_t = torch.tensor(idx, device=device)
            batch = {k: v[idx_t] for k, v in train_batch.items()}
            target = train_target[idx_t]

            out = model(batch)
            loss = asymmetric_huber_loss(
                out["lap_time_hat"], target,
                delta=cfg.huber_delta_s, asymmetric_ratio=cfg.asymmetric_ratio,
            )
            penalty = model.traffic_zero_penalty(batch["gap_ahead_s"], cfg.traffic_zero_penalty_gap_s)
            evo_penalty = (out["evo"] ** 2).mean()
            smoothness_penalty = model.tyre_smoothness_penalty(
                batch["compound"], batch["circuit_id"], batch["track_temp"]
            )
            l1_penalty = model.tyre_l1_penalty(
                batch["compound"], batch["circuit_id"], batch["track_temp"]
            )
            total = (
                loss
                + cfg.traffic_zero_penalty_weight * penalty
                + cfg.evo_shrinkage_weight * evo_penalty
                + cfg.tyre_smoothness_weight * smoothness_penalty
                + cfg.tyre_l1_weight * l1_penalty
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_losses.append(total.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_batch)
            val_loss = huber_loss(val_out["lap_time_hat"], val_target, delta=cfg.huber_delta_s).item()

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_huber"].append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if verbose and (epoch % 10 == 0 or epoch == cfg.max_epochs - 1):
            print(f"epoch {epoch:3d}  train {history['train_loss'][-1]:.4f}  val_huber {val_loss:.4f}"
                  f"  best {best_val:.4f} @ {best_epoch}")

        if epochs_since_best >= cfg.early_stop_patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (no improvement for {cfg.early_stop_patience})")
            break

    model.load_state_dict(best_state)
    # Recompute CondHead's center against the final, trained weights.
    with torch.no_grad():
        model.heads["cond"].calibrate(
            train_batch["track_temp"], train_batch["air_temp"],
            train_batch["wind_speed"], train_batch["humidity"],
        )

    history["best_epoch"] = best_epoch
    history["best_val_huber"] = best_val
    return history
