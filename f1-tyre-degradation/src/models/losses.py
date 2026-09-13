"""Phase 1.5 — the loss.

Lap-time noise is asymmetric: a driver loses time to a mistake far more often
than they gain it. Plain MSE (or symmetric Huber) drags every fitted curve
upward toward those rare slow outliers, because minimising squared error means
matching the (outlier-inflated) mean, not the typical (clean) lap.

Residual convention: `r = actual - predicted`.
  - `r > 0` ("under-prediction") is the mistake-lap case: the model predicted
    the normal pace and the driver was slower than that. We do not want the
    fitted curve to chase this.
  - `r < 0` ("over-prediction") is the model predicting a slower lap than
    actually happened. Penalised MORE (at `asymmetric_ratio`), so the model is
    kept honest to the typical/fast side of the distribution rather than
    creeping upward to reduce squared error on rare mistakes.
"""

from __future__ import annotations

import torch


def huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_r = residual.abs()
    quadratic = torch.clamp(abs_r, max=delta)
    linear = abs_r - quadratic
    return 0.5 * quadratic**2 + delta * linear


def asymmetric_huber_loss(predicted: torch.Tensor, actual: torch.Tensor, *,
                            delta: float = 0.25, asymmetric_ratio: float = 1.5) -> torch.Tensor:
    """Mean asymmetric Huber loss. See module docstring for the sign convention."""
    residual = actual - predicted
    base = huber(residual, delta)
    weight = torch.where(residual < 0, asymmetric_ratio, 1.0)
    return (base * weight).mean()


def mse_loss(predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return ((actual - predicted) ** 2).mean()


def huber_loss(predicted: torch.Tensor, actual: torch.Tensor, *, delta: float = 0.25) -> torch.Tensor:
    return huber(actual - predicted, delta).mean()
