"""Loss functions used by Stage 1 experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss: ``-(1 - p_t) ** gamma * log(p_t)``.

    The modulation suppresses gradients from already-correct samples so the
    optimizer spends relatively more capacity on visually ambiguous examples.
    Class weights remain optional because the real class distribution is not
    available in the supplied public example.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        *,
        alpha: Sequence[float] | torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.gamma = float(gamma)
        self.reduction = reduction
        alpha_tensor = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        if alpha_tensor is not None and (alpha_tensor.ndim != 1 or len(alpha_tensor) < 1):
            raise ValueError("alpha must be a non-empty one-dimensional sequence")
        self.register_buffer("alpha", alpha_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [batch, classes], got {tuple(logits.shape)}")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(
                "targets must have shape [batch] matching logits, "
                f"got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
            )
        if self.alpha is not None and len(self.alpha) != logits.shape[1]:
            raise ValueError(
                f"alpha has {len(self.alpha)} entries but logits have {logits.shape[1]} classes"
            )

        log_probabilities = F.log_softmax(logits, dim=1)
        log_pt = log_probabilities.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        losses = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            losses = losses * self.alpha.to(logits)[targets]
        if self.reduction == "sum":
            return losses.sum()
        if self.reduction == "mean":
            return losses.mean()
        return losses
