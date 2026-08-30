"""Loss functions used by Stage 1 experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss: ``-(1 - p_t) ** gamma * log(p_t)``.

    The modulation suppresses gradients from already-correct samples so the
    optimizer spends relatively more capacity on visually ambiguous examples.
    Optional per-class ``alpha`` weights can be derived from each training fold
    without using validation labels.
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
        if alpha_tensor is not None and (
            not torch.isfinite(alpha_tensor).all() or bool((alpha_tensor <= 0).any())
        ):
            raise ValueError("alpha entries must be finite and > 0")
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


def truncated_temporal_mse(
    frame_logits: torch.Tensor,
    *,
    truncation: float = 4.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize abrupt changes between adjacent frame log probabilities.

    The previous frame is detached as a stable local target, following the
    truncated temporal MSE used by temporal segmentation models. Squared
    differences above ``truncation ** 2`` are clipped so true scene changes do
    not dominate training.
    """

    if frame_logits.ndim != 3:
        raise ValueError(
            "frame_logits must have shape [batch, time, classes], "
            f"got {tuple(frame_logits.shape)}"
        )
    if frame_logits.shape[-1] < 2:
        raise ValueError("frame_logits must contain at least two classes")
    if truncation <= 0.0:
        raise ValueError("truncation must be > 0")
    if frame_logits.shape[1] < 2:
        return frame_logits.sum() * 0.0

    log_probabilities = F.log_softmax(frame_logits, dim=-1)
    differences = (
        log_probabilities[:, 1:]
        - log_probabilities[:, :-1].detach()
    ).square()
    losses = differences.clamp_max(float(truncation) ** 2)
    if valid_mask is None:
        return losses.mean()
    if valid_mask.shape != frame_logits.shape[:2]:
        raise ValueError(
            "valid_mask must have shape [batch, time], "
            f"got {tuple(valid_mask.shape)}"
        )
    transition_mask = (valid_mask[:, 1:] & valid_mask[:, :-1]).to(losses)
    transition_mask = transition_mask.unsqueeze(-1).expand_as(losses)
    denominator = transition_mask.sum().clamp_min(1.0)
    return (losses * transition_mask).sum() / denominator


def explainability_reconstruction_terms(
    reconstructed_targets: torch.Tensor,
    target_frames: torch.Tensor,
    explainability_masks: torch.Tensor,
    *,
    explainability_mask_logits: torch.Tensor | None = None,
    charbonnier_epsilon: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mask-weighted photometric loss and BCE-to-one regularization."""

    if reconstructed_targets.shape != target_frames.shape:
        raise ValueError(
            "reconstructed_targets and target_frames must have identical shapes: "
            f"{tuple(reconstructed_targets.shape)} != {tuple(target_frames.shape)}"
        )
    if reconstructed_targets.ndim != 5 or reconstructed_targets.shape[1] != 3:
        raise ValueError(
            "reconstruction tensors must have shape [batch, 3, time, height, width]"
        )
    expected_mask_shape = (
        reconstructed_targets.shape[0],
        1,
        reconstructed_targets.shape[2],
        reconstructed_targets.shape[3],
        reconstructed_targets.shape[4],
    )
    if tuple(explainability_masks.shape) != expected_mask_shape:
        raise ValueError(
            "explainability_masks must have shape [batch, 1, time, height, width], "
            f"expected {expected_mask_shape}, got {tuple(explainability_masks.shape)}"
        )
    if charbonnier_epsilon <= 0.0:
        raise ValueError("charbonnier_epsilon must be > 0")
    if not torch.isfinite(explainability_masks).all():
        raise ValueError("explainability masks must be finite")

    photometric = torch.sqrt(
        (reconstructed_targets - target_frames).square()
        + float(charbonnier_epsilon) ** 2
    ).mean(dim=1, keepdim=True)
    weighted_reconstruction = (explainability_masks * photometric).mean()
    mask_logits = (
        torch.logit(explainability_masks.float().clamp(1e-6, 1.0 - 1e-6))
        if explainability_mask_logits is None
        else explainability_mask_logits.float()
    )
    if mask_logits.shape != explainability_masks.shape:
        raise ValueError("explainability_mask_logits must match explainability_masks")
    mask_regularization = F.binary_cross_entropy_with_logits(
        mask_logits,
        torch.ones_like(mask_logits),
    )
    return weighted_reconstruction, mask_regularization


class Stage1MultiTaskLoss(nn.Module):
    """Combine classification, smoothing, and balanced explainability losses.

    The explainability term is deliberately grouped as
    ``lambda_x * (reconstruction + lambda_e * BCE-to-one)``.  With the defaults
    this makes the effective BCE coefficient ``0.05 * 0.02 = 0.001``.  A
    separate ``1e-3 * mean(mask)`` term gives dynamic/occluded regions a small
    incentive to lower their masks without allowing the all-zero solution.
    """

    def __init__(
        self,
        *,
        focal_gamma: float = 2.0,
        focal_alpha: Sequence[float] | torch.Tensor | None = None,
        frame_classification_weight: float = 0.25,
        smoothing_weight: float = 0.05,
        smoothing_truncation: float = 4.0,
        explainability_weight: float = 0.05,
        mask_regularization_weight: float = 0.02,
        mask_sparsity_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        weights = {
            "frame_classification_weight": frame_classification_weight,
            "smoothing_weight": smoothing_weight,
            "explainability_weight": explainability_weight,
            "mask_regularization_weight": mask_regularization_weight,
            "mask_sparsity_weight": mask_sparsity_weight,
        }
        invalid = {name: value for name, value in weights.items() if value < 0.0}
        if invalid:
            raise ValueError(f"Stage 1 auxiliary loss weights must be >= 0: {invalid}")
        if smoothing_truncation <= 0.0:
            raise ValueError("smoothing_truncation must be > 0")
        self.classification = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.frame_classification_weight = float(frame_classification_weight)
        self.smoothing_weight = float(smoothing_weight)
        self.smoothing_truncation = float(smoothing_truncation)
        self.explainability_weight = float(explainability_weight)
        self.mask_regularization_weight = float(mask_regularization_weight)
        self.mask_sparsity_weight = float(mask_sparsity_weight)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        required = {
            "logits",
            "frame_logits",
            "reconstructed_targets",
            "target_frames",
            "explainability_masks",
            "explainability_mask_logits",
        }
        missing = sorted(required - set(outputs))
        if missing:
            raise ValueError(f"Stage 1 multi-task outputs are missing keys: {missing}")
        clip_classification = self.classification(outputs["logits"], targets)
        frame_logits = outputs["frame_logits"]
        if frame_logits.ndim != 3 or frame_logits.shape[0] != targets.shape[0]:
            raise ValueError("frame_logits must have shape [batch, time, classes]")
        stage_frame_logits = outputs.get("stage_frame_logits")
        if stage_frame_logits is None:
            supervised_frame_logits = frame_logits[:, None]
        else:
            if (
                stage_frame_logits.ndim != 4
                or stage_frame_logits.shape[0] != targets.shape[0]
                or stage_frame_logits.shape[2:] != frame_logits.shape[1:]
            ):
                raise ValueError(
                    "stage_frame_logits must have shape "
                    "[batch, stages, time, classes] matching frame_logits"
                )
            supervised_frame_logits = stage_frame_logits
        batch, stages, time, classes = supervised_frame_logits.shape
        repeated_targets = (
            targets[:, None, None]
            .expand(batch, stages, time)
            .reshape(-1)
        )
        frame_classification = self.classification(
            supervised_frame_logits.reshape(-1, classes),
            repeated_targets,
        )
        smoothing = truncated_temporal_mse(
            supervised_frame_logits.reshape(batch * stages, time, classes),
            truncation=self.smoothing_truncation,
        )
        reconstruction, mask_regularization = explainability_reconstruction_terms(
            outputs["reconstructed_targets"],
            outputs["target_frames"],
            outputs["explainability_masks"],
            explainability_mask_logits=outputs["explainability_mask_logits"],
        )
        mask_sparsity = outputs["explainability_masks"].mean()
        total = (
            clip_classification
            + self.frame_classification_weight * frame_classification
            + self.smoothing_weight * smoothing
            + self.explainability_weight
            * (reconstruction + self.mask_regularization_weight * mask_regularization)
            + self.mask_sparsity_weight * mask_sparsity
        )
        return {
            "total": total,
            "clip_classification": clip_classification,
            "frame_classification": frame_classification,
            "smoothing": smoothing,
            "reconstruction": reconstruction,
            "mask_regularization": mask_regularization,
            "mask_sparsity": mask_sparsity,
        }
