"""Physics-aware transition constraints for Stage 3 10 Hz predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


ACCEL_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "ACCELERATING": frozenset({"ACCELERATING", "CONSTANT"}),
    "DECELERATING": frozenset({"DECELERATING", "CONSTANT", "STOPPED"}),
    "CONSTANT": frozenset({"ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"}),
    "STOPPED": frozenset({"ACCELERATING", "CONSTANT", "STOPPED"}),
}
STEER_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "LEFT": frozenset({"LEFT", "STRAIGHT"}),
    "STRAIGHT": frozenset({"LEFT", "STRAIGHT", "RIGHT"}),
    "RIGHT": frozenset({"STRAIGHT", "RIGHT"}),
}


def transition_mask(
    labels: Sequence[str],
    transitions: Mapping[str, frozenset[str]],
) -> np.ndarray:
    """Return ``[previous, next]`` booleans for an explicit label graph."""

    ordered = tuple(labels)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("labels must be non-empty and unique")
    if set(transitions) != set(ordered):
        raise ValueError("transition graph must define every label exactly once")
    unknown = set().union(*transitions.values()) - set(ordered)
    if unknown:
        raise ValueError(f"transition graph contains unknown labels: {sorted(unknown)}")
    return np.asarray(
        [[next_label in transitions[previous_label] for next_label in ordered] for previous_label in ordered],
        dtype=np.bool_,
    )


def viterbi_transition_logits(
    probabilities: np.ndarray,
    *,
    allowed_transitions: np.ndarray,
    forbidden_penalty: float = -1e9,
) -> np.ndarray:
    """Return logits whose argmax is the best transition-valid Viterbi path.

    Emission probabilities are converted to log space.  Every forbidden edge
    receives ``forbidden_penalty`` during dynamic programming.  The returned
    tensor keeps the chosen path's emission logit and assigns the same large
    penalty to every non-path class, making downstream ``argmax`` deterministic.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    transitions = np.asarray(allowed_transitions, dtype=np.bool_)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [time, classes]")
    if transitions.shape != (values.shape[1], values.shape[1]):
        raise ValueError("allowed_transitions must have shape [classes, classes]")
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = values.sum(axis=1, keepdims=True)
    if bool((row_sums <= 0.0).any()):
        raise ValueError("each probability row must have positive mass")
    if not np.isfinite(forbidden_penalty) or forbidden_penalty >= 0.0:
        raise ValueError("forbidden_penalty must be a finite negative value")

    normalized = values / row_sums
    emissions = np.log(normalized.clip(min=np.finfo(np.float64).tiny))
    transition_logits = np.where(transitions, 0.0, forbidden_penalty)
    time, classes = emissions.shape
    scores = np.empty_like(emissions)
    backpointers = np.zeros((time, classes), dtype=np.int64)
    scores[0] = emissions[0]
    for index in range(1, time):
        candidates = scores[index - 1, :, None] + transition_logits
        backpointers[index] = candidates.argmax(axis=0)
        scores[index] = emissions[index] + candidates.max(axis=0)

    path = np.empty(time, dtype=np.int64)
    path[-1] = int(scores[-1].argmax())
    for index in range(time - 1, 0, -1):
        path[index - 1] = backpointers[index, path[index]]

    constrained = np.full_like(emissions, forbidden_penalty)
    constrained[np.arange(time), path] = emissions[np.arange(time), path]
    return constrained.astype(np.float32)


def constrain_stage3_scores(
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    accel_labels: Sequence[str],
    steer_labels: Sequence[str],
    forbidden_penalty: float = -1e9,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Apply acceleration and steering transition graphs per video."""

    accel_mask = transition_mask(accel_labels, ACCEL_TRANSITIONS)
    steer_mask = transition_mask(steer_labels, STEER_TRANSITIONS)
    constrained: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for video_id, (accel, steer) in scores.items():
        if len(accel) != len(steer):
            raise ValueError(f"Stage 3 accel/steer length differs for {video_id}")
        constrained[video_id] = (
            viterbi_transition_logits(
                accel,
                allowed_transitions=accel_mask,
                forbidden_penalty=forbidden_penalty,
            ),
            viterbi_transition_logits(
                steer,
                allowed_transitions=steer_mask,
                forbidden_penalty=forbidden_penalty,
            ),
        )
    return constrained
