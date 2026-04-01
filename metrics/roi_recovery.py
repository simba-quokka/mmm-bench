"""
ROI recovery metrics.

Two metrics are computed:

1. Absolute ROI accuracy (MAPE)
   Compares estimated ROI directly against ground truth.
   Sensitive to parameterisation differences between tools — a tool that
   uses a different saturation formula may produce correct relative efficiency
   but wrong absolute scale.

2. Relative ROI accuracy (MAPE on normalised ROIs)
   Both true and estimated ROIs are divided by their own mean before comparing.
   Measures whether the tool correctly identifies *which channels are most
   efficient*, regardless of the absolute scale of its estimates.
   This is the primary metric for budget allocation decisions and is fair
   across tools with different saturation parameterisations.

   Example: if true ROIs are [0.28, 1.36, 0.75] and estimated are [1.08, 1.37, 1.35],
   normalised true = [0.43, 2.08, 1.15], normalised est = [0.91, 1.15, 1.13].
   The relative accuracy captures the ordering and relative gaps.
"""

from __future__ import annotations

import numpy as np


def _normalise(rois: dict[str, float]) -> dict[str, float]:
    """Divide each ROI by the mean of all valid ROIs."""
    valid = {ch: v for ch, v in rois.items() if v is not None and v > 0}
    if not valid:
        return rois
    mean_roi = float(np.mean(list(valid.values())))
    if mean_roi == 0:
        return rois
    return {ch: v / mean_roi for ch, v in valid.items()}


def mean_absolute_percentage_error(
    true: dict[str, float], estimated: dict[str, float]
) -> float:
    """MAPE across channels (absolute scale)."""
    errors = []
    for ch, true_val in true.items():
        est = estimated.get(ch)
        if est is None or true_val == 0:
            continue
        errors.append(abs(true_val - est) / abs(true_val))
    return float(np.mean(errors)) if errors else float("inf")


def roi_recovery_score(
    true: dict[str, float], estimated: dict[str, float]
) -> dict[str, float]:
    """
    Per-channel and overall ROI recovery metrics — both absolute and relative.

    Returns
    -------
    dict with keys:
        abs_mape           : MAPE on raw ROI values
        abs_accuracy       : 1 - abs_mape (capped at 0)
        rel_mape           : MAPE on mean-normalised ROI values (primary metric)
        rel_accuracy       : 1 - rel_mape (capped at 0) — primary accuracy metric
        per_channel_abs    : {channel: absolute % error}
        per_channel_rel    : {channel: relative % error after normalisation}
        failed_channels    : channels where tool returned None
        n_valid            : number of channels with valid estimates
        n_failed           : number of channels that failed
    """
    failed = [ch for ch, v in estimated.items() if v is None]
    valid_channels = [ch for ch in true if estimated.get(ch) is not None]

    # --- Absolute ---
    per_channel_abs = {}
    for ch in valid_channels:
        t = true[ch]
        e = estimated[ch]
        per_channel_abs[ch] = abs(t - e) / abs(t) if t != 0 else float("inf")

    abs_mape = float(np.mean(list(per_channel_abs.values()))) if per_channel_abs else float("inf")
    abs_accuracy = max(0.0, 1.0 - abs_mape)

    # --- Relative (normalised by mean) ---
    true_norm = _normalise({ch: true[ch] for ch in valid_channels})
    est_norm = _normalise({ch: estimated[ch] for ch in valid_channels})

    per_channel_rel = {}
    for ch in valid_channels:
        t = true_norm.get(ch, 0)
        e = est_norm.get(ch, 0)
        per_channel_rel[ch] = abs(t - e) / abs(t) if t != 0 else float("inf")

    rel_mape = float(np.mean(list(per_channel_rel.values()))) if per_channel_rel else float("inf")
    rel_accuracy = max(0.0, 1.0 - rel_mape)

    return {
        "abs_mape": round(abs_mape, 4),
        "abs_accuracy": round(abs_accuracy, 4),
        "rel_mape": round(rel_mape, 4),
        "rel_accuracy": round(rel_accuracy, 4),
        "per_channel_abs": {ch: round(v, 4) for ch, v in per_channel_abs.items()},
        "per_channel_rel": {ch: round(v, 4) for ch, v in per_channel_rel.items()},
        "failed_channels": failed,
        "n_valid": len(valid_channels),
        "n_failed": len(failed),
    }
