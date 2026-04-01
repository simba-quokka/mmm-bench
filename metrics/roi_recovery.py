"""
ROI recovery metrics.

Measures how accurately a tool recovers the true channel ROI
from the synthetic dataset's known ground truth.
"""

from __future__ import annotations

import numpy as np


def mean_absolute_percentage_error(
    true: dict[str, float], estimated: dict[str, float]
) -> float:
    """
    MAPE across channels.
    Channels where estimated is None (tool failed) are excluded and penalised separately.
    Returns value in [0, 1+] where 0 = perfect, 1 = 100% error on average.
    """
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
    Per-channel and overall ROI recovery metrics.

    Returns
    -------
    dict with keys:
        mape          : mean absolute percentage error across channels
        accuracy      : 1 - mape (capped at 0)
        per_channel   : {channel: % error}
        failed        : list of channels where tool returned None
    """
    failed = [ch for ch, v in estimated.items() if v is None]
    valid_channels = [ch for ch in true if estimated.get(ch) is not None]

    per_channel = {}
    for ch in valid_channels:
        t = true[ch]
        e = estimated[ch]
        per_channel[ch] = abs(t - e) / abs(t) if t != 0 else float("inf")

    mape = float(np.mean(list(per_channel.values()))) if per_channel else float("inf")
    accuracy = max(0.0, 1.0 - mape)

    return {
        "mape": round(mape, 4),
        "accuracy": round(accuracy, 4),
        "per_channel": {ch: round(v, 4) for ch, v in per_channel.items()},
        "failed_channels": failed,
        "n_valid": len(valid_channels),
        "n_failed": len(failed),
    }
