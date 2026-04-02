"""
Contribution Share Accuracy.

Measures how accurately the tool recovers the true percentage share of total
media contribution per channel. More robust than absolute ROI across tools
with different saturation forms — the shares drive budget allocation decisions.
"""

from __future__ import annotations

import numpy as np


def contribution_share_accuracy(
    true_shares: dict[str, float],
    estimated_shares: dict[str, float],
) -> dict:
    """
    Compare true vs estimated contribution shares.

    Parameters
    ----------
    true_shares : from ground_truth['true_contribution_share']
    estimated_shares : from RunResult.estimated_contribution_share

    Returns
    -------
    dict with:
        share_mape     : MAPE on share values
        share_accuracy : max(0, 1 - share_mape)
        per_channel    : {channel: absolute error in share}
    """
    errors = []
    per_channel: dict[str, float] = {}

    for ch, true_share in true_shares.items():
        est_share = estimated_shares.get(ch)
        if est_share is None or true_share == 0:
            continue
        err = abs(true_share - est_share) / abs(true_share)
        errors.append(err)
        per_channel[ch] = round(err, 4)

    if not errors:
        return {"share_mape": None, "share_accuracy": None, "per_channel": {}}

    share_mape = float(np.mean(errors))

    return {
        "share_mape": round(share_mape, 4),
        "share_accuracy": round(max(0.0, 1.0 - share_mape), 4),
        "per_channel": per_channel,
    }
