"""
Business Sense Score.

Checks whether estimated ROIs fall within plausible industry ranges,
independent of whether they match ground truth exactly. A tool can
converge cleanly and produce confident but nonsensical ROIs — this catches it.
"""

from __future__ import annotations


# Industry plausibility ranges (spend-denominated ROI)
INDUSTRY_RANGES: dict[str, tuple[float, float]] = {
    "tv":         (0.05, 0.80),
    "ooh":        (0.03, 0.40),
    "paid_search": (0.30, 4.00),
    "paid_social": (0.10, 2.00),
    "display":    (0.05, 0.80),
    "email":      (0.50, 8.00),
    "youtube":    (0.10, 1.50),
    "affiliates": (0.30, 4.00),
    "tiktok":     (0.20, 2.50),
}


def business_sense_score(estimated_rois: dict[str, float]) -> dict:
    """
    Score how many estimated ROIs fall within plausible industry ranges.

    Channels not in the lookup table are excluded from the denominator.
    Returns score in [0, 1] plus per-channel pass/fail detail.
    """
    per_channel: dict[str, bool] = {}

    for ch, roi in estimated_rois.items():
        if roi is None:
            continue
        key = ch.lower().strip()
        if key not in INDUSTRY_RANGES:
            continue
        lo, hi = INDUSTRY_RANGES[key]
        per_channel[ch] = lo <= roi <= hi

    n_total = len(per_channel)
    n_pass = sum(per_channel.values())

    return {
        "score": n_pass / n_total if n_total > 0 else None,
        "per_channel": per_channel,
        "n_pass": n_pass,
        "n_fail": n_total - n_pass,
    }
