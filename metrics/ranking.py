"""
Channel ranking metrics.

Measures whether a tool correctly identifies which channels are most efficient.
Getting the ranking right matters more than exact ROI numbers for budget decisions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _to_ranked_list(roi_dict: dict[str, float]) -> list[str]:
    """Sort channels by ROI descending, skipping None values."""
    valid = {ch: v for ch, v in roi_dict.items() if v is not None}
    return sorted(valid, key=valid.get, reverse=True)


def ranking_accuracy(
    true_rois: dict[str, float], estimated_rois: dict[str, float]
) -> dict[str, float]:
    """
    Compare channel rankings between ground truth and tool estimates.

    Returns
    -------
    dict with:
        top1_correct  : bool — did it correctly identify the #1 channel?
        top2_correct  : fraction of top-2 channels correctly identified
        full_accuracy : fraction of all pairwise orderings that are correct
        spearman_rho  : Spearman rank correlation (1 = perfect)
    """
    true_ranking = _to_ranked_list(true_rois)
    est_ranking = _to_ranked_list(estimated_rois)

    if not true_ranking or not est_ranking:
        return {"top1_correct": False, "top2_correct": 0.0, "full_accuracy": 0.0, "spearman_rho": 0.0}

    # Top-1
    top1_correct = true_ranking[0] == est_ranking[0] if est_ranking else False

    # Top-2 overlap (Jaccard on top-2 set)
    n = min(2, len(true_ranking), len(est_ranking))
    top2_overlap = len(set(true_ranking[:n]) & set(est_ranking[:n])) / n if n > 0 else 0.0

    # Pairwise ordering accuracy
    channels = [ch for ch in true_ranking if ch in estimated_rois and estimated_rois[ch] is not None]
    n_correct = 0
    n_pairs = 0
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            a, b = channels[i], channels[j]
            true_a_better = true_rois[a] > true_rois[b]
            est_a_better = estimated_rois[a] > estimated_rois[b]
            n_correct += int(true_a_better == est_a_better)
            n_pairs += 1
    full_accuracy = n_correct / n_pairs if n_pairs > 0 else 0.0

    return {
        "top1_correct": top1_correct,
        "top2_overlap": round(top2_overlap, 3),
        "pairwise_accuracy": round(full_accuracy, 3),
    }


def spearman_rank_correlation(
    true_rois: dict[str, float], estimated_rois: dict[str, float]
) -> float:
    """Spearman rank correlation between true and estimated ROIs."""
    channels = [ch for ch in true_rois if estimated_rois.get(ch) is not None]
    if len(channels) < 2:
        return 0.0
    true_vals = [true_rois[ch] for ch in channels]
    est_vals = [estimated_rois[ch] for ch in channels]
    rho, _ = spearmanr(true_vals, est_vals)
    return round(float(rho), 3)
