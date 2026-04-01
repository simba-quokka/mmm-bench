"""Compute all metrics for a single run result against ground truth."""

from __future__ import annotations

from runners.base import RunResult
from .roi_recovery import roi_recovery_score
from .ranking import ranking_accuracy, spearman_rank_correlation


def compute_all_metrics(result: RunResult, ground_truth: dict) -> dict:
    """
    Compute all benchmark metrics for a RunResult against ground truth.

    Primary metric: rel_accuracy — relative ROI accuracy after mean-normalisation.
    This is fair across tools with different saturation parameterisations.

    Secondary metric: abs_accuracy — absolute ROI accuracy (MAPE).
    Sensitive to parameterisation differences; included for completeness.
    """
    true_rois = ground_truth["true_rois"]
    est_rois = result.estimated_rois

    roi = roi_recovery_score(true_rois, est_rois)
    ranking = ranking_accuracy(true_rois, est_rois)
    spearman = spearman_rank_correlation(true_rois, est_rois)

    return {
        # Identity
        "tool": result.tool_name,
        "version": result.tool_version,
        "scenario": ground_truth["scenario_name"],

        # Primary metric: relative ROI accuracy
        "rel_roi_mape": roi["rel_mape"],
        "rel_roi_accuracy": roi["rel_accuracy"],

        # Secondary metric: absolute ROI accuracy
        "abs_roi_mape": roi["abs_mape"],
        "abs_roi_accuracy": roi["abs_accuracy"],

        # Per-channel detail — raw ROI values for granular table
        "true_rois": true_rois,
        "estimated_rois": {ch: v for ch, v in est_rois.items() if v is not None},
        "per_channel_rel": roi["per_channel_rel"],
        "per_channel_abs": roi["per_channel_abs"],

        # Ranking
        "top1_correct": ranking["top1_correct"],
        "top2_overlap": ranking["top2_overlap"],
        "pairwise_accuracy": ranking["pairwise_accuracy"],
        "spearman_rho": spearman,

        # Convergence
        "converged": result.converged,
        "n_convergence_warnings": len(result.convergence_warnings),

        # Runtime
        "runtime_seconds": round(result.runtime_seconds, 1),
        "estimated_cost_usd": result.estimated_cost_usd,

        # Failures
        "n_channels_failed": roi["n_failed"],
        "n_channels_valid": roi["n_valid"],
    }
