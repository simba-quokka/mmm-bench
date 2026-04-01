"""Compute all metrics for a single run result against ground truth."""

from __future__ import annotations

from runners.base import RunResult
from .roi_recovery import roi_recovery_score
from .ranking import ranking_accuracy, spearman_rank_correlation


def compute_all_metrics(result: RunResult, ground_truth: dict) -> dict:
    """
    Compute all benchmark metrics for a RunResult against ground truth.

    Returns a flat dict suitable for writing to results/leaderboard.
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
        # ROI recovery
        "roi_mape": roi["mape"],
        "roi_accuracy": roi["accuracy"],
        "roi_per_channel": roi["per_channel"],
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
