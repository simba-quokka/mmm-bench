"""Compute all metrics for a single run result against ground truth."""

from __future__ import annotations

from runners.base import RunResult
from .roi_recovery import roi_recovery_score
from .ranking import ranking_accuracy, spearman_rank_correlation
from .business_sense import business_sense_score
from .contribution_share import contribution_share_accuracy
from .fit_index import compute_fit_index


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

    # Business sense
    biz = business_sense_score(est_rois)

    # Contribution share accuracy
    true_shares = ground_truth.get("true_contribution_share", {})
    est_shares = result.estimated_contribution_share or {}
    share = contribution_share_accuracy(true_shares, est_shares)

    # In-sample fit index
    fitted_kpi = result.raw_output.get("fitted_kpi")
    actual_kpi = result.raw_output.get("actual_kpi")
    if fitted_kpi is not None and actual_kpi is not None:
        fit = compute_fit_index(actual_kpi, fitted_kpi)
    else:
        fit = {"r_squared": None, "in_sample_mape": None, "wape": None, "fit_index": None}

    # --- Holdout accuracy ---
    holdout_mape_val = result.raw_output.get("holdout_mape")
    holdout_accuracy = (
        round(max(0.0, 1.0 - holdout_mape_val), 4)
        if holdout_mape_val is not None
        else None
    )

    # --- Composite score ---
    # Weighted combination of all metrics. Components that are None are
    # excluded and their weight redistributed proportionally.
    _weights = {
        "rel_roi_accuracy": 0.30,
        "holdout_accuracy": 0.20,
        "contribution_share_accuracy": 0.20,
        "business_sense_score": 0.15,
        "fit_index": 0.15,
    }
    _values = {
        "rel_roi_accuracy": roi["rel_accuracy"],
        "holdout_accuracy": holdout_accuracy,
        "contribution_share_accuracy": share["share_accuracy"],
        "business_sense_score": biz["score"],
        "fit_index": fit["fit_index"],
    }
    available = {k: v for k, v in _values.items() if v is not None}
    if available:
        total_weight = sum(_weights[k] for k in available)
        composite = sum(_weights[k] * v / total_weight for k, v in available.items())
        composite = round(composite, 4)
    else:
        composite = None

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

        # Business sense
        "business_sense_score": biz["score"],
        "business_sense_per_channel": biz["per_channel"],

        # Contribution share accuracy
        "contribution_share_mape": share["share_mape"],
        "contribution_share_accuracy": share["share_accuracy"],

        # In-sample fit
        "r_squared": fit["r_squared"],
        "in_sample_mape": fit["in_sample_mape"],
        "wape": fit["wape"],
        "fit_index": fit["fit_index"],

        # Out-of-sample holdout
        "holdout_mape": holdout_mape_val,
        "holdout_accuracy": holdout_accuracy,

        # Composite
        "composite_score": composite,

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
