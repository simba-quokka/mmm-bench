"""
In-sample Fit Index.

Three components combined into a single score:
- R²: coefficient of determination
- MAPE: mean absolute percentage error on fitted vs actual KPI
- WAPE: weighted absolute percentage error (more robust than MAPE)

fit_index = (R² + (1 - MAPE) + (1 - WAPE)) / 3, all clipped to [0, 1].
"""

from __future__ import annotations

import numpy as np


def compute_fit_index(actual: np.ndarray, fitted: np.ndarray) -> dict:
    """
    Compute in-sample fit metrics.

    Parameters
    ----------
    actual : actual KPI values (training period)
    fitted : model-predicted KPI values (same period)

    Returns
    -------
    dict with r_squared, in_sample_mape, wape, fit_index
    """
    actual = np.asarray(actual, dtype=float)
    fitted = np.asarray(fitted, dtype=float)

    if len(actual) != len(fitted) or len(actual) == 0:
        return {
            "r_squared": None,
            "in_sample_mape": None,
            "wape": None,
            "fit_index": None,
        }

    # R²
    ss_res = float(np.sum((actual - fitted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # MAPE (skip near-zero actuals)
    mask = np.abs(actual) > 1e-6
    if mask.sum() > 0:
        mape = float(np.mean(np.abs(actual[mask] - fitted[mask]) / np.abs(actual[mask])))
    else:
        mape = float("inf")

    # WAPE
    total_actual = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(actual - fitted))) / total_actual if total_actual > 0 else float("inf")

    # Composite: clip each to [0, 1] before averaging
    r2_clipped = max(0.0, min(1.0, r_squared))
    mape_clipped = max(0.0, min(1.0, 1.0 - mape))
    wape_clipped = max(0.0, min(1.0, 1.0 - wape))
    fit_index = (r2_clipped + mape_clipped + wape_clipped) / 3.0

    return {
        "r_squared": round(r_squared, 4),
        "in_sample_mape": round(mape, 4),
        "wape": round(wape, 4),
        "fit_index": round(fit_index, 4),
    }
