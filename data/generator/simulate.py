"""
Synthetic MMM dataset generator.

Generates datasets with known ground-truth parameters for benchmarking.
The data generating process mirrors what real MMM tools try to recover:

  1. Simulate spend per channel (correlated where specified)
  2. Apply adstock transform (geometric or delayed)
  3. Apply tanh saturation
  4. Multiply by true coefficient
  5. Sum contributions + baseline + trend + seasonality + noise

Ground truth returned alongside data so benchmarks can measure recovery accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scenario import ChannelConfig, Scenario


def _geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock to a spend series."""
    out = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + decay * out[t - 1]
    return out


def _delayed_adstock(x: np.ndarray, decay: float, peak: int) -> np.ndarray:
    """Apply delayed (Weibull-like) adstock: effect peaks at `peak` weeks after spend."""
    n = len(x)
    # Build a finite impulse response kernel
    max_lag = min(peak * 4, 12)
    lags = np.arange(max_lag + 1)
    # Triangular rise to peak then geometric decay
    kernel = np.where(
        lags <= peak,
        lags / peak if peak > 0 else 1.0,
        (decay ** (lags - peak)),
    )
    kernel = kernel / kernel.sum()  # normalise so total effect sums to 1

    out = np.zeros(n)
    for t in range(n):
        for lag, w in enumerate(kernel):
            if t - lag >= 0:
                out[t] += x[t - lag] * w
    return out


def _tanh_saturation(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply tanh saturation.
    scalar = max(x) so the transform is always in [0, tanh(1/alpha)] range.
    alpha controls shape: lower alpha = more saturation.
    """
    scalar = np.max(x) if np.max(x) > 0 else 1.0
    return np.tanh(x / (scalar * alpha + 1e-9))


def simulate_dataset(scenario: Scenario) -> tuple[pd.DataFrame, dict]:
    """
    Simulate a synthetic MMM dataset from a scenario definition.

    Returns
    -------
    df : pd.DataFrame
        Columns: date, kpi, <channel_name> per channel (raw spend).
    ground_truth : dict
        True parameters and per-channel contributions for metric calculation.
    """
    rng = np.random.default_rng(scenario.seed)
    n = scenario.n_weeks
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")

    # --- Simulate spend ---
    spends_raw: dict[str, np.ndarray] = {}
    channels = scenario.channels

    for i, ch in enumerate(channels):
        if ch.correlated_with is not None and ch.correlated_with < i:
            # Correlated with a previously-simulated channel
            base_ch = channels[ch.correlated_with].name
            base = spends_raw[base_ch]
            noise = rng.normal(0, ch.spend_std, n)
            spend = ch.spend_mean + ch.correlation * (base - channels[ch.correlated_with].spend_mean) + noise
        else:
            spend = rng.normal(ch.spend_mean, ch.spend_std, n)
        spends_raw[ch.name] = np.clip(spend, 0, None)

    # --- Apply transforms and compute contributions ---
    contributions: dict[str, np.ndarray] = {}
    true_rois: dict[str, float] = {}
    adstocked: dict[str, np.ndarray] = {}
    saturated: dict[str, np.ndarray] = {}

    for ch in channels:
        raw = spends_raw[ch.name]

        # Adstock
        if ch.adstock_type == "geometric":
            ads = _geometric_adstock(raw, ch.decay)
        else:
            ads = _delayed_adstock(raw, ch.decay, ch.peak_delay)
        adstocked[ch.name] = ads

        # Saturation
        sat = _tanh_saturation(ads, ch.alpha)
        saturated[ch.name] = sat

        # Contribution
        contrib = ch.true_coefficient * sat
        contributions[ch.name] = contrib

        # True ROI = total KPI contribution / total spend
        total_spend = raw.sum()
        total_contrib = contrib.sum()
        true_rois[ch.name] = total_contrib / total_spend if total_spend > 0 else 0.0

    # --- Baseline + trend ---
    t = np.arange(n)
    baseline = scenario.baseline + scenario.trend_slope * t

    # --- Seasonality (annual Fourier, 2 terms) ---
    freq = 2 * np.pi / 52.18  # annual cycle in weeks
    seasonality = (
        scenario.seasonality_amplitude
        * scenario.baseline
        * (np.sin(freq * t) + 0.3 * np.sin(2 * freq * t))
    )

    # --- Noise ---
    noise = rng.normal(0, scenario.noise_sigma, n)

    # --- KPI ---
    media_total = sum(contributions.values())
    kpi = baseline + seasonality + media_total + noise

    # --- Build DataFrame ---
    df = pd.DataFrame({"date": dates, "kpi": kpi})
    for ch in channels:
        df[ch.name] = spends_raw[ch.name]

    # --- Ground truth dict ---
    total_budget = scenario.total_budget or sum(
        ch.spend_mean * n for ch in channels
    )

    ground_truth = {
        "true_rois": true_rois,
        # Channel ranking by ROI (descending)
        "true_ranking": sorted(true_rois, key=true_rois.get, reverse=True),
        # True contribution share per channel
        "true_contribution_share": {
            ch: contributions[ch].sum() / media_total.sum()
            for ch in contributions
        },
        # Raw contributions time series (for advanced metrics)
        "contributions": {ch: contributions[ch].copy() for ch in contributions},
        "adstocked": {ch: adstocked[ch].copy() for ch in adstocked},
        "saturated": {ch: saturated[ch].copy() for ch in saturated},
        # Scenario metadata
        "scenario_name": scenario.name,
        "n_weeks": n,
        "channels": [ch.name for ch in channels],
        "true_params": {
            ch.name: {
                "decay": ch.decay,
                "alpha": ch.alpha,
                "true_coefficient": ch.true_coefficient,
                "adstock_type": ch.adstock_type,
            }
            for ch in channels
        },
        "total_budget": total_budget,
    }

    return df, ground_truth
