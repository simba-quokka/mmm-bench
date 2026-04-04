"""
Synthetic CPG MMM dataset generator.

Generates datasets with known ground-truth parameters for benchmarking.

Data generating process:

  1. Simulate weekly spend per channel (flighted / always-on / seasonal-burst)
  2. Derive media activity (impressions / GRPs) from spend via CPM with noise
  3. Apply adstock transform to impressions (geometric or delayed)
  4. Apply tanh saturation to adstocked impressions
  5. Multiply by true coefficient → channel contribution
  6. Sum contributions + baseline + trend + seasonality + controls + noise = KPI

Returned DataFrame:
  - date
  - kpi
  - {channel}          : weekly impressions (the media activity variable)
  - {channel}_spend    : weekly spend in $ (used only as ROI denominator)
  - {control}          : each control variable

Ground truth:
  - true_rois          : contribution.sum() / spend.sum()  (spend-denominated)
  - true_contribution_share, contributions, adstocked, saturated, true_params
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scenario import ChannelConfig, ControlConfig, Scenario


# ---------------------------------------------------------------------------
# Adstock transforms
# ---------------------------------------------------------------------------

def _geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock to an impressions series."""
    out = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + decay * out[t - 1]
    return out


def _delayed_adstock(x: np.ndarray, decay: float, peak: int) -> np.ndarray:
    """Delayed (Weibull-like) adstock: effect peaks at `peak` weeks after exposure."""
    n = len(x)
    max_lag = min(peak * 4, 12)
    lags = np.arange(max_lag + 1)
    kernel = np.where(
        lags <= peak,
        lags / peak if peak > 0 else 1.0,
        decay ** (lags - peak),
    )
    kernel = kernel / kernel.sum()

    out = np.zeros(n)
    for t in range(n):
        for lag, w in enumerate(kernel):
            if t - lag >= 0:
                out[t] += x[t - lag] * w
    return out


# ---------------------------------------------------------------------------
# Saturation transform
# ---------------------------------------------------------------------------

def _tanh_saturation(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    tanh saturation.
    scalar = max(x) so the argument spans [0, 1/alpha] at the maximum.
    alpha < 1 → saturates aggressively; alpha > 1 → more linear range.
    """
    scalar = float(np.max(x)) if np.max(x) > 0 else 1.0
    return np.tanh(x / (scalar * alpha + 1e-9))


# ---------------------------------------------------------------------------
# Spend-pattern generators
# ---------------------------------------------------------------------------

def _spend_always_on(rng: np.random.Generator, ch: ChannelConfig, n: int) -> np.ndarray:
    """Continuous weekly spend drawn from N(mean, std), clipped at 0."""
    return np.clip(rng.normal(ch.spend_mean, ch.spend_std, n), 0, None)


def _spend_flighted(
    rng: np.random.Generator,
    ch: ChannelConfig,
    n: int,
) -> np.ndarray:
    """
    Flighted spend: on for `flight_on_rate` fraction of weeks in alternating blocks.
    Typical TV/OOH pattern — burst, go dark, burst again.
    """
    # Build flight mask: blocks of ~4-week duration
    block_size = max(1, round(4 / ch.flight_on_rate))  # avg block period
    on_weeks = max(1, round(block_size * ch.flight_on_rate))
    off_weeks = block_size - on_weeks

    mask = np.zeros(n, dtype=bool)
    t = 0
    on = True
    while t < n:
        length = on_weeks if on else off_weeks
        if on:
            mask[t : t + length] = True
        t += length
        on = not on

    spend = np.where(mask, rng.normal(ch.spend_mean, ch.spend_std, n), 0.0)
    return np.clip(spend, 0, None)


def _spend_seasonal_burst(
    rng: np.random.Generator,
    ch: ChannelConfig,
    n: int,
) -> np.ndarray:
    """
    Seasonal burst: spend concentrated in Q4 (weeks 40-52 each year).
    Think holiday / back-to-school CPG campaigns.
    """
    t = np.arange(n)
    week_in_year = t % 52
    in_burst = (week_in_year >= 39) & (week_in_year <= 51)  # Q4 weeks
    spend = np.where(
        in_burst,
        rng.normal(ch.spend_mean * 1.8, ch.spend_std * 1.5, n),
        rng.normal(ch.spend_mean * 0.3, ch.spend_std * 0.5, n),
    )
    return np.clip(spend, 0, None)


def _generate_spend(rng: np.random.Generator, ch: ChannelConfig, n: int) -> np.ndarray:
    if ch.spend_pattern == "flighted":
        return _spend_flighted(rng, ch, n)
    elif ch.spend_pattern == "seasonal_burst":
        return _spend_seasonal_burst(rng, ch, n)
    else:
        return _spend_always_on(rng, ch, n)


# ---------------------------------------------------------------------------
# Impressions generation
# ---------------------------------------------------------------------------

def _spend_to_impressions(
    rng: np.random.Generator,
    spend: np.ndarray,
    cpm: float,
    cpm_cv: float,
) -> np.ndarray:
    """
    Convert spend to impressions (or GRPs) with realistic CPM variation.

    impressions_t = spend_t / cpm_t * 1000
    cpm_t = cpm * (1 + epsilon_t),  epsilon_t ~ N(0, cpm_cv)

    On dark weeks (spend=0) impressions=0 automatically.
    """
    n = len(spend)
    cpm_noise = 1.0 + rng.normal(0, cpm_cv, n)
    cpm_noise = np.clip(cpm_noise, 0.3, 3.0)   # guard against extreme values
    effective_cpm = cpm * cpm_noise
    return np.where(effective_cpm > 0, spend / effective_cpm * 1000.0, 0.0)


# ---------------------------------------------------------------------------
# Control variable generators
# ---------------------------------------------------------------------------

def _generate_control(
    rng: np.random.Generator,
    ctrl: ControlConfig,
    n: int,
    t: np.ndarray,
) -> np.ndarray:
    freq = 2 * np.pi / 52.18

    if ctrl.pattern == "ar1":
        series = np.zeros(n)
        innov = rng.normal(0, ctrl.ar1_std, n)
        for i in range(1, n):
            series[i] = ctrl.ar1_rho * series[i - 1] + innov[i]
        return series

    elif ctrl.pattern == "trend":
        return ctrl.trend_slope * t  # starts at 0, drifts up/down

    elif ctrl.pattern == "seasonal":
        return ctrl.seasonality_amplitude * (
            np.sin(freq * t) + 0.4 * np.sin(2 * freq * t)
        )

    elif ctrl.pattern == "binary":
        # Random promotion/event weeks
        return rng.binomial(1, ctrl.binary_rate, n).astype(float)

    else:
        return np.zeros(n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def simulate_dataset(scenario: Scenario) -> tuple[pd.DataFrame, dict]:
    """
    Simulate a synthetic CPG MMM dataset from a scenario definition.

    Returns
    -------
    df : pd.DataFrame
        Columns:
          - date
          - kpi
          - {channel}        weekly impressions / GRPs (media activity variable)
          - {channel}_spend  weekly spend in $ (for ROI denominator)
          - {control}        each control variable
    ground_truth : dict
        True parameters and per-channel contributions for metric calculation.
        ROI is always spend-denominated: contribution.sum() / spend.sum().
    """
    rng = np.random.default_rng(scenario.seed)
    n = scenario.n_weeks
    t = np.arange(n)
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")

    channels = scenario.channels
    controls = scenario.controls

    # -------------------------------------------------------------------
    # 1. Simulate spend
    # -------------------------------------------------------------------
    spends_raw: dict[str, np.ndarray] = {}

    for i, ch in enumerate(channels):
        if ch.correlated_with is not None and ch.correlated_with < i:
            # Correlated with an already-simulated channel
            base_ch = channels[ch.correlated_with].name
            base_spend = spends_raw[base_ch]
            base_cfg = channels[ch.correlated_with]
            noise = rng.normal(0, ch.spend_std, n)
            spend = (
                ch.spend_mean
                + ch.correlation * (base_spend - base_cfg.spend_mean)
                + noise
            )
            spends_raw[ch.name] = np.clip(spend, 0, None)
        else:
            spends_raw[ch.name] = _generate_spend(rng, ch, n)

    # -------------------------------------------------------------------
    # 2. Derive impressions from spend (CPM with week-to-week variation)
    # -------------------------------------------------------------------
    impressions_raw: dict[str, np.ndarray] = {}
    for ch in channels:
        impressions_raw[ch.name] = _spend_to_impressions(
            rng, spends_raw[ch.name], ch.cpm, ch.cpm_cv
        )

    # -------------------------------------------------------------------
    # 3. Apply adstock → saturation → coefficient on impressions
    # -------------------------------------------------------------------
    contributions: dict[str, np.ndarray] = {}
    true_rois: dict[str, float] = {}
    adstocked: dict[str, np.ndarray] = {}
    saturated: dict[str, np.ndarray] = {}

    for ch in channels:
        imps = impressions_raw[ch.name]

        # Adstock on impressions
        if ch.adstock_type == "geometric":
            ads = _geometric_adstock(imps, ch.decay)
        else:
            ads = _delayed_adstock(imps, ch.decay, ch.peak_delay)
        adstocked[ch.name] = ads

        # Saturation
        sat = _tanh_saturation(ads, ch.alpha)
        saturated[ch.name] = sat

        # Contribution
        contrib = ch.true_coefficient * sat
        contributions[ch.name] = contrib

        # True ROI = total KPI contribution / total spend (spend-denominated)
        total_spend = float(spends_raw[ch.name].sum())
        total_contrib = float(contrib.sum())
        true_rois[ch.name] = total_contrib / total_spend if total_spend > 0 else 0.0

    # -------------------------------------------------------------------
    # 4. Control variables
    # -------------------------------------------------------------------
    control_series: dict[str, np.ndarray] = {}
    control_contributions: dict[str, np.ndarray] = {}

    for ctrl in controls:
        series = _generate_control(rng, ctrl, n, t)
        control_series[ctrl.name] = series
        control_contributions[ctrl.name] = ctrl.true_coefficient * series

    # -------------------------------------------------------------------
    # 5. Baseline + trend + seasonality + noise
    # -------------------------------------------------------------------
    freq = 2 * np.pi / 52.18
    baseline = scenario.baseline + scenario.trend_slope * t
    seasonality = (
        scenario.seasonality_amplitude
        * scenario.baseline
        * (np.sin(freq * t) + 0.3 * np.sin(2 * freq * t))
    )
    noise = rng.normal(0, scenario.noise_sigma, n)

    # -------------------------------------------------------------------
    # 6. KPI
    # -------------------------------------------------------------------
    media_total = sum(contributions.values())
    ctrl_total = (
        sum(control_contributions.values())
        if control_contributions
        else np.zeros(n)
    )
    kpi = baseline + seasonality + media_total + ctrl_total + noise

    # -------------------------------------------------------------------
    # 7. Build DataFrame
    # -------------------------------------------------------------------
    df = pd.DataFrame({"date": dates, "kpi": kpi})
    for ch in channels:
        df[ch.name] = impressions_raw[ch.name]          # activity variable
        df[f"{ch.name}_spend"] = spends_raw[ch.name]    # spend (ROI denominator)
    for ctrl in controls:
        df[ctrl.name] = control_series[ctrl.name]

    # -------------------------------------------------------------------
    # 8. Ground truth
    # -------------------------------------------------------------------
    total_budget = scenario.total_budget or sum(
        ch.spend_mean * n for ch in channels
    )

    ground_truth = {
        # Primary benchmark signal
        "true_rois": true_rois,
        "true_ranking": sorted(true_rois, key=true_rois.__getitem__, reverse=True),
        "true_contribution_share": {
            ch: float(contributions[ch].sum() / media_total.sum())
            for ch in contributions
        },

        # Raw time series (for advanced diagnostics)
        "contributions": {ch: contributions[ch].copy() for ch in contributions},
        "adstocked": {ch: adstocked[ch].copy() for ch in adstocked},
        "saturated": {ch: saturated[ch].copy() for ch in saturated},
        "impressions": {ch: impressions_raw[ch].copy() for ch in impressions_raw},
        "spends": {ch: spends_raw[ch].copy() for ch in spends_raw},
        "control_series": {ctrl: control_series[ctrl].copy() for ctrl in control_series},

        # Scenario metadata
        "scenario_name": scenario.name,
        "n_weeks": n,
        "channels": [ch.name for ch in channels],
        "control_cols": [ctrl.name for ctrl in controls],
        "spend_cols": [f"{ch.name}_spend" for ch in channels],
        "total_budget": total_budget,

        # True DGP parameters
        "true_params": {
            ch.name: {
                "decay": ch.decay,
                "alpha": ch.alpha,
                "true_coefficient": ch.true_coefficient,
                "adstock_type": ch.adstock_type,
                "cpm": ch.cpm,
                "spend_pattern": ch.spend_pattern,
            }
            for ch in channels
        },
        "true_control_params": {
            ctrl.name: {
                "true_coefficient": ctrl.true_coefficient,
                "pattern": ctrl.pattern,
            }
            for ctrl in controls
        },
    }

    return df, ground_truth


# ---------------------------------------------------------------------------
# Lift test generation
# ---------------------------------------------------------------------------

def generate_lift_tests(
    scenario: Scenario,
    df: pd.DataFrame,
    ground_truth: dict,
    channels: list[str] | None = None,
    start_week: int = 40,
    end_week: int = 52,
    spend_uplift: float = 0.20,
    noise_cv: float = 0.10,
) -> pd.DataFrame:
    """
    Generate synthetic geo-holdout lift test data from the known DGP.

    Simulates an experiment where the treatment geo receives `spend_uplift`
    more spend for weeks `start_week` to `end_week`. The true incremental
    outcome is computed from the known saturation curve, with measurement
    noise added.

    Returns a DataFrame compatible with PyMC-Marketing's
    ``add_lift_test_measurements()``:
        - channel: channel name
        - x: baseline media activity (mean adstocked impressions in control period)
        - delta_x: additional media activity in treatment
        - delta_y: observed incremental KPI contribution (with noise)
        - sigma: measurement noise standard deviation

    Parameters
    ----------
    scenario : Scenario definition (for channel configs)
    df : Full simulated DataFrame
    ground_truth : Ground truth dict from simulate_dataset()
    channels : Which channels to generate lift tests for (default: all)
    start_week, end_week : Experiment period (row indices)
    spend_uplift : Fractional increase in spend for treatment (0.20 = +20%)
    noise_cv : Coefficient of variation for measurement noise
    """
    rng = np.random.default_rng(scenario.seed + 999)

    if channels is None:
        channels = [ch.name for ch in scenario.channels]

    ch_configs = {ch.name: ch for ch in scenario.channels}
    rows = []

    for ch_name in channels:
        ch = ch_configs[ch_name]

        # Baseline: mean adstocked impressions during the experiment period
        adstocked = ground_truth["adstocked"][ch_name]
        experiment_mask = slice(start_week, end_week + 1)
        x_baseline = float(np.mean(adstocked[experiment_mask]))

        # Treatment: additional impressions from spend uplift
        # extra_spend → extra_impressions → adstock → saturation difference
        spend_in_period = df[f"{ch_name}_spend"].values[experiment_mask]
        extra_spend = spend_in_period * spend_uplift
        extra_impressions = extra_spend / ch.cpm * 1000.0
        x_treatment = x_baseline + float(np.mean(extra_impressions))
        delta_x = x_treatment - x_baseline

        # True incremental outcome from DGP saturation curve
        scalar = float(np.max(adstocked)) if np.max(adstocked) > 0 else 1.0
        sat_baseline = np.tanh(x_baseline / (scalar * ch.alpha + 1e-9))
        sat_treatment = np.tanh(x_treatment / (scalar * ch.alpha + 1e-9))
        true_incremental = ch.true_coefficient * (sat_treatment - sat_baseline)

        # Add measurement noise
        sigma = abs(true_incremental) * noise_cv if true_incremental != 0 else 1.0
        observed_incremental = true_incremental + rng.normal(0, sigma)

        rows.append({
            "channel": ch_name,
            "x": x_baseline,
            "delta_x": delta_x,
            "delta_y": float(observed_incremental),
            "sigma": float(sigma),
        })

    return pd.DataFrame(rows)
