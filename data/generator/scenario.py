"""Scenario and channel configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ChannelConfig:
    """True parameters for a single media channel."""

    name: str

    # Spend distribution
    spend_mean: float        # Average weekly spend ($)
    spend_std: float         # Standard deviation of spend

    # Spend pattern — how budgets are allocated over time
    spend_pattern: Literal["always_on", "flighted", "seasonal_burst"] = "always_on"
    flight_on_rate: float = 1.0   # fraction of weeks channel is active (flighted only)

    # Impressions / activity variable generation
    # impressions = spend / cpm * 1000 * (1 + cpm_noise)
    # TV: GRPs per $1000 spend.  Digital: thousands of impressions per $1000 spend.
    cpm: float = 10.0         # cost-per-mille ($) — determines spend→impressions scale
    cpm_cv: float = 0.15      # week-to-week CPM variation (coefficient of variation)

    # Adstock parameters
    adstock_type: Literal["geometric", "delayed"] = "geometric"
    decay: float = 0.4        # Geometric decay rate (0–1). Higher = longer carryover.
    peak_delay: int = 1       # For delayed adstock: week of peak effect

    # Saturation parameters (tanh)
    # true response = tanh(adstock(impressions) / (scalar * alpha))
    # scalar = max(adstock(impressions)) so response ∈ [0, tanh(1/alpha)]
    alpha: float = 1.0        # Saturation shape. Lower = saturates faster.

    # True channel effect (units: KPI per unit of saturated, adstocked impressions)
    true_coefficient: float = 1.0

    # Spend correlation with another channel (index into scenario channel list)
    correlated_with: int | None = None
    correlation: float = 0.0


@dataclass
class ControlConfig:
    """Configuration for a single CPG control variable."""

    name: str

    # True effect: KPI = ... + true_coefficient * control_value + ...
    true_coefficient: float = 0.0

    # Time-series pattern for the control variable
    pattern: Literal["ar1", "trend", "seasonal", "binary"] = "ar1"

    # AR(1) parameters (for pattern="ar1")
    ar1_rho: float = 0.7
    ar1_std: float = 1.0     # innovation std

    # Trend parameters (for pattern="trend")
    trend_slope: float = 0.01   # per-week trend

    # Seasonal parameters (for pattern="seasonal")
    seasonality_amplitude: float = 1.0   # peak-to-trough half-amplitude

    # Binary / promotion parameters (for pattern="binary")
    binary_rate: float = 0.2   # fraction of weeks the event is active


@dataclass
class Scenario:
    """A complete benchmark scenario definition."""

    name: str
    description: str

    # Time series config
    n_weeks: int = 104           # 2 years default
    seed: int = 42

    # Channels
    channels: list[ChannelConfig] = field(default_factory=list)

    # Control variables (CPG: price, distribution, competitor spend, …)
    controls: list[ControlConfig] = field(default_factory=list)

    # Baseline / trend / seasonality
    baseline: float = 50_000     # Intercept (KPI units, e.g. revenue $)
    trend_slope: float = 0.0     # Weekly trend (0 = no trend)
    seasonality_amplitude: float = 0.1   # Fraction of baseline for seasonal swing
    noise_sigma: float = 2_000   # Gaussian noise std on KPI

    # Budget constraint for optimization metrics
    total_budget: float | None = None   # If None, uses sum of mean spends
