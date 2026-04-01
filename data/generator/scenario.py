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

    # Adstock parameters (geometric)
    adstock_type: Literal["geometric", "delayed"] = "geometric"
    decay: float = 0.4       # Geometric decay rate (0-1). Higher = longer carryover.
    peak_delay: int = 1      # For delayed adstock: week of peak effect

    # Saturation parameters (tanh)
    # true response = tanh(x / (scalar * alpha))
    # scalar is fixed to max activity in data; alpha controls shape
    alpha: float = 1.0       # Saturation shape. Lower = saturates faster.

    # Channel ROI (true coefficient — units of KPI per unit of saturated, adstocked spend)
    true_coefficient: float = 1.0

    # Spend correlation with other channels (index into scenario channel list)
    correlated_with: int | None = None
    correlation: float = 0.0


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

    # Baseline / trend / seasonality
    baseline: float = 50_000     # Intercept (KPI units, e.g. revenue)
    trend_slope: float = 0.0     # Weekly trend (0 = no trend)
    seasonality_amplitude: float = 0.1   # Fraction of baseline for seasonal swing
    noise_sigma: float = 2_000   # Gaussian noise std on KPI

    # Budget constraint for optimization metrics
    total_budget: float | None = None   # If None, uses sum of mean spends
