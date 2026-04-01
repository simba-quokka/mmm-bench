"""Abstract base class for all benchmark runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import pandas as pd


@dataclass
class RunResult:
    """
    Standardised output from a benchmark runner.

    Every runner must produce estimated ROIs per channel.
    All other fields are optional enrichment.
    """
    tool_name: str
    tool_version: str
    scenario_name: str

    # Core outputs — required
    estimated_rois: dict[str, float]        # channel -> estimated ROI

    # Optional outputs
    estimated_contribution_share: dict[str, float] = field(default_factory=dict)
    credible_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    converged: bool = True                  # False if sampler failed convergence checks
    convergence_warnings: list[str] = field(default_factory=list)

    # Runtime
    runtime_seconds: float = 0.0
    estimated_cost_usd: float | None = None

    # Raw output for debugging
    raw_output: dict = field(default_factory=dict)


class BenchmarkRunner(ABC):
    """Base class for all MMM tool runners."""

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Human-readable tool name."""

    @property
    @abstractmethod
    def tool_version(self) -> str:
        """Version string of the tool being benchmarked."""

    @abstractmethod
    def _run(self, df: pd.DataFrame, channels: list[str], kpi_col: str) -> RunResult:
        """Run the tool and return results. Implement per-tool logic here."""

    def run(self, df: pd.DataFrame, channels: list[str], kpi_col: str = "kpi") -> RunResult:
        """Timed wrapper around _run."""
        t0 = time.perf_counter()
        result = self._run(df, channels, kpi_col)
        result.runtime_seconds = time.perf_counter() - t0
        return result
