"""
Meridian runner (Google).

Meridian is Google's open-source Bayesian MMM built on TensorFlow Probability.
https://github.com/google/meridian

Installation: pip install google-meridian
Requires: TensorFlow >= 2.12, TensorFlow Probability

Status: IMPLEMENTED (stub — full integration pending Meridian API stabilisation)
Version tracking: monitors google-meridian on PyPI
"""

from __future__ import annotations

import pandas as pd

from .base import BenchmarkRunner, RunResult

try:
    import meridian
    MERIDIAN_AVAILABLE = True
    _version = meridian.__version__
except ImportError:
    MERIDIAN_AVAILABLE = False
    _version = "not installed"


class MeridianRunner(BenchmarkRunner):

    tool_name = "meridian"

    @property
    def tool_version(self) -> str:
        return _version

    def _run(self, df: pd.DataFrame, channels: list[str], kpi_col: str) -> RunResult:
        if not MERIDIAN_AVAILABLE:
            return RunResult(
                tool_name=self.tool_name,
                tool_version=self.tool_version,
                scenario_name="",
                estimated_rois={ch: None for ch in channels},
                converged=False,
                convergence_warnings=["meridian not installed — pip install google-meridian"],
            )

        # TODO: Full Meridian integration
        # Meridian uses a different data format (xarray-based InputData)
        # and a different model specification API.
        #
        # Key integration points:
        #   from meridian.data import InputData
        #   from meridian.model import Meridian
        #   mmm = Meridian(input_data=input_data)
        #   mmm.fit(...)
        #   roi = mmm.roi(...)
        #
        # See: https://github.com/google/meridian/tree/main/meridian/docs

        raise NotImplementedError(
            "Meridian full integration coming soon. "
            "Track progress: https://github.com/simba-quokka/mmm-bench/issues"
        )
