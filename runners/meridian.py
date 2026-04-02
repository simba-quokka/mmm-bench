"""
Meridian runner (Google, v1.5+).

Meridian is Google's open-source Bayesian MMM built on TensorFlow Probability.
https://github.com/google/meridian

CPG conventions:
- df[ch]           : media activity (impressions / GRPs) → passed as media_cols
- df[f'{ch}_spend']: weekly spend in $               → passed as media_spend_cols
- df[ctrl]         : control variables               → passed as controls (if supported)

ROI definition: Meridian's native Analyzer.roi() computes leave-one-out incremental
outcome divided by total spend — exactly matching our spend-denominated ROI definition.

Single-geo setup: population = 1, revenue_per_kpi = 1 (KPI already in revenue units).
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .base import BenchmarkRunner, RunResult

try:
    import meridian
    from meridian.data.data_frame_input_data_builder import DataFrameInputDataBuilder
    from meridian.model.model import Meridian
    from meridian.analysis.analyzer import Analyzer
    AVAILABLE = True
    _version = meridian.__version__
except ImportError:
    AVAILABLE = False
    _version = "not installed"


class MeridianRunner(BenchmarkRunner):

    tool_name = "meridian"

    @property
    def tool_version(self) -> str:
        return _version

    def _run(
        self,
        df: pd.DataFrame,
        channels: list[str],
        kpi_col: str,
        control_cols: list[str],
        df_test: pd.DataFrame | None = None,
    ) -> RunResult:
        if not AVAILABLE:
            return RunResult(
                tool_name=self.tool_name,
                tool_version=self.tool_version,
                scenario_name="",
                estimated_rois={ch: None for ch in channels},
                converged=False,
                convergence_warnings=["meridian not installed — pip install google-meridian"],
            )

        convergence_warnings = []

        # Spend columns — Meridian natively supports separate impressions and spend
        spend_cols = [f"{ch}_spend" for ch in channels]

        # Build the input data frame — add single-geo scaffolding
        data = df.copy()
        data["geo"] = "total"
        data["population"] = 1.0
        data["revenue_per_kpi"] = 1.0   # KPI already in revenue units

        builder = DataFrameInputDataBuilder(kpi_type="revenue")
        build_chain = (
            builder
            .with_kpi(data, kpi_col=kpi_col, time_col="date", geo_col="geo")
            .with_media(
                data,
                media_cols=channels,           # impressions / GRPs (activity variable)
                media_spend_cols=spend_cols,   # weekly spend in $ (ROI denominator)
                media_channels=channels,
                time_col="date",
                geo_col="geo",
            )
            .with_population(data, population_col="population", geo_col="geo")
            .with_revenue_per_kpi(
                data,
                revenue_per_kpi_col="revenue_per_kpi",
                time_col="date",
                geo_col="geo",
            )
        )

        # Add control variables via with_controls (non-media regressors).
        # Note: with_organic_media is wrong here — it applies adstock/saturation
        # to controls, which inflates their effect and distorts ROI attribution.
        if control_cols:
            try:
                build_chain = build_chain.with_controls(
                    data,
                    control_cols=control_cols,
                    time_col="date",
                    geo_col="geo",
                )
            except Exception as e:
                convergence_warnings.append(
                    f"Controls skipped ({len(control_cols)} cols): {e}"
                )

        input_data = build_chain.build()

        # --- Fit ---
        mmm = Meridian(input_data=input_data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            mmm.sample_prior(n_draws=500, seed=42)
            mmm.sample_posterior(
                n_chains=4,
                n_adapt=1000,
                n_burnin=500,
                n_keep=1000,
                seed=42,
            )

            for w in caught:
                msg = str(w.message).lower()
                if any(k in msg for k in ("divergen", "rhat", "converge")):
                    convergence_warnings.append(str(w.message))

        # --- Check R-hat ---
        analyzer = Analyzer(mmm)
        try:
            rhat_summary = analyzer.rhat_summary()
            if hasattr(rhat_summary, "values"):
                max_rhat = float(np.nanmax(rhat_summary.values))
            else:
                max_rhat = float(max(rhat_summary.values()))
            if max_rhat > 1.05:
                convergence_warnings.append(f"Max R-hat = {max_rhat:.3f} > 1.05")
        except Exception:
            pass

        converged = len(convergence_warnings) == 0

        # --- Extract ROI ---
        # Analyzer.roi() returns tensor of shape (chains, draws, channels)
        # Uses spend as denominator — exactly matches our CPG ROI definition
        estimated_rois = {}
        credible_intervals = {}
        estimated_shares = {}

        try:
            roi_samples = analyzer.roi(use_posterior=True)
            roi_array = np.array(roi_samples)

            # Normalise to (samples, channels)
            if roi_array.ndim == 3:
                roi_flat = roi_array.reshape(-1, roi_array.shape[-1])
            elif roi_array.ndim == 2:
                roi_flat = roi_array
            else:
                roi_flat = roi_array.reshape(1, -1)

            for i, ch in enumerate(channels):
                samples = roi_flat[:, i]
                estimated_rois[ch] = float(np.mean(samples))
                credible_intervals[ch] = (
                    float(np.percentile(samples, 3)),
                    float(np.percentile(samples, 97)),
                )

        except Exception as e:
            convergence_warnings.append(f"ROI extraction failed: {e}")
            estimated_rois = {ch: None for ch in channels}

        valid = {ch: v for ch, v in estimated_rois.items() if v is not None}
        total = sum(valid.values())
        if total > 0:
            estimated_shares = {ch: v / total for ch, v in valid.items()}

        # --- Fitted KPI for in-sample fit index ---
        raw_output = {}
        try:
            outcome = analyzer.expected_outcome(use_posterior=True)
            outcome_arr = np.array(outcome)
            # Shape: (samples, times) or (chains, draws, times)
            if outcome_arr.ndim == 3:
                fitted_kpi = outcome_arr.reshape(-1, outcome_arr.shape[-1]).mean(axis=0)
            elif outcome_arr.ndim == 2:
                fitted_kpi = outcome_arr.mean(axis=0)
            else:
                fitted_kpi = outcome_arr
            raw_output["fitted_kpi"] = fitted_kpi
            raw_output["actual_kpi"] = df[kpi_col].values
        except Exception:
            pass

        # --- Holdout prediction ---
        # Meridian 1.5.x doesn't support predict-on-new-data directly.
        # Holdout MAPE will be None for Meridian until a prediction API is available.
        if df_test is not None:
            raw_output["holdout_mape"] = None

        return RunResult(
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            scenario_name="",
            estimated_rois=estimated_rois,
            estimated_contribution_share=estimated_shares,
            credible_intervals=credible_intervals,
            converged=converged,
            convergence_warnings=convergence_warnings,
            raw_output=raw_output,
        )
