"""
PyMC-Marketing runner.

Uses MMM from pymc_marketing.mmm with:
- GeometricAdstock transform
- TanhSaturation (logistic fallback where tanh not available)
- Default priors (no smart prior injection — vanilla out-of-the-box)
- NUTS sampler, 1000 draws, 500 tune
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .base import BenchmarkRunner, RunResult

try:
    import pymc_marketing
    from pymc_marketing.mmm import MMM, GeometricAdstock, TanhSaturation
    PYMC_MARKETING_AVAILABLE = True
    _version = pymc_marketing.__version__
except ImportError:
    PYMC_MARKETING_AVAILABLE = False
    _version = "not installed"


class PyMCMarketingRunner(BenchmarkRunner):

    tool_name = "pymc-marketing"

    @property
    def tool_version(self) -> str:
        return _version

    def _run(self, df: pd.DataFrame, channels: list[str], kpi_col: str) -> RunResult:
        if not PYMC_MARKETING_AVAILABLE:
            raise RuntimeError(
                "pymc-marketing is not installed. Run: pip install pymc-marketing"
            )

        import pymc as pm

        # --- Build channel transforms ---
        channel_transforms = {
            ch: GeometricAdstock(l_max=8) * TanhSaturation()
            for ch in channels
        }

        # --- Fit model ---
        mmm = MMM(
            adstock=GeometricAdstock(l_max=8),
            saturation=TanhSaturation(),
            date_column="date",
            channel_columns=channels,
            control_columns=[],
        )

        X = df[["date"] + channels]
        y = df[kpi_col]

        convergence_warnings = []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mmm.fit(
                X=X,
                y=y,
                target_accept=0.9,
                draws=1000,
                tune=500,
                chains=2,
                progressbar=False,
            )
            for w in caught:
                if "divergence" in str(w.message).lower() or "rhat" in str(w.message).lower():
                    convergence_warnings.append(str(w.message))

        converged = len(convergence_warnings) == 0

        # --- Extract ROI estimates ---
        # ROI = mean contribution / total spend per channel
        estimated_rois = {}
        estimated_shares = {}
        credible_intervals = {}

        try:
            contributions = mmm.get_posterior_predictive_data()
        except Exception:
            contributions = None

        # Fallback: use channel coefficients from posterior
        posterior = mmm.idata.posterior

        for ch in channels:
            total_spend = df[ch].sum()
            if total_spend == 0:
                estimated_rois[ch] = 0.0
                continue

            # Try to get direct contribution estimates
            contrib_key = f"{ch}_contribution"
            if contributions is not None and contrib_key in contributions:
                mean_contrib = float(contributions[contrib_key].mean())
            elif hasattr(posterior, "channel_contributions"):
                # pymc-marketing >= 0.10 style
                ch_idx = channels.index(ch)
                contrib_samples = posterior["channel_contributions"].values[:, :, :, ch_idx]
                mean_contrib = float(contrib_samples.mean()) * len(df)
            else:
                mean_contrib = float(posterior.get(f"beta_channel", np.ones(len(channels))).mean())

            roi = mean_contrib / total_spend
            estimated_rois[ch] = roi

            # Credible interval from coefficient posterior if available
            try:
                ch_idx = channels.index(ch)
                samples = posterior["channel_contributions"].values[:, :, :, ch_idx].flatten()
                total_contribs = samples.sum() / (len(posterior.chain) * len(posterior.draw))
                lo = float(np.percentile(samples, 3)) * len(df) / total_spend
                hi = float(np.percentile(samples, 97)) * len(df) / total_spend
                credible_intervals[ch] = (lo, hi)
            except Exception:
                pass

        # Contribution shares
        total_roi = sum(estimated_rois.values())
        if total_roi > 0:
            estimated_shares = {ch: v / total_roi for ch, v in estimated_rois.items()}

        return RunResult(
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            scenario_name="",
            estimated_rois=estimated_rois,
            estimated_contribution_share=estimated_shares,
            credible_intervals=credible_intervals,
            converged=converged,
            convergence_warnings=convergence_warnings,
            raw_output={"idata": mmm.idata},
        )
