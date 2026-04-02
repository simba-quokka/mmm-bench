"""
PyMC-Marketing runner (v0.19+).

Follows the recommended best practices from the official migration guide:
https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_migration_guide.html

Key choices:
- LogisticSaturation: the primary saturation function in the docs
- GeometricAdstock(l_max=8): standard carryover
- 4 chains, 1000 draws, 1000 tune, target_accept=0.95
- build_model() + add_original_scale_contribution_variable() before fit()

CPG conventions:
- df[ch]           : media activity (impressions / GRPs) — passed as channel input
- df[f'{ch}_spend']: weekly spend in $ — used as ROI denominator
- df[ctrl]         : control variables — passed via control_columns

ROI = sum(channel_contribution_original_scale) / sum({ch}_spend)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .base import BenchmarkRunner, RunResult

try:
    import pymc_marketing
    from pymc_marketing.mmm.multidimensional import MMM
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    AVAILABLE = True
    _version = pymc_marketing.__version__
except ImportError:
    AVAILABLE = False
    _version = "not installed"


class PyMCMarketingRunner(BenchmarkRunner):

    tool_name = "pymc-marketing"

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
            raise RuntimeError("pip install pymc-marketing")

        convergence_warnings = []

        # channels = impressions activity columns
        # spend columns for ROI: {ch}_spend
        spend_cols = {ch: f"{ch}_spend" for ch in channels}

        # --- Initialise model ---
        mmm_kwargs = dict(
            date_column="date",
            channel_columns=channels,
            target_column=kpi_col,
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
        )
        if control_cols:
            mmm_kwargs["control_columns"] = control_cols

        mmm = MMM(**mmm_kwargs)

        feature_cols = ["date"] + channels + control_cols
        X = df[feature_cols].copy()
        y = df[kpi_col].rename(kpi_col)

        # --- Build model and register original-scale contribution variables ---
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(
            var=["channel_contribution", "intercept_contribution"]
        )

        # --- Fit ---
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mmm.fit(
                X=X,
                y=y,
                progressbar=False,
                random_seed=42,
                draws=1000,
                tune=1000,
                chains=4,
                target_accept=0.95,
            )
            for w in caught:
                msg = str(w.message).lower()
                if any(k in msg for k in ("divergen", "rhat", "converge", "effective")):
                    convergence_warnings.append(str(w.message))

        # --- Check R-hat convergence ---
        try:
            import arviz as az
            rhat = az.rhat(mmm.idata.posterior)
            max_rhat = float(max(float(v.max()) for v in rhat.data_vars.values()))
            if max_rhat > 1.05:
                convergence_warnings.append(f"Max R-hat = {max_rhat:.3f} > 1.05")
        except Exception:
            pass

        converged = len(convergence_warnings) == 0

        # --- Extract contributions and compute ROI (spend-denominated) ---
        estimated_rois = {}
        estimated_shares = {}
        credible_intervals = {}

        try:
            contrib_df = mmm.compute_mean_contributions_over_time()
            num_contrib = contrib_df.select_dtypes(include="number")

            for ch in channels:
                sc = spend_cols[ch]
                total_spend = float(df[sc].sum()) if sc in df.columns else float(df[ch].sum())
                if total_spend == 0:
                    estimated_rois[ch] = 0.0
                    continue
                if ch in num_contrib.columns:
                    estimated_rois[ch] = float(num_contrib[ch].sum()) / total_spend
                else:
                    estimated_rois[ch] = None

        except Exception as e:
            convergence_warnings.append(f"Contribution extraction failed: {e}")
            estimated_rois = {ch: None for ch in channels}

        # --- 94% HDI credible intervals from posterior ---
        try:
            posterior = mmm.idata.posterior
            os_key = "channel_contribution_original_scale"
            if os_key in posterior:
                samples = posterior[os_key]
                for i, ch in enumerate(channels):
                    sc = spend_cols[ch]
                    total_spend = float(df[sc].sum()) if sc in df.columns else float(df[ch].sum())
                    if total_spend == 0:
                        continue
                    ch_total = samples.isel(channel=i).sum(dim="date")
                    flat = ch_total.values.flatten()
                    credible_intervals[ch] = (
                        float(np.percentile(flat, 3)) / total_spend,
                        float(np.percentile(flat, 97)) / total_spend,
                    )
        except Exception:
            pass

        # --- Contribution shares (from actual contribution sums, not ROIs) ---
        try:
            contrib_sums = {}
            for ch in channels:
                if ch in num_contrib.columns:
                    contrib_sums[ch] = float(num_contrib[ch].sum())
            total_contrib = sum(contrib_sums.values())
            if total_contrib > 0:
                estimated_shares = {ch: v / total_contrib for ch, v in contrib_sums.items()}
        except Exception:
            # Fallback: derive from ROIs (less accurate but functional)
            valid = {ch: v for ch, v in estimated_rois.items() if v is not None}
            total = sum(valid.values())
            if total > 0:
                estimated_shares = {ch: v / total for ch, v in valid.items()}

        # --- Fitted KPI for in-sample fit index ---
        raw_output = {"idata": mmm.idata}
        try:
            fitted_kpi = num_contrib.sum(axis=1).values
            raw_output["fitted_kpi"] = fitted_kpi
            raw_output["actual_kpi"] = df[kpi_col].values
        except Exception:
            pass

        # --- Holdout prediction ---
        if df_test is not None:
            try:
                feature_cols_test = ["date"] + channels + control_cols
                X_test = df_test[feature_cols_test].copy()
                ppc = mmm.predict(X_test)
                predicted_kpi = ppc.mean(("chain", "draw")).values
                actual_test = df_test[kpi_col].values
                mask = np.abs(actual_test) > 1e-6
                if mask.sum() > 0:
                    holdout_mape = float(np.mean(
                        np.abs(actual_test[mask] - predicted_kpi[mask]) / np.abs(actual_test[mask])
                    ))
                else:
                    holdout_mape = None
                raw_output["holdout_mape"] = holdout_mape
                raw_output["holdout_actual"] = actual_test
                raw_output["holdout_predicted"] = predicted_kpi
            except Exception as e:
                convergence_warnings.append(f"Holdout prediction failed: {e}")

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
