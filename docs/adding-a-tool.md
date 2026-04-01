# Adding a Tool

This guide explains how to add a new MMM tool to mmm-bench. Read it fully before opening a PR — there are column conventions and ROI definition requirements that are easy to get wrong.

---

## Requirements

Before starting implementation, verify the tool meets these requirements:

| Requirement | Required | Notes |
|-------------|---------|-------|
| pip-installable Python library | Yes | Must be available on PyPI or installable via pip |
| Channel-level contribution or ROI output | Yes | The tool must produce per-channel attribution, not just totals |
| Runs on a standard laptop | Yes | No mandatory cloud compute. GPU optional but must run on CPU. |
| Python 3.11 compatible | Yes | Benchmark environment uses Python 3.11 |
| Open source or freely available | Strongly preferred | Closed-source tools with free tiers are acceptable if reproducible |
| Accepts impressions/GRPs as activity input | Preferred | Tools that only accept spend can still run — see below |

If the tool requires cloud compute, it can still be included if the runner handles credential setup cleanly and the compute cost is reasonable (e.g., Decision-packs via Modal). Document this in your runner's docstring.

---

## Step 1: Read the Base Class

All runners implement the `BenchmarkRunner` abstract base class defined in `runners/base.py`. Read this file before writing your runner:

```python
# runners/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class RunResult:
    """Results from a single tool run on a single scenario."""

    tool_name: str
    tool_version: str
    scenario_name: str

    # Core outputs — required
    estimated_rois: dict[str, float]
    converged: bool

    # Optional enrichment — include what your tool provides
    estimated_contribution_share: Optional[dict[str, float]] = None
    credible_intervals: Optional[dict[str, tuple[float, float]]] = None
    convergence_warnings: list[str] = field(default_factory=list)
    runtime_seconds: Optional[float] = None

    # Model diagnostics — include if your tool provides them
    r_hat_max: Optional[float] = None
    n_divergences: Optional[int] = None
    ess_min: Optional[float] = None

    # Fit metrics — computed by benchmark.py from holdout predictions
    # Do not set these in your runner; they are computed externally
    holdout_mape: Optional[float] = None
    in_sample_r2: Optional[float] = None

    def validate(self) -> None:
        """Called by benchmark.py after your _run() returns. Raises ValueError if invalid."""
        if not self.estimated_rois:
            raise ValueError("estimated_rois must not be empty")
        for ch, roi in self.estimated_rois.items():
            if roi < 0:
                raise ValueError(f"ROI for {ch} is negative ({roi}). ROIs must be >= 0.")
            if roi > 50:
                raise ValueError(f"ROI for {ch} is implausibly large ({roi}). Check denominator.")


class BenchmarkRunner(ABC):
    """Abstract base class for all tool runners."""

    tool_name: str  # Set as class attribute

    @property
    @abstractmethod
    def tool_version(self) -> str:
        """Return the installed version string of the tool."""
        ...

    @abstractmethod
    def _run(
        self,
        df: pd.DataFrame,
        channels: list[str],
        kpi_col: str,
        control_cols: list[str],
        ground_truth,  # GroundTruth dataclass — available for logging only, not for fitting
    ) -> RunResult:
        """
        Fit the tool and return a RunResult.

        Args:
            df: The full scenario DataFrame (training + holdout weeks).
                Columns include:
                  - df[ch]             Weekly impressions/GRPs for channel ch
                  - df[f'{ch}_spend']  Weekly spend in $ for channel ch
                  - df[ctrl]           Control variable values
                  - df[kpi_col]        Weekly KPI (revenue in $)
                  - df['week']         Integer week index (1-indexed)
                  - df['date']         datetime column
                The last 13 rows are the holdout period. Your runner should
                fit on df[:-13] and not look at df[-13:].

            channels: List of channel names matching df column names.

            kpi_col: Name of the KPI column.

            control_cols: List of control variable column names.

            ground_truth: Available for logging and debugging ONLY.
                          Do not use ground truth parameters in your fitting code.

        Returns:
            RunResult with estimated_rois populated. All other fields are optional
            but including them enables richer benchmark analysis.
        """
        ...

    def run(self, df, channels, kpi_col, control_cols, ground_truth) -> RunResult:
        """Public entry point — wraps _run with timing and validation."""
        import time
        t0 = time.time()
        result = self._run(df, channels, kpi_col, control_cols, ground_truth)
        result.runtime_seconds = time.time() - t0
        result.validate()
        return result
```

---

## Step 2: Understand the DataFrame Columns

**This is the most important section.** Getting the column conventions wrong produces silently wrong ROIs.

### Column naming

```python
# For a channel named 'paid_search':

df['paid_search']        # Weekly impressions or GRPs — the ACTIVITY variable
                         # Use this as input to adstock and saturation

df['paid_search_spend']  # Weekly spend in $ — the ROI DENOMINATOR
                         # Use this ONLY for computing ROI, not as an activity input
```

### Why this matters

The benchmark generates impressions separately from spend (via a noisy CPM process). This reflects industry practice:
- Media planners buy GRPs and reach, not raw dollars
- Spend varies based on CPM fluctuations independent of actual audience delivery
- Tools like Meridian are designed to receive impressions/reach as the primary input

If your tool only accepts spend (not impressions), use `df[f'{ch}_spend']` as the activity variable — but document this clearly in your runner's docstring, because it means the tool is not using the impressions signal.

### ROI computation

**Always use spend as the ROI denominator.** This is not optional:

```python
# Correct — spend-denominated ROI
estimated_rois = {
    ch: channel_contribution[ch].sum() / df[f'{ch}_spend'].sum()
    for ch in channels
}

# WRONG — impressions-denominated ROI (produces incomparable numbers)
estimated_rois = {
    ch: channel_contribution[ch].sum() / df[ch].sum()  # DO NOT DO THIS
    for ch in channels
}

# WRONG — using in-sample contribution only when holdout is in df
# Use only training data contribution (df[:-13]) over total spend (df)
# OR use full period consistently — but be explicit in your docstring
```

**Important:** The `estimated_rois` you return are the values compared against `ground_truth.true_rois`. The ground truth ROIs are computed as:

```python
true_ROI(ch) = sum_t(contribution_t(ch)) / sum_t(spend_t(ch))
```

where `contribution_t(ch)` includes adstock carryover across the full scenario. Match this definition as closely as your tool's output allows.

### Holdout handling

The benchmark fits on `df[:-13]` and evaluates holdout on `df[-13:]`. Your runner **must not fit on the holdout rows.** The convention:

```python
train_df = df.iloc[:-13]  # fit on this
# holdout = df.iloc[-13:]  # do NOT use this in fitting

# But compute ROI over the full period (including holdout carryover effects)
# if your tool naturally produces a contribution series over all rows
```

If your tool only produces contributions for the training period, compute ROI over `df[:-13]`:

```python
estimated_rois = {
    ch: contribution_train[ch].sum() / df.iloc[:-13][f'{ch}_spend'].sum()
    for ch in channels
}
```

Be consistent and document your choice.

---

## Step 3: Create runners/your_tool.py

```python
# runners/your_tool.py

"""
Runner for YourTool — brief description of the tool and version tested.

Notes on activity variable handling:
  - df[ch] (impressions) is passed as the activity variable to YourTool's adstock.
  - df[f'{ch}_spend'] is used ONLY for ROI computation.

Notes on holdout:
  - Training window: df[:-13]
  - ROI computed over training window only.

Known limitations:
  - YourTool requires at least 52 weeks of data (will fail on data_scarce scenario
    if below 52 weeks — raises YourToolDataError, caught and returned as converged=False).
"""

from __future__ import annotations
import pandas as pd
from runners.base import BenchmarkRunner, RunResult


class YourToolRunner(BenchmarkRunner):
    tool_name = "your-tool"

    @property
    def tool_version(self) -> str:
        import your_tool
        return your_tool.__version__

    def _run(
        self,
        df: pd.DataFrame,
        channels: list[str],
        kpi_col: str,
        control_cols: list[str],
        ground_truth,
    ) -> RunResult:

        train_df = df.iloc[:-13].copy()

        # --- Build your model ---
        # Use df[ch] as the impressions/activity variable
        # Use df[kpi_col] as the target
        # Use df[ctrl] as control variables

        try:
            model = your_tool.MMM(
                # ... your tool's configuration ...
            )
            model.fit(
                media=train_df[[ch for ch in channels]].values,       # impressions
                media_spend=train_df[[f'{ch}_spend' for ch in channels]].values,
                target=train_df[kpi_col].values,
                extra_features=train_df[control_cols].values if control_cols else None,
                # ... any other required arguments ...
            )
            converged = True
            convergence_warnings = []

        except Exception as e:
            # Return a failed result rather than crashing the benchmark
            return RunResult(
                tool_name=self.tool_name,
                tool_version=self.tool_version,
                scenario_name="",
                estimated_rois={ch: 0.0 for ch in channels},
                converged=False,
                convergence_warnings=[f"Exception during fitting: {str(e)}"],
            )

        # --- Extract channel-level contributions ---
        # Your tool should provide these; exact API depends on the library
        contributions = model.get_contributions()  # shape: (n_train_weeks, n_channels)

        # --- Compute spend-denominated ROI ---
        # Denominator: total spend over training period
        estimated_rois = {
            ch: float(contributions[:, i].sum()) / float(train_df[f'{ch}_spend'].sum())
            for i, ch in enumerate(channels)
        }

        # --- Optional: extract credible intervals ---
        # If your tool provides posterior samples, compute the 3rd and 97th percentiles
        # to match the 94% HDI convention used by Simba/PyMC-Marketing
        credible_intervals = None
        try:
            samples = model.get_roi_samples()  # shape: (n_samples, n_channels)
            credible_intervals = {
                ch: (
                    float(samples[:, i].quantile(0.03)),
                    float(samples[:, i].quantile(0.97)),
                )
                for i, ch in enumerate(channels)
            }
        except AttributeError:
            pass  # Tool doesn't provide posterior samples

        # --- Optional: extract contribution shares ---
        total_media_contribution = sum(contributions[:, i].sum() for i in range(len(channels)))
        estimated_contribution_share = {
            ch: float(contributions[:, i].sum()) / total_media_contribution
            for i, ch in enumerate(channels)
        } if total_media_contribution > 0 else None

        # --- Optional: extract convergence diagnostics ---
        r_hat_max = None
        n_divergences = None
        try:
            diagnostics = model.get_diagnostics()
            r_hat_max = float(diagnostics['r_hat'].max())
            n_divergences = int(diagnostics['divergences'])
            if r_hat_max > 1.05:
                convergence_warnings.append(f"R-hat max: {r_hat_max:.3f}")
            if n_divergences > 0:
                convergence_warnings.append(f"Divergences: {n_divergences}")
        except (AttributeError, KeyError):
            pass

        return RunResult(
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            scenario_name="",  # set by benchmark.py
            estimated_rois=estimated_rois,
            estimated_contribution_share=estimated_contribution_share,
            credible_intervals=credible_intervals,
            converged=converged,
            convergence_warnings=convergence_warnings,
            r_hat_max=r_hat_max,
            n_divergences=n_divergences,
        )
```

---

## Step 4: Register in benchmark.py

Add your runner to the `RUNNERS` dictionary:

```python
# benchmark.py

from runners.pymc_marketing import PyMCMarketingRunnerLogistic, PyMCMarketingRunnerTanh
from runners.meridian import MeridianRunner
from runners.decision_packs import DecisionPacksRunner
from runners.your_tool import YourToolRunner  # ADD THIS

RUNNERS: dict[str, type[BenchmarkRunner]] = {
    "pymc-marketing-logistic": PyMCMarketingRunnerLogistic,
    "pymc-marketing-tanh": PyMCMarketingRunnerTanh,
    "meridian": MeridianRunner,
    "decision-packs": DecisionPacksRunner,
    "your-tool": YourToolRunner,  # ADD THIS
}
```

The key (`"your-tool"`) is the value used with `--tool your-tool` on the CLI.

---

## Step 5: Add to requirements.txt

```
# Add your tool's package with a minimum version pin
your-tool>=X.Y.Z
```

Pin the minimum version to the version you tested. Use `>=` rather than `==` so that the benchmark automatically tests new versions as they ship.

---

## Step 6: Add to the tools table in README.md

Open `README.md` and add your tool to the "Tools Benchmarked" table:

```markdown
| [YourTool](https://github.com/org/your-tool) | Your Org | Backend | Brief description |
```

---

## Step 7: Test Your Runner

```bash
# Activate your venv
.venv\Scripts\activate

# Smoke test — import only
python -c "from runners.your_tool import YourToolRunner; r = YourToolRunner(); print(r.tool_version)"

# Single scenario run (~5-15 min depending on tool)
python benchmark.py --scenario simple --tool your-tool

# If it completes, check the output table for sensible ROIs
# Then run on a second scenario
python benchmark.py --scenario complex --tool your-tool

# Run the unit tests (data generation and metrics only — does not test your runner)
pytest tests/ -v
```

**Expected output for the simple scenario:**

Your runner should produce:
- Three ROIs in the roughly correct order (search > social > TV)
- Holdout MAPE < 25% (anything higher suggests a fitting problem)
- No `RunResult.validate()` failures (negative ROIs, implausibly large ROIs)

If you get `converged=False` consistently, check the exception message in `convergence_warnings`.

---

## Handling Tools That Don't Separate Impressions from Spend

Some tools are designed to receive only spend as input — they do not model the impressions/CPM layer. This is a valid (if less precisely specified) approach. You can still benchmark it:

```python
def _run(self, df, channels, kpi_col, control_cols, ground_truth):
    train_df = df.iloc[:-13].copy()

    # Use spend as the activity variable instead of impressions
    # Document this clearly — it means the tool is using a different
    # input than the benchmark's intended design
    media_data = train_df[[f'{ch}_spend' for ch in channels]].values
    # Rename to match channel names for tool API compatibility
    media_data_df = pd.DataFrame(media_data, columns=channels)

    model = YourSpendOnlyTool(...)
    model.fit(media=media_data_df, target=train_df[kpi_col], ...)

    # ROI denominator is still spend (same as always)
    contributions = model.get_contributions()
    estimated_rois = {
        ch: float(contributions[:, i].sum()) / float(train_df[f'{ch}_spend'].sum())
        for i, ch in enumerate(channels)
    }

    return RunResult(
        ...,
        convergence_warnings=["NOTE: This runner uses spend as activity variable, not impressions."],
    )
```

The benchmark will include the tool's results in the leaderboard with a footnote explaining the activity variable difference.

---

## Submitting a PR

Open a PR against `main` with:

1. `runners/your_tool.py` — the runner implementation
2. `requirements.txt` — updated with your tool's dependency
3. `README.md` — updated tools table
4. Result outputs from at least two scenarios (paste the per-channel tables in the PR description)

**PR description template:**

```markdown
## Adding [YourTool vX.Y.Z]

### Tool summary
[2-3 sentences about the tool, who maintains it, and what's distinctive about its approach]

### Activity variable handling
- [x] Accepts impressions/GRPs as activity variable
- [ ] Uses spend as activity variable (document why)

### Results

**simple scenario:**
[paste per-channel output table]

**complex scenario:**
[paste per-channel output table]

### Known limitations
[Any scenarios that fail, any platform limitations, any cloud compute requirements]

### Codebase files checked
- runners/base.py — read and understood the ABC
- runners/pymc_marketing.py — used as implementation reference
```

A Quokka maintainer will review and run the PR on the full benchmark suite before merging. The leaderboard will update automatically after merge.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using impressions as ROI denominator | ROIs 100-1000× too small | Change denominator to `df[f'{ch}_spend'].sum()` |
| Fitting on full df (including holdout) | Holdout MAPE = 0% or suspiciously low | Use `df.iloc[:-13]` for fitting |
| Not handling fitting exceptions | Benchmark crashes mid-run | Wrap `_run` in try/except, return `converged=False` |
| Returning negative ROIs | `RunResult.validate()` raises | Check that contributions are non-negative; some tools can return negative coefficients |
| Not documenting activity variable | Confusing benchmark comparisons | Add docstring explaining whether impressions or spend was used |
| Using ground truth in fitting | Inflated accuracy that doesn't generalise | `ground_truth` argument is for logging/debugging only — never use in fitting |

---

*See also:*
- *[Running locally](running-locally.md) — environment setup and CLI reference*
- *[Methodology](methodology.md) — ROI definition and metric design*
- *[Data-generating process](data-generating-process.md) — column conventions and DGP specification*
