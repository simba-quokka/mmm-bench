"""
Decision-packs runner (PyMC Labs).

Decision-packs is a multi-agent MMM framework that runs multiple structurally
different models in parallel and checks for convergence across models.
https://github.com/pymc-labs/decision-lab/tree/main/decision-packs

Status: STUB — requires Docker and Modal setup.
The decision-packs MMM pack runs on Modal serverless compute and is not
directly invocable as a Python library. This runner shells out to the
decision-lab CLI.

Cost: ~$7 per run on Modal (19 min for 2-year weekly dataset).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .base import BenchmarkRunner, RunResult

# decision-lab CLI — install from https://github.com/pymc-labs/decision-lab
DECISION_LAB_CMD = "decision-lab"

try:
    result = subprocess.run([DECISION_LAB_CMD, "--version"], capture_output=True, text=True, timeout=5)
    DECISION_LAB_AVAILABLE = result.returncode == 0
    _version = result.stdout.strip() or "unknown"
except (FileNotFoundError, subprocess.TimeoutExpired):
    DECISION_LAB_AVAILABLE = False
    _version = "not installed"


class DecisionPacksRunner(BenchmarkRunner):

    tool_name = "decision-packs"

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
        lift_test_df: pd.DataFrame | None = None,
    ) -> RunResult:
        if not DECISION_LAB_AVAILABLE:
            return RunResult(
                tool_name=self.tool_name,
                tool_version=self.tool_version,
                scenario_name="",
                estimated_rois={ch: None for ch in channels},
                converged=False,
                convergence_warnings=[
                    "decision-lab CLI not found. "
                    "Install from https://github.com/pymc-labs/decision-lab"
                ],
            )

        # Write dataset to temp CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            df.to_csv(data_path, index=False)

            # Build prompt for the MMM pack
            channel_list = ", ".join(channels)
            spend_list = ", ".join(f"{ch}_spend" for ch in channels)
            ctrl_list = ", ".join(control_cols) if control_cols else "none"
            prompt = (
                f"Run an MMM analysis on this dataset. "
                f"KPI column: {kpi_col}. "
                f"Media activity columns (impressions/GRPs): {channel_list}. "
                f"Media spend columns ($ spend, ROI denominator): {spend_list}. "
                f"Control variables: {ctrl_list}. "
                f"Return estimated ROI per channel (KPI contribution / spend)."
            )

            # Shell out to decision-lab CLI
            cmd = [
                DECISION_LAB_CMD, "run",
                "--pack", "mmm",
                "--data", str(data_path),
                "--prompt", prompt,
                "--output-format", "json",
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if proc.returncode != 0:
                return RunResult(
                    tool_name=self.tool_name,
                    tool_version=self.tool_version,
                    scenario_name="",
                    estimated_rois={ch: None for ch in channels},
                    converged=False,
                    convergence_warnings=[f"decision-lab exited with code {proc.returncode}: {proc.stderr}"],
                )

            # Parse JSON output
            try:
                output = json.loads(proc.stdout)
                roi_data = output.get("roi", {})
                estimated_rois = {ch: roi_data.get(ch) for ch in channels}
                converged = output.get("converged", True)
                warnings_list = output.get("warnings", [])
            except json.JSONDecodeError:
                estimated_rois = {ch: None for ch in channels}
                converged = False
                warnings_list = [f"Could not parse decision-lab output: {proc.stdout[:500]}"]

        return RunResult(
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            scenario_name="",
            estimated_rois=estimated_rois,
            converged=converged,
            convergence_warnings=warnings_list,
            estimated_cost_usd=7.0,  # approximate Modal cost
            raw_output={"stdout": proc.stdout, "stderr": proc.stderr},
        )
