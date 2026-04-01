"""
mmm-bench: Open benchmark for Marketing Mix Modeling tools.

Usage:
    python benchmark.py                          # run all scenarios, all tools
    python benchmark.py --scenario simple        # one scenario
    python benchmark.py --tool pymc-marketing    # one tool
    python benchmark.py --update-readme          # update README leaderboard only
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from data.generator import simulate_dataset
from scenarios import load_scenario, load_all_scenarios
from runners import PyMCMarketingRunner, MeridianRunner, DecisionPacksRunner
from metrics import compute_all_metrics

app = typer.Typer(help="MMM benchmark runner")
console = Console()

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")

RUNNERS = {
    "pymc-marketing": PyMCMarketingRunner,
    "meridian": MeridianRunner,
    "decision-packs": DecisionPacksRunner,
}

SCENARIO_NAMES = ["simple", "complex", "data_scarce", "adversarial"]


@app.command()
def run(
    scenario: str = typer.Option(None, help="Scenario to run (default: all)"),
    tool: str = typer.Option(None, help="Tool to benchmark (default: all)"),
    update_readme: bool = typer.Option(False, "--update-readme", help="Update README leaderboard after run"),
    output_dir: Path = typer.Option(RESULTS_DIR, help="Directory to write results JSON"),
):
    """Run the benchmark suite."""
    scenarios_to_run = [scenario] if scenario else SCENARIO_NAMES
    tools_to_run = [tool] if tool else list(RUNNERS.keys())

    RESULTS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir()

    all_metrics = []

    for scenario_name in scenarios_to_run:
        console.rule(f"[bold blue]Scenario: {scenario_name}")

        try:
            sc = load_scenario(scenario_name)
        except FileNotFoundError:
            console.print(f"[red]Scenario '{scenario_name}' not found.[/red]")
            continue

        df, ground_truth = simulate_dataset(sc)
        channels = [ch.name for ch in sc.channels]

        console.print(f"  {len(df)} weeks · {len(channels)} channels: {', '.join(channels)}")

        for tool_name in tools_to_run:
            runner_cls = RUNNERS.get(tool_name)
            if runner_cls is None:
                console.print(f"[yellow]Unknown tool: {tool_name}[/yellow]")
                continue

            runner = runner_cls()
            console.print(f"\n  Running [bold]{tool_name}[/bold] v{runner.tool_version}...")

            try:
                result = runner.run(df, channels)
                result.scenario_name = scenario_name
            except Exception as e:
                console.print(f"  [red]FAILED: {e}[/red]")
                continue

            metrics = compute_all_metrics(result, ground_truth)
            all_metrics.append(metrics)

            # Print quick summary
            _print_result_row(metrics)

        # Save scenario results
        scenario_results_path = run_dir / f"{scenario_name}.json"
        with open(scenario_results_path, "w") as f:
            json.dump(
                [m for m in all_metrics if m["scenario"] == scenario_name],
                f, indent=2, default=str
            )

    # Save full run summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "results": all_metrics
        }, f, indent=2, default=str)

    console.print(f"\n[green]Results written to {run_dir}[/green]")

    # Print leaderboard table
    _print_leaderboard(all_metrics)

    if update_readme:
        _update_readme(all_metrics, run_id)
        console.print(f"[green]README updated.[/green]")

    return all_metrics


def _print_result_row(m: dict):
    status = "[green]OK[/green]" if m["converged"] else "[yellow]WARN[/yellow]"
    console.print(
        f"    {status} ROI accuracy: {m['roi_accuracy']:.1%} | "
        f"Ranking: {m['pairwise_accuracy']:.1%} pairwise | "
        f"Spearman ρ: {m['spearman_rho']:.2f} | "
        f"{m['runtime_seconds']:.0f}s"
    )


def _print_leaderboard(all_metrics: list[dict]):
    if not all_metrics:
        return

    console.rule("[bold]Leaderboard")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Tool")
    table.add_column("Scenario")
    table.add_column("ROI Acc.")
    table.add_column("Pairwise Rank")
    table.add_column("Spearman ρ")
    table.add_column("Converged")
    table.add_column("Runtime")

    for m in all_metrics:
        converged = "[green]✓[/green]" if m["converged"] else "[yellow]⚠[/yellow]"
        table.add_row(
            m["tool"],
            m["scenario"],
            f"{m['roi_accuracy']:.1%}",
            f"{m['pairwise_accuracy']:.1%}",
            f"{m['spearman_rho']:.2f}",
            converged,
            f"{m['runtime_seconds']:.0f}s",
        )

    console.print(table)


def _update_readme(all_metrics: list[dict], run_id: str):
    """Rewrite the leaderboard section of README.md with fresh results."""
    if not README_PATH.exists():
        return

    readme = README_PATH.read_text()
    leaderboard_md = _build_leaderboard_md(all_metrics, run_id)

    start_marker = "<!-- LEADERBOARD_START -->"
    end_marker = "<!-- LEADERBOARD_END -->"

    if start_marker in readme and end_marker in readme:
        before = readme[:readme.index(start_marker) + len(start_marker)]
        after = readme[readme.index(end_marker):]
        readme = before + "\n" + leaderboard_md + "\n" + after
    else:
        readme += "\n" + start_marker + "\n" + leaderboard_md + "\n" + end_marker

    README_PATH.write_text(readme)


def _build_leaderboard_md(all_metrics: list[dict], run_id: str) -> str:
    run_date = run_id[:10]
    lines = [
        f"*Last updated: {run_date} · Run ID: `{run_id}`*",
        "",
    ]

    for scenario_name in SCENARIO_NAMES:
        scenario_metrics = [m for m in all_metrics if m["scenario"] == scenario_name]
        if not scenario_metrics:
            continue

        lines.append(f"### {scenario_name.replace('_', '-').title()}")
        lines.append("")
        lines.append("| Tool | Version | ROI Accuracy | Pairwise Ranking | Spearman ρ | Top-1 Correct | Converged | Runtime |")
        lines.append("|------|---------|-------------|-----------------|------------|---------------|-----------|---------|")

        for m in sorted(scenario_metrics, key=lambda x: x["roi_accuracy"], reverse=True):
            converged = "✓" if m["converged"] else "⚠"
            top1 = "✓" if m["top1_correct"] else "✗"
            cost = f" (~${m['estimated_cost_usd']:.0f})" if m["estimated_cost_usd"] else ""
            lines.append(
                f"| {m['tool']} | {m['version']} "
                f"| {m['roi_accuracy']:.1%} "
                f"| {m['pairwise_accuracy']:.1%} "
                f"| {m['spearman_rho']:.2f} "
                f"| {top1} "
                f"| {converged} "
                f"| {m['runtime_seconds']:.0f}s{cost} |"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    app()
