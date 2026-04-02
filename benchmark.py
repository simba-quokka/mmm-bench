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
from runners import PyMCMarketingRunner, PyMCMarketingTanhRunner, MeridianRunner, DecisionPacksRunner
from metrics import compute_all_metrics

app = typer.Typer(help="MMM benchmark runner")
console = Console()

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")

RUNNERS = {
    "pymc-marketing": PyMCMarketingRunner,
    "pymc-marketing-tanh": PyMCMarketingTanhRunner,
    "meridian": MeridianRunner,
    "decision-packs": DecisionPacksRunner,
}

SCENARIO_NAMES = [
    "simple", "simple_no_controls", "simple_high_seasonality",
    "complex", "data_scarce", "adversarial",
]
HOLDOUT_WEEKS = 13


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
        control_cols = ground_truth.get("control_cols", [])

        console.print(
            f"  {len(df)} weeks · {len(channels)} channels: {', '.join(channels)}"
            + (f" · controls: {', '.join(control_cols)}" if control_cols else "")
        )

        # Split train/test for holdout evaluation
        # ROI ground truth uses full data; holdout only affects fit/predict
        if len(df) > HOLDOUT_WEEKS + 20:  # need enough training data
            df_train = df.iloc[:-HOLDOUT_WEEKS].copy()
            df_test = df.iloc[-HOLDOUT_WEEKS:].copy()
        else:
            df_train = df
            df_test = None

        for tool_name in tools_to_run:
            runner_cls = RUNNERS.get(tool_name)
            if runner_cls is None:
                console.print(f"[yellow]Unknown tool: {tool_name}[/yellow]")
                continue

            runner = runner_cls()
            console.print(f"\n  Running [bold]{tool_name}[/bold] v{runner.tool_version}...")

            try:
                result = runner.run(df_train, channels, control_cols=control_cols, df_test=df_test)
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
        f"    {status} Rel ROI acc: {m['rel_roi_accuracy']:.1%} | "
        f"Abs ROI acc: {m['abs_roi_accuracy']:.1%} | "
        f"Ranking: {m['pairwise_accuracy']:.1%} pairwise | "
        f"Spearman rho: {m['spearman_rho']:.2f} | "
        f"{m['runtime_seconds']:.0f}s"
    )
    _print_per_channel(m)


def _print_per_channel(m: dict):
    """Per-channel ROI breakdown table printed immediately after the summary line."""
    true_rois = m.get("true_rois", {})
    est_rois = m.get("estimated_rois", {})
    per_rel = m.get("per_channel_rel", {})
    per_abs = m.get("per_channel_abs", {})

    if not true_rois:
        return

    # Header
    console.print(
        f"    {'':2}{'Channel':<16} {'True ROI':>9} {'Est ROI':>9} "
        f"{'Abs Err':>9} {'Rel Err':>9}"
    )
    console.print(f"    {'':2}{'-'*54}")

    channels = list(true_rois.keys())
    for ch in channels:
        t = true_rois.get(ch)
        e = est_rois.get(ch)
        abs_err = per_abs.get(ch)
        rel_err = per_rel.get(ch)

        t_str = f"{t:.3f}" if t is not None else "   -"
        e_str = f"{e:.3f}" if e is not None else "   -"

        # colour abs error: green <30%, yellow <75%, red >=75%
        if abs_err is None:
            abs_str = "    -"
        elif abs_err < 0.30:
            abs_str = f"[green]{abs_err:.1%}[/green]"
        elif abs_err < 0.75:
            abs_str = f"[yellow]{abs_err:.1%}[/yellow]"
        else:
            abs_str = f"[red]{abs_err:.1%}[/red]"

        if rel_err is None:
            rel_str = "    -"
        elif rel_err < 0.20:
            rel_str = f"[green]{rel_err:.1%}[/green]"
        elif rel_err < 0.50:
            rel_str = f"[yellow]{rel_err:.1%}[/yellow]"
        else:
            rel_str = f"[red]{rel_err:.1%}[/red]"

        console.print(
            f"    {'':2}{ch:<16} {t_str:>9} {e_str:>9} "
            f"{abs_str:>9} {rel_str:>9}"
        )


def _print_leaderboard(all_metrics: list[dict]):
    if not all_metrics:
        return

    console.rule("[bold]Leaderboard")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Tool")
    table.add_column("Scenario")
    table.add_column("Composite", justify="right")
    table.add_column("Rel ROI", justify="right")
    table.add_column("Holdout", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("Biz Sense", justify="right")
    table.add_column("Fit Idx", justify="right")
    table.add_column("Pairwise", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("Conv.", justify="center")
    table.add_column("Runtime", justify="right")

    sort_key = lambda x: x.get("composite_score") or x["rel_roi_accuracy"]
    for m in sorted(all_metrics, key=sort_key, reverse=True):
        converged = "[green]Y[/green]" if m["converged"] else "[yellow]![/yellow]"
        cs = f"{m['composite_score']:.1%}" if m.get("composite_score") is not None else "-"
        ho = f"{m['holdout_accuracy']:.1%}" if m.get("holdout_accuracy") is not None else "-"
        sh = f"{m['contribution_share_accuracy']:.1%}" if m.get("contribution_share_accuracy") is not None else "-"
        bs = f"{m['business_sense_score']:.1%}" if m.get("business_sense_score") is not None else "-"
        fi = f"{m['fit_index']:.1%}" if m.get("fit_index") is not None else "-"
        table.add_row(
            m["tool"],
            m["scenario"],
            cs,
            f"{m['rel_roi_accuracy']:.1%}",
            ho,
            sh,
            bs,
            fi,
            f"{m['pairwise_accuracy']:.1%}",
            f"{m['spearman_rho']:.2f}",
            converged,
            f"{m['runtime_seconds']:.0f}s",
        )

    console.print(table)

    # Per-channel detail — one narrow table per tool per scenario
    scenarios_seen = dict.fromkeys(m["scenario"] for m in all_metrics)
    for scenario_name in scenarios_seen:
        scenario_metrics = [m for m in all_metrics if m["scenario"] == scenario_name]
        if not any(m.get("true_rois") for m in scenario_metrics):
            continue

        channels = list(next(m["true_rois"] for m in scenario_metrics if m.get("true_rois")).keys())

        for m in scenario_metrics:
            console.rule(f"[bold]{m['tool']}[/bold] | {scenario_name}")

            ch_table = Table(show_header=True, header_style="bold magenta")
            ch_table.add_column("Channel", style="bold")
            ch_table.add_column("True ROI", justify="right")
            ch_table.add_column("Est ROI", justify="right")
            ch_table.add_column("Abs Err", justify="right")
            ch_table.add_column("Rel Err", justify="right")

            for ch in channels:
                true_roi = m["true_rois"].get(ch)
                e = m.get("estimated_rois", {}).get(ch)
                abs_err = m.get("per_channel_abs", {}).get(ch)
                rel_err = m.get("per_channel_rel", {}).get(ch)

                t_str = f"{true_roi:.3f}" if true_roi is not None else "-"
                e_str = f"{e:.3f}" if e is not None else "-"

                if abs_err is None:
                    abs_str = "-"
                elif abs_err < 0.30:
                    abs_str = f"[green]{abs_err:.1%}[/green]"
                elif abs_err < 0.75:
                    abs_str = f"[yellow]{abs_err:.1%}[/yellow]"
                else:
                    abs_str = f"[red]{abs_err:.1%}[/red]"

                if rel_err is None:
                    rel_str = "-"
                elif rel_err < 0.20:
                    rel_str = f"[green]{rel_err:.1%}[/green]"
                elif rel_err < 0.50:
                    rel_str = f"[yellow]{rel_err:.1%}[/yellow]"
                else:
                    rel_str = f"[red]{rel_err:.1%}[/red]"

                ch_table.add_row(ch, t_str, e_str, abs_str, rel_str)

            console.print(ch_table)


def _update_readme(all_metrics: list[dict], run_id: str):
    """Rewrite the leaderboard section of README.md with fresh results."""
    if not README_PATH.exists():
        return

    readme = README_PATH.read_text(encoding="utf-8")
    leaderboard_md = _build_leaderboard_md(all_metrics, run_id)

    start_marker = "<!-- LEADERBOARD_START -->"
    end_marker = "<!-- LEADERBOARD_END -->"

    if start_marker in readme and end_marker in readme:
        before = readme[:readme.index(start_marker) + len(start_marker)]
        after = readme[readme.index(end_marker):]
        readme = before + "\n" + leaderboard_md + "\n" + after
    else:
        readme += "\n" + start_marker + "\n" + leaderboard_md + "\n" + end_marker

    README_PATH.write_text(readme, encoding="utf-8")


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
        lines.append("| Tool | Composite | Rel ROI | Holdout | Share | Biz Sense | Fit Idx | Pairwise | Spearman | Conv | Runtime |")
        lines.append("|------|-----------|---------|---------|-------|-----------|---------|----------|----------|------|---------|")

        sort_key = lambda x: x.get("composite_score") or x["rel_roi_accuracy"]
        for m in sorted(scenario_metrics, key=sort_key, reverse=True):
            converged = "Y" if m["converged"] else "!"
            cs = f"{m['composite_score']:.1%}" if m.get("composite_score") is not None else "-"
            ho = f"{m['holdout_accuracy']:.1%}" if m.get("holdout_accuracy") is not None else "-"
            sh = f"{m['contribution_share_accuracy']:.1%}" if m.get("contribution_share_accuracy") is not None else "-"
            bs = f"{m['business_sense_score']:.1%}" if m.get("business_sense_score") is not None else "-"
            fi = f"{m['fit_index']:.1%}" if m.get("fit_index") is not None else "-"
            cost = f" (~${m['estimated_cost_usd']:.0f})" if m["estimated_cost_usd"] else ""
            lines.append(
                f"| {m['tool']} "
                f"| {cs} "
                f"| {m['rel_roi_accuracy']:.1%} "
                f"| {ho} "
                f"| {sh} "
                f"| {bs} "
                f"| {fi} "
                f"| {m['pairwise_accuracy']:.1%} "
                f"| {m['spearman_rho']:.2f} "
                f"| {converged} "
                f"| {m['runtime_seconds']:.0f}s{cost} |"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    app()
