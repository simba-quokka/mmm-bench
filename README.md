# mmm-bench

**The open benchmark for Marketing Mix Modeling tools.**

Synthetic datasets with known ground-truth parameters. Each tool gets the same data. Results are updated automatically when new versions ship.

Maintained by [Quokka](https://github.com/simba-quokka) — an autonomous agent that monitors PyPI for new releases and re-runs the benchmark suite when tools update.

---

## Tools benchmarked

| Tool | Who | Language | Approach |
|------|-----|----------|----------|
| [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) | PyMC Labs | Python | Bayesian MMM, NUTS sampler |
| [Meridian](https://github.com/google/meridian) | Google | Python/JAX | Hierarchical Bayesian, HMC |
| [Decision-packs MMM](https://github.com/pymc-labs/decision-lab/tree/main/decision-packs) | PyMC Labs | Python (multi-agent) | Parallel models, consensus checking |

> LightweightMMM (Google) and Robyn (Meta) are no longer actively maintained and are excluded.

---

## Results

<!-- LEADERBOARD_START -->
*No results yet. Run `python benchmark.py` to generate.*
<!-- LEADERBOARD_END -->

---

## How it works

### Ground truth

Each benchmark scenario is a synthetic dataset generated with **known true parameters**: decay rates per channel, saturation shape, and channel ROI. The data generating process is:

```
spend → adstock (geometric or delayed) → tanh saturation → × coefficient → sum → + baseline + seasonality + noise
```

Because we control the parameters, we can measure exactly how close each tool gets.

### Scenarios

| Scenario | Weeks | Channels | Difficulty | Tests |
|----------|-------|----------|-----------|-------|
| **simple** | 104 | 3 | Easy | Baseline accuracy, all tools should pass |
| **complex** | 156 | 8 | Medium | Correlated spend, mixed adstock types |
| **data-scarce** | 78 | 4 | Medium | Short history, new channel, prior handling |
| **adversarial** | 130 | 4 | Hard | High noise, high multicollinearity — tests epistemic honesty |

### Metrics

| Metric | Description |
|--------|-------------|
| **ROI Accuracy** | `1 - MAPE` across channels. 100% = perfect ROI recovery. |
| **Pairwise Ranking** | Fraction of channel pairs ranked in the correct order. |
| **Spearman ρ** | Rank correlation between estimated and true ROIs. 1.0 = perfect. |
| **Top-1 Correct** | Did the tool identify the highest-ROI channel? |
| **Converged** | Did the sampler converge cleanly (no divergences, R-hat < 1.01)? |
| **Runtime** | Wall-clock time in seconds. |

### What "Accuracy" means on the adversarial scenario

The adversarial scenario has high multicollinearity and noise where no tool can reliably recover true ROIs. Here, **lower accuracy with a wide credible interval is the correct answer** — a tool that says "I don't know" is doing better than one that confidently returns wrong ROIs.

Decision-packs explicitly implements this: its consensus checker can return "models disagree, results unreliable" rather than forcing a recommendation.

---

## Running it yourself

```bash
git clone https://github.com/simba-quokka/mmm-bench
cd mmm-bench
pip install -r requirements.txt

# Run all scenarios, all tools
python benchmark.py

# One scenario, one tool
python benchmark.py --scenario simple --tool pymc-marketing

# Update README leaderboard after run
python benchmark.py --update-readme
```

### Adding a new tool

1. Create `runners/your_tool.py` implementing `BenchmarkRunner`
2. Add it to `RUNNERS` in `benchmark.py`
3. Submit a PR

---

## Contributing

PRs welcome for:
- New tool runners
- New scenarios (different industries, panel data, daily granularity)
- Metric improvements
- Bug fixes

Open an issue to propose a new scenario or metric before building.

---

## Methodology notes

- All scenarios use a fixed random seed for reproducibility
- Spend is simulated from Normal distributions with realistic means/std
- Saturation uses `tanh(x / (scalar × α))` matching PyMC-Marketing's implementation
- True ROI = total KPI contribution / total spend (unitless)
- Benchmarks run with default hyperparameters for each tool — no tuning
- PyMC-Marketing: 1000 draws, 500 tune, 2 chains, target_accept=0.9

---

*Maintained by [Quokka](https://github.com/simba-quokka) · Built on [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) · Part of the [Simba](https://simba-mmm.com) ecosystem*
