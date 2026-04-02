# mmm-bench

**The first AI-blind benchmark for Marketing Mix Modeling tools.**

An autonomous agent monitors PyPI for new MMM package releases, reads the latest documentation, examines synthetic CPG data *without any knowledge of how it was generated*, and builds the best model it can using each tool's API. The resulting ROI estimates are scored against ground truth that only the benchmark harness knows.

This makes mmm-bench a dual benchmark: it measures both the **modeling tool** and the **AI agent's ability to use it** — how well [Claude Opus 4.6](https://docs.anthropic.com/en/docs/about-claude/models) can navigate documentation, understand data patterns, select features, configure priors, and produce accurate marketing ROI estimates under realistic conditions.

---

## How the blind benchmark works

```
                                    ┌─────────────────────┐
                                    │   PyPI monitoring    │
                                    │   (new release?)     │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  Read latest docs &  │
                                    │  changelog for tool  │
                                    └──────────┬──────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
          ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
          │  PyMC-Marketing │      │    Meridian      │      │ Decision-packs  │
          │  (Logistic)     │      │    (Google)      │      │  (Multi-agent)  │
          └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
                   │                        │                        │
                   │    Agent reads docs,   │                        │
                   │    examines data,      │                        │
                   │    builds best model   │                        │
                   │    it can — NO access  │                        │
                   │    to true parameters  │                        │
                   ▼                        ▼                        ▼
          ┌─────────────────────────────────────────────────────────────┐
          │              Estimated ROIs per channel                     │
          └──────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
          ┌─────────────────────────────────────────────────────────────┐
          │    Score against ground truth (known only to harness)       │
          │    Composite score, ranking, holdout, business sense...     │
          └──────────────────────────────────────────────────────────────┘
```

### What "blind" means

The agent operating each tool sees:

- The **raw dataset** — weekly KPI, media impressions, spend, and control variables
- The **tool's documentation** — latest API reference, tutorials, and migration guides
- The **tool's changelog** — what changed in this release

The agent does **not** see:

- The data generating process (adstock form, saturation function, true coefficients)
- The true ROI of any channel
- The true contribution shares or ranking
- How noise, seasonality, or controls were simulated
- Any ground truth whatsoever

This mirrors real-world conditions exactly. A practitioner using an MMM tool has data and documentation — never ground truth. The benchmark measures what you'd actually get if you handed the data to an expert AI agent and said "build me the best MMM you can."

### What this benchmarks

**The tool:** Does PyMC-Marketing recover ROIs more accurately than Meridian? Does TanhSaturation outperform LogisticSaturation? Do control variables actually help?

**The AI agent:** Can Claude Opus 4.6 correctly interpret the documentation for each tool? Does it choose appropriate priors? Does it handle always-on vs flighted channels differently? Does it configure adstock and saturation correctly given the data patterns it observes?

**The combination:** A powerful tool with confusing documentation may score worse than a simpler tool with clear APIs. A tool that exposes more configuration surface area only helps if the agent can navigate it correctly. The benchmark captures this interaction — which is exactly what matters for AI-assisted marketing analytics.

---

## The agent: Quokka

[Quokka](https://github.com/simba-quokka) is an autonomous agent built on [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Anthropic's agentic coding tool) powered by Claude Opus 4.6.

**Monitoring loop:**
1. Watches PyPI for new releases of benchmarked packages (pymc-marketing, google-meridian, etc.)
2. On new release: pulls latest documentation, reads changelog, updates its understanding
3. Generates synthetic data from scenario definitions (agent never sees the generation code)
4. For each tool: reads docs, examines the dataset, writes and executes the modeling code
5. Scores results against ground truth, updates the leaderboard, opens a PR if results change

**Why an AI agent, not a static script?** Static benchmark runners freeze a single "best practice" configuration at the time of writing. When PyMC-Marketing ships a new API or Meridian adds a feature, a static runner doesn't adapt. Quokka re-reads the documentation every time and builds fresh — capturing whether new features actually improve real-world outcomes, and whether the documentation is clear enough for an expert user (human or AI) to use them correctly.

---

## Why this exists

MMM tools make claims about marketing ROI that directly inform multi-million-dollar budget decisions. But there's no standard way to evaluate whether those claims are correct, because real-world ground truth is never fully known.

mmm-bench solves this with synthetic data where the true ROI of every channel is known exactly. The data generating process is realistic — CPG-style weekly data with adstock carryover, saturation, seasonality, trend, control variables, and noise — but because the harness controls every parameter, it can measure exactly how close each tool (and agent) gets.

The goal is not to crown a winner. It's to give practitioners an honest, reproducible signal about where each tool excels, where it struggles, and how much to trust its outputs in different conditions.

---

## Results

<!-- LEADERBOARD_START -->
*No results yet. Run `python benchmark.py` to generate.*
<!-- LEADERBOARD_END -->

---

## Tools benchmarked

| Tool | Maintainer | Approach | Status |
|------|-----------|----------|--------|
| [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) | PyMC Labs | Bayesian MMM, NUTS sampler, LogisticSaturation | Active |
| [PyMC-Marketing (Tanh)](https://github.com/pymc-labs/pymc-marketing) | PyMC Labs | Same as above but TanhSaturation (matches DGP) | Active |
| [Meridian](https://github.com/google/meridian) | Google | Hierarchical Bayesian, HMC via TFP | Active |
| [Decision-packs MMM](https://github.com/pymc-labs/decision-lab/tree/main/decision-packs) | PyMC Labs | Multi-agent parallel models, consensus checking | Stub |

> LightweightMMM (Google) and Robyn (Meta) are no longer actively maintained and are excluded.

**Adding a new tool:** Create `runners/your_tool.py` implementing the `BenchmarkRunner` ABC, register it in `benchmark.py`, and submit a PR. See [docs/adding-a-tool.md](docs/adding-a-tool.md) for the full guide.

---

## Data generating process

Every scenario follows the same pipeline. The DGP is intentionally transparent in the codebase — but the agent building each model never sees it. Only the benchmark harness uses the true parameters for scoring.

```
Weekly spend (flighted / always-on / seasonal-burst)
  → Impressions (spend / CPM with weekly noise)
    → Adstock (geometric or delayed/Weibull-like)
      → Saturation (tanh)
        → × true_coefficient = channel contribution

KPI = baseline + trend + seasonality + Σ channel_contributions + Σ control_effects + noise
```

**Key design choices:**

- **Spend and impressions are separate columns.** `df[ch]` = impressions (the activity variable, input to adstock/saturation). `df[f'{ch}_spend']` = weekly spend in $ (always the ROI denominator, never impressions).
- **ROI = total contribution / total spend** — spend-denominated, cumulative over all weeks, includes carryover via adstock. This matches how practitioners define ROI.
- **Controls are realistic.** Price index (AR1), distribution/ACV (trend), competitor spend (AR1), temperature (seasonal), trade promotions (binary). They create confounders that the model must account for — without them, always-on channels absorb the signal.
- **Correlated channels test multicollinearity.** TV and OOH often share campaign timing. Search and social budgets shift together. The scenarios encode this with explicit correlation coefficients.
- **Noise is calibrated.** Simple scenario: $12k sigma on $500k baseline (2.4%). Adversarial: $60k on $800k (7.5%). This spans the range practitioners see in real CPG data.

Full specification: [docs/data-generating-process.md](docs/data-generating-process.md)

---

## Scenarios

### simple — the baseline

**104 weeks, 3 channels, 3 controls.** TV is flighted (on ~55% of weeks), paid search and paid social are always-on. Low noise, no multicollinearity, standard CPG controls.

| Channel | Pattern | Spend/wk | True ROI |
|---------|---------|----------|----------|
| TV | Flighted | $80k | 0.28 |
| Paid Search | Always-on | $30k | 1.35 |
| Paid Social | Always-on | $25k | 0.75 |

**What it tests:** Can the tool recover ROIs at all? TV should be easiest (flighting creates natural before/after contrast). Always-on channels are harder — the model must separate their steady contribution from the structural baseline. For the agent: does it correctly configure adstock length and saturation for each channel based on the data patterns it observes?

**Good performance:** >70% relative ROI accuracy, 100% pairwise ranking, Spearman > 0.9.

### complex — real-world scale

**156 weeks, 8 channels, 5 controls.** TV correlated with OOH (r=0.65), search correlated with social (r=0.55). Email has tiny budget but highest ROI (2.50). YouTube uses delayed adstock (peak at week 1). Affiliates have near-zero decay.

| Channel | Pattern | Spend/wk | True ROI | Notes |
|---------|---------|----------|----------|-------|
| TV | Flighted | $120k | 0.22 | Delayed adstock, peak at week 2 |
| OOH | Flighted | $40k | 0.15 | Correlated with TV (r=0.65) |
| Paid Search | Always-on | $50k | 1.40 | |
| Paid Social | Always-on | $45k | 0.90 | Correlated with search (r=0.55) |
| Display | Always-on | $20k | 0.45 | |
| Email | Always-on | $5k | 2.50 | Tiny budget, high ROI |
| YouTube | Always-on | $35k | 0.65 | Delayed adstock, peak at week 1 |
| Affiliates | Always-on | $15k | 2.00 | Near-instant decay |

**What it tests:** Can the tool find the high-efficiency small channels (email, affiliates) or does it get distracted by the big spenders? Can it separate TV from OOH despite correlation? For the agent: does it recognise from the data that YouTube has delayed response? Does it handle 8 channels without running into convergence issues?

**Good performance:** >60% relative ROI accuracy, top-1 correct on email.

### data_scarce — limited history

**78 weeks, 4 channels.** TikTok is a new channel with shorter, sparser data. Only 1.5 years of history total.

**What it tests:** Prior quality. Tools with informative default priors for new channels should outperform. For the agent: does it recognise that TikTok has limited data and set appropriately wider priors? Does it reduce model complexity to match the data budget?

### adversarial — epistemic honesty

**130 weeks, 4 channels. High noise ($60k sigma), high multicollinearity (TV/OOH r=0.85, search/social r=0.75), TV spend barely varies (std=$8k vs mean=$100k).** Slight downward trend simulating brand erosion.

| Channel | Pattern | True ROI | Identification challenge |
|---------|---------|----------|------------------------|
| TV | Flighted | 0.20 | Spend CV = 8% — nearly flat, very hard to identify |
| OOH | Flighted | 0.13 | Highly correlated with TV (r=0.85) |
| Paid Search | Always-on | 1.15 | Correlated with social (r=0.75) |
| Paid Social | Always-on | 0.72 | Correlated with search (r=0.75) |

**What it tests:** The correct answer here is wide credible intervals and acknowledged uncertainty. A tool that says "I'm not sure" is doing better than one that confidently returns wrong ROIs. For the agent: does it detect the multicollinearity and flat spend from the data alone? Does it warn that results are unreliable? Decision-packs explicitly implements this — its consensus checker can return "models disagree, results unreliable" rather than forcing a recommendation.

**Good performance:** Wide CIs, Spearman > 0 (at least directionally correct), honest convergence warnings. High point accuracy here is likely a red flag, not a virtue.

### Ablation scenarios

| Scenario | Based on | Change | Tests |
|----------|----------|--------|-------|
| **simple_no_controls** | simple | Controls removed entirely | How much control variables help identify always-on ROIs |
| **simple_high_seasonality** | simple | 3.5x seasonal amplitude | Whether always-on channels absorb seasonal peaks |

Full scenario specifications: [docs/scenarios.md](docs/scenarios.md)

---

## Metrics

### Primary: Composite Score

A weighted combination of all metrics, used for leaderboard ranking:

```
composite = 0.30 × rel_roi_accuracy
          + 0.20 × holdout_accuracy
          + 0.20 × contribution_share_accuracy
          + 0.15 × business_sense_score
          + 0.15 × fit_index
```

Components that are unavailable (e.g. holdout for Meridian) are excluded and their weight redistributed.

### Component metrics

| Metric | Range | What it measures |
|--------|-------|-----------------|
| **Relative ROI Accuracy** | 0-100% | Mean-normalised MAPE across channels. Both true and estimated ROIs are divided by their own mean before comparing. Fair across tools with different saturation forms — captures whether the tool gets the *relative efficiency* right, which is what drives budget allocation. **Primary metric.** |
| **Holdout Accuracy** | 0-100% | `1 - MAPE` on last 13 held-out weeks. Train on weeks 1 to N-13, predict weeks N-12 to N. The most trusted practitioner metric for model quality. |
| **Contribution Share Accuracy** | 0-100% | MAPE on the percentage share of total media contribution per channel. More robust than absolute ROI — shares drive budget allocation. |
| **Business Sense Score** | 0-100% | Fraction of channels whose estimated ROI falls within plausible industry ranges (e.g. TV: 0.05-0.80, paid search: 0.30-4.00). Catches tools that converge cleanly but produce nonsensical estimates. |
| **Fit Index** | 0-100% | Average of R², (1-MAPE), and (1-WAPE) on in-sample fitted vs actual KPI. A model with poor fit has meaningless ROI estimates regardless. |

### Ranking metrics (reported separately)

| Metric | What it measures |
|--------|-----------------|
| **Pairwise Ranking** | Fraction of channel pairs ranked in the correct order. 100% = perfect. |
| **Spearman rho** | Rank correlation between true and estimated ROIs. 1.0 = perfect monotonic agreement. |
| **Top-1 Correct** | Did the tool correctly identify the highest-ROI channel? |

### Not in composite (reported separately)

| Metric | Why separate |
|--------|-------------|
| **Absolute ROI Accuracy** | Sensitive to saturation parameterisation — a tool using logistic saturation vs tanh may produce correct relative efficiency but different absolute scale. Included for completeness. |
| **Runtime** | Important for practitioner workflow but not a quality metric. |
| **Convergence** | Binary flag — did the sampler converge cleanly (no divergences, R-hat < 1.05)? |

Full methodology: [docs/methodology.md](docs/methodology.md) | Interpreting results: [docs/interpreting-results.md](docs/interpreting-results.md)

---

## Running locally

### Quick start

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

### CLI options

```
python benchmark.py --scenario simple          # one scenario
python benchmark.py --tool meridian            # one tool
python benchmark.py --update-readme            # rewrite leaderboard section
python benchmark.py --output-dir ./my-results  # custom output directory
```

### Expected runtimes

| Scenario | Per tool | Full suite (3 tools) |
|----------|----------|---------------------|
| simple | 3-6 min | 10-18 min |
| complex | 10-20 min | 30-60 min |
| data_scarce | 3-8 min | 10-25 min |
| adversarial | 8-15 min | 25-45 min |

All scenarios, all tools: ~2-3 hours on a standard laptop (4 cores, 16GB RAM).

### Results output

Results are saved as JSON to `results/{timestamp}/{scenario}.json`. Each file contains per-tool metrics, estimated ROIs, per-channel breakdowns, and convergence diagnostics.

Full setup guide (including Windows gotchas, reduced-resource options): [docs/running-locally.md](docs/running-locally.md)

---

## Project structure

```
mmm-bench/
├── benchmark.py                    # Main CLI runner
├── data/generator/
│   ├── scenario.py                 # ChannelConfig, ControlConfig, Scenario dataclasses
│   └── simulate.py                 # CPG data generator (DGP — never seen by agent)
├── metrics/
│   ├── roi_recovery.py             # Absolute + relative ROI MAPE
│   ├── ranking.py                  # Pairwise, Spearman, top-1
│   ├── business_sense.py           # Industry plausibility ranges
│   ├── contribution_share.py       # Share-of-contribution accuracy
│   ├── fit_index.py                # R², MAPE, WAPE → fit index
│   └── summary.py                  # compute_all_metrics() + composite score
├── runners/
│   ├── base.py                     # BenchmarkRunner ABC + RunResult dataclass
│   ├── pymc_marketing.py           # LogisticSaturation
│   ├── pymc_marketing_tanh.py      # TanhSaturation
│   ├── meridian.py                 # Google Meridian
│   └── decision_packs.py           # Stub — requires decision-lab CLI
├── scenarios/
│   ├── simple.yaml                 # 3 channels, 104 weeks, 3 controls
│   ├── simple_no_controls.yaml     # Same as simple, no controls
│   ├── simple_high_seasonality.yaml # Same as simple, 3.5× seasonal amplitude
│   ├── complex.yaml                # 8 channels, 156 weeks, 5 controls
│   ├── data_scarce.yaml            # 4 channels, 78 weeks, TikTok new
│   └── adversarial.yaml            # 4 channels, 130 weeks, high noise
├── docs/                           # Full documentation suite
│   ├── methodology.md
│   ├── data-generating-process.md
│   ├── scenarios.md
│   ├── interpreting-results.md
│   ├── running-locally.md
│   └── adding-a-tool.md
└── results/                        # JSON per run (gitignored)
```

---

## Design principles

**Blind evaluation.** The agent building each model never sees the data generating process, the true parameters, or the ground truth. It has exactly what a real practitioner has: data and documentation. This eliminates the most common criticism of MMM benchmarks — that the benchmark author tuned the runner to match the DGP.

**Known ground truth.** Synthetic data is the only way to know true ROI. Real-world holdout tests (geo experiments, lift studies) only give partial ground truth for specific channels and time windows.

**CPG-realistic.** National weekly data, flighted TV, always-on digital, price/distribution/competitor controls, seasonal patterns. The most common MMM use case.

**Fair comparison.** Same dataset, same seed, same agent (Claude Opus 4.6) for every tool. No per-tool tuning. The benchmark measures what the tool + agent combination produces out of the box.

**Continuous evaluation.** When a new version of any tool ships to PyPI, Quokka re-reads the latest documentation and re-runs the full suite. This captures regressions, improvements, and whether new features are discoverable enough to be used correctly.

**Relative accuracy is primary.** Absolute ROI recovery is sensitive to saturation form — a tool using logistic vs tanh may get different absolute numbers but correct relative ordering. Since budget allocation depends on *which channels are more efficient*, relative accuracy is what matters.

**Adversarial honesty matters.** When data can't support identification (flat spend, high collinearity), the right answer is uncertainty, not false precision. The adversarial scenario explicitly tests this.

---

## Contributing

PRs welcome for:

- **New tool runners** — see [docs/adding-a-tool.md](docs/adding-a-tool.md)
- **New scenarios** — different industries, panel data, daily granularity
- **Metric improvements** — new metrics or refinements to existing ones
- **Bug fixes**

Open an issue to propose a new scenario or metric before building.

---

*Maintained by [Quokka](https://github.com/simba-quokka) · Powered by [Claude Opus 4.6](https://docs.anthropic.com/en/docs/about-claude/models) via [Claude Code](https://docs.anthropic.com/en/docs/claude-code) · Part of the [Simba](https://getsimba.ai) ecosystem*
