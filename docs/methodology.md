# Benchmark Methodology

This document explains the design philosophy behind mmm-bench: why the benchmark is structured the way it is, what choices were made and why, and what the metrics are actually measuring. Read this before drawing conclusions from the leaderboard.

---

## Design Principles

### Ground truth is knowable

The only way to measure accuracy is to know the answer. mmm-bench uses synthetic data with a fully specified, recorded data-generating process (DGP). Every parameter — channel ROI, adstock decay, saturation shape, baseline — is saved before any tool sees the data. There is no argument about what the "correct" answer is.

This is a deliberate departure from benchmarks that use historical real-world data and proxy metrics (e.g., holdout lift). Holdout metrics test generalization, not attribution accuracy. A model can generalise well while attributing revenue to the wrong channels. mmm-bench measures both: direct ROI accuracy against ground truth, and holdout MAPE as a generalization check.

### CPG-realistic (not toy data)

The data generating process is calibrated to match what practitioners encounter in fast-moving consumer goods (FMCG) brand marketing:

- Budgets in the range of $50k–$500k per channel per week
- TV and OOH as flighted brand channels (on/off flight patterns, 4-week cycles)
- Paid search and paid social as always-on performance channels
- CPM variation reflecting real programmatic price fluctuations (10–25% coefficient of variation)
- Correlated channel spend (TV and OOH planned together, search and social often scaled together)
- Control variables: price index, distribution ACV, category seasonality, competitor spend

The goal is not to make the benchmark easy to pass. The goal is to make it representative of what tools will encounter in production.

### Fair comparison

All tools run on:
- The same dataset (same random seed, same DGP)
- Default hyperparameters (no per-tool tuning — see [What default hyperparameters means](#what-default-hyperparameters-means))
- The same hardware (standard Windows laptop, no GPU, no cloud)

No tool gets a home-field advantage. If a tool's defaults perform poorly, that is meaningful signal — practitioners using that tool out of the box will also get poor results.

### Spend-denominated ROI throughout

All ROI calculations use spend in the denominator, not impressions:

```
ROI(ch) = total KPI contribution(ch) / total spend(ch)
```

This is the standard industry definition. It is what practitioners report to stakeholders ("we spent $X on TV and got $Y in incremental revenue"). It allows fair comparison across channels with radically different impression volumes — email and TV might have similar spend but order-of-magnitude differences in impressions. Using impressions as the denominator would make email appear far more efficient and TV far less, which is an artefact of the media buying model, not the marketing effectiveness.

---

## Why CPG / FMCG Data?

CPG is the most common real-world MMM use case. Most MMM practitioners work in FMCG, alcohol, personal care, or adjacent categories with:

- National TV spend as the dominant brand channel
- Significant always-on performance media (search, social, display)
- Strong seasonal effects (Christmas, summer, category-specific)
- Mature measurement infrastructure (Nielsen/IRI sales data, GRP tracking)
- Active interest in MMM for budget allocation

CPG data also has all the features that make MMM genuinely hard:

1. **Flighted brand spend** — TV goes dark for 8–12 weeks at a time. This creates identification power but also means the model must handle structural zeros in the activity variable.
2. **Always-on performance channels** — search never goes dark. This is the core identification challenge (see below).
3. **Correlated channels** — TV and OOH are often planned by the same agency on the same cycle. Search and social scale together as digital performance budgets move. This creates multicollinearity that degrades attribution.
4. **Control variables that matter** — price promotions, distribution changes, and competitor activity are real confounders that MMM must handle.
5. **Impressions ≠ spend** — TV GRPs and digital impressions are the natural activity variables for adstock and saturation, but spend is the right ROI denominator. Tools that conflate these will produce biased ROIs.

---

## Why Separate Impressions from Spend?

This is one of the most important design decisions in mmm-bench, and it is worth explaining carefully.

**The industry standard** is to pass impressions, GRPs, or reach as the media activity variable into an MMM model. Spend is used to compute ROI at the end. This matters because:

1. **Meridian is designed this way.** Google's Meridian natively accepts reach and frequency as inputs. Passing only spend into Meridian misses the design intent.

2. **CPM varies week to week.** A $100k TV spend in Q4 buys fewer GRPs than $100k in Q3 because inventory costs more. If you pass spend directly into adstock/saturation, you are conflating budget decisions with actual audience exposure. The activity variable should reflect actual exposure.

3. **Spend and activity can diverge dramatically.** Programmatic display spend can increase 3× during peak season while impressions only increase 1.5× because CPMs rise. Passing spend as the activity variable would overstate the saturation of display during peak.

4. **ROI must still be spend-denominated** for it to be actionable. The question a planner asks is: "If I spend $1M more on TV, what incremental revenue do I get?" — not "If I deliver 1M more GRPs...".

The DGP generates both: `df[ch]` is weekly impressions/GRPs (the activity variable), and `df[f'{ch}_spend']` is weekly spend in dollars (the ROI denominator). Tools that only accept spend can still be benchmarked — but this is noted as a limitation.

---

## The Always-On Identification Problem

This is the most important concept for interpreting mmm-bench results.

**The problem:** When a media channel runs every single week without variation, the model cannot distinguish its contribution from an elevated baseline. If search spend is $50k±$5k every week and revenue averages $5M/week, the model has no way to know whether search contributes $0.5M/week or $0/week (with the baseline being $0.5M higher). The data is consistent with both interpretations.

**Why flighted channels are easier:** TV goes dark for 8 weeks, then spends $150k for 4 weeks. Revenue rises and falls in rough synchrony (accounting for adstock lag). This before/after contrast gives the model strong signal about TV's contribution.

**Consequences for the benchmark:**
- Tools will consistently over- or under-attribute always-on channels
- Absolute ROI accuracy on search and social is fundamentally limited without additional data
- Relative ROI accuracy is more robust — even if both search and social are biased, their *ratio* may be approximately correct
- The adversarial scenario makes this worse deliberately: TV barely varies too, so no channel provides clear identification contrast

**The practitioner solution: lift tests.** Geo-holdout experiments that dark out a channel in one region provide the direct attribution signal that time-series variation cannot. This is why the planned test dimension "with lift tests" is important — it tests whether tools correctly integrate geo-experiment data to constrain always-on channel estimates. It is the one situation where absolute ROI accuracy on always-on channels should be achievable.

**How to read results given this:** When you see high absolute ROI error on search/social but lower relative ROI error, and good pairwise ranking, the tool is likely behaving correctly given the fundamental identification limitation. This is not a model failure — it is an honest reflection of what time-series data can and cannot identify.

---

## Metric Design

### Relative ROI Accuracy (primary)

The primary metric normalises both true and estimated ROIs by their cross-channel mean before computing MAPE:

```
rel_true_ROI(ch)  = true_ROI(ch) / mean(true_ROI)
rel_est_ROI(ch)   = est_ROI(ch)  / mean(est_ROI)

Relative ROI Accuracy = 1 - mean_ch( |rel_true - rel_est| / rel_true )
```

**Why normalise?** Different saturation functional forms produce different absolute ROI scales. A model using LogisticSaturation and a model using TanhSaturation, both fitted to the same data, will produce systematically different absolute ROIs even if they agree perfectly on the relative ordering and gaps between channels. Normalising by the mean removes this saturation-scale confound.

**Interpretation:** A relative ROI accuracy of 80% means that, after removing the saturation scale difference, the tool's channel ROI estimates are on average 20% off from the truth in relative terms.

### Absolute ROI Accuracy (secondary)

The raw MAPE on spend-denominated ROIs, without normalisation:

```
Absolute ROI Accuracy = 1 - mean_ch( |true_ROI(ch) - est_ROI(ch)| / true_ROI(ch) )
```

This is meaningful when:
- The DGP and tool saturation functional form match (e.g., benchmarking PyMC-Marketing Tanh against a TanhSaturation DGP)
- Lift tests are available to constrain the absolute scale
- The channel is flighted (before/after contrast provides absolute identification)

It is *not* expected to be high for always-on channels without lift tests. Treat low absolute ROI accuracy on always-on channels as expected, not as a tool failure.

### Holdout MAPE (primary)

The last 13 weeks of each scenario are withheld from fitting. After fitting, each tool produces a forecast for these 13 weeks. MAPE is computed against actual KPI values.

This is the most trusted metric in practitioner MMM because:
1. It does not require ground truth ROIs — it is observable from real data
2. It tests whether the model captures the genuine data-generating structure (good models generalise; overfit or underspecified models do not)
3. A low holdout MAPE is necessary but not sufficient for correct attribution. A model can fit and predict well while attributing revenue to the wrong channels. But a model with high holdout MAPE is almost certainly wrong.

### Contribution Share Accuracy

Measures how accurately the tool recovers the true percentage share of total media contribution per channel:

```
true_share(ch) = contribution(ch).sum() / Σ_all_ch contribution(ch).sum()
est_share(ch)  = estimated_contribution(ch) / Σ_all_ch estimated_contribution(ch)

Contribution Share MAPE = mean_ch( |true_share - est_share| / true_share )
Contribution Share Accuracy = 1 - Contribution Share MAPE
```

This metric is more robust than absolute ROI across tools with different saturation parameterisations. Even when absolute ROIs are biased by the always-on identification problem, the *shares* of total contribution may still be approximately correct — and shares are what drive budget allocation decisions.

- **> 80%**: Excellent — share estimates are reliable for allocation
- **60–80%**: Good — directionally useful
- **< 60%**: Poor — contribution decomposition is unreliable

### In-sample Fit Index

Three components combined into a single score measuring how well the model fits the training data:

```
R² = 1 - Σ(KPI_actual - KPI_fitted)² / Σ(KPI_actual - mean(KPI_actual))²
MAPE = mean_t( |KPI_actual_t - KPI_fitted_t| / KPI_actual_t )
WAPE = Σ|KPI_actual - KPI_fitted| / Σ KPI_actual

Fit Index = (clip(R², 0, 1) + clip(1 - MAPE, 0, 1) + clip(1 - WAPE, 0, 1)) / 3
```

A model with poor fit has meaningless ROI estimates regardless of what they say. The fit index provides a basic quality gate.

- **> 85%**: Good — model captures the data structure
- **70–85%**: Acceptable — some residual structure unexplained
- **< 70%**: Poor — model is misspecified; ROI estimates should not be trusted

### Composite Score

The single headline metric for the leaderboard, combining all component metrics:

```
Composite = 0.30 × Relative ROI Accuracy
          + 0.20 × Holdout Accuracy
          + 0.20 × Contribution Share Accuracy
          + 0.15 × Business Sense Score
          + 0.15 × Fit Index
```

If any component is unavailable (e.g. holdout prediction not supported by the tool), its weight is redistributed proportionally to the remaining components. This means no single missing capability disqualifies a tool from scoring, but tools that support more metrics have a truer composite.

Speed (runtime) is reported separately — it is not in the composite because it is an operational metric, not a quality metric.

### Business Sense Score

For each tool and scenario, we check whether each channel's estimated ROI falls within an industry-plausible range:

| Channel type | Plausible ROI range | Rationale |
|-------------|--------------------|-|
| TV (national brand) | 0.05 – 0.80 | High cost, broad reach; lower direct ROI typical |
| OOH | 0.05 – 0.60 | Similar economics to TV |
| Paid search | 0.30 – 4.00 | High-intent, measurable; wide range reflects category differences |
| Paid social | 0.20 – 3.00 | Variable by format and category |
| Display | 0.05 – 1.50 | Low attention, broad reach |
| Email | 0.50 – 8.00 | Very low cost, high relative ROI |
| YouTube / video | 0.10 – 1.50 | Branding + direct hybrid |
| Affiliates | 0.30 – 5.00 | Performance-based, variable |

A tool with business_sense_score = 50% means half its channel ROIs are outside these ranges. This often indicates:
- A convergence failure where the sampler got stuck at a degenerate solution
- A saturation parameterisation that produces negative or explosive contributions
- A prior that allows economically impossible values without penalising them

The business sense score is independent of accuracy — a tool can have perfect relative ROI accuracy on a simple scenario but fail the business sense check on adversarial (because the correct answer on adversarial involves wide, uncertain estimates that may stray outside plausible ranges).

---

## What Default Hyperparameters Means

The benchmark tests out-of-the-box usability, not ceiling performance. Every tool runs with its default settings as shipped, with only the following adjustments that any practitioner would make:

**PyMC-Marketing:**
- 1000 draws, 1000 tune, 4 chains, `target_accept=0.95`
- `GeometricAdstock(l_max=8)` — standard practitioner choice
- `LogisticSaturation` for the Logistic variant, `TanhSaturation` for the Tanh variant
- No custom prior specifications

**Meridian:**
- 4 chains, 1000 adapt steps, 500 burnin, 1000 keep samples
- Default prior configuration
- Reach/frequency input (not spend-as-activity)

**Decision-packs:**
- Default ensemble configuration
- No custom consensus thresholds

**Why this matters:** MMM practitioners at most companies do not spend days tuning sampler settings and prior specifications for each model run. They use the defaults. If a tool requires expert tuning to produce good results, that is a usability limitation that the benchmark should surface, not paper over.

Ceiling performance benchmarks (where each tool is hand-tuned by an expert) are valuable but answer a different question. mmm-bench answers: *What will a competent analyst get when they use this tool correctly but without heroic tuning effort?*

---

## Benchmark Dimensions

### Implemented

1. **Seasonality stress test** — `simple_high_seasonality` scenario: 3.5× seasonal amplitude vs the baseline. Tests whether always-on channels absorb seasonal peaks and get over-attributed.

2. **With/without control variables** — `simple_no_controls` scenario: same data as simple but with all control variables removed. Quantifies the confounding cost of omitting price, distribution, and competitor data.

3. **Holdout evaluation** — 13-week train/test split on all scenarios. Tests generalisation independently of attribution accuracy.

### Planned

1. **Lift test integration** — Geo holdout experiments injected as likelihood observations (not priors). Tests whether tools correctly use this additional data to constrain always-on channels. This is the single highest-value extension because it directly addresses the fundamental limitation of time-series MMM.

2. **Structural break** — Baseline shift at the dataset midpoint (e.g., 20% step change). Tests whether tools can handle non-stationarity without attributing the break to a media channel.

3. **Data-sparse only** — 52 weeks only. Shorter than `data_scarce`. Tests the minimum viable data requirement for each tool.

---

*See also:*
- *[Data-generating process](data-generating-process.md) — full technical specification*
- *[Scenarios](scenarios.md) — per-scenario detail*
- *[Interpreting results](interpreting-results.md) — how to read and act on benchmark output*
