# Scenarios

mmm-bench has four benchmark scenarios, ranging from identifiable and clean to adversarial and noisy. All scenarios simulate CPG (FMCG) brand marketing data. This document provides the full channel-by-channel specification for each scenario, the control variable setup, the known identification challenges, and what good tool performance looks like.

---

## How to Read This Document

Each scenario section contains:
1. **Overview** — what the scenario tests and why it is difficult
2. **Channel table** — spend parameters, CPM, adstock, saturation, and true ROI for each channel
3. **Control variables table** — the confounders the model must handle
4. **Known challenges** — what makes attribution hard on this scenario
5. **What good performance looks like** — concrete metric thresholds that define success

---

## Scenario 1: `simple`

### Overview

**104 weeks · 3 channels · 3 controls · Difficulty: Easy**

The simple scenario is the baseline sanity check. Three channels: one clearly flighted brand channel (TV), and two always-on performance channels (paid search, paid social). TV is strongly identifiable — it alternates between large on-bursts and near-zero off-periods, giving the model clear before/after contrast.

The always-on channels are harder. Search and social overlap almost perfectly in time and have similar spend variability. Their absolute ROIs are difficult to separate, but their relative ROI ordering (search >> social) should be recoverable.

All tools that claim to be production-ready should pass this scenario comfortably.

### Channel Table

| Channel | Spend pattern | Spend mean (on) | Spend mean (off) | Spend sigma | CPM mean | CPM_cv | Adstock type | Decay | Alpha (α) | True ROI |
|---------|--------------|----------------|-----------------|-------------|----------|--------|-------------|-------|-----------|---------|
| `tv` | flighted | $150,000 | $5,000 | $20,000 | $20 | 0.12 | delayed (peak=2) | 0.60 | 0.70 | 0.280 |
| `paid_search` | always_on | $80,000 | — | $12,000 | $8 | 0.20 | geometric | 0.35 | 1.20 | 1.350 |
| `paid_social` | always_on | $60,000 | — | $10,000 | $10 | 0.22 | geometric | 0.25 | 1.10 | 0.750 |

**Notes:**
- TV flights: on-rate = 0.50 (runs approximately 52 weeks out of 104, in 4-week blocks)
- Total weekly spend across all channels: ~$290k on-flight, ~$145k off-flight
- Search ROI (1.35) >> social ROI (0.75) >> TV ROI (0.28) — the correct ranking order

### Control Variables

| Control | Pattern | Parameters | True coefficient | Interpretation |
|---------|---------|------------|----------------|---------------|
| `price_index` | AR1 | rho=0.85, sigma=0.04 | -0.12 | 1% price increase → 12% revenue decline |
| `distribution_acv` | trend | slope=0.08 | +0.25 | Distribution builds steadily over 2 years |
| `category_seasonality` | seasonal | amplitude=0.20 | +1.00 | Annual seasonal cycle, peak at Christmas |

### Known Challenges

1. **Search vs social attribution.** Both are always-on; their contributions are difficult to disentangle absolutely. Expect over-attribution to at least one of them.

2. **TV delayed adstock.** Tools that use geometric adstock (not delayed) will slightly misfit TV — the sales effect peaks 2 weeks after the TV burst, not in the same week. Tools with delayed adstock options should outperform on TV specifically.

3. **Price effect confounding.** The price index is highly persistent (rho=0.85). If the model excludes price or handles it poorly, the price effect will be partially attributed to the always-on channels (search is always on, and price moves persistently — they correlate over time).

### What Good Performance Looks Like

| Metric | Good | Excellent |
|--------|------|-----------|
| Relative ROI Accuracy | > 70% | > 85% |
| Absolute ROI Accuracy | > 50% | > 70% |
| Pairwise Ranking | > 80% | 100% (correct: search > social > TV) |
| Spearman rho | > 0.90 | 1.00 |
| Top-1 Correct | Yes (search) | Yes |
| Holdout MAPE | < 10% | < 6% |
| Business Sense Score | 100% | 100% |

Any production-ready tool should achieve "Good" on all metrics for this scenario. "Excellent" is achievable with well-specified priors and correct adstock type selection.

---

## Scenario 2: `complex`

### Overview

**156 weeks · 8 channels · 5 controls · Difficulty: Medium**

The complex scenario adds multicollinearity, delayed adstock, a high-ROI small channel, and a greater diversity of channel types. Three years of weekly data provides more statistical power, but the identification challenges are meaningfully harder:

- **TV and OOH are correlated** (r=0.65) — they share a planning cycle, so when TV spend rises, OOH spend rises too. The model must disentangle two positively correlated flighted channels.
- **Search and social are correlated** (r=0.55) — digital performance budgets scale together.
- **Email has very high ROI and tiny spend** ($5k/week, ROI=2.50). It is cheap, effective, and easy to ignore statistically because it barely moves the KPI needle in absolute terms. Tests whether tools identify high-efficiency small channels.
- **Affiliates have near-zero adstock** (decay=0.05) — almost entirely last-click. Very different decay structure from brand channels.
- **YouTube uses delayed adstock** (peak=3 weeks) — longer brand-building lag than TV.

### Channel Table

| Channel | Spend pattern | Spend mean (on) | Spend mean (off) | Spend sigma | CPM mean | CPM_cv | Adstock type | Decay | Alpha (α) | True ROI | Correlation |
|---------|--------------|----------------|-----------------|-------------|----------|--------|-------------|-------|-----------|---------|-------------|
| `tv` | flighted | $180,000 | $8,000 | $25,000 | $20 | 0.12 | delayed (peak=2) | 0.62 | 0.65 | 0.250 | r=0.65 with OOH |
| `ooh` | flighted | $80,000 | $3,000 | $12,000 | $6 | 0.15 | geometric | 0.55 | 0.80 | 0.180 | r=0.65 with TV |
| `paid_search` | always_on | $90,000 | — | $14,000 | $8 | 0.20 | geometric | 0.30 | 1.15 | 1.400 | r=0.55 with social |
| `paid_social` | always_on | $70,000 | — | $12,000 | $10 | 0.22 | geometric | 0.22 | 1.05 | 0.720 | r=0.55 with search |
| `youtube` | flighted | $60,000 | $2,000 | $10,000 | $12 | 0.15 | delayed (peak=3) | 0.45 | 0.90 | 0.350 | — |
| `display` | always_on | $30,000 | — | $6,000 | $4 | 0.25 | geometric | 0.10 | 1.50 | 0.320 | — |
| `email` | always_on | $5,000 | — | $800 | $0.5 | 0.10 | geometric | 0.05 | 2.00 | 2.500 | — |
| `affiliates` | always_on | $20,000 | — | $4,000 | $3 | 0.20 | geometric | 0.05 | 1.80 | 1.100 | — |

**Notes:**
- Total weekly spend: ~$555k on-flight, ~$228k off-flight
- Correct ROI ranking: email (2.50) > search (1.40) > affiliates (1.10) > youtube (0.35) > display (0.32) > social (0.72) > TV (0.25) > OOH (0.18)
- Email has the highest ROI but the smallest spend — it contributes little to total KPI despite its efficiency

### Control Variables

| Control | Pattern | Parameters | True coefficient | Interpretation |
|---------|---------|------------|----------------|---------------|
| `price_index` | AR1 | rho=0.85, sigma=0.04 | -0.15 | Price sensitivity slightly higher than simple scenario |
| `distribution_acv` | trend | slope=0.06 | +0.20 | Slower distribution build over 3 years |
| `competitor_spend` | AR1 | rho=0.70, sigma=0.08 | -0.08 | Competitor spend depresses own revenue |
| `category_seasonality` | seasonal | amplitude=0.25 | +1.00 | Stronger seasonal amplitude |
| `trade_promotion` | binary | p=0.15 | +0.35 | Periodic trade deals (15% of weeks) |

### Known Challenges

1. **TV vs OOH decomposition.** With r=0.65, the model sees two channels that almost always move together. It must rely on the residual variation (the 35% of spend that does not correlate) to identify each channel's individual contribution. Tools with stronger regularisation or hierarchical priors across brand channels will fare better.

2. **Email identification.** Email's contribution is real but tiny in absolute dollar terms (~$12.5k/week on a $5M+/week KPI). Its coefficient is well-identified if the model has low noise in that coefficient's posterior — but weak priors may let the email coefficient drift to near-zero.

3. **Affiliates vs search confusion.** Both have very low adstock (nearly immediate response). If the model's adstock prior forces some minimum decay, affiliates may be misfit.

4. **YouTube delayed adstock.** Tools using geometric adstock for YouTube will attribute its contribution to the wrong weeks, slightly degrading both fit and attribution.

5. **Correlated controls.** Trade promotions (binary) correlate weakly with the seasonal indicator (promotions happen more in peak season). Models that cannot handle mild multicollinearity between controls may partially confound these two.

### What Good Performance Looks Like

| Metric | Good | Excellent |
|--------|------|-----------|
| Relative ROI Accuracy | > 60% | > 78% |
| Absolute ROI Accuracy | > 40% | > 60% |
| Pairwise Ranking | > 70% (20 of 28 pairs correct) | > 85% |
| Spearman rho | > 0.80 | > 0.95 |
| Top-1 Correct | Yes (email) | Yes |
| Holdout MAPE | < 15% | < 9% |
| Business Sense Score | > 80% | 100% |

**Top-1 Correct is a high bar here.** Email has the highest ROI but the smallest spend and contribution. Many tools will over-attribute to the large-spend channels (TV, search) and under-identify email. Correctly identifying email as #1 ROI channel requires a model that is not dominated by spend scale.

---

## Scenario 3: `data_scarce`

### Overview

**78 weeks · 4 channels · 3 controls · Difficulty: Hard**

Seventy-eight weeks (18 months) is the shortest data window in the benchmark. In practice, many brands launch MMM when they have 18–24 months of clean data. This scenario tests what tools can do with limited data.

It includes **TikTok as a new channel** — one that did not exist 3 years ago, has no historical context for the model to learn from, and has a very different audience profile from the other channels. Tools that rely entirely on data-driven prior estimation will struggle with TikTok; tools that allow practitioners to inject informative priors (or that have sensible defaults for new channels) should outperform.

### Channel Table

| Channel | Spend pattern | Spend mean | Spend sigma | CPM mean | CPM_cv | Adstock type | Decay | Alpha (α) | True ROI |
|---------|--------------|------------|-------------|----------|--------|-------------|-------|-----------|---------|
| `tv` | flighted | $120,000 (on) / $4,000 (off) | $18,000 | $20 | 0.12 | delayed (peak=2) | 0.58 | 0.72 | 0.260 |
| `paid_search` | always_on | $70,000 | $10,000 | $8 | 0.20 | geometric | 0.32 | 1.18 | 1.300 |
| `paid_social` | always_on | $55,000 | $9,000 | $10 | 0.22 | geometric | 0.24 | 1.08 | 0.680 |
| `tiktok` | always_on | $25,000 | $8,000 | $7 | 0.25 | geometric | 0.18 | 1.30 | 0.950 |

**Notes:**
- TikTok: launched at week 1 of the dataset — there is no prior period data. Decay=0.18, more scroll-immediate than search, less than TV.
- TikTok true ROI (0.95) is higher than TV and social but below search. A tool should ideally rank TikTok correctly.
- 78 weeks means only ~9 TV flights if on-rate=0.50 — relatively few before/after contrasts.

### Control Variables

| Control | Pattern | Parameters | True coefficient | Interpretation |
|---------|---------|------------|----------------|---------------|
| `price_index` | AR1 | rho=0.85, sigma=0.04 | -0.13 | Standard price effect |
| `distribution_acv` | trend | slope=0.12 | +0.30 | Faster distribution build (brand is newer) |
| `category_seasonality` | seasonal | amplitude=0.20 | +1.00 | Annual seasonal cycle |

### Known Challenges

1. **Short history reduces statistical power.** With 78 observations and 4 channels plus 3 controls, the data-to-parameter ratio is tight. Saturation shape parameters (alpha) and adstock decay rates are hard to identify jointly with limited data.

2. **TikTok: zero historical context.** The model has never seen TikTok at different spend levels in a prior period. If TikTok spend is roughly constant over the 78 weeks, it faces the same always-on identification problem as search and social.

3. **Prior quality matters.** Tools with uninformative or flat priors will produce wide, poorly-centred estimates on TikTok. Tools with sensible channel-type priors (e.g., a decay prior centred around 0.2–0.3 for short-video social channels) will produce more useful estimates. This is one scenario where the choice of default priors directly affects benchmark performance.

4. **Saturation shape uncertainty.** Alpha is fundamentally harder to identify from 78 weeks than from 156. Expect wider credible intervals on alpha (and derived ROI) for all channels.

5. **Distribution trend confounding.** With a steep distribution trend (slope=0.12 over 78 weeks), the rising baseline can be partially attributed to media if the trend is mis-specified.

### What Good Performance Looks Like

| Metric | Good | Excellent |
|--------|------|-----------|
| Relative ROI Accuracy | > 55% | > 72% |
| Absolute ROI Accuracy | > 35% | > 55% |
| Pairwise Ranking | > 65% (4 of 6 pairs correct) | > 85% |
| Spearman rho | > 0.75 | > 0.92 |
| Top-1 Correct | Yes (search) | Yes |
| Holdout MAPE | < 20% | < 12% |
| Business Sense Score | > 75% | 100% |

**Wider credible intervals are desirable here.** A tool that produces narrow, overconfident intervals on a data-scarce scenario is likely overfitting. Calibration of uncertainty — not just point estimate accuracy — matters. Future benchmark phases will include proper scoring rules (e.g., interval coverage rates) to measure calibration explicitly.

---

## Scenario 4: `adversarial`

### Overview

**130 weeks · 4 channels · 4 controls · Difficulty: Very Hard**

The adversarial scenario is designed to be genuinely difficult for all tools. It is not designed to be "passable with the right settings" — it is designed to test whether tools produce honest uncertainty when faced with data that does not support confident attribution.

The key adversarial properties:
- **TV barely varies.** Spend std / spend mean = 8%. Compared to always-on channels, TV has essentially no variation — it provides none of the identification contrast that makes flighted TV identifiable in the simple scenario.
- **All channels are highly correlated.** TV and OOH: r=0.85. Search and social: r=0.75.
- **High noise.** The iid noise sigma ($60,000) is large relative to the KPI variation, degrading signal-to-noise.
- **Downward trend.** The KPI is trending down over the 130 weeks. A model that fails to capture this trend will attribute the revenue decline to media reductions (or, worse, attribute the trend as a negative contribution from a media channel).

**What good looks like here is different.** The correct answer for the adversarial scenario is not a confident ROI estimate — it is honest uncertainty. A tool that says "I cannot reliably attribute TV because it doesn't vary enough" (wide CI on TV ROI, R-hat warning, possibly a note to the user) is performing better than a tool that confidently returns a wrong TV ROI of 0.85 (more than 3× the true value).

### Channel Table

| Channel | Spend pattern | Spend mean | Spend sigma | CV (σ/μ) | CPM mean | CPM_cv | Adstock type | Decay | Alpha (α) | True ROI | Correlation |
|---------|--------------|------------|-------------|----------|----------|--------|-------------|-------|-----------|---------|-------------|
| `tv` | flighted | $100,000 | $8,000 | **8%** | $20 | 0.12 | geometric | 0.58 | 0.72 | 0.260 | r=0.85 with OOH |
| `ooh` | flighted | $50,000 | $5,000 | 10% | $6 | 0.15 | geometric | 0.50 | 0.80 | 0.180 | r=0.85 with TV |
| `paid_search` | always_on | $75,000 | $8,000 | 11% | $8 | 0.20 | geometric | 0.30 | 1.15 | 1.200 | r=0.75 with social |
| `paid_social` | always_on | $60,000 | $7,000 | 12% | $10 | 0.22 | geometric | 0.22 | 1.05 | 0.680 | r=0.75 with search |

**Notes:**
- TV flight on-rate = 0.55, but spend barely changes between on and off periods (mean $100k on vs $92k off after adding correlation noise). The flight pattern provides almost no identification power.
- noise_sigma = $60,000 (high — roughly 1.2% of weekly KPI)
- KPI trend_slope = -0.003 (approximately 30% KPI decline over 130 weeks — a struggling brand)

### Control Variables

| Control | Pattern | Parameters | True coefficient | Interpretation |
|---------|---------|------------|----------------|---------------|
| `price_index` | AR1 | rho=0.85, sigma=0.04 | -0.18 | Elevated price sensitivity (elastic category) |
| `distribution_acv` | trend | slope=-0.05 | +0.20 | Losing distribution (matches downward trend story) |
| `competitor_spend` | AR1 | rho=0.70, sigma=0.10 | -0.10 | Competitor activity increases as market share falls |
| `trade_promotion` | binary | p=0.12 | +0.30 | Periodic deals attempting to arrest decline |

### Known Challenges

1. **TV identification failure.** With CV=8%, TV is practically an always-on channel in this scenario. There is insufficient variation to separate TV contribution from the structural baseline. No amount of modelling sophistication fully overcomes this — it is a fundamental identification failure from the data itself.

2. **High inter-channel correlation.** With TV-OOH at r=0.85 and search-social at r=0.75, the VIF (variance inflation factor) for attribution is extremely high. The model's posterior for individual channel ROIs will be wide and potentially bimodal.

3. **Trend misattribution.** The downward KPI trend can be absorbed by the trend control — or it can leak into media channel coefficients. A model that excludes the trend control will partially attribute the revenue decline to media. A model that includes it but forces positive-only media coefficients may produce a different but equally wrong attribution.

4. **High noise.** The signal-to-noise ratio is low. Even with infinite data, the noise makes precise attribution difficult. With only 130 weeks, it is essentially impossible to recover true ROIs within 30% accuracy for highly correlated channels.

5. **The "confident wrong answer" failure mode.** Some tools converge cleanly (no divergences, low R-hat) but produce ROIs that are far from truth with narrow credible intervals. This is the most dangerous failure mode — it gives practitioners false confidence. The business sense score and R-hat diagnostics are the main catches.

### What Good Performance Looks Like

| Metric | Good (honest uncertainty) | Excellent |
|--------|--------------------------|-----------|
| Relative ROI Accuracy | > 40% | > 60% |
| Absolute ROI Accuracy | > 20% | > 40% |
| Pairwise Ranking | > 55% | > 75% |
| Spearman rho | > 0.50 | > 0.80 |
| Top-1 Correct | Yes (search) | Yes |
| Holdout MAPE | < 25% | < 15% |
| Business Sense Score | > 60% | > 85% |
| CI width (TV ROI) | > 0.20 | > 0.30 |

**On the adversarial scenario, a tool that achieves moderate accuracy with wide CIs is better than one that achieves slightly higher accuracy with narrow, overconfident CIs.** Proper Bayesian tools should automatically produce wide CIs when the data is non-informative — this is a feature, not a failure. Future benchmark phases will formally score calibration (whether the 94% HDI contains the true value 94% of the time).

**The Decision-packs consensus check is designed for exactly this scenario.** When its parallel model ensemble disagrees substantially, it reports "models disagree, attribution unreliable" rather than averaging divergent estimates into a false consensus. This honest failure mode should score highly in any benchmark that values epistemic honesty.

---

*See also:*
- *[Data-generating process](data-generating-process.md) — full DGP technical specification*
- *[Methodology](methodology.md) — metric definitions and design philosophy*
- *[Interpreting results](interpreting-results.md) — how to read benchmark output*
