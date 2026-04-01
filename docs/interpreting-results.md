# Interpreting Results

This document explains how to read mmm-bench output — the per-channel tables, metric summaries, convergence diagnostics, and the decision framework for when to trust different types of estimates. It also documents the Quokka workflow: how Claude Code is used to interpret results conversationally.

---

## The Quokka Workflow

mmm-bench is designed to be run and interpreted by **Quokka**, a Claude Code-based autonomous agent that monitors PyPI for new tool releases and re-runs the benchmark when versions change.

The workflow is:

1. **Quokka detects a new release** (e.g., `pymc-marketing==0.20.0` appears on PyPI)
2. **Quokka runs the benchmark** via `python benchmark.py --tool pymc-marketing`
3. **Claude Code reads the result JSON** and the per-channel output table
4. **Claude Code interprets the results** — identifies patterns, explains root causes, flags regressions vs the previous version
5. **Quokka posts to GitHub Discussions** — a summary with the leaderboard update, diagnostic commentary, and any notable changes
6. **Quokka updates the README leaderboard** via `python benchmark.py --update-readme`

Human practitioners can follow the same workflow locally — run the benchmark, then ask Claude Code to interpret the output.

### Typical Claude Code session

```
User: I ran pymc-marketing on the complex scenario. Here are the results:
      [paste results/YYYY-MM-DDTHH-MM-SSZ/complex_pymc-marketing_logistic.json]
      What does this tell us?

Claude Code: [reads per-channel table, identifies always-on over-attribution,
              explains TV-OOH correlation impact, flags email under-identification,
              notes search is over-attributed due to always-on problem,
              suggests lift test experiment to constrain always-on channels]
```

---

## Reading the Per-Channel Output Table

After each run, the benchmark prints a per-channel table to stdout:

```
Scenario: simple | Tool: pymc-marketing (logistic) | Version: 0.19.2
Seed: 42 | Runtime: 322s | Converged: YES | Divergences: 0

Channel          True ROI   Est ROI    Abs Err   Rel Err   Status
──────────────────────────────────────────────────────────────────
tv                  0.280     0.261      6.8%     12.3%    🟢
paid_search         1.350     2.395     77.4%     15.0%    🟡
paid_social         0.750     2.525    236.7%     42.0%    🔴

Pairwise ranking: 3/3 (100.0%)   ← correct: search > social > TV
Spearman rho: 1.00               ← perfect rank correlation
Top-1 correct: YES (search)
Holdout MAPE: 6.7%
Business sense: 3/3 (100.0%)

Relative ROI Accuracy: 74.3%
Absolute ROI Accuracy: 38.2%   ← low due to always-on over-attribution
```

### Colour coding

| Colour | Abs error | Rel error | Meaning |
|--------|-----------|-----------|---------|
| 🟢 Green | < 30% | < 20% | Good attribution — trust this channel's ROI |
| 🟡 Yellow | 30–75% | 20–50% | Moderate error — directionally correct, absolute scale uncertain |
| 🔴 Red | > 75% | > 50% | Poor attribution — use only for ranking, not planning |

**Read the colour coding at the channel level, not the scenario level.** In the example above:
- TV is green: the absolute and relative errors are both low. Trust TV's ROI for planning.
- Search is yellow: the estimated ROI (2.40) is nearly double the true ROI (1.35) in absolute terms, but the relative error is only 15% — search is correctly identified as the highest-ROI channel.
- Social is red: both absolute and relative errors are high. The model cannot cleanly separate social from search.

This pattern — flighted channel green, always-on channels yellow/red — is the expected result for the simple scenario without lift tests. It is not a model failure; it is the always-on identification problem in practice.

---

## The Absolute vs Relative Distinction

Understanding this distinction is the most important skill for interpreting mmm-bench results.

### When absolute ROI accuracy is low but relative ROI accuracy is high

**Example:** True ROIs = [0.28, 1.35, 0.75]. Estimated ROIs = [0.22, 2.10, 1.15].

- Absolute MAPE: mean(|0.28-0.22|/0.28, |1.35-2.10|/1.35, |0.75-1.15|/0.75) = 46%
- Normalised true ROIs: [0.37, 1.80, 1.00] (dividing by mean 0.793)
- Normalised est ROIs: [0.19, 1.82, 1.00] (dividing by mean 1.157)
- Relative MAPE: mean(|0.37-0.19|/0.37, |1.80-1.82|/1.80, |1.00-1.00|/1.00) = 16%

**Interpretation:** The model has correctly identified the relative ordering and the relative gaps between channels. Search is about 4.8× more efficient than TV; the model estimates 9.5× — wrong in absolute terms but still ranking correctly. Social is correctly identified as the midpoint channel.

**What this means for decision-making:**
- **Budget allocation decisions:** Use relative ROI. If you are deciding how to rebalance $500k between search, social, and TV, the model correctly tells you to move money toward search and away from TV.
- **Payback calculations:** Do not use absolute ROI when accuracy is low. If you tell the CFO "search ROI is 2.10" when the true value is 1.35, you will over-promise on the incremental revenue from search investment.
- **New channel evaluation:** Relative ROI is sufficient to decide if TikTok is more or less efficient than social. Absolute ROI matters if you are trying to model the specific payback period.

### When both absolute and relative accuracy are low

This is a genuine model failure. It occurs on:
- The adversarial scenario (expected — the data does not support attribution)
- Convergence failures (R-hat > 1.05, many divergences)
- Misspecified models (e.g., wrong adstock form, omitted trend control)

In this case, neither the absolute nor relative ROI estimates should be used for planning. The correct response is to investigate convergence, add data sources (lift tests), or run a simpler model.

---

## Diagnosing Convergence Warnings

### R-hat (Gelman-Rubin diagnostic)

R-hat measures whether multiple MCMC chains have converged to the same distribution:

| R-hat | Status | Action |
|-------|--------|--------|
| < 1.01 | Excellent convergence | Results are reliable |
| 1.01–1.05 | Good convergence | Minor warnings; results usable |
| 1.05–1.10 | Poor convergence | Treat results with caution; increase draws or tune |
| > 1.10 | Failed convergence | Do not use results; diagnose the model |

**What causes high R-hat in MMM?**

1. **Identifiability issues.** Two parameters that are not individually identified (e.g., two correlated channels' coefficients) will produce bimodal posteriors where chains get stuck in different modes. Always-on channels under high correlation are the most common trigger.

2. **Prior-likelihood mismatch.** If the prior is strongly concentrated in a region that the likelihood strongly disagrees with, the sampler struggles to bridge the gap. Common with very tight priors on saturation shape.

3. **Too few tune steps.** The default 1000 tune steps is usually sufficient, but complex models with many parameters may need 2000. Try `--tune 2000` if R-hat is borderline.

4. **Collinear controls.** If two control variables are strongly correlated (e.g., distribution trend and category seasonality both trending up), their coefficients can be non-identified individually.

### Divergences

Divergences are proposals that violate the Hamiltonian trajectory — they indicate that the sampler's step size is too large for the local geometry of the posterior.

| Divergences | Status | Action |
|------------|--------|--------|
| 0 | Excellent | No action needed |
| 1–10 | Acceptable | Monitor; increase `target_accept` to 0.99 |
| 10–100 | Concerning | Likely prior-likelihood mismatch; check model specification |
| > 100 | Failed | Do not use results; reparameterise or adjust priors |

**Common fix:** Most divergences in PyMC-Marketing MMM are resolved by increasing `target_accept` from 0.95 to 0.99, which forces smaller NUTS steps at the cost of slower sampling. The benchmark uses 0.95 as the default.

### Effective Sample Size (ESS)

ESS measures how many independent samples the chains effectively produced, accounting for autocorrelation:

| ESS | Status | Implication |
|-----|--------|-------------|
| > 400 per chain | Good | Reliable posterior summaries |
| 100–400 | Acceptable | Slightly noisy credible intervals |
| < 100 | Poor | Credible intervals unreliable; increase draws |

ESS is reported in the convergence_warnings field of the result JSON.

### PyMC-Marketing specific: `gamma_control` identification

When control variables are collinear or weakly identified, `gamma_control` parameters (the control variable coefficients in PyMC-Marketing) sometimes produce high R-hat warnings independently of the media channel parameters. This is not necessarily a problem for ROI recovery — it affects the control decomposition, not the media attribution, if the collinearity is between controls rather than between controls and media.

**Check:** If only `gamma_control_*` parameters have R-hat > 1.05 but all `beta_channel_*` parameters are well-converged, the media ROI estimates may still be reliable. If `beta_channel_*` also has high R-hat, the attribution is compromised.

---

## Metric Reference

### Relative ROI Accuracy (primary)

The most important metric for most use cases.

```
rel_true_ROI(ch)  = true_ROI(ch)  / mean_ch(true_ROI)
rel_est_ROI(ch)   = est_ROI(ch)   / mean_ch(est_ROI)

Relative MAPE = mean_ch( |rel_true - rel_est| / rel_true )
Relative ROI Accuracy = 1 - Relative MAPE
```

- **100%**: Perfect relative attribution
- **> 80%**: Excellent — suitable for confident budget allocation
- **60–80%**: Good — directionally reliable, some channel-level uncertainty
- **40–60%**: Marginal — use rankings only, not specific ROI values
- **< 40%**: Poor — results should not be used for decisions

### Absolute ROI Accuracy (secondary)

```
Absolute MAPE = mean_ch( |true_ROI(ch) - est_ROI(ch)| / true_ROI(ch) )
Absolute ROI Accuracy = 1 - Absolute MAPE
```

Low absolute accuracy on always-on channels without lift tests is expected and does not indicate a model failure. Absolute accuracy below 50% on flighted channels (TV, OOH, YouTube) is a meaningful warning.

### Pairwise Ranking

```
Pairwise Ranking = #{pairs (i,j) where sign(est_i - est_j) = sign(true_i - true_j)}
                   / #{all pairs}
```

For N channels there are N×(N-1)/2 pairs. A random ranker scores 50%.

- **100%**: Perfect ranking — use for confident channel priority decisions
- **> 80%**: Good — most allocation decisions will be correct
- **50–80%**: Marginal — chance of inverting important channel pairs
- **< 50%**: Worse than random — something is fundamentally wrong

### Spearman rho

The rank correlation between true and estimated ROI vectors. Equivalent information to pairwise ranking but more sensitive to large rank inversions.

- **1.00**: Perfect rank ordering
- **0.90+**: Excellent
- **0.70–0.90**: Good
- **0.50–0.70**: Marginal
- **< 0.50**: Poor (note: negative rho means inverted ranking)

### Holdout MAPE

Out-of-sample forecast error on the last 13 weeks:

```
Holdout MAPE = mean_t( |actual_t - predicted_t| / actual_t )
```

This is the most trusted metric in practitioner MMM because it does not require ground truth ROIs. It tests real generalization.

- **< 8%**: Excellent — suitable for forecasting and scenario planning
- **8–15%**: Good — adequate for most planning purposes
- **15–25%**: Marginal — model may be overfit or misspecified
- **> 25%**: Poor — model does not generalise; do not use for forecasting

**Important:** Low holdout MAPE does not guarantee good attribution. A model can predict well while attributing revenue to the wrong channels (e.g., by using the trend as a proxy for always-on channels). Always check both holdout MAPE and ROI accuracy.

### Business Sense Score

The fraction of channel ROIs within industry-plausible ranges:

| Channel type | Plausible range | Rationale |
|-------------|----------------|-----------|
| TV (national) | 0.05–0.80 | High cost per impression |
| OOH | 0.05–0.60 | Broad reach, limited targeting |
| Paid search | 0.30–4.00 | High intent, measurable |
| Paid social | 0.20–3.00 | Variable by format |
| Display | 0.05–1.50 | Low attention, broad reach |
| Email | 0.50–8.00 | Very low cost |
| YouTube | 0.10–1.50 | Brand + direct hybrid |
| Affiliates | 0.30–5.00 | Performance-based |
| TikTok | 0.20–2.50 | Social video; similar to YouTube |

A business sense score of 67% means 2 of 3 channels have ROIs in the plausible range. The one outside is likely a convergence artefact.

---

## What Good Results Look Like Per Scenario

### `simple` scenario

Expected pattern for any production-quality tool:
- TV: 🟢 green (flighted, identifiable)
- Search: 🟡 yellow abs, 🟢 green relative (over-attributed absolutely, correct ranking)
- Social: 🟡 or 🔴 (hardest to separate from search)
- Pairwise ranking 3/3 (search > social > TV is the correct order)
- Holdout MAPE < 10%

If TV is red or search and social are inverted in ranking, that is a genuine failure.

### `complex` scenario

Expected pattern:
- TV: 🟢 green (flighted, but degraded by OOH correlation)
- OOH: 🟡 yellow (correlated with TV, harder to isolate)
- Search: 🟡 yellow (always-on over-attribution)
- Social: 🟡 to 🔴 (correlated with search)
- YouTube: 🟢 to 🟡 (flighted, somewhat identifiable)
- Display: 🟡 (always-on, but low spend limits over-attribution magnitude)
- Email: 🟡 to 🔴 (tiny spend, hard to identify)
- Affiliates: 🟢 (instant decay, distinctive pattern)
- Holdout MAPE < 15%

**Top-1 correct (email)** is the hard test here. Most tools will not correctly identify email as the highest-ROI channel because its absolute contribution is tiny.

### `data_scarce` scenario

Expected pattern:
- TV: 🟡 (fewer flights = less identification power)
- Search: 🟡 (standard always-on)
- Social: 🟡 to 🔴
- TikTok: 🔴 (new channel, minimal historical variation)
- Holdout MAPE < 20%
- CI widths should be noticeably wider than in the simple scenario

Tools with informative priors for new/unknown channels should produce narrower, better-centred TikTok estimates.

### `adversarial` scenario

Expected pattern (all channels yellow or red):
- All channels: 🟡 to 🔴 (this is the correct outcome — the data does not support confident attribution)
- Wide credible intervals on all channels (CI width > 0.20 for TV ROI)
- Holdout MAPE < 25%
- **Key test:** Does the tool report wide uncertainty or falsely confident estimates?

For the adversarial scenario, a tool that reports R-hat warnings, high posterior variance, or explicitly flags "poor identification" is performing *better* than a tool that converges confidently to wrong ROIs.

---

## When to Trust Absolute vs Relative ROI

### Trust relative ROI when:
- Relative ROI Accuracy > 60%
- Spearman rho > 0.80
- Pairwise ranking > 70%
- Holdout MAPE < 20%

Use case: Budget allocation decisions — which channels to grow and which to reduce.

### Trust absolute ROI when:
- All of the above, plus:
- Absolute ROI Accuracy > 60%
- Business Sense Score = 100%
- No convergence warnings (R-hat < 1.05, divergences = 0)
- Channel has a flighted pattern (TV, OOH, YouTube) providing before/after identification, OR lift tests were included in the run

Use case: Payback modelling, investment cases, "will this campaign pay back in 12 months?"

### Trust neither when:
- R-hat > 1.05 on media channel parameters
- Divergences > 50
- Holdout MAPE > 25%
- Business Sense Score < 60%

Action: Diagnose the model before using results for any decision.

---

## Example: Interpreting a Regression

Quokka's post-release interpretation workflow. Suppose `pymc-marketing==0.20.0` is released and the benchmark shows:

```
v0.19.2:  Relative ROI Accuracy (simple) = 74.3%
v0.20.0:  Relative ROI Accuracy (simple) = 61.1%
```

A 13-percentage-point regression on the primary metric. Quokka's Claude Code analysis:

1. **Read the per-channel table for both versions.** Check which channels regressed.
2. **Check convergence.** Did v0.20.0 have more divergences? Higher R-hat on specific parameters?
3. **Check changelog.** What changed in v0.20.0 — prior specification, default adstock settings, sampling parameters?
4. **Form a hypothesis.** If the regression is concentrated on search and social (always-on channels), and v0.20.0 changed default priors, the new priors may be pulling always-on coefficients in the wrong direction.
5. **Propose a test.** Run v0.20.0 with custom priors matching v0.19.2's defaults to isolate the cause.
6. **Post to GitHub Discussions.** Document the regression, the hypothesis, and the proposed test. Tag the PyMC-Labs team.

This conversational interpretation loop — benchmark output → Claude Code analysis → hypothesis → experiment → GitHub Discussion — is the core value proposition of mmm-bench.

---

*See also:*
- *[Methodology](methodology.md) — full metric definitions*
- *[Scenarios](scenarios.md) — what good performance looks like on each scenario*
- *[Running locally](running-locally.md) — setup and execution*
