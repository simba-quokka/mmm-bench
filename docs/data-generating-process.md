# Data-Generating Process

This document is the full technical specification of how mmm-bench synthetic datasets are produced. Every parameter listed here is saved as ground truth before any benchmarked tool sees the data. Familiarity with this document is essential for interpreting benchmark results correctly.

---

## Overview

The DGP simulates a CPG brand selling a consumer product across multiple media channels over 2–3 years of weekly data. It generates:

- **Weekly spend per channel** — with realistic patterns (always-on, flighted, seasonal burst)
- **Weekly impressions / GRPs** — derived from spend via a noisy CPM process
- **Control variables** — price index, distribution, competitor spend, seasonal indicators
- **KPI (revenue in $)** — assembled from known components with known coefficients
- **Ground truth file** — all true parameters recorded: coefficients, decay rates, saturation alphas, baseline, trend

The DGP is implemented in `data/generator.py`. All scenarios call `simulate_dataset(scenario_config, seed=42)`, which returns a `(DataFrame, GroundTruth)` tuple.

---

## Step 1: Spend Generation

Each channel is assigned one of three spend patterns:

### `always_on`

Used for digital performance channels: paid search, paid social, display, affiliates.

```python
spend_t = max(0, Normal(mu=spend_mean, sigma=spend_sigma))
```

The channel is active every week. Spend varies modestly around the mean. `spend_sigma / spend_mean` (the coefficient of variation) is typically 0.10–0.25 for always-on channels. This low variation is the root of the always-on identification problem — there is little contrast for the model to attribute.

### `flighted`

Used for brand channels: TV, OOH.

Spend alternates between "on" bursts and "off" periods in 4-week blocks:

```python
# flight schedule: on for 4 weeks, off for 4 weeks, repeat
on_indicator = flight_pattern(t, flight_on_rate)  # 0 or 1

if on_indicator:
    spend_t = max(0, Normal(mu=spend_mean_on, sigma=spend_sigma_on))
else:
    spend_t = max(0, Normal(mu=spend_mean_off, sigma=spend_sigma_off))
    # spend_mean_off << spend_mean_on; often 0 or residual
```

`flight_on_rate` controls what fraction of periods are "on" — typically 0.4–0.6 for TV (campaigns run roughly half the year). This before/after contrast is what makes flighted channels identifiable.

### `seasonal_burst`

Used for holiday-heavy categories (e.g., confectionery, gifts). Spend concentrates in weeks 40–52 (Q4) each year:

```python
q4_indicator = (week_of_year >= 40)
spend_t = max(0, Normal(
    mu = spend_mean_peak if q4_indicator else spend_mean_base,
    sigma = spend_sigma
))
```

### Correlated channels

In the `complex` and `adversarial` scenarios, some channels have correlated spend (shared planning cycles):

```python
spend_i_t = spend_mean_i + rho * (spend_j_t - spend_mean_j) + epsilon_i_t
```

where `rho` is the cross-channel correlation coefficient (e.g., 0.65 for TV-OOH, 0.55 for search-social, 0.85 for the adversarial scenario). The correlation is introduced at the spend level, which propagates through to impressions.

---

## Step 2: Impressions from Spend

Impressions (or GRPs) are derived from spend via a noisy CPM process:

```
impressions_t = spend_t / CPM_t × 1000

where CPM_t = CPM_mean × (1 + epsilon_t),
      epsilon_t ~ Normal(0, CPM_cv),
      clipped to [0.3 × CPM_mean, 3.0 × CPM_mean]
```

`CPM_cv` is the coefficient of variation of CPM — the week-to-week efficiency noise. This reflects:
- Programmatic auction dynamics (display, social)
- Scatter vs upfront TV buys (different CPM volatility)
- Seasonal CPM inflation (Q4 programmatic CPMs rise 30–60%)

The ±3× clipping prevents extreme outliers that would produce implausible impression spikes.

### Typical CPM values by channel

| Channel | CPM mean | CPM_cv | Notes |
|---------|----------|--------|-------|
| TV (national) | $20 | 0.12 | Broadcast CPM; upfront buys are more stable than scatter |
| OOH (national) | $6 | 0.15 | Out-of-home; lower CPM, broad reach, less auction volatility |
| Paid search | $8 | 0.20 | CPC-to-impression equivalent; auction volatility meaningful |
| Paid social | $10 | 0.22 | Facebook/Instagram CPM; high Q4 inflation |
| Display (programmatic) | $4 | 0.25 | Highest auction volatility; CPM swings substantially |
| Email | $0.5 | 0.10 | Cost per send; very stable |
| YouTube / video | $12 | 0.15 | Video CPM; TrueView pricing adds some volatility |
| Affiliates | $3 | 0.20 | CPA-adjacent; variable efficiency |

**Why this matters for benchmarking:** Tools that pass spend directly as their activity variable (rather than impressions) will see a cleaner, smoother signal than tools using impressions. The CPM noise creates realistic divergence between the spend series and the impressions series. Tools designed to use impressions (like Meridian) are working with a noisier but more correctly-specified input.

---

## Step 3: Adstock (Carryover)

Adstock transforms the weekly impressions series into a distributed-lag "effective impressions" series that captures carryover effects.

### Geometric adstock

Simple exponential decay. Used for most channels.

```
adstock_t = impressions_t + decay × adstock_{t-1}
```

Equivalently, `adstock_t = Σ_{l=0}^{∞} decay^l × impressions_{t-l}`, which is a weighted sum of all past impressions with exponentially declining weights.

- `decay ∈ [0.05, 0.15]` — fast decay, close to zero carryover (display, affiliates, email)
- `decay ∈ [0.30, 0.50]` — medium carryover (paid search, paid social)
- `decay ∈ [0.55, 0.70]` — slow decay, long carryover (TV brand building, OOH)

### Delayed adstock

Used for brand-building channels where awareness builds before the sales effect peaks (TV, YouTube). The impulse response function first rises to a peak, then decays.

Kernel weights for lag `l`:

```
w_l = l / peak_delay          if l <= peak_delay
w_l = decay ^ (l - peak_delay) if l > peak_delay

adstock_t = Σ_l  w_l × impressions_{t-l}  (weights normalised to sum = 1)
```

`peak_delay` is the lag at which the effect peaks — typically 1–3 weeks for TV (awareness registered, purchase follows). This means the DGP TV contribution in week `t` depends on impressions in weeks `t-3` through `t+0`, with the effect of impressions from `t-2` being strongest.

Geometric adstock is a special case of delayed adstock with `peak_delay = 0`.

---

## Step 4: Saturation

Saturation transforms the adstock series through a diminishing-returns function:

```
sat_t = tanh(adstock_t / (scalar × α))

where scalar = max(adstock_series)     [computed from the data before fitting]
      α      = saturation shape parameter (true value recorded; tools must estimate it)
```

**The scalar:** Computed as the maximum of the channel's adstock series. This fixes the scale of the saturation function to the data range, ensuring `sat_t ∈ [0, tanh(1/α)]` regardless of the absolute level of impressions. This is mathematically identical to PyMC-Marketing's TanhSaturation implementation.

**The alpha parameter (α):**
- `α < 1` (e.g., 0.5): steep saturation — the channel saturates quickly. Typical for large-reach brand channels (TV, OOH) where additional GRPs deliver progressively less incremental reach.
- `α = 1`: standard saturation curve.
- `α > 1` (e.g., 2.0): more linear, less saturation — typical for small-volume always-on channels (email, affiliates) that are far from their saturation point.

**Clipping:** The input to `tanh` is clipped to `[-20, 20]` to prevent numerical overflow.

---

## Step 5: Channel Contribution

```
contribution_t(ch) = true_coefficient(ch) × sat_t(ch)
```

`true_coefficient(ch)` is calibrated such that the total channel contribution over the scenario period, divided by total spend, equals the target ROI:

```
true_coefficient(ch) = target_ROI(ch) × total_spend(ch) / sum_t(sat_t(ch))
```

This calibration step ensures that the true ROI (as defined for the benchmark) is exactly the target ROI specified in the scenario configuration. The `true_coefficient` and `target_ROI` are both recorded as ground truth.

---

## Step 6: Control Variables

Each control variable is assigned a pattern:

### `ar1` — Autoregressive order 1

```
ctrl_t = rho × ctrl_{t-1} + sigma × epsilon_t,  epsilon_t ~ Normal(0, 1)
```

Used for: price index (rho=0.85), competitor spend (rho=0.70). AR1 creates a persistent, slowly-evolving series that resembles real price and competitive dynamics.

### `trend` — Linear trend

```
ctrl_t = intercept + slope × t / T
```

Used for: distribution ACV (the fraction of stores carrying the product builds over time). `T` is the total number of periods. The trend is normalised so that `ctrl` moves from `intercept` to `intercept + slope` over the scenario.

### `seasonal` — Annual Fourier seasonality

```
ctrl_t = amplitude × (sin(2π t / 52) + 0.4 × sin(4π t / 52))
```

Used for: category temperature effect, advertising seasonality index. Two Fourier terms (fundamental + first harmonic) produce a realistic annual cycle with asymmetric peaks.

### `binary` — Event indicator

```
ctrl_t ~ Bernoulli(p)
```

Used for: trade promotions, listing events, major sporting occasions. `p` is the probability of the event occurring in any given week (typically 0.10–0.20).

---

## Step 7: KPI Assembly

The full KPI is assembled additively:

```
KPI_t = baseline
      + trend_slope × t
      + seasonality_amplitude × baseline × (sin(2π t / 52) + 0.3 × sin(4π t / 52))
      + Σ_ch   true_coefficient(ch) × sat_t(ch)
      + Σ_ctrl true_coefficient(ctrl) × ctrl_t
      + ε_t,   ε_t ~ Normal(0, noise_sigma)
```

**baseline:** The intercept — the KPI level with zero media spend and controls at zero. Represents structural brand equity, repeat purchasing, and non-modelled effects. Calibrated as a fraction of total KPI (typically 40–60%).

**trend_slope:** A linear trend in the KPI over time. Can be positive (growing brand) or negative (mature declining category — the adversarial scenario uses a downward trend to create structural confounding).

**seasonality_amplitude:** Controls the peak-to-trough variation in the KPI's seasonal pattern. At amplitude=0.20 (the default for most scenarios), the KPI varies by approximately ±20% of baseline due to seasonality.

**noise_sigma:** The standard deviation of the iid noise term. Calibrated as a fraction of the KPI standard deviation. Higher noise makes all channels harder to identify. The adversarial scenario uses high noise ($60k sigma on a ~$5M/week KPI) to stress-test tools.

---

## Ground Truth ROI Definition

```
true_ROI(ch) = Σ_t contribution_t(ch) / Σ_t spend_t(ch)
```

This is the **spend-denominated ROI**: total incremental KPI contribution divided by total dollars spent. It is:

- **Unitless** when KPI is in dollars (contribution in $ / spend in $)
- **Spend-denominated** (never impressions-denominated)
- **Cumulative over the full scenario period** (not a weekly figure)
- **Inclusive of adstock carryover** (contribution includes the carried-over effect of past impressions)

The carryover point is important. If TV runs a 4-week flight and then goes dark, the TV contribution continues for several weeks after the flight ends (depending on the decay rate). The true ROI accounts for this full contribution including the tail.

---

## Reproducibility

All scenarios use fixed random seeds:
- `seed=42` is the default for benchmark runs
- Changing to `seed=43` produces a different noise draw (different `ε_t`) but uses the **same DGP parameters** (same true ROIs, same decay rates, same saturation alphas)
- This allows "same DGP, different data" experiments to test noise sensitivity

The ground truth file for each run is saved as `results/YYYY-MM-DDTHH-MM-SSZ/ground_truth_{scenario}.json` alongside the tool results.

---

## Scenario Configuration Format

Scenarios are defined in YAML files in `scenarios/`. Each file specifies:

```yaml
scenario_name: simple
n_weeks: 104
seed: 42
baseline: 3_000_000
trend_slope: 0
noise_sigma: 30_000
seasonality_amplitude: 0.15

channels:
  - name: tv
    spend_pattern: flighted
    spend_mean_on: 150_000
    spend_mean_off: 5_000
    spend_sigma: 20_000
    flight_on_rate: 0.5
    cpm_mean: 20
    cpm_cv: 0.12
    adstock_type: delayed
    peak_delay: 2
    decay: 0.60
    alpha: 0.70
    target_roi: 0.280

  - name: paid_search
    spend_pattern: always_on
    spend_mean: 80_000
    spend_sigma: 12_000
    cpm_mean: 8
    cpm_cv: 0.20
    adstock_type: geometric
    decay: 0.35
    alpha: 1.20
    target_roi: 1.350

  # ... additional channels

controls:
  - name: price_index
    pattern: ar1
    rho: 0.85
    sigma: 0.04
    true_coefficient: -0.12

  # ... additional controls
```

See [docs/scenarios.md](scenarios.md) for the full per-scenario configurations.

---

*See also:*
- *[Scenarios](scenarios.md) — per-scenario parameter tables*
- *[Methodology](methodology.md) — design philosophy and metric definitions*
