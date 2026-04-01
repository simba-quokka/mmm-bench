# Running mmm-bench Locally

This guide covers everything you need to run the full benchmark suite on Windows. It is intentionally thorough — MMM tools have complicated dependency chains, and several Windows-specific issues can derail a first-time setup.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11.x | 3.11.8 |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB free | 10 GB free |
| OS | Windows 10 (64-bit) | Windows 11 |
| CPU cores | 4 | 8+ |

**Why Python 3.11 specifically?**

- Meridian (Google) depends on TensorFlow, which has limited support for Python 3.12+
- PyMC / Aesara has tested compatibility on 3.11
- Decision-packs is tested on 3.11

**Do not use Python 3.12 or later** until you have confirmed that the specific versions of all three tools support it. If you must use a different Python version, run the smoke test first and check for import errors.

**Why 16 GB RAM?**

PyMC-Marketing runs 4 MCMC chains simultaneously. Each chain maintains its own copy of the model state, gradient computations, and sample arrays. For the `complex` scenario (8 channels, 156 weeks), peak memory usage can reach 6–8 GB. With 8 GB total RAM, you will hit swap and runtimes will increase dramatically (10× or more). If you must run on 8 GB, use `--chains 2` — see [Running with reduced resources](#running-with-reduced-resources).

---

## Installation (Windows Native Python)

### Step 1: Clone the repository

```bash
git clone https://github.com/simba-quokka/mmm-bench
cd mmm-bench
```

### Step 2: Create a virtual environment

```bash
# Using Python 3.11 explicitly (adjust path if your Python 3.11 is elsewhere)
py -3.11 -m venv .venv

# Or if python 3.11 is your default:
python -m venv .venv

# Activate
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your prompt.

### Step 3: Install the benchmarked tools

Install tools individually so that version conflicts surface clearly:

```bash
# PyMC-Marketing (installs PyMC, Aesara, Bambi, etc.)
pip install pymc-marketing==0.19.2

# Meridian (installs TensorFlow, TensorFlow Probability, JAX)
pip install google-meridian==1.5.3

# Decision-packs (requires a Modal account for cloud execution)
pip install decision-packs
```

**On Meridian:** The `google-meridian` package pulls in TensorFlow, which is a large download (200–500 MB). This is normal. TensorFlow will produce warning messages on import — these are harmless (see [Known Gotchas](#known-windows-gotchas) below).

**On Decision-packs:** This runner requires a Modal account and API token for cloud execution. Set up your Modal credentials with `modal token new` after installation. Decision-packs can be excluded from local runs with `--skip decision-packs`.

### Step 4: Install remaining dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers: `typer`, `pyyaml`, `pandas`, `numpy`, `scipy`, `matplotlib`, `pymc-marketing` optional extras, and `pytest` for the test suite.

### Step 5: Smoke test

Verify that all imports and data generation work without running any model:

```bash
python -c "
from scenarios import load_scenario
from data.generator import simulate_dataset

sc = load_scenario('simple')
df, gt = simulate_dataset(sc)
print('Dataset shape:', df.shape)
print('Channels:', list(gt.true_rois.keys()))
print('Ground truth ROIs:', gt.true_rois)
print('OK')
"
```

Expected output:
```
Dataset shape: (104, 19)
Channels: ['tv', 'paid_search', 'paid_social']
Ground truth ROIs: {'tv': 0.28, 'paid_search': 1.35, 'paid_social': 0.75}
OK
```

If you see import errors, check that your venv is active and re-run the installation steps.

---

## Known Windows Gotchas

### 1. PyMC multiprocessing guard

PyMC uses Python's `multiprocessing` module for parallel NUTS chains. On Windows, `multiprocessing` spawns new processes by default (rather than forking), which means the top-level script is re-imported in each child process. If you run `benchmark.py` directly, this is handled correctly via Typer's CLI entry point.

**If you write your own scripts** that call PyMC directly, you must guard with:

```python
if __name__ == '__main__':
    # your PyMC fitting code here
```

Without this, Windows will recursively spawn processes and hang.

### 2. TensorFlow warning spam

When Meridian is imported, TensorFlow prints several lines like:

```
2026-04-01 10:00:00.000000: W tensorflow/stream_executor/...
I tensorflow/core/util/port.cc:...
...
```

These are **completely harmless**. TensorFlow is logging its hardware detection and optimization settings. They do not indicate errors. To suppress them before running:

```bash
set TF_CPP_MIN_LOG_LEVEL=3
python benchmark.py --scenario simple --tool meridian
```

Or set the environment variable permanently in your session.

### 3. Path separators

The benchmark uses `pathlib.Path` throughout, which handles Windows paths (`C:\Users\...`) natively in Python. You should not encounter path separator issues running through the CLI.

However, if you are writing helper scripts, always use `pathlib.Path` or forward slashes in f-strings, not raw backslash paths.

### 4. `/tmp/` vs `C:\tmp\` for chart output

Python's `tempfile.gettempdir()` on Windows returns `C:\Users\<username>\AppData\Local\Temp`, not `/tmp/`. If you save charts or intermediate files using `/tmp/` paths in your scripts (a common Unix habit), they will go to `AppData\Local\Temp`, which is different from what MSYS2 or Git Bash considers `/tmp/`.

**Recommendation:** When saving output files from Python scripts, use `C:\tmp\` explicitly (create it first with `mkdir C:\tmp`) or use `pathlib.Path.home() / 'tmp'`. The benchmark itself saves all results to the `results/` directory in the repo — no `/tmp/` usage in the core code.

### 5. Screen sleep during long runs

PyMC-Marketing can take 5–15 minutes per scenario. Windows power management may throttle CPU or reduce clock speed when your display sleeps, causing 5–10× slowdowns in sampling.

**Solution:** Before a long benchmark run, disable screen sleep for the session:

```cmd
powercfg /change standby-timeout-ac 0
```

Re-enable when done:

```cmd
powercfg /change standby-timeout-ac 15
```

Or simply keep your screen on. This is the most common cause of unexpectedly slow PyMC runs on Windows laptops.

### 6. Encoding errors

All file I/O in mmm-bench uses `encoding="utf-8"` explicitly. However, if your Windows terminal is set to code page 1252 (the default in older Windows installs), you may see `UnicodeDecodeError` when reading result files or config YAMLs.

Check and fix:

```cmd
# Check current code page
chcp

# Switch to UTF-8
chcp 65001
```

Or add to your `.bashrc` / PowerShell profile:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## Running the Benchmark

### Basic usage

```bash
# Activate venv first
.venv\Scripts\activate

# Smoke test — no model fitting (~5 seconds)
python -c "from scenarios import load_scenario; from data.generator import simulate_dataset; sc = load_scenario('simple'); df, gt = simulate_dataset(sc); print('OK', df.shape)"

# Single tool, single scenario
python benchmark.py --scenario simple --tool pymc-marketing

# Single scenario, all tools
python benchmark.py --scenario simple

# All scenarios, all tools (long run)
python benchmark.py

# Skip Decision-packs (requires Modal account)
python benchmark.py --skip decision-packs

# After completing runs, update README leaderboard
python benchmark.py --update-readme
```

### All CLI options

```
python benchmark.py [OPTIONS]

Options:
  --scenario TEXT        Run only this scenario [simple|complex|data_scarce|adversarial]
  --tool TEXT            Run only this tool [pymc-marketing|meridian|decision-packs]
  --skip TEXT            Skip this tool (repeatable: --skip tool1 --skip tool2)
  --seed INTEGER         Override random seed [default: 42]
  --chains INTEGER       Override number of MCMC chains [default: 4]
  --draws INTEGER        Override number of posterior draws [default: 1000]
  --tune INTEGER         Override number of tuning steps [default: 1000]
  --update-readme        Update README.md leaderboard from latest results
  --results-dir PATH     Custom results directory [default: results/]
  --no-holdout           Skip holdout evaluation (faster, for debugging)
  --help                 Show this message and exit.
```

### Results directory structure

Each run creates a timestamped directory:

```
results/
└── 2026-04-01T17-59-12Z/
    ├── run_metadata.json          # Tool versions, seed, hardware info
    ├── ground_truth_simple.json   # True parameters for the simple scenario
    ├── ground_truth_complex.json
    ├── simple_pymc-marketing_logistic.json    # Per-tool results
    ├── simple_pymc-marketing_tanh.json
    ├── simple_meridian.json
    ├── simple_decision-packs.json
    ├── complex_pymc-marketing_logistic.json
    ├── ...
    └── summary.json               # Aggregated metrics table (used by --update-readme)
```

Each per-tool result JSON contains:
```json
{
  "tool_name": "pymc-marketing",
  "tool_version": "0.19.2",
  "scenario_name": "simple",
  "estimated_rois": {"tv": 0.261, "paid_search": 2.395, "paid_social": 2.525},
  "estimated_contribution_share": {"tv": 0.22, "paid_search": 0.48, "paid_social": 0.30},
  "credible_intervals": {"tv": [0.14, 0.42], "paid_search": [1.80, 3.10], "paid_social": [1.90, 3.20]},
  "converged": true,
  "convergence_warnings": [],
  "runtime_seconds": 322,
  "holdout_mape": 0.067,
  "in_sample_r2": 0.94,
  "metrics": {
    "relative_roi_accuracy": 0.743,
    "absolute_roi_accuracy": 0.382,
    "pairwise_ranking": 1.0,
    "spearman_rho": 1.0,
    "top1_correct": true,
    "holdout_mape": 0.067,
    "business_sense_score": 1.0
  }
}
```

---

## Expected Runtimes

Measured on a modern Windows laptop (Intel Core i7-1270P, 16 GB RAM, screen on, no GPU).

| Tool | `simple` | `complex` | `data_scarce` | `adversarial` | Notes |
|------|----------|-----------|--------------|--------------|-------|
| PyMC-Marketing (Logistic) | 5–10 min | 12–25 min | 4–8 min | 8–15 min | 4 chains × 1000 draws |
| PyMC-Marketing (Tanh) | 5–10 min | 12–25 min | 4–8 min | 8–15 min | Same settings |
| Meridian | 2–4 min | 5–10 min | 2–4 min | 4–8 min | TF/JAX gradient-based HMC is faster |
| Decision-packs | Varies | Varies | Varies | Varies | Cloud execution via Modal |

**Total benchmark suite (all tools, all scenarios):** approximately 2.5–4 hours on recommended hardware.

**Note on Meridian speed:** Meridian uses TensorFlow's implementation of HMC, which benefits from XLA compilation. The first run for each session includes JIT compilation overhead (~30–60 seconds extra). Subsequent runs in the same Python session are faster.

---

## Running with Reduced Resources

If you have limited RAM or CPU:

```bash
# 2 chains instead of 4 (halves memory usage, doubles variance in estimates)
python benchmark.py --chains 2

# Fewer draws (faster but noisier posterior)
python benchmark.py --draws 500 --tune 500

# Run only the simple scenario (the fastest)
python benchmark.py --scenario simple

# Skip holdout evaluation (faster, but holdout MAPE won't be available)
python benchmark.py --no-holdout
```

Results from reduced-resource runs are valid but will show higher variance in metrics. For leaderboard-quality results, use the default settings (4 chains, 1000 draws, 1000 tune).

---

## Using Claude Code to Interpret Results

After running the benchmark, use Claude Code to interpret the per-channel output:

```bash
# In Claude Code, point it at your results directory:
# "Read results/2026-04-01T17-59-12Z/summary.json and interpret the per-channel
#  attribution for pymc-marketing on the complex scenario"
```

Claude Code will:
1. Read the per-channel ROI table and colour-code errors (green/yellow/red)
2. Identify patterns — which channels are over- or under-attributed and why
3. Explain root causes — always-on identification, correlated channels, saturation mismatch
4. Propose next experiments — e.g., "run with lift tests to constrain always-on channels"

See [docs/interpreting-results.md](interpreting-results.md) for the full workflow.

---

## Running the Test Suite

The benchmark includes unit tests for the DGP and metric computations:

```bash
# Run all tests (~30 seconds, no model fitting)
pytest tests/

# Verbose output
pytest tests/ -v

# Run only DGP tests
pytest tests/test_generator.py

# Run only metric tests
pytest tests/test_metrics.py
```

Tests do not require any benchmarked tool to be installed — they only test the `data/` and `metrics/` modules.

---

*See also:*
- *[Interpreting results](interpreting-results.md) — how to read and act on benchmark output*
- *[Methodology](methodology.md) — what the metrics measure*
- *[Adding a tool](adding-a-tool.md) — how to contribute a new runner*
