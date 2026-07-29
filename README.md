# PulseBO: Gaussian-Process Models for Cobalt Selectivity in Pulsed Electrodeposition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PulseBO** collects Gaussian-process (GP) models for **cobalt selectivity** in Co/Ni pulsed electrodeposition. It brings two complementary approaches to the same problem together under one project, sharing a dataset and the same engineered pulse-shape features (applied voltage, duty cycle, period, total on-time):

1. **Constrained Bayesian optimization** — a selectivity GP and a deposition GP drive a batch-BO loop that recommends the next experiments, treating deposition as a feasibility *constraint*.
2. **Deposition-informed selectivity model (DSMM)** — a stacked model that predicts deposition first and feeds that prediction in as an *input* to the selectivity model, giving a prospectively deployable selectivity predictor.

This repository currently implements approach 1 in the `pulse_bo` package (below); approach 2 is the companion model, summarized under Method Overview and added to the repository alongside it.

## 🎯 Key Features

**Approach 1 — Constrained Bayesian optimization**

- **Dual-GP, feasibility-aware surrogate**: a selectivity GP trained on feasible runs only, and a deposition GP trained on every run to learn where deposition fails.
- **Constrained Expected Improvement**: candidates are ranked by `CEI = EI × P(feasible)`, coupling the two GPs so exploration is penalized where deposition is likely to fall short.
- **Two-phase candidate generation**: a Sobol quasi-random sweep followed by L-BFGS-B refinement over the continuous pulse parameters, per voltage on a discrete grid.
- **Diverse explore/exploit batches**: greedy selection with a minimum normalised-distance filter and a guard that reallocates explore slots when the pool is prior-dominated (no data nearby).
- **Model diagnostics**: repeated k-fold cross-validation with calibration metrics (90% coverage, NLPD), a length-scale audit that flags effectively-ignored features, and out-of-fold parity plots.

**Approach 2 — Deposition-informed selectivity model (DSMM)**

- **Stacked predictor**: a deposition GP's prediction is added as an input feature to the selectivity model, so it stays usable *before* an experiment is run (unlike a model that needs measured deposition).
- **Benchmarked**: compared against a controls-only selectivity model and a measured-deposition upper bound to isolate how much deposition information helps.
- **Uncertainty propagation**: error from the upstream deposition model is carried into the selectivity prediction and quantified by nested bootstrap.

## 📋 Requirements

### System Requirements

- **OS**: Linux / macOS / Windows (CPU only — no GPU required)
- **Memory**: 4 GB+ RAM
- **Runtime**: the cross-validation grid fits many GPs and can take tens of minutes on a laptop; candidate generation adds a few minutes.

### Software Dependencies

- Python 3.10+
- NumPy, pandas, SciPy
- scikit-learn 1.3+ (Gaussian processes)
- openpyxl (Excel IO)
- matplotlib, seaborn (parity plotting)

## 🛠️ Installation

### Quick Start with Poetry

```bash
# Clone the repository
git clone https://github.com/<org>/PulseBO.git
cd PulseBO

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies (registers the pulse-bo-* console scripts)
poetry install

# Activate the virtual environment
poetry shell
```

### Alternative: pip + pinned requirements

```bash
python -m venv pulsebo_env
source pulsebo_env/bin/activate        # Linux/Mac
# pulsebo_env\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt        # exact, pinned versions
pip install -e .                       # editable install of the pulse_bo package
```

### Reproducibility

Dependencies are pinned to exact versions so results don't drift between library
releases: Poetry users get this from the committed `poetry.lock`, and pip users
from `requirements.txt`. To reproduce the **published** numbers exactly, install
from these pinned files rather than the version ranges in `pyproject.toml`. If
you re-run the analysis in a different environment, regenerate the pins
(`poetry lock`, or `pip freeze > requirements.txt`) so they match the environment
that produced your results.

## 📂 Project Structure

```
PulseBO/
├── pulse_bo/                     # Core implementation package
│   ├── config.py                 # All constants: columns, bounds, thresholds, CV/BO settings
│   ├── data/                     # Data loading and feature engineering
│   │   ├── parse.py              # Raw workbook -> single clean sheet (column-alias handling)
│   │   └── features.py           # Pulse-shape features, dataset extraction, scaling
│   ├── models/                   # Gaussian-process models
│   │   ├── gp.py                 # Kernels, GP construction, final fitting
│   │   └── evaluation.py         # Repeated k-fold CV, calibration, length-scale audit
│   ├── optimization/             # Constrained Bayesian optimization
│   │   ├── acquisition.py        # EI, probability of feasibility, CEI, UCB
│   │   ├── candidates.py         # Two-phase Sobol + L-BFGS-B candidate generation
│   │   └── batch.py              # Explore/exploit batch selection with diversity filter
│   ├── plotting.py               # Out-of-fold selectivity parity plot
│   ├── pipeline.py               # End-to-end workflow (CV -> fit -> candidates -> batch)
│   └── cli.py                    # pulse-bo-parse / pulse-bo-run / pulse-bo-parity
├── data/                         # cleaned_data.xlsx (model input) + raw_example_v6.xlsx (raw parser input)
│   ├── cleaned_data.xlsx         # Cleaned single-sheet input consumed by the GPs
│   └── raw_example_v6.xlsx       # Example raw multi-sheet workbook for the parser
├── experiments/                  # Reproducible runs that accompany the paper
│   └── v6_batch/                 # Batch recommendation on the v6 dataset
└── results/                      # Output CSVs, run log, and figures
```

## 🔬 Method Overview

Both approaches use Gaussian processes over engineered pulse-shape features — applied voltage, duty cycle (on-time / period), period (on + off time), and total on-time — which are less collinear than the raw on/off timings. A run is **feasible** if its total deposition clears the threshold in `config.DEP_BAD_THRESH`.

### Approach 1 — Constrained Bayesian optimization

The selectivity GP is fit on feasible runs only, while the deposition GP is fit on all runs so it can predict the feasibility probability of untested conditions. Each candidate is scored by **Constrained Expected Improvement**:

```
CEI(x) = EI_selectivity(x) × P(deposition(x) ≥ threshold)
```

Candidates are generated per voltage by a Sobol sweep refined with L-BFGS-B, then a batch is assembled from the highest-CEI (exploit) and highest-uncertainty (explore) points under a diversity filter.

### Approach 2 — Deposition-informed selectivity model (DSMM)

Rather than using deposition as a constraint, this approach uses it as information. A deposition GP predicts total deposition from the pulse parameters, and that prediction is added as an input to the selectivity model (a stacked model). Because it relies on *predicted* rather than *measured* deposition, it can be applied prospectively. It is benchmarked against a controls-only selectivity model and a measured-deposition upper bound, and uncertainty from the upstream deposition model is propagated into the selectivity prediction via nested bootstrap.

## 🚀 Usage

The commands below drive **approach 1** (the constrained-BO workflow). It has three steps; paths default to `data/cleaned_data.xlsx` and `results/`, and can be overridden with flags.

```bash
# 1. Clean a raw experiment workbook into a single sheet
pulse-bo-parse data/raw_example_v6.xlsx data/cleaned_data.xlsx

# 2. Run the full BO workflow (CV, fit, candidate generation, batch selection)
pulse-bo-run --data data/cleaned_data.xlsx --results results/

# 3. (Optional) Generate the out-of-fold selectivity parity plot
pulse-bo-parity --data data/cleaned_data.xlsx --out results/selectivity_gp_parity.pdf
```

`pulse-bo-run` writes `batch_recommendations.csv` (the next experiments to run) plus
`cv_selectivity_gp.csv`, `cv_deposition_gp.csv`, `length_scales_*.csv`, and the full scored
`candidates_full.csv` (pass `--no-candidates` to skip the large file).

**Data format.** The cleaned workbook (`data/cleaned_data.xlsx`) has one row per
experiment with columns: `Solution Label`, `Applied V`, `Von (ms)`, `Voff (ms)`,
`Total Von (s)`, `Total Von (ms)`, `Co selectivity`,
`Total amount of deposition (ppm)`, `(moles)`. The GPs consume four engineered
features derived from these (see Method Overview); `pulse-bo-parse` produces this
sheet from a raw workbook and drops any sheet using a non-zero stripping voltage.

### Updating the model with a new batch

1. Add the completed experiments as a new sheet in the next raw workbook (each row: ID, applied voltage, on-time (ms), off-time (ms), total on-time (s), Co selectivity (%), total deposition).
2. `pulse-bo-parse ED_QDPM_v7.xlsx data/cleaned_data.xlsx` — cleans the file, maps renamed columns, computes total on-time in ms, and drops sheets that use a non-zero stripping voltage.
3. `pulse-bo-run` — re-tunes with cross-validation, refits on the full dataset, and writes the next recommended batch.
4. Review the outputs: `batch_recommendations.csv` for the next experiments, `cv_selectivity_gp.csv` to confirm accuracy hasn't dropped, and `length_scales_selectivity.csv` to see whether any input has become uninformative.

### Library usage

```python
from pulse_bo import run_bo
batch, candidates = run_bo(data_file="data/cleaned_data.xlsx", results_dir="results")
```

## 📊 Reproducing Paper Results

The `experiments/` directory holds a self-contained script that regenerates the results reported in the paper:

```bash
cd experiments/v6_batch
python script.py            # equivalent to `pulse-bo-run`, writing into this folder
```

This writes, into the experiment folder:

| File                                                              | Description                                        |
| ----------------------------------------------------------------- | -------------------------------------------------- |
| `batch_recommendations.csv`                                       | Recommended next experiments (exploit + explore).  |
| `cv_selectivity_gp.csv` / `cv_deposition_gp.csv`                  | Cross-validation grids for each GP.                |
| `length_scales_selectivity.csv` / `length_scales_deposition.csv` | Per-feature length-scale audit.                    |
| `candidates_full.csv`                                             | Full scored candidate pool (large; git-ignored).   |
| `run.log`                                                         | Timestamped log of the run (metrics, kernels).     |

> The cross-validation grid fits many GPs and can take tens of minutes on a laptop.

## ⭐ Acknowledgements

This work builds on several open-source projects:

- [scikit-learn](https://scikit-learn.org/) — Gaussian processes and cross-validation
- [SciPy](https://scipy.org/) — Sobol sampling and L-BFGS-B optimization
- [NumPy](https://numpy.org/) / [pandas](https://pandas.pydata.org/) — numerics and data handling
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — figures

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{<citekey>,
  title   = {<Paper title>},
  author  = {<Author One> and <Author Two> and Yue, Shuwen},
  journal = {<Journal / Preprint>},
  year    = {2026},
  doi     = {<DOI>},
  note    = {Preprint}
}
```

## 📫 Contact

For questions and feedback:

- **Corresponding author**: `<Name>` (`<email>`)

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
