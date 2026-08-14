# Gaussian-process workflow for pulsed electrodeposition

This repository contains an internal scientific Python workflow for modeling cobalt/nickel separation experiments using pulsed electrodeposition features, Gaussian-process regression, and Bayesian-optimization candidate scoring.


## Repository layout

```text
src/          Shared Python modules for features, models, validation, plotting, and BO logic
scripts/      Command-line entry points for validation, configuration selection, plotting, and candidates
data/         Internal input data
results/      Generated validation outputs, candidate tables, and figures
tests/        Characterization tests for current behavior
archive/      Historical working files retained for reference
```

## Environment

Create and activate a Python environment, then install the recorded dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The repository has been developed on Windows with Excel files read through `openpyxl`.

## Data Configuration

All scripts read the shared repository dataset through `src/data.py`. The paths,
column names, and feasibility threshold are configured in `src/config.py`:

```text
workbook            ../data/cleaned_data.xlsx   (285 runs)
deposition column   Total amount of deposition (ppm)
target column       Co selectivity
threshold           0.2942 ppm  (equivalent to 5e-9 mol)
```

Models are trained on the 190 runs at or above the deposition threshold.
Feasibility is defined by that threshold rather than by a missing selectivity
value: the shared dataset reports a selectivity number for every run, including
sub-threshold runs where that number is not physically meaningful, so a
null-based filter would silently train on all 285 rows.

Deposition MAE and RMSE are therefore reported in ppm. R2 is unaffected by the
choice of deposition units.

Change these values only when intentionally starting a new scientific run.

## Main Workflow

Run the core reproducibility workflow in this order:

```powershell
python scripts\01_validate_models.py
python scripts\02_select_best_gp.py
python scripts\03_evaluate_deployed_stack_config.py
python scripts\04_compare_validation_folds.py
```

The default validation run selects component models only:

- Model A: controls-only selectivity GP
- Model B: controls plus measured-deposition selectivity GP
- deposition GP trained and validated on raw deposition

The deployed stack is evaluated separately with:

- deposition GP configuration from `BEST_DEPOSITION_GP`
- raw deposition target transform
- selectivity GP configuration from `BEST_MODEL_B_GP`

Run the old stacked-kernel grid only as an explicit diagnostic:

```powershell
python scripts\01_validate_models.py --include-stack-grid
```

## Routine Retraining

For routine BO updates after adding new completed experiments, keep the deployed
GP configurations in `src/validated_gp_config.py` fixed and refit the deployed
stack on the updated dataset. Do not rerun model selection unless the goal is
to intentionally revalidate kernels and hyperparameters.

Routine update:

```powershell
python scripts\03_evaluate_deployed_stack_config.py
python scripts\04_compare_validation_folds.py
python scripts\05_run_bo_candidates.py
```

Architecture revalidation:

```powershell
python scripts\01_validate_models.py
python scripts\02_select_best_gp.py
python scripts\03_evaluate_deployed_stack_config.py
python scripts\04_compare_validation_folds.py
python scripts\05_run_bo_candidates.py
```

`scripts\01_validate_models.py` and `scripts\02_select_best_gp.py` may change
the selected kernel, `alpha`, restart settings, and deployed-stack
configuration. Use them only when changing or revalidating the model-selection
basis, not for ordinary retraining after a new experimental batch.

## Analyses

```powershell
python scripts\05_run_bo_candidates.py
python scripts\06_baseline_comparison.py
python scripts\07_ablation_tests.py
python scripts\08_monte_carlo_error_propagation.py
python scripts\09_bootstrap_error_propagation.py
python scripts\10_make_plots.py
python scripts\11_plot_deployed_stack_parity.py
python scripts\12_check_validation_outputs.py
```


## Tests

The tests are characterization tests. They are intended to protect current scientific behavior, not redefine it.

```powershell
python -m unittest discover -s tests
```

For reviewer-facing checks that do not modify scientific outputs, run:

```powershell
python -m compileall -q src scripts tests
python scripts\12_check_validation_outputs.py
```

