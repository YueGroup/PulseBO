# Reproducibility notes

This document records the current internal workflow and assumptions needed to reproduce the saved model-selection and deployed-stack validation outputs.

## Data input

The configured input file is:

```text
data/raw/cleaned_data.xlsx
```

The configured sheet is `0`. Column names are stripped before feature engineering.

Core columns used by the workflow include:

- `Applied V`
- `Von (ms)`
- `Voff (ms)`
- `Total Von (s)`
- `Total amount of deposition (moles)`
- `Co selectivity`

Rows with missing values in required model-specific columns are dropped inside the relevant training or validation function.

## Randomness

The central random seed is configured in `src/config.py` as `RANDOM_SEED`.

Repeated cross-validation uses:

- `CV_FOLDS`
- `CV_REPEATS`
- `RANDOM_SEED`

Candidate generation uses NumPy's `default_rng` initialized from `RANDOM_SEED`.

## Model-selection workflow

Run:

```powershell
python scripts\01_validate_models.py
python scripts\02_select_best_gp.py
```

This selects:

- `BEST_MODEL_A_GP`
- `BEST_MODEL_B_GP`
- `BEST_DEPOSITION_GP`

The deposition model is selected from raw-deposition validation rows.

## Deployed-stack workflow

Run:

```powershell
python scripts\03_evaluate_deployed_stack_config.py
python scripts\04_compare_validation_folds.py
```

The deployed stack uses:

- `BEST_DEPLOYED_DEPOSITION_GP`
- `BEST_DEPLOYED_SELECTIVITY_GP`

The deployed deposition GP uses raw deposition with the configured inverse method. The selectivity GP uses the Model B selected configuration.

## Recorded validation artifacts

Current key artifact hashes are:

```text
results/validation/validation_summary.xlsx
SHA256 0545AB056C6244EE515D1E59B2A96F67E56F9E75CD97F31136E39769A380C70C

results/validation/validation_all_fold_results.xlsx
SHA256 D535FDDC01F6CA98FE2B026846E50170064C928E167471C1C36C968AE0CB498A

results/validation/deployed_stack_config/deployed_stack_summary.xlsx
SHA256 03EDEAE8D1E851AFE5BC5B0F3AFA82ADAB821B0063E479F2B59849D3857F18D4

results/validation/validation_per_fold_comparison_summary.xlsx
SHA256 859CD2F8E93A9649F333FB76D3785D63A056872DC743C1BAEB152F46FE3A0B48
```

Refresh these hashes only after an intentional rerun that changes the saved validation artifacts.


