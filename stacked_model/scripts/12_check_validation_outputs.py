import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
outdir = PROJECT_ROOT / "results" / "validation"

summary = pd.read_excel(outdir / "validation_summary.xlsx")
folds = pd.read_excel(outdir / "validation_all_fold_results.xlsx")
preds = pd.read_excel(outdir / "validation_all_predictions.xlsx")

print("Summary rows:", len(summary))
print(summary.groupby("model_name").size())

print("\nEvaluations per model/setup:")
check = (
    folds
    .groupby(["model_name", "kernel_type", "n_restarts_optimizer", "alpha"])
    .size()
    .reset_index(name="n_fold_evaluations")
)

print(check)

print("\nAny setup not equal to 50?")
print(check[check["n_fold_evaluations"] != 50])

print("\nPrediction counts per model/setup:")
pred_check = (
    preds
    .groupby(["model_name", "kernel_type", "n_restarts_optimizer", "alpha"])
    .size()
    .reset_index(name="n_predictions")
)

print(pred_check)

print("\nBest setup per model by RMSE:")
best = (
    summary
    .sort_values("rmse_mean", ascending=True)
    .groupby("model_name", as_index=False)
    .first()
)

print(best[
    [
        "model_name",
        "kernel_type",
        "n_restarts_optimizer",
        "alpha",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "coverage_90_mean",
        "nlpd_mean",
        "lml_mean",
    ]
])

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

best = (
    summary
    .sort_values("rmse_mean", ascending=True)
    .groupby("model_name", as_index=False)
    .first()
)

print(best[
    [
        "model_name",
        "kernel_type",
        "n_restarts_optimizer",
        "alpha",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
        "coverage_90_mean",
        "coverage_90_std",
        "nlpd_mean",
        "nlpd_std",
        "lml_mean",
        "lml_std",
    ]
].to_string(index=False))
