from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CV_FOLDS, CV_REPEATS, RANDOM_SEED, VALIDATION_RESULTS_DIR
from src.validated_gp_config import (
    BEST_DEPLOYED_DEPOSITION_GP,
    BEST_DEPLOYED_SELECTIVITY_GP,
    BEST_MODEL_A_GP,
    BEST_MODEL_B_GP,
)


FOLD_RESULTS_FILE = VALIDATION_RESULTS_DIR / "validation_all_fold_results.xlsx"
DEPLOYED_STACK_FOLD_RESULTS_FILE = (
    VALIDATION_RESULTS_DIR
    / "deployed_stack_config"
    / "deployed_stack_fold_results.xlsx"
)

OUTPUT_FILE = VALIDATION_RESULTS_DIR / "validation_per_fold_comparison.xlsx"
SUMMARY_FILE = (
    VALIDATION_RESULTS_DIR
    / "validation_per_fold_comparison_summary.xlsx"
)

CONTROLS_NAME = "Model A: controls only"
MEASURED_NAME = "Model B: controls + measured deposition"
STACKED_NAME = (
    "Stacked deployed pipeline: controls -> predicted deposition -> "
    "mechanistic selectivity"
)

EXPECTED_EVALUATIONS = CV_FOLDS * CV_REPEATS


def config_label(config: dict) -> str:
    return (
        f"kernel={config['kernel_type']}, "
        f"restarts={config['n_restarts_optimizer']}, "
        f"alpha={config['alpha']}"
    )


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {purpose}: {path}. Run the validation step that "
            f"creates this file before running per-fold comparison."
        )


def select_model_rows(
        fold_df: pd.DataFrame,
        model_name: str,
        config: dict,
) -> pd.DataFrame:
    selected = fold_df[
        (fold_df["model_name"] == model_name)
        & (fold_df["kernel_type"] == config["kernel_type"])
        & (
            fold_df["n_restarts_optimizer"].astype(int)
            == int(config["n_restarts_optimizer"])
        )
        & np.isclose(fold_df["alpha"].astype(float), float(config["alpha"]))
    ].copy()

    selected = selected.sort_values("evaluation").reset_index(drop=True)
    if len(selected) != EXPECTED_EVALUATIONS:
        raise ValueError(
            f"Expected {EXPECTED_EVALUATIONS} evaluations for "
            f"{model_name} ({config_label(config)}), but found "
            f"{len(selected)}."
        )

    return selected


def require_unique_config_value(
        df: pd.DataFrame,
        column: str,
        expected,
) -> None:
    if column not in df.columns:
        raise ValueError(
            f"Missing required deployed-stack column '{column}' in "
            f"{DEPLOYED_STACK_FOLD_RESULTS_FILE}."
        )

    values = df[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"Expected one value for deployed-stack column '{column}', "
            f"but found {values.tolist()}."
        )

    actual = values.iloc[0]
    if isinstance(expected, float):
        matches = np.isclose(float(actual), expected)
    else:
        matches = actual == expected

    if not matches:
        raise ValueError(
            f"Deployed-stack fold file does not match current config for "
            f"'{column}': expected {expected!r}, found {actual!r}. "
            f"Rerun scripts/03_evaluate_deployed_stack_config.py."
        )


def select_deployed_stack_rows(fold_df: pd.DataFrame) -> pd.DataFrame:
    selected = fold_df[fold_df["model_name"] == STACKED_NAME].copy()

    if len(selected) != EXPECTED_EVALUATIONS:
        raise ValueError(
            f"Expected {EXPECTED_EVALUATIONS} deployed-stack evaluations "
            f"in {DEPLOYED_STACK_FOLD_RESULTS_FILE}, but found "
            f"{len(selected)}."
        )

    require_unique_config_value(
        selected,
        "deposition_kernel_type",
        BEST_DEPLOYED_DEPOSITION_GP["kernel_type"],
    )
    require_unique_config_value(
        selected,
        "deposition_n_restarts_optimizer",
        BEST_DEPLOYED_DEPOSITION_GP["n_restarts_optimizer"],
    )
    require_unique_config_value(
        selected,
        "deposition_alpha",
        BEST_DEPLOYED_DEPOSITION_GP["alpha"],
    )
    require_unique_config_value(
        selected,
        "deposition_transform",
        BEST_DEPLOYED_DEPOSITION_GP["transform"],
    )
    require_unique_config_value(
        selected,
        "deposition_inverse_method",
        BEST_DEPLOYED_DEPOSITION_GP["inverse_method"],
    )
    require_unique_config_value(
        selected,
        "selectivity_kernel_type",
        BEST_DEPLOYED_SELECTIVITY_GP["kernel_type"],
    )
    require_unique_config_value(
        selected,
        "selectivity_n_restarts_optimizer",
        BEST_DEPLOYED_SELECTIVITY_GP["n_restarts_optimizer"],
    )
    require_unique_config_value(
        selected,
        "selectivity_alpha",
        BEST_DEPLOYED_SELECTIVITY_GP["alpha"],
    )

    return selected.sort_values("evaluation").reset_index(drop=True)


def assert_aligned_evaluations(
        controls: pd.DataFrame,
        measured: pd.DataFrame,
        stacked: pd.DataFrame,
) -> None:
    controls_eval = controls["evaluation"].to_numpy()
    if not np.array_equal(controls_eval, measured["evaluation"].to_numpy()):
        raise ValueError("Controls and measured-deposition folds are misaligned.")
    if not np.array_equal(controls_eval, stacked["evaluation"].to_numpy()):
        raise ValueError("Controls and deployed-stack folds are misaligned.")


def bootstrap_mean_ci(
        values: np.ndarray,
        n_bootstrap: int = 10000,
        random_state: int = RANDOM_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    bootstrap_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means[i] = np.mean(sample)

    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def summarize_rmse_comparison(
        delta: pd.Series,
        comparison_name: str,
        random_state: int,
) -> dict:
    """Positive delta means the alternative has lower RMSE than controls."""

    values = delta.dropna().to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(values, random_state=random_state)

    if np.allclose(values, 0):
        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0
    else:
        result = wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
        )
        wilcoxon_statistic = float(result.statistic)
        wilcoxon_p = float(result.pvalue)

    value_std = np.std(values, ddof=1)
    paired_cohens_d = (
        float(np.mean(values) / value_std)
        if value_std > 0
        else np.nan
    )

    return {
        "comparison": comparison_name,
        "n_paired_evaluations": len(values),
        "mean_rmse_improvement": float(np.mean(values)),
        "median_rmse_improvement": float(np.median(values)),
        "rmse_improvement_std": float(value_std),
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "fraction_evaluations_improved": float(np.mean(values > 0)),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_one_sided": wilcoxon_p,
        "paired_cohens_d": paired_cohens_d,
    }


def run_per_fold_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_file(FOLD_RESULTS_FILE, "full validation fold results")
    require_file(
        DEPLOYED_STACK_FOLD_RESULTS_FILE,
        "deployed-stack fold results",
    )

    fold_df = pd.read_excel(FOLD_RESULTS_FILE)
    deployed_stack_fold_df = pd.read_excel(DEPLOYED_STACK_FOLD_RESULTS_FILE)

    controls = select_model_rows(fold_df, CONTROLS_NAME, BEST_MODEL_A_GP)
    measured = select_model_rows(fold_df, MEASURED_NAME, BEST_MODEL_B_GP)
    stacked = select_deployed_stack_rows(deployed_stack_fold_df)
    assert_aligned_evaluations(controls, measured, stacked)

    paired = pd.DataFrame(
        {
            "evaluation": controls["evaluation"],
            "controls_rmse": controls["rmse"],
            "measured_deposition_rmse": measured["rmse"],
            "stacked_rmse": stacked["rmse"],
            "controls_mae": controls["mae"],
            "measured_mae": measured["mae"],
            "stacked_mae": stacked["mae"],
            "controls_r2": controls["r2"],
            "measured_r2": measured["r2"],
            "stacked_r2": stacked["r2"],
            "controls_coverage_90": controls["coverage_90"],
            "measured_coverage_90": measured["coverage_90"],
            "stacked_coverage_90": stacked["coverage_90"],
            "controls_nlpd": controls["nlpd"],
            "measured_nlpd": measured["nlpd"],
            "stacked_nlpd": stacked["nlpd"],
            "controls_config": config_label(BEST_MODEL_A_GP),
            "measured_config": config_label(BEST_MODEL_B_GP),
            "deployed_deposition_config": config_label(
                BEST_DEPLOYED_DEPOSITION_GP
            ),
            "deployed_deposition_transform": (
                BEST_DEPLOYED_DEPOSITION_GP["transform"]
            ),
            "deployed_deposition_inverse_method": (
                BEST_DEPLOYED_DEPOSITION_GP["inverse_method"]
            ),
            "deployed_selectivity_config": config_label(
                BEST_DEPLOYED_SELECTIVITY_GP
            ),
        }
    )

    paired["rmse_improvement_measured_vs_controls"] = (
        paired["controls_rmse"] - paired["measured_deposition_rmse"]
    )
    paired["rmse_improvement_stacked_vs_controls"] = (
        paired["controls_rmse"] - paired["stacked_rmse"]
    )
    paired["mae_improvement_measured_vs_controls"] = (
        paired["controls_mae"] - paired["measured_mae"]
    )
    paired["mae_improvement_stacked_vs_controls"] = (
        paired["controls_mae"] - paired["stacked_mae"]
    )
    paired["r2_improvement_measured_vs_controls"] = (
        paired["measured_r2"] - paired["controls_r2"]
    )
    paired["r2_improvement_stacked_vs_controls"] = (
        paired["stacked_r2"] - paired["controls_r2"]
    )
    paired["nlpd_improvement_measured_vs_controls"] = (
        paired["controls_nlpd"] - paired["measured_nlpd"]
    )
    paired["nlpd_improvement_stacked_vs_controls"] = (
        paired["controls_nlpd"] - paired["stacked_nlpd"]
    )

    summary = pd.DataFrame(
        [
            summarize_rmse_comparison(
                delta=paired["rmse_improvement_measured_vs_controls"],
                comparison_name="Measured deposition vs controls-only",
                random_state=RANDOM_SEED,
            ),
            summarize_rmse_comparison(
                delta=paired["rmse_improvement_stacked_vs_controls"],
                comparison_name="Deployed stacked vs controls-only",
                random_state=RANDOM_SEED + 1,
            ),
        ]
    )

    paired.to_excel(OUTPUT_FILE, index=False)
    summary.to_excel(SUMMARY_FILE, index=False)

    print("\nCompared current configured models:")
    print(f"Controls: {config_label(BEST_MODEL_A_GP)}")
    print(f"Measured deposition: {config_label(BEST_MODEL_B_GP)}")
    print(
        "Deployed deposition: "
        f"{config_label(BEST_DEPLOYED_DEPOSITION_GP)}, "
        f"transform={BEST_DEPLOYED_DEPOSITION_GP['transform']}, "
        f"inverse={BEST_DEPLOYED_DEPOSITION_GP['inverse_method']}"
    )
    print(f"Deployed selectivity: {config_label(BEST_DEPLOYED_SELECTIVITY_GP)}")

    print("\nSaved per-fold comparison files:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print("\nSaved per-fold comparison summary:")
    print(summary.to_string(index=False))
    return paired, summary


if __name__ == "__main__":
    run_per_fold_comparison()
