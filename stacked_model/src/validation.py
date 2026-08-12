"""
Runs cross validation analyses for selectivity, deposition, and the stacked deployed model.

Active model selection is component based:
- Model A selects the controls-only selectivity GP.
- Model B selects the measured-deposition selectivity GP.
- The deposition model selects the raw-deposition surrogate GP.

The deployed stack is evaluated separately with the selected deposition GP and
selected Model B GP so that per-fold comparisons reflect the actual deployed
pipeline rather than a shared stacked-kernel grid.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from itertools import product

from src.config import (
    EXCEL_FILE,
    SHEET_NAME,
    CV_FOLDS,
    CV_REPEATS,
    RANDOM_SEED,
    TARGET_COL,
    RAW_DEPOSITION_COL,
    VALIDATION_RESULTS_DIR,
)
from src.features import (
    add_engineered_features,
    make_base_controllable_input_cols,
    make_model_input_cols,
    make_deposition_input_cols,
)
from src.models import (
    train_selectivity_model,
    train_deposition_model,
    predict_gp,
    predict_deposition_logspace,
    deposition_prediction_to_raw,
)

GP_KERNEL_TYPES = [
    "rbf",
    "matern_1.5",
]

GP_RESTART_VALUES = [
    0,
    10,
]

GP_ALPHA_VALUES = [
    1e-10,
    1e-6,
    1e-2,
]

DEPLOYED_DEPOSITION_TRANSFORM = "raw"
DEPLOYED_DEPOSITION_INVERSE_METHOD = "mean"

def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2

def coverage_90_from_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        z_value: float = 1.645,
) -> float:
    y_std = np.maximum(y_std, 1e-12)

    lower = y_pred - z_value * y_std
    upper = y_pred + z_value * y_std

    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def nlpd_from_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        eps: float = 1e-12,
) -> float:
    y_var = np.maximum(y_std ** 2, eps)

    nlpd = (
        0.5 * np.log(2.0 * np.pi * y_var)
        + 0.5 * ((y_true - y_pred) ** 2 / y_var)
    )

    return float(np.mean(nlpd))


def summarize_fold_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_df
        .groupby(["model_name", "kernel_type", "n_restarts_optimizer", "alpha"], as_index=False)
        .agg(
            n_evaluations=("evaluation", "count"),
            n_total_test_points=("test_n", "sum"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            coverage_90_mean=("coverage_90", "mean"),
            coverage_90_std=("coverage_90", "std"),
            nlpd_mean=("nlpd", "mean"),
            nlpd_std=("nlpd", "std"),
            lml_mean=("lml", "mean"),
            lml_std=("lml", "std"),
        )
        .sort_values("rmse_mean", ascending=True)
        .reset_index(drop=True)
    )

def inverse_log1p_mean(mu_log: np.ndarray) -> np.ndarray:
    return np.expm1(mu_log)

def monte_carlo_propagate_deposition_uncertainty(
        mech_bundle,
        test_df: pd.DataFrame,
        controllable_cols: list[str],
        mech_cols: list[str],
        dep_mu: np.ndarray,
        dep_sigma: np.ndarray,
        deposition_transform: str = DEPLOYED_DEPOSITION_TRANSFORM,
        n_mc: int = 2000,
        random_state: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """
    Propagate the deposition GP posterior predictive distribution through the downstream selectivity GP.

    The returned total selectivity variance follows the law of total variance
    """

    if n_mc < 2:
        raise ValueError("n_mc must be at least 2.")

    rng = np.random.default_rng(random_state)

    dep_mu = np.array(dep_mu, dtype=float).reshape(-1)
    dep_sigma = np.maximum(
        np.asarray(dep_sigma, dtype=float).reshape(-1),
        1e-12,
    )

    n_test = len(test_df)

    if len(dep_mu) != n_test or len(dep_sigma) != n_test:
        raise ValueError(
            "Deposition prediction arrays must match the test-fold size."
        )

    # Shape: n_test x n_mc
    deposition_target_samples = rng.normal(
        loc=dep_mu[:, None],
        scale=dep_sigma[:, None],
        size=(n_test, n_mc),
    )

    if deposition_transform == "log1p":
        deposition_samples = np.maximum(
            np.expm1(deposition_target_samples),
            0.0,
        )
    elif deposition_transform == "raw":
        deposition_samples = np.maximum(deposition_target_samples, 0.0)
    else:
        raise ValueError(f"Unknown deposition_transform: {deposition_transform}")

    selectivity_mean_samples = np.empty(
        (n_test, n_mc),
        dtype=float,
    )

    selectivity_variance_samples = np.empty(
        (n_test, n_mc),
        dtype=float,
    )

    base_X = test_df[controllable_cols].copy()

    for sample_idx in range(n_mc):
        X_mc = base_X.copy()

        X_mc[RAW_DEPOSITION_COL] = (
            deposition_samples[:, sample_idx]
        )

        mu_mc, sigma_mc = predict_gp(
            mech_bundle,
            X_mc[mech_cols].values.astype(float),
        )

        selectivity_mean_samples[:, sample_idx] = mu_mc

        selectivity_variance_samples[:, sample_idx] = (
            np.maximum(sigma_mc, 1e-12) ** 2
        )

    # Predictive mean averaged over uncertain deposition.
    selectivity_mc_mean = np.mean(
        selectivity_mean_samples,
        axis=1,
    )

    # Variance introduced by uncertainty in predicted deposition.
    propagated_deposition_variance = np.var(
        selectivity_mean_samples,
        axis=1,
        ddof=1,
    )

    # Average conditional uncertainty from the selectivity GP.
    conditional_selectivity_variance = np.mean(
        selectivity_variance_samples,
        axis=1,
    )

    # Law of total variance.
    total_selectivity_variance = (
        propagated_deposition_variance
        + conditional_selectivity_variance
    )

    selectivity_std_from_deposition = np.sqrt(
        np.maximum(propagated_deposition_variance, 0.0)
    )

    conditional_selectivity_std = np.sqrt(
        np.maximum(conditional_selectivity_variance, 0.0)
    )

    total_selectivity_std = np.sqrt(
        np.maximum(total_selectivity_variance, 1e-12)
    )

    # Draw from the conditional selectivity distributions to create the full posterior predictive mixture.
    selectivity_draws = rng.normal(
        loc=selectivity_mean_samples,
        scale=np.sqrt(selectivity_variance_samples),
    )

    return {
        "deposition_mc_mean": np.mean(
            deposition_samples,
            axis=1,
        ),
        "deposition_mc_std": np.std(
            deposition_samples,
            axis=1,
            ddof=1,
        ),
        "selectivity_mc_mean": selectivity_mc_mean,
        "selecitivity_mc_mean": selectivity_mc_mean,
        "selectivity_std_from_deposition": (
            selectivity_std_from_deposition
        ),
        "conditional_selectivity_std": (
            conditional_selectivity_std
        ),
        "total_selectivity_std": total_selectivity_std,
        "selectivity_mc_p05": np.quantile(
            selectivity_draws,
            0.05,
            axis=1,
        ),
        "selectivity_mc_p50": np.quantile(
            selectivity_draws,
            0.50,
            axis=1,
        ),
        "selectivity_mc_p95": np.quantile(
            selectivity_draws,
            0.95,
            axis=1,
        ),
    }



def make_repeated_cv() -> RepeatedKFold:
    """
    5-fold cross-validation repeated 10 times.
    Gives 50 held-out evaluations total.
    """
    return RepeatedKFold(
        n_splits=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_SEED,
    )


def summarize_single_repeated_cv(
        fold_results_df: pd.DataFrame,
        model_name: str,
        n: int,
        kernel_type: str,
        n_restarts_optimizer: int,
        alpha: float,
) -> pd.DataFrame:
    """
    Summarize one repeated-CV model result.

    This is used by deposition and stacked models,
    where each function only returns one model/setup at a time.
    """

    return pd.DataFrame([
        {
            "model_name": model_name,
            "n": n,
            "kernel_type": kernel_type,
            "n_restarts_optimizer": n_restarts_optimizer,
            "alpha": alpha,
            "n_evaluations": len(fold_results_df),
            "n_total_test_points": int(fold_results_df["test_n"].sum()),

            "mae_mean": fold_results_df["mae"].mean(),
            "mae_std": fold_results_df["mae"].std(),

            "rmse_mean": fold_results_df["rmse"].mean(),
            "rmse_std": fold_results_df["rmse"].std(),

            "r2_mean": fold_results_df["r2"].mean(),
            "r2_std": fold_results_df["r2"].std(),

            "coverage_90_mean": fold_results_df["coverage_90"].mean(),
            "coverage_90_std": fold_results_df["coverage_90"].std(),

            "nlpd_mean": fold_results_df["nlpd"].mean(),
            "nlpd_std": fold_results_df["nlpd"].std(),

            "lml_mean": fold_results_df["lml"].mean(),
            "lml_std": fold_results_df["lml"].std(),
        }
    ])

def summarize_bootstrap_distribution(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)),
        f"{prefix}_p025": float(np.quantile(values, 0.025)),
        f"{prefix}_p500": float(np.quantile(values, 0.500)),
        f"{prefix}_p975": float(np.quantile(values, 0.975)),
    }

def bootstrap_stacked_error_propagation_for_fold(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        controllable_cols: list[str],
        dep_cols: list[str],
        kernel_type: str,
        n_restarts_optimizer: int,
        alpha: float,
        deposition_kernel_type: str | None = None,
        deposition_n_restarts_optimizer: int | None = None,
        deposition_alpha: float | None = None,
        deposition_transform: str = DEPLOYED_DEPOSITION_TRANSFORM,
        deposition_inverse_method: str = DEPLOYED_DEPOSITION_INVERSE_METHOD,
        selectivity_kernel_type: str | None = None,
        selectivity_n_restarts_optimizer: int | None = None,
        selectivity_alpha: float | None = None,
        n_bootstrap: int = 200,
        random_state: int = RANDOM_SEED,
) -> tuple[dict, pd.DataFrame]:
    """
    Bootstrap error propagation for one held-out fold.

    This bootstraps the full deployed stacked pipeline:

        bootstrap training fold
        -> train deposition GP
        -> predict test-fold deposition
        -> train selectivity GP using measured deposition in bootstrap sample
        -> predict test-fold selectivity using predicted deposition
        -> compare against diagnostic prediction using measured deposition

    The measured-deposition test prediction is diagnostic only.
    It is not the deployed model.
    """

    rng = np.random.default_rng(random_state)
    deposition_kernel_type = deposition_kernel_type or kernel_type
    deposition_n_restarts_optimizer = (
        n_restarts_optimizer
        if deposition_n_restarts_optimizer is None
        else deposition_n_restarts_optimizer
    )
    deposition_alpha = alpha if deposition_alpha is None else deposition_alpha

    selectivity_kernel_type = selectivity_kernel_type or kernel_type
    selectivity_n_restarts_optimizer = (
        n_restarts_optimizer
        if selectivity_n_restarts_optimizer is None
        else selectivity_n_restarts_optimizer
    )
    selectivity_alpha = alpha if selectivity_alpha is None else selectivity_alpha

    y_test = test_df[TARGET_COL].values.astype(float)
    measured_dep = test_df[RAW_DEPOSITION_COL].values.astype(float)

    n_test = len(test_df)

    pred_selectivity_pred_dep = np.empty((n_test, n_bootstrap), dtype=float)
    pred_selectivity_measured_dep = np.empty((n_test, n_bootstrap), dtype=float)
    pred_deposition = np.empty((n_test, n_bootstrap), dtype=float)

    rmse_deployed_boot = np.empty(n_bootstrap, dtype=float)
    rmse_measured_dep_input_boot = np.empty(n_bootstrap, dtype=float)
    rmse_substitution_penalty_boot = np.empty(n_bootstrap, dtype=float)

    r2_deployed_boot = np.empty(n_bootstrap, dtype=float)
    r2_measured_dep_input_boot = np.empty(n_bootstrap, dtype=float)
    r2_substitution_delta_boot = np.empty(n_bootstrap, dtype=float)

    deposition_error_rmse_boot = np.empty(n_bootstrap, dtype=float)
    deposition_error_mae_boot = np.empty(n_bootstrap, dtype=float)
    deposition_error_bias_boot = np.empty(n_bootstrap, dtype=float)

    selectivity_shift_rmse_boot = np.empty(n_bootstrap, dtype=float)
    selectivity_shift_mae_boot = np.empty(n_bootstrap, dtype=float)
    selectivity_shift_bias_boot = np.empty(n_bootstrap, dtype=float)

    mech_cols = controllable_cols + [RAW_DEPOSITION_COL]

    for b in range(n_bootstrap):
        # --------------------------------------------------
        # Bootstrap sample from training fold only
        # --------------------------------------------------
        boot_idx = rng.integers(
            low=0,
            high=len(train_df),
            size=len(train_df),
        )

        boot_df = train_df.iloc[boot_idx].copy().reset_index(drop=True)


        # Train first-stage deposition GP on bootstrap sample
        dep_bundle, _ = train_deposition_model(
            boot_df,
            controllable_cols=dep_cols,
            deposition_col=RAW_DEPOSITION_COL,
            transform=deposition_transform,
            kernel_type=deposition_kernel_type,
            n_restarts_optimizer=deposition_n_restarts_optimizer,
            alpha=deposition_alpha,
        )

        test_dep_mu, test_dep_sigma = predict_deposition_logspace(
            dep_bundle,
            test_df[dep_cols].values.astype(float),
        )

        test_dep_pred, _ = deposition_prediction_to_raw(
            test_dep_mu,
            test_dep_sigma,
            transform=deposition_transform,
            inverse_method=deposition_inverse_method,
        )

        pred_deposition[:, b] = test_dep_pred

        # Train second-stage selectivity GP on bootstrap sample
        # using measured deposition from the training data
        mech_bundle, _ = train_selectivity_model(
            boot_df,
            input_cols=mech_cols,
            target_col=TARGET_COL,
            kernel_type=selectivity_kernel_type,
            n_restarts_optimizer=selectivity_n_restarts_optimizer,
            alpha=selectivity_alpha,
        )

        # Deployed prediction:
        # controls_test + predicted deposition_test
        X_pred_dep = test_df[controllable_cols].copy()
        X_pred_dep[RAW_DEPOSITION_COL] = test_dep_pred

        mu_pred_dep, _ = predict_gp(
            mech_bundle,
            X_pred_dep[mech_cols].values.astype(float),
        )

        pred_selectivity_pred_dep[:, b] = mu_pred_dep

        # Diagnostic prediction:
        # controls_test + measured deposition_test
        # Not deployable. Used only to quantify substitution penalty.
        X_measured_dep = test_df[controllable_cols].copy()
        X_measured_dep[RAW_DEPOSITION_COL] = measured_dep

        mu_measured_dep, _ = predict_gp(
            mech_bundle,
            X_measured_dep[mech_cols].values.astype(float),
        )

        pred_selectivity_measured_dep[:, b] = mu_measured_dep


        # Bootstrap-level metrics
        _, rmse_deployed, r2_deployed = metrics_from_predictions(
            y_test,
            mu_pred_dep,
        )

        _, rmse_measured_dep_input, r2_measured_dep_input = metrics_from_predictions(
            y_test,
            mu_measured_dep,
        )

        dep_error = test_dep_pred - measured_dep
        sel_shift = mu_pred_dep - mu_measured_dep

        rmse_deployed_boot[b] = rmse_deployed
        rmse_measured_dep_input_boot[b] = rmse_measured_dep_input
        rmse_substitution_penalty_boot[b] = rmse_deployed - rmse_measured_dep_input

        r2_deployed_boot[b] = r2_deployed
        r2_measured_dep_input_boot[b] = r2_measured_dep_input
        r2_substitution_delta_boot[b] = r2_deployed - r2_measured_dep_input

        deposition_error_rmse_boot[b] = float(np.sqrt(np.mean(dep_error ** 2)))
        deposition_error_mae_boot[b] = float(np.mean(np.abs(dep_error)))
        deposition_error_bias_boot[b] = float(np.mean(dep_error))

        selectivity_shift_rmse_boot[b] = float(np.sqrt(np.mean(sel_shift ** 2)))
        selectivity_shift_mae_boot[b] = float(np.mean(np.abs(sel_shift)))
        selectivity_shift_bias_boot[b] = float(np.mean(sel_shift))


    # Point-level bootstrap summaries
    pred_mean = np.mean(pred_selectivity_pred_dep, axis=1)
    pred_std = np.std(pred_selectivity_pred_dep, axis=1, ddof=1)
    pred_p025 = np.quantile(pred_selectivity_pred_dep, 0.025, axis=1)
    pred_p050 = np.quantile(pred_selectivity_pred_dep, 0.500, axis=1)
    pred_p975 = np.quantile(pred_selectivity_pred_dep, 0.975, axis=1)

    measured_dep_input_mean = np.mean(pred_selectivity_measured_dep, axis=1)

    dep_pred_mean = np.mean(pred_deposition, axis=1)
    dep_pred_std = np.std(pred_deposition, axis=1, ddof=1)
    dep_pred_p025 = np.quantile(pred_deposition, 0.025, axis=1)
    dep_pred_p975 = np.quantile(pred_deposition, 0.975, axis=1)

    selectivity_shift = pred_selectivity_pred_dep - pred_selectivity_measured_dep

    shift_mean = np.mean(selectivity_shift, axis=1)
    shift_std = np.std(selectivity_shift, axis=1, ddof=1)
    shift_p025 = np.quantile(selectivity_shift, 0.025, axis=1)
    shift_p975 = np.quantile(selectivity_shift, 0.975, axis=1)

    dep_error_boot = pred_deposition - measured_dep[:, None]

    dep_error_mean = np.mean(dep_error_boot, axis=1)
    dep_error_std = np.std(dep_error_boot, axis=1, ddof=1)

    # Performance of the bootstrap-averaged deployed prediction
    mae_mean_pred, rmse_mean_pred, r2_mean_pred = metrics_from_predictions(
        y_test,
        pred_mean,
    )

    # Empirical 95% bootstrap interval coverage.
    # This is a bootstrap model-uncertainty interval, not a full experimental-noise interval.
    coverage_95_bootstrap = float(
        np.mean((y_test >= pred_p025) & (y_test <= pred_p975))
    )

    fold_summary = {
        "mae_bootstrap_mean_prediction": mae_mean_pred,
        "rmse_bootstrap_mean_prediction": rmse_mean_pred,
        "r2_bootstrap_mean_prediction": r2_mean_pred,
        "coverage_95_bootstrap": coverage_95_bootstrap,
        "bootstrap_prediction_std_mean": float(np.mean(pred_std)),
        "bootstrap_prediction_std_median": float(np.median(pred_std)),
    }

    fold_summary.update(
        summarize_bootstrap_distribution(
            rmse_deployed_boot,
            "rmse_deployed_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            rmse_measured_dep_input_boot,
            "rmse_measured_dep_input_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            rmse_substitution_penalty_boot,
            "rmse_substitution_penalty_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            r2_substitution_delta_boot,
            "r2_substitution_delta_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            deposition_error_rmse_boot,
            "deposition_error_rmse_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            deposition_error_mae_boot,
            "deposition_error_mae_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            deposition_error_bias_boot,
            "deposition_error_bias_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            selectivity_shift_rmse_boot,
            "selectivity_shift_rmse_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            selectivity_shift_mae_boot,
            "selectivity_shift_mae_bootstrap",
        )
    )

    fold_summary.update(
        summarize_bootstrap_distribution(
            selectivity_shift_bias_boot,
            "selectivity_shift_bias_bootstrap",
        )
    )

    point_summary = pd.DataFrame(
        {
            "row_index": test_df.index.values,
            "Observed_Selectivity": y_test,

            "Predicted_Selectivity_BootstrapMean": pred_mean,
            "Predicted_Selectivity_BootstrapStd": pred_std,
            "Predicted_Selectivity_BootstrapP025": pred_p025,
            "Predicted_Selectivity_BootstrapP050": pred_p050,
            "Predicted_Selectivity_BootstrapP975": pred_p975,

            "Predicted_Selectivity_MeasuredDepInput_BootstrapMean": measured_dep_input_mean,

            "Measured_Deposition": measured_dep,
            "Predicted_Deposition_BootstrapMean": dep_pred_mean,
            "Predicted_Deposition_BootstrapStd": dep_pred_std,
            "Predicted_Deposition_BootstrapP025": dep_pred_p025,
            "Predicted_Deposition_BootstrapP975": dep_pred_p975,

            "Deposition_Error_BootstrapMean": dep_error_mean,
            "Deposition_Error_BootstrapStd": dep_error_std,

            "Selectivity_Shift_PredDepMinusMeasuredDep_BootstrapMean": shift_mean,
            "Selectivity_Shift_PredDepMinusMeasuredDep_BootstrapStd": shift_std,
            "Selectivity_Shift_PredDepMinusMeasuredDep_BootstrapP025": shift_p025,
            "Selectivity_Shift_PredDepMinusMeasuredDep_BootstrapP975": shift_p975,
        }
    )

    return fold_summary, point_summary

def cross_validate_model(
        df: pd.DataFrame,
        input_cols: list[str],
        target_col: str,
        model_name: str,
        kernel_type: str,
        n_restarts_optimizer: int,
        alpha: float,
) -> dict:
    use_df = df.dropna(subset=input_cols + [target_col]).copy().reset_index(drop=True)

    X_all = use_df[input_cols].values.astype(float)
    y_all = use_df[target_col].values.astype(float)

    all_preds = []
    fold_rows = []

    rkf = RepeatedKFold(
        n_splits=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_SEED,
    )

    for evaluation, (train_idx, test_idx) in enumerate(rkf.split(X_all), start=1):
        train_df = use_df.iloc[train_idx].copy()
        test_df = use_df.iloc[test_idx].copy()

        bundle, _ = train_selectivity_model(
            train_df,
            input_cols,
            target_col=target_col,
            kernel_type=kernel_type,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
        )

        mu, sigma = predict_gp(
            bundle,
            test_df[input_cols].values.astype(float),
        )

        y_test = y_all[test_idx]

        mae, rmse, r2 = metrics_from_predictions(y_test, mu)

        coverage_90 = coverage_90_from_predictions(
            y_true=y_test,
            y_pred=mu,
            y_std=sigma,
        )

        nlpd = nlpd_from_predictions(
            y_true=y_test,
            y_pred=mu,
            y_std=sigma,
        )

        lml = float(getattr(bundle.gp, "log_marginal_likelihood_value_", np.nan))

        fold_rows.append(
            {
                "model_name": model_name,
                "kernel_type": kernel_type,
                "n_restarts_optimizer": n_restarts_optimizer,
                "alpha": alpha,
                "evaluation": evaluation,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "coverage_90": coverage_90,
                "nlpd": nlpd,
                "lml": lml,
            }
        )

        fold_pred_df = pd.DataFrame(
            {
                "model_name": model_name,
                "kernel_type": kernel_type,
                "n_restarts_optimizer": n_restarts_optimizer,
                "alpha": alpha,
                "evaluation": evaluation,
                "row_index": test_idx,
                "Observed": y_test,
                "Predicted": mu,
                "Pred_Std": sigma,
            }
        )

        all_preds.append(fold_pred_df)

    fold_results_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(all_preds, axis=0, ignore_index=True)

    summary_df = summarize_single_repeated_cv(
        fold_results_df=fold_results_df,
        model_name=model_name,
        n=len(use_df),
        kernel_type=kernel_type,
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=alpha,
    )

    row = summary_df.iloc[0]

    return {
        "model_name": model_name,
        "n": len(use_df),
        "kernel_type": kernel_type,
        "n_restarts_optimizer": n_restarts_optimizer,
        "alpha": alpha,

        "fold_results": fold_results_df,
        "predictions": predictions_df,
        "summary": summary_df,

        "mae": row["mae_mean"],
        "rmse": row["rmse_mean"],
        "r2": row["r2_mean"],
        "coverage_90": row["coverage_90_mean"],
        "nlpd": row["nlpd_mean"],
        "lml": row["lml_mean"],
    }


def cross_validate_deposition_model(
        df: pd.DataFrame,
        controllable_cols: list[str],
        kernel_type: str = "matern_1.5",
        n_restarts_optimizer: int = 10,
        alpha: float = 1e-4,
        deposition_transform: str = DEPLOYED_DEPOSITION_TRANSFORM,
        deposition_inverse_method: str = DEPLOYED_DEPOSITION_INVERSE_METHOD,
) -> dict:
    """
    Repeated-CV evaluation for the first-stage deposition surrogate.

    By default this validates raw deposition, matching the deployed stack. The
    transform metadata is written into fold and summary outputs so selected GP
    configurations can be traced back to the target scale used during CV.
    """
    model_name = "Deposition model: controls -> deposition"

    use_df = (
        df
        .dropna(subset=controllable_cols + [RAW_DEPOSITION_COL])
        .copy()
        .reset_index(drop=True)
    )

    X_all = use_df[controllable_cols].values.astype(float)
    y_all = use_df[RAW_DEPOSITION_COL].values.astype(float)

    rkf = make_repeated_cv()

    fold_rows = []
    prediction_rows = []

    for evaluation, (train_idx, test_idx) in enumerate(rkf.split(X_all), start=1):
        train_df = use_df.iloc[train_idx].copy()
        test_df = use_df.iloc[test_idx].copy()

        dep_bundle, _ = train_deposition_model(
            train_df,
            controllable_cols=controllable_cols,
            deposition_col=RAW_DEPOSITION_COL,
            transform=deposition_transform,
            kernel_type=kernel_type,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
        )

        mu_dep, sigma_dep = predict_deposition_logspace(
            dep_bundle,
            test_df[controllable_cols].values.astype(float),
        )

        mu_raw, sigma_raw = deposition_prediction_to_raw(
            mu_dep,
            sigma_dep,
            transform=deposition_transform,
            inverse_method=deposition_inverse_method,
        )

        y_test = y_all[test_idx]

        mae, rmse, r2 = metrics_from_predictions(y_test, mu_raw)

        coverage_90 = coverage_90_from_predictions(
            y_true=y_test,
            y_pred=mu_raw,
            y_std=sigma_raw,
        )

        nlpd = nlpd_from_predictions(
            y_true=y_test,
            y_pred=mu_raw,
            y_std=sigma_raw,
        )

        lml = float(getattr(dep_bundle.gp, "log_marginal_likelihood_value_", np.nan))

        fold_rows.append(
            {
                "model_name": model_name,
                "kernel_type": kernel_type,
                "n_restarts_optimizer": n_restarts_optimizer,
                "alpha": alpha,
                "deposition_transform": deposition_transform,
                "deposition_inverse_method": deposition_inverse_method,
                "evaluation": evaluation,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "coverage_90": coverage_90,
                "nlpd": nlpd,
                "lml": lml,
            }
        )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "kernel_type": kernel_type,
                    "n_restarts_optimizer": n_restarts_optimizer,
                    "alpha": alpha,
                    "deposition_transform": deposition_transform,
                    "deposition_inverse_method": deposition_inverse_method,
                    "evaluation": evaluation,
                    "row_index": test_idx,
                    "Observed": y_test,
                    "Predicted": mu_raw,
                    "Pred_Std": sigma_raw,
                }
            )
        )

    fold_results_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_rows, axis=0, ignore_index=True)

    summary_df = summarize_single_repeated_cv(
        fold_results_df=fold_results_df,
        model_name=model_name,
        n=len(use_df),
        kernel_type=kernel_type,
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=alpha,
    )
    summary_df["deposition_transform"] = deposition_transform
    summary_df["deposition_inverse_method"] = deposition_inverse_method

    row = summary_df.iloc[0]

    return {
        "model_name": model_name,
        "n": len(use_df),
        "kernel_type": kernel_type,
        "n_restarts_optimizer": n_restarts_optimizer,
        "alpha": alpha,
        "mae": fold_results_df["mae"].mean(),
        "rmse": row["rmse_mean"],
        "r2": row["r2_mean"],
        "coverage_90": row["coverage_90_mean"],
        "nlpd": row["nlpd_mean"],
        "lml": row["lml_mean"],
        "fold_results": fold_results_df,
        "predictions": predictions_df,
        "summary": summary_df,
    }


def cross_validate_stacked_deployed_pipeline(
        df: pd.DataFrame,
        controllable_cols: list[str],
        dep_cols: list[str],
        kernel_type: str = "matern_1.5",
        n_restarts_optimizer: int = 10,
        alpha: float = 1e-4,
        deposition_kernel_type: str | None = None,
        deposition_n_restarts_optimizer: int | None = None,
        deposition_alpha: float | None = None,
        deposition_transform: str = DEPLOYED_DEPOSITION_TRANSFORM,
        deposition_inverse_method: str = DEPLOYED_DEPOSITION_INVERSE_METHOD,
        selectivity_kernel_type: str | None = None,
        selectivity_n_restarts_optimizer: int | None = None,
        selectivity_alpha: float | None = None,
) -> dict:
    """
    Repeated-CV evaluation of the deployed two-stage stack.

    Each fold trains the deposition GP on the training fold, predicts deposition
    for the held-out fold, trains the selectivity GP on measured training-fold
    deposition, and evaluates selectivity on held-out rows using predicted
    deposition. This mirrors test-time deployment, where true deposition is not
    available for candidate conditions.
    """
    model_name = "Stacked deployed pipeline: controls -> predicted deposition -> mechanistic selectivity"

    deposition_kernel_type = deposition_kernel_type or kernel_type
    deposition_n_restarts_optimizer = (
        n_restarts_optimizer
        if deposition_n_restarts_optimizer is None
        else deposition_n_restarts_optimizer
    )
    deposition_alpha = alpha if deposition_alpha is None else deposition_alpha

    selectivity_kernel_type = selectivity_kernel_type or kernel_type
    selectivity_n_restarts_optimizer = (
        n_restarts_optimizer
        if selectivity_n_restarts_optimizer is None
        else selectivity_n_restarts_optimizer
    )
    selectivity_alpha = alpha if selectivity_alpha is None else selectivity_alpha

    needed_cols = list(dict.fromkeys(
        controllable_cols + dep_cols + [RAW_DEPOSITION_COL, TARGET_COL]
    ))

    use_df = df.dropna(subset=needed_cols).copy().reset_index(drop=True)

    X_ctrl_all = use_df[controllable_cols].values.astype(float)
    y_sel_all = use_df[TARGET_COL].values.astype(float)

    rkf = make_repeated_cv()

    fold_rows = []
    prediction_rows = []

    for evaluation, (train_idx, test_idx) in enumerate(rkf.split(X_ctrl_all), start=1):
        train_df = use_df.iloc[train_idx].copy()
        test_df = use_df.iloc[test_idx].copy()

        # 1) Train deposition model on the training fold only
        dep_bundle, _ = train_deposition_model(
            train_df,
            controllable_cols=dep_cols,
            deposition_col=RAW_DEPOSITION_COL,
            transform=deposition_transform,
            kernel_type=deposition_kernel_type,
            n_restarts_optimizer=deposition_n_restarts_optimizer,
            alpha=deposition_alpha,
        )

        # 2) Predict deposition for the held-out test fold
        test_dep_mu, test_dep_sigma = predict_deposition_logspace(
            dep_bundle,
            test_df[dep_cols].values.astype(float),
        )

        test_dep_pred, dep_unc_proxy = deposition_prediction_to_raw(
            test_dep_mu,
            test_dep_sigma,
            transform=deposition_transform,
            inverse_method=deposition_inverse_method,
        )

        # 3) Train selectivity model using measured deposition on training fold
        mech_cols = controllable_cols + [RAW_DEPOSITION_COL]

        mech_bundle, _ = train_selectivity_model(
            train_df,
            input_cols=mech_cols,
            target_col=TARGET_COL,
            kernel_type=selectivity_kernel_type,
            n_restarts_optimizer=selectivity_n_restarts_optimizer,
            alpha=selectivity_alpha,
        )

        # 4) Test deployed stack using predicted deposition
        mech_test_X = test_df[controllable_cols].copy()
        mech_test_X[RAW_DEPOSITION_COL] = test_dep_pred

        mu, sigma = predict_gp(
            mech_bundle,
            mech_test_X[mech_cols].values.astype(float),
        )

        stacked_sigma = np.maximum(sigma, 1e-12) + 0.05 * dep_unc_proxy

        y_test = y_sel_all[test_idx]

        mae, rmse, r2 = metrics_from_predictions(y_test, mu)

        coverage_90 = coverage_90_from_predictions(
            y_true=y_test,
            y_pred=mu,
            y_std=stacked_sigma,
        )

        nlpd = nlpd_from_predictions(
            y_true=y_test,
            y_pred=mu,
            y_std=stacked_sigma,
        )

        dep_lml = float(getattr(dep_bundle.gp, "log_marginal_likelihood_value_", np.nan))
        sel_lml = float(getattr(mech_bundle.gp, "log_marginal_likelihood_value_", np.nan))

        # This is a useful bookkeeping value, but do not rank stack vs single-GP models by summed LML.
        lml_sum = dep_lml + sel_lml

        fold_rows.append(
            {
                "model_name": model_name,
                "kernel_type": kernel_type,
                "n_restarts_optimizer": n_restarts_optimizer,
                "alpha": alpha,
                "deposition_kernel_type": deposition_kernel_type,
                "deposition_n_restarts_optimizer": deposition_n_restarts_optimizer,
                "deposition_alpha": deposition_alpha,
                "deposition_transform": deposition_transform,
                "deposition_inverse_method": deposition_inverse_method,
                "selectivity_kernel_type": selectivity_kernel_type,
                "selectivity_n_restarts_optimizer": selectivity_n_restarts_optimizer,
                "selectivity_alpha": selectivity_alpha,
                "evaluation": evaluation,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "coverage_90": coverage_90,
                "nlpd": nlpd,
                "lml": lml_sum,
                "deposition_lml": dep_lml,
                "selectivity_lml": sel_lml,
            }
        )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "kernel_type": kernel_type,
                    "n_restarts_optimizer": n_restarts_optimizer,
                    "alpha": alpha,
                    "deposition_kernel_type": deposition_kernel_type,
                    "deposition_n_restarts_optimizer": deposition_n_restarts_optimizer,
                    "deposition_alpha": deposition_alpha,
                    "deposition_transform": deposition_transform,
                    "deposition_inverse_method": deposition_inverse_method,
                    "selectivity_kernel_type": selectivity_kernel_type,
                    "selectivity_n_restarts_optimizer": selectivity_n_restarts_optimizer,
                    "selectivity_alpha": selectivity_alpha,
                    "evaluation": evaluation,
                    "row_index": test_idx,
                    "Observed": y_test,
                    "Predicted": mu,
                    "Pred_Std": stacked_sigma,
                    "Predicted_Deposition": test_dep_pred,
                    "Deposition_Std_Proxy": dep_unc_proxy,
                }
            )
        )

    fold_results_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_rows, axis=0, ignore_index=True)

    summary_df = summarize_single_repeated_cv(
        fold_results_df=fold_results_df,
        model_name=model_name,
        n=len(use_df),
        kernel_type=kernel_type,
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=alpha,
    )
    summary_df["deposition_kernel_type"] = deposition_kernel_type
    summary_df["deposition_n_restarts_optimizer"] = deposition_n_restarts_optimizer
    summary_df["deposition_alpha"] = deposition_alpha
    summary_df["deposition_transform"] = deposition_transform
    summary_df["deposition_inverse_method"] = deposition_inverse_method
    summary_df["selectivity_kernel_type"] = selectivity_kernel_type
    summary_df["selectivity_n_restarts_optimizer"] = selectivity_n_restarts_optimizer
    summary_df["selectivity_alpha"] = selectivity_alpha

    row = summary_df.iloc[0]

    return {
        "model_name": model_name,
        "n": len(use_df),
        "kernel_type": kernel_type,
        "n_restarts_optimizer": n_restarts_optimizer,
        "alpha": alpha,
        "mae": fold_results_df["mae"].mean(),
        "rmse": row["rmse_mean"],
        "r2": row["r2_mean"],
        "coverage_90": row["coverage_90_mean"],
        "nlpd": row["nlpd_mean"],
        "lml": row["lml_mean"],
        "fold_results": fold_results_df,
        "predictions": predictions_df,
        "summary": summary_df,
    }
def cross_validate_stacked_monte_carlo_error_propagation(
        df: pd.DataFrame,
        controllable_cols: list[str],
        dep_cols: list[str],
        kernel_type: str,
        n_restarts_optimizer: int,
        alpha: float,
        deposition_kernel_type: str | None = None,
        deposition_n_restarts_optimizer: int | None = None,
        deposition_alpha: float | None = None,
        deposition_transform: str = DEPLOYED_DEPOSITION_TRANSFORM,
        deposition_inverse_method: str = DEPLOYED_DEPOSITION_INVERSE_METHOD,
        selectivity_kernel_type: str | None = None,
        selectivity_n_restarts_optimizer: int | None = None,
        selectivity_alpha: float | None = None,
        n_mc: int = 2000,
) -> dict:

    model_name = "Monte Carlo error propagation: stacked deployed pipeline"
    deposition_kernel_type = deposition_kernel_type or kernel_type
    deposition_n_restarts_optimizer = (
        n_restarts_optimizer
        if deposition_n_restarts_optimizer is None
        else deposition_n_restarts_optimizer
    )
    deposition_alpha = alpha if deposition_alpha is None else deposition_alpha

    selectivity_kernel_type = selectivity_kernel_type or kernel_type
    selectivity_n_restarts_optimizer = (
        n_restarts_optimizer
        if selectivity_n_restarts_optimizer is None
        else selectivity_n_restarts_optimizer
    )
    selectivity_alpha = alpha if selectivity_alpha is None else selectivity_alpha

    needed_cols = list(dict.fromkeys(
        controllable_cols
        + dep_cols
        + [RAW_DEPOSITION_COL, TARGET_COL]
    ))

    use_df = (
        df
        .dropna(subset=needed_cols)
        .copy().reset_index(drop=True)
    )

    X_ctrl_all = use_df[controllable_cols].values.astype(float)
    y_sel_all = use_df[TARGET_COL].values.astype(float)
    rkf = make_repeated_cv()

    fold_rows = []
    point_rows = []

    for evaluation, (train_idx, test_idx) in enumerate(
        rkf.split(X_ctrl_all),
        start=1,
    ):
        train_df = use_df.iloc[train_idx].copy()
        test_df = use_df.iloc[test_idx].copy()

        # Fit upstream deposition GP on training fold
        dep_bundle, _ = train_deposition_model(
            train_df,
            controllable_cols = dep_cols,
            deposition_col = RAW_DEPOSITION_COL,
            transform = deposition_transform,
            kernel_type = deposition_kernel_type,
            n_restarts_optimizer = deposition_n_restarts_optimizer,
            alpha = deposition_alpha,
        )

        dep_mu, dep_sigma = predict_deposition_logspace(
            dep_bundle,
            test_df[dep_cols].values.astype(float),
        )

        # Existing deployed point estimate
        dep_point, _ = deposition_prediction_to_raw(
            dep_mu,
            dep_sigma,
            transform=deposition_transform,
            inverse_method=deposition_inverse_method,
        )

        # Fit downstream selectivity GP on training fold using measured training fold deposition
        mech_cols = controllable_cols + [RAW_DEPOSITION_COL]
        mech_bundle, _ = train_selectivity_model(
            train_df,
            input_cols = mech_cols,
            target_col = TARGET_COL,
            kernel_type = selectivity_kernel_type,
            n_restarts_optimizer = selectivity_n_restarts_optimizer,
            alpha = selectivity_alpha,
        )

        # Existing point prediction used by validation
        point_X = test_df[controllable_cols].copy()
        point_X[RAW_DEPOSITION_COL] = dep_point

        point_mu, point_sigma = predict_gp(
            mech_bundle,
            point_X[mech_cols].values.astype(float),
        )

        # Monte Carlo propagation
        rng = np.random.default_rng(
            RANDOM_SEED + evaluation
        )

        dep_target_samples = rng.normal(
            loc = np.asarray(dep_mu)[:, None],
            scale = np.maximum(
                np.asarray(dep_sigma)[:, None],
                1e-12,),
            size = (len(test_df), n_mc),
        )

        if deposition_transform == "log1p":
            dep_samples = np.maximum(
                np.expm1(dep_target_samples),
                0.0,
            )
        elif deposition_transform == "raw":
            dep_samples = np.maximum(dep_target_samples, 0.0)
        else:
            raise ValueError(
                f"Unknown deposition_transform: {deposition_transform}"
            )

        selectivity_mu_samples = np.empty(
            (len(test_df), n_mc),
            dtype = float,
        )

        for mc_idx in range(n_mc):
            mc_X = test_df[controllable_cols].copy()
            mc_X[RAW_DEPOSITION_COL] = dep_samples[:, mc_idx]
            mc_mu, _ = predict_gp(
                mech_bundle,
                mc_X[mech_cols].values.astype(float),
            )
            selectivity_mu_samples[:, mc_idx] = mc_mu

        propagated_mean = np.mean(
            selectivity_mu_samples,
            axis=1,
        )

        propagated_std = np.std(
            selectivity_mu_samples,
            axis=1,
            ddof=1,
        )

        propagated_p025 = np.quantile(
            selectivity_mu_samples,
            0.025,
            axis=1,
        )

        propagated_p975 = np.quantile(
            selectivity_mu_samples,
            0.975,
            axis=1,
        )

        # Change caused by uncertain deposition input
        mean_shift = propagated_mean - point_mu
        y_test = y_sel_all[test_idx]
        mean_absolute_shift = float(np.mean(np.abs(mean_shift)))
        rmse_shift = float(np.sqrt(np.mean(mean_shift ** 2)))

        assert rmse_shift + 1e-12 >= mean_absolute_shift, (
            f"Invalid shift metrics in evaluation {evaluation}:"
            f"MAE shift = {mean_absolute_shift:.6f},"
            f"RMSE shift = {rmse_shift:.6f}"
        )
        fold_rows.append(
            {
                "model name": model_name,
                "evaluation": evaluation,
                "train_n": len(train_df),
                "kernel type": kernel_type,
                "n_restarts_optimizer": n_restarts_optimizer,
                "alpha": alpha,
                "deposition_kernel_type": deposition_kernel_type,
                "deposition_n_restarts_optimizer": deposition_n_restarts_optimizer,
                "deposition_alpha": deposition_alpha,
                "deposition_transform": deposition_transform,
                "deposition_inverse_method": deposition_inverse_method,
                "selectivity_kernel_type": selectivity_kernel_type,
                "selectivity_n_restarts_optimizer": selectivity_n_restarts_optimizer,
                "selectivity_alpha": selectivity_alpha,
                "n_mc": n_mc,

                "mean_propagated_selectivity_std": float(
                    np.mean(propagated_std)
                ),

                "median_propagated_selectivity_std": float(
                    np.median(propagated_std)
                ),

                "mean_absolute_mc_shift": float(
                    np.mean(np.abs(mean_shift))
                ),

                "rmse_mc_shift": float(
                    np.sqrt(np.mean(mean_shift ** 2))
                ),

                "max_absolute_mc_shift": float(
                    np.max(np.abs(mean_shift))
                ),

                "mean_deposition_mc_std": float(
                    np.mean(
                        np.std(
                            dep_samples,
                            axis=1,
                            ddof=1,
                        )
                    )
                ),
            }
        )


        point_rows.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "evaluation": evaluation,
                    "row_index": test_idx,
                    "Observed_Selectivity": y_test,

                    "Point_Predicted_Selectivity": point_mu,
                    "Point_Selectivity_Std": point_sigma,

                    "MC_Propagated_Selectivity_Mean": (
                        propagated_mean
                    ),

                    "MC_Propagated_Selectivity_Std": (
                        propagated_std
                    ),

                    "MC_Propagated_Selectivity_P025": (
                        propagated_p025
                    ),

                    "MC_Propagated_Selectivity_P975": (
                        propagated_p975
                    ),

                    "MC_Mean_Shift_From_Point_Prediction": (
                        mean_shift
                    ),

                    "Point_Predicted_Deposition": dep_point,

                    "MC_Deposition_Mean": np.mean(
                        dep_samples,
                        axis=1,
                    ),

                    "MC_Deposition_Std": np.std(
                        dep_samples,
                        axis=1,
                        ddof=1,
                    ),
                }
            )
        )

        print(
            f"Monte Carlo propagation evaluation"
            f"{evaluation}/50 complete"
        )

    fold_results_df = pd.DataFrame(fold_rows)
    point_results_df = pd.concat(
        point_rows,
        axis=0,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "n": len(use_df),
                "kernel type": kernel_type,
                "n_restarts_optimizer": n_restarts_optimizer,
                "alpha": alpha,
                "deposition_kernel_type": deposition_kernel_type,
                "deposition_n_restarts_optimizer": deposition_n_restarts_optimizer,
                "deposition_alpha": deposition_alpha,
                "deposition_transform": deposition_transform,
                "deposition_inverse_method": deposition_inverse_method,
                "selectivity_kernel_type": selectivity_kernel_type,
                "selectivity_n_restarts_optimizer": selectivity_n_restarts_optimizer,
                "selectivity_alpha": selectivity_alpha,
                "n_mc": n_mc,
                "n_evaluations": len(fold_results_df),

                "propagated_selectivity_std_mean": (
                    fold_results_df[
                        "mean_propagated_selectivity_std"
                    ].mean()
                ),

                "propagated_selectivity_std_std": (
                    fold_results_df[
                        "mean_propagated_selectivity_std"
                    ].std()
                ),

                "mc_mean_absolute_shift_mean": (
                    fold_results_df[
                        "mean_absolute_mc_shift"
                    ].mean()
                ),

                "mc_mean_absolute_shift_std": (
                    fold_results_df[
                        "mean_absolute_mc_shift"
                    ].std()
                ),

                "mc_shift_rmse_mean": (
                    fold_results_df[
                        "rmse_mc_shift"
                    ].mean()
                ),

                "mc_shift_rmse_std": (
                    fold_results_df[
                        "rmse_mc_shift"
                    ].std()
                ),

                "deposition_mc_std_mean": (
                    fold_results_df[
                        "mean_deposition_mc_std"
                    ].mean()
                ),
            }
        ]
    )

    return {
        "fold_results": fold_results_df,
        "point_results": point_results_df,
        "summary": summary_df,
    }

def cross_validate_stacked_bootstrap_error_propagation(
        df: pd.DataFrame,
        controllable_cols: list[str],
        dep_cols: list[str],
        kernel_type: str,
        n_restarts_optimizer: int,
        alpha: float,
        deposition_kernel_type: str | None = None,
        deposition_n_restarts_optimizer: int | None = None,
        deposition_alpha: float | None = None,
        deposition_transform: str = DEPLOYED_DEPOSITION_TRANSFORM,
        deposition_inverse_method: str = DEPLOYED_DEPOSITION_INVERSE_METHOD,
        selectivity_kernel_type: str | None = None,
        selectivity_n_restarts_optimizer: int | None = None,
        selectivity_alpha: float | None = None,
        n_bootstrap: int = 200,
) -> dict:
    model_name = "Bootstrap stacked deployed pipeline"
    deposition_kernel_type = deposition_kernel_type or kernel_type
    deposition_n_restarts_optimizer = (
        n_restarts_optimizer
        if deposition_n_restarts_optimizer is None
        else deposition_n_restarts_optimizer
    )
    deposition_alpha = alpha if deposition_alpha is None else deposition_alpha

    selectivity_kernel_type = selectivity_kernel_type or kernel_type
    selectivity_n_restarts_optimizer = (
        n_restarts_optimizer
        if selectivity_n_restarts_optimizer is None
        else selectivity_n_restarts_optimizer
    )
    selectivity_alpha = alpha if selectivity_alpha is None else selectivity_alpha

    needed_cols = list(dict.fromkeys(
        controllable_cols + dep_cols + [RAW_DEPOSITION_COL, TARGET_COL]
    ))

    use_df = df.dropna(subset=needed_cols).copy().reset_index(drop=True)

    X_ctrl_all = use_df[controllable_cols].values.astype(float)

    rkf = make_repeated_cv()

    fold_rows = []
    point_rows = []

    for evaluation, (train_idx, test_idx) in enumerate(rkf.split(X_ctrl_all), start=1):
        train_df = use_df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = use_df.iloc[test_idx].copy()
        test_df.index = test_idx

        fold_summary, point_summary = bootstrap_stacked_error_propagation_for_fold(
            train_df=train_df,
            test_df=test_df,
            controllable_cols=controllable_cols,
            dep_cols=dep_cols,
            kernel_type=kernel_type,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            deposition_kernel_type=deposition_kernel_type,
            deposition_n_restarts_optimizer=deposition_n_restarts_optimizer,
            deposition_alpha=deposition_alpha,
            deposition_transform=deposition_transform,
            deposition_inverse_method=deposition_inverse_method,
            selectivity_kernel_type=selectivity_kernel_type,
            selectivity_n_restarts_optimizer=selectivity_n_restarts_optimizer,
            selectivity_alpha=selectivity_alpha,
            n_bootstrap=n_bootstrap,
            random_state=RANDOM_SEED + evaluation,
        )

        fold_summary.update(
            {
                "model_name": model_name,
                "evaluation": evaluation,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "kernel_type": kernel_type,
                "n_restarts_optimizer": n_restarts_optimizer,
                "alpha": alpha,
                "deposition_kernel_type": deposition_kernel_type,
                "deposition_n_restarts_optimizer": deposition_n_restarts_optimizer,
                "deposition_alpha": deposition_alpha,
                "deposition_transform": deposition_transform,
                "deposition_inverse_method": deposition_inverse_method,
                "selectivity_kernel_type": selectivity_kernel_type,
                "selectivity_n_restarts_optimizer": selectivity_n_restarts_optimizer,
                "selectivity_alpha": selectivity_alpha,
                "n_bootstrap": n_bootstrap,
            }
        )

        point_summary.insert(0, "model_name", model_name)
        point_summary.insert(1, "evaluation", evaluation)
        point_summary.insert(2, "kernel_type", kernel_type)
        point_summary.insert(3, "n_restarts_optimizer", n_restarts_optimizer)
        point_summary.insert(4, "alpha", alpha)
        point_summary.insert(5, "deposition_kernel_type", deposition_kernel_type)
        point_summary.insert(6, "deposition_n_restarts_optimizer", deposition_n_restarts_optimizer)
        point_summary.insert(7, "deposition_alpha", deposition_alpha)
        point_summary.insert(8, "deposition_transform", deposition_transform)
        point_summary.insert(9, "deposition_inverse_method", deposition_inverse_method)
        point_summary.insert(10, "selectivity_kernel_type", selectivity_kernel_type)
        point_summary.insert(11, "selectivity_n_restarts_optimizer", selectivity_n_restarts_optimizer)
        point_summary.insert(12, "selectivity_alpha", selectivity_alpha)
        point_summary.insert(13, "n_bootstrap", n_bootstrap)

        fold_rows.append(fold_summary)
        point_rows.append(point_summary)

        print(
            f"Bootstrap propagation evaluation {evaluation}: "
            f"RMSE={fold_summary['rmse_bootstrap_mean_prediction']:.3f}, "
            f"substitution penalty={fold_summary['rmse_substitution_penalty_bootstrap_mean']:.3f}"
        )

    fold_results_df = pd.DataFrame(fold_rows)
    point_results_df = pd.concat(point_rows, axis=0, ignore_index=True)

    overall_summary = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "n": len(use_df),
                "kernel_type": kernel_type,
                "n_restarts_optimizer": n_restarts_optimizer,
                "alpha": alpha,
                "deposition_kernel_type": deposition_kernel_type,
                "deposition_n_restarts_optimizer": deposition_n_restarts_optimizer,
                "deposition_alpha": deposition_alpha,
                "deposition_transform": deposition_transform,
                "deposition_inverse_method": deposition_inverse_method,
                "selectivity_kernel_type": selectivity_kernel_type,
                "selectivity_n_restarts_optimizer": selectivity_n_restarts_optimizer,
                "selectivity_alpha": selectivity_alpha,
                "n_bootstrap": n_bootstrap,
                "n_evaluations": len(fold_results_df),

                "rmse_bootstrap_mean_prediction_mean": fold_results_df[
                    "rmse_bootstrap_mean_prediction"
                ].mean(),
                "rmse_bootstrap_mean_prediction_std": fold_results_df[
                    "rmse_bootstrap_mean_prediction"
                ].std(),

                "r2_bootstrap_mean_prediction_mean": fold_results_df[
                    "r2_bootstrap_mean_prediction"
                ].mean(),
                "r2_bootstrap_mean_prediction_std": fold_results_df[
                    "r2_bootstrap_mean_prediction"
                ].std(),

                "coverage_95_bootstrap_mean": fold_results_df[
                    "coverage_95_bootstrap"
                ].mean(),
                "coverage_95_bootstrap_std": fold_results_df[
                    "coverage_95_bootstrap"
                ].std(),

                "bootstrap_prediction_std_mean": fold_results_df[
                    "bootstrap_prediction_std_mean"
                ].mean(),

                "rmse_substitution_penalty_bootstrap_mean": fold_results_df[
                    "rmse_substitution_penalty_bootstrap_mean"
                ].mean(),
                "rmse_substitution_penalty_bootstrap_std": fold_results_df[
                    "rmse_substitution_penalty_bootstrap_mean"
                ].std(),

                "deposition_error_rmse_bootstrap_mean": fold_results_df[
                    "deposition_error_rmse_bootstrap_mean"
                ].mean(),
                "deposition_error_rmse_bootstrap_std": fold_results_df[
                    "deposition_error_rmse_bootstrap_mean"
                ].std(),

                "selectivity_shift_rmse_bootstrap_mean": fold_results_df[
                    "selectivity_shift_rmse_bootstrap_mean"
                ].mean(),
                "selectivity_shift_rmse_bootstrap_std": fold_results_df[
                    "selectivity_shift_rmse_bootstrap_mean"
                ].std(),
            }
        ]
    )

    return {
        "fold_results": fold_results_df,
        "point_results": point_results_df,
        "summary": overall_summary,
    }


def print_summary(result: dict) -> None:
    print("\n" + "=" * 80)
    print(result["model_name"])
    print("=" * 80)

    print(f"N:             {result['n']}")
    print(f"Kernel:        {result.get('kernel_type', 'default')}")
    print(f"Restarts:      {result.get('n_restarts_optimizer', 'default')}")
    print(f"Alpha:         {result.get('alpha', 'default')}")

    if "summary" in result:
        row = result["summary"].iloc[0]

        print(f"Evaluations:   {int(row['n_evaluations'])}")
        print(f"MAE:           {row['mae_mean']:.4f} +/- {row['mae_std']:.4f}")
        print(f"RMSE:          {row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f}")
        print(f"R2:            {row['r2_mean']:.4f} +/- {row['r2_std']:.4f}")
        print(f"Coverage 90%:  {row['coverage_90_mean']:.4f} +/- {row['coverage_90_std']:.4f}")
        print(f"NLPD:          {row['nlpd_mean']:.4f} +/- {row['nlpd_std']:.4f}")
        print(f"LML:           {row['lml_mean']:.4f} +/- {row['lml_std']:.4f}")

    else:
        print(f"MAE:           {result['mae']:.4f}")
        print(f"RMSE:          {result['rmse']:.4f}")
        print(f"R2:            {result['r2']:.4f}")


def save_predictions_excel(filename: str, result: dict) -> None:
    if "predictions" in result:
        result["predictions"].to_excel(filename, index=False)
    else:
        out = {
            "Observed": result["y_true"],
            "Predicted": result["y_pred"],
            "Pred_Std": result["y_std"],
        }

        if "pred_dep" in result:
            out["Predicted_Deposition"] = result["pred_dep"]

        pd.DataFrame(out).to_excel(filename, index=False)

    print(f"Saved: {filename}")


def run_validation_analysis(
    outdir: Path = VALIDATION_RESULTS_DIR,
    include_stack_grid: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Run the component model-selection grid.

    The stacked-kernel grid is disabled by default because the active deployed
    stack combines the selected raw-deposition GP with the selected Model B
    selectivity GP. Use `include_stack_grid=True` only for diagnostic comparisons
    that are not part of the default model-selection workflow.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df = add_engineered_features(df)

    base_cols = make_base_controllable_input_cols(df)
    controllable_cols = make_model_input_cols(base_cols)
    mechanistic_cols = controllable_cols + [RAW_DEPOSITION_COL]

    dep_cols = make_deposition_input_cols(df)

    print("BASE COLS:", base_cols)
    print("MODEL INPUT COLS:", controllable_cols)
    print("DEPOSITION INPUT COLS:", dep_cols)
    print("MECHANISTIC COLS:", mechanistic_cols)

    all_fold_results = []
    all_predictions = []
    all_summaries = []

    for kernel_type, n_restarts_optimizer, alpha in product(
            GP_KERNEL_TYPES,
            GP_RESTART_VALUES,
            GP_ALPHA_VALUES,
    ):
        print("\n" + "#" * 90)
        print(
            f"GP setup: kernel={kernel_type}, "
            f"restarts={n_restarts_optimizer}, "
            f"alpha={alpha}"
        )
        print("#" * 90)

        result_a = cross_validate_model(
            df=df,
            input_cols=controllable_cols,
            target_col=TARGET_COL,
            model_name="Model A: controls only",
            kernel_type=kernel_type,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
        )

        result_b = cross_validate_model(
            df=df,
            input_cols=mechanistic_cols,
            target_col=TARGET_COL,
            model_name="Model B: controls + measured deposition",
            kernel_type=kernel_type,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
        )

        result_dep = cross_validate_deposition_model(
            df=df,
            controllable_cols=dep_cols,
            kernel_type=kernel_type,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            deposition_transform=DEPLOYED_DEPOSITION_TRANSFORM,
            deposition_inverse_method=DEPLOYED_DEPOSITION_INVERSE_METHOD,
        )

        results = [result_a, result_b, result_dep]

        if include_stack_grid:
            result_stack = cross_validate_stacked_deployed_pipeline(
                df=df,
                controllable_cols=controllable_cols,
                dep_cols=dep_cols,
                kernel_type=kernel_type,
                n_restarts_optimizer=n_restarts_optimizer,
                alpha=alpha,
                deposition_transform=DEPLOYED_DEPOSITION_TRANSFORM,
                deposition_inverse_method=DEPLOYED_DEPOSITION_INVERSE_METHOD,
            )
            results.append(result_stack)

        for result in results:
            print_summary(result)

            all_fold_results.append(result["fold_results"])
            all_predictions.append(result["predictions"])
            all_summaries.append(result["summary"])

    fold_results_df = pd.concat(
        all_fold_results,
        axis=0,
        ignore_index=True,
    )

    predictions_df = pd.concat(
        all_predictions,
        axis=0,
        ignore_index=True,
    )

    summary_df = pd.concat(
        all_summaries,
        axis=0,
        ignore_index=True,
    )

    summary_df = summary_df.sort_values(
        ["model_name", "rmse_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    fold_results_df.to_excel(
        outdir / "validation_all_fold_results.xlsx",
        index=False,
    )

    predictions_df.to_excel(
        outdir / "validation_all_predictions.xlsx",
        index=False,
    )

    summary_df.to_excel(
        outdir / "validation_summary.xlsx",
        index=False,
    )


    for model_name, model_pred_df in predictions_df.groupby("model_name"):
        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace(":", "")
            .replace("+", "plus")
            .replace("->", "to")
            .replace("/", "_")
        )

        model_pred_df.to_excel(
            outdir / f"{safe_name}_predictions.xlsx",
            index=False,
        )

    # Optional: save one summary file per model type
    for model_name, model_summary_df in summary_df.groupby("model_name"):
        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace(":", "")
            .replace("+", "plus")
            .replace("->", "to")
            .replace("/", "_")
        )

        model_summary_df.to_excel(
            outdir / f"{safe_name}_summary.xlsx",
            index=False,
        )

    print("\nSaved full repeated-CV validation results:")
    print(outdir / "validation_all_fold_results.xlsx")
    print(outdir / "validation_all_predictions.xlsx")
    print(outdir / "validation_summary.xlsx")

    return {
        "fold_results": fold_results_df,
        "predictions": predictions_df,
        "summary": summary_df,
    }

def run_monte_carlo_error_propagation() -> dict:
    from src.validated_gp_config import (
        BEST_DEPLOYED_DEPOSITION_GP,
        BEST_DEPLOYED_SELECTIVITY_GP,
    )

    outdir = (
        VALIDATION_RESULTS_DIR
        / "monte_carlo_error_propagation"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df = add_engineered_features(df)

    base_cols = make_base_controllable_input_cols(df)
    controllable_cols = make_model_input_cols(base_cols)
    dep_cols = make_deposition_input_cols(df)

    dep_gp_config = {
        key: value
        for key, value in BEST_DEPLOYED_DEPOSITION_GP.items()
        if key not in {"transform", "inverse_method"}
    }

    result = (
        cross_validate_stacked_monte_carlo_error_propagation(
            df=df,
            controllable_cols=controllable_cols,
            dep_cols=dep_cols,
            kernel_type=BEST_DEPLOYED_SELECTIVITY_GP["kernel_type"],
            n_restarts_optimizer=BEST_DEPLOYED_SELECTIVITY_GP[
                "n_restarts_optimizer"
            ],
            alpha=BEST_DEPLOYED_SELECTIVITY_GP["alpha"],
            deposition_kernel_type=dep_gp_config["kernel_type"],
            deposition_n_restarts_optimizer=dep_gp_config[
                "n_restarts_optimizer"
            ],
            deposition_alpha=dep_gp_config["alpha"],
            deposition_transform=BEST_DEPLOYED_DEPOSITION_GP["transform"],
            deposition_inverse_method=BEST_DEPLOYED_DEPOSITION_GP[
                "inverse_method"
            ],
            selectivity_kernel_type=BEST_DEPLOYED_SELECTIVITY_GP[
                "kernel_type"
            ],
            selectivity_n_restarts_optimizer=BEST_DEPLOYED_SELECTIVITY_GP[
                "n_restarts_optimizer"
            ],
            selectivity_alpha=BEST_DEPLOYED_SELECTIVITY_GP["alpha"],
            n_mc=2000,
        )
    )

    result["fold_results"].to_excel(
        outdir / "mc_error_propagation_fold_results.xlsx",
        index=False,
    )
    result["point_results"].to_excel(
        outdir / "mc_error_propagation_point_results.xlsx",
        index=False,
    )
    result["summary"].to_excel(
        outdir / "mc_error_propagation_summary.xlsx",
        index=False,
    )

    print("\nSaved Monte Carlo error propagation results:")
    print(outdir)
    return result

def run_bootstrap_error_propagation_best_stacked_model() -> dict:
    from src.validated_gp_config import (
        BEST_DEPLOYED_DEPOSITION_GP,
        BEST_DEPLOYED_SELECTIVITY_GP,
    )

    outdir = VALIDATION_RESULTS_DIR / "bootstrap_error_propagation"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data exactly the same way as run_validation_analysis()
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df = add_engineered_features(df)

    base_cols = make_base_controllable_input_cols(df)
    controllable_cols = make_model_input_cols(base_cols)
    dep_cols = make_deposition_input_cols(df)

    print("BASE COLS:", base_cols)
    print("MODEL INPUT COLS:", controllable_cols)
    print("DEPOSITION INPUT COLS:", dep_cols)
    print("BOOTSTRAP DEPLOYED DEPOSITION GP CONFIG:", BEST_DEPLOYED_DEPOSITION_GP)
    print("BOOTSTRAP DEPLOYED SELECTIVITY GP CONFIG:", BEST_DEPLOYED_SELECTIVITY_GP)

    dep_gp_config = {
        key: value
        for key, value in BEST_DEPLOYED_DEPOSITION_GP.items()
        if key not in {"transform", "inverse_method"}
    }

    result = cross_validate_stacked_bootstrap_error_propagation(
        df=df,
        controllable_cols=controllable_cols,
        dep_cols=dep_cols,
        kernel_type=BEST_DEPLOYED_SELECTIVITY_GP["kernel_type"],
        n_restarts_optimizer=BEST_DEPLOYED_SELECTIVITY_GP[
            "n_restarts_optimizer"
        ],
        alpha=BEST_DEPLOYED_SELECTIVITY_GP["alpha"],
        deposition_kernel_type=dep_gp_config["kernel_type"],
        deposition_n_restarts_optimizer=dep_gp_config[
            "n_restarts_optimizer"
        ],
        deposition_alpha=dep_gp_config["alpha"],
        deposition_transform=BEST_DEPLOYED_DEPOSITION_GP["transform"],
        deposition_inverse_method=BEST_DEPLOYED_DEPOSITION_GP[
            "inverse_method"
        ],
        selectivity_kernel_type=BEST_DEPLOYED_SELECTIVITY_GP["kernel_type"],
        selectivity_n_restarts_optimizer=BEST_DEPLOYED_SELECTIVITY_GP[
            "n_restarts_optimizer"
        ],
        selectivity_alpha=BEST_DEPLOYED_SELECTIVITY_GP["alpha"],
        n_bootstrap=200,
    )

    result["fold_results"].to_excel(
        outdir / "bootstrap_error_propagation_fold_results.xlsx",
        index=False,
    )

    result["point_results"].to_excel(
        outdir / "bootstrap_error_propagation_point_results.xlsx",
        index=False,
    )

    result["summary"].to_excel(
        outdir / "bootstrap_error_propagation_summary.xlsx",
        index=False,
    )

    print("\nSaved bootstrap error propagation results:")
    print(outdir / "bootstrap_error_propagation_fold_results.xlsx")
    print(outdir / "bootstrap_error_propagation_point_results.xlsx")
    print(outdir / "bootstrap_error_propagation_summary.xlsx")

    return result
