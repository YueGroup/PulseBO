"""Model evaluation: repeated k-fold CV, calibration, and length-scale audit."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor

from ..config import (
    N_SPLITS,
    N_REPEATS,
    KERNEL_NAMES,
    N_RESTART_GRID,
    ALPHA_GRID,
)
from ..data.features import fit_scaler, scale
from .gp import make_gpr

logger = logging.getLogger(__name__)


def calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_std: np.ndarray) -> dict:
    """90% prediction-interval coverage and NLPD (both lower-is-better for NLPD)."""
    z90 = 1.645
    coverage = float(np.mean((y_true >= y_pred - z90 * y_std) &
                             (y_true <= y_pred + z90 * y_std)))
    y_std_safe = np.where(y_std < 1e-12, 1e-12, y_std)
    nlpd = float(np.mean(
        0.5 * np.log(2 * np.pi * y_std_safe ** 2) +
        0.5 * ((y_true - y_pred) / y_std_safe) ** 2
    ))
    return {"coverage_90": coverage, "nlpd": nlpd}


def evaluate_hyperparameters(X_raw: np.ndarray, y: np.ndarray,
                             label: str = "model") -> tuple[pd.DataFrame, dict]:
    """Grid-search kernel x n_restarts x alpha with repeated k-fold CV.

    Scaling is re-fit inside every fold on the training split only. Returns the
    full results table (sorted by mean RMSE) and the best-row dict.
    """
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    results = []

    for kernel_name in KERNEL_NAMES:
        for n_restarts in N_RESTART_GRID:
            for alpha in ALPHA_GRID:
                fold_rmse, fold_r2, fold_cov, fold_nlpd, fold_lml = [], [], [], [], []

                for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_raw)):
                    X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]

                    mean, std = fit_scaler(X_tr_raw)
                    X_tr = scale(X_tr_raw, mean, std)
                    X_te = scale(X_te_raw, mean, std)

                    gpr = make_gpr(kernel_name, X_tr.shape[1], alpha,
                                   n_restarts, seed=fold_idx)
                    gpr.fit(X_tr, y_tr)
                    y_pred, y_std = gpr.predict(X_te, return_std=True)

                    fold_rmse.append(np.sqrt(mean_squared_error(y_te, y_pred)))
                    fold_r2.append(r2_score(y_te, y_pred))
                    fold_lml.append(gpr.log_marginal_likelihood(gpr.kernel_.theta))
                    cal = calibration_metrics(y_te, y_pred, y_std)
                    fold_cov.append(cal["coverage_90"])
                    fold_nlpd.append(cal["nlpd"])

                results.append({
                    "label": label,
                    "kernel": kernel_name,
                    "n_restarts": n_restarts,
                    "alpha": alpha,
                    "mean_rmse": float(np.mean(fold_rmse)),
                    "std_rmse": float(np.std(fold_rmse, ddof=0)),
                    "mean_r2": float(np.mean(fold_r2)),
                    "std_r2": float(np.std(fold_r2, ddof=0)),
                    "mean_coverage_90": float(np.mean(fold_cov)),
                    "mean_nlpd": float(np.mean(fold_nlpd)),
                    "mean_lml": float(np.mean(fold_lml)),
                })

    df = (pd.DataFrame(results)
          .sort_values(["mean_rmse", "std_rmse"])
          .reset_index(drop=True))
    best = df.iloc[0].to_dict()
    return df, best


def inspect_kernel_length_scales(gp: GaussianProcessRegressor,
                                 feature_names: list[str],
                                 X_df: dict,
                                 scaler_std: np.ndarray,
                                 label: str = "GP") -> pd.DataFrame:
    """Convert learned length scales back to raw units and flag ignored features.

    A length scale much larger than a feature's search range means the GP surface
    is nearly flat in that feature, i.e. the feature is effectively ignored.
    """
    kernel = gp.kernel_
    ls_std = None
    for component in [kernel, getattr(kernel, "k1", None), getattr(kernel, "k2", None)]:
        if component is not None and hasattr(component, "length_scale"):
            ls_std = np.atleast_1d(component.length_scale)
            break

    if ls_std is None:
        logger.warning("[%s] Could not extract length scales from kernel: %s",
                       label, kernel)
        return pd.DataFrame()

    if len(ls_std) == 1:
        ls_std = np.repeat(ls_std, len(feature_names))

    ls_raw = ls_std * scaler_std
    data_ranges = X_df[feature_names].max().to_numpy() - X_df[feature_names].min().to_numpy()
    ratio = ls_raw / data_ranges

    flags = []
    for r in ratio:
        if r > 5.0:
            flags.append("EFFECTIVELY IGNORED  (ls > 5x range)")
        elif r > 2.0:
            flags.append("weakly informative   (ls > 2x range)")
        else:
            flags.append("informative")

    df = pd.DataFrame({
        "feature": feature_names,
        "ls_std_space": ls_std,
        "ls_raw_units": ls_raw,
        "data_ranges": data_ranges,
        "ls_ratio": ratio,
        "verdict": flags,
    })

    lines = [
        "=" * 70,
        f"Length-scale audit - {label}",
        "=" * 70,
        f"{'Feature':<18} {'ls (std)':<12} {'ls (raw)':<14} "
        f"{'range':<14} {'ratio':<8} Verdict",
        "-" * 70,
    ]
    for _, row in df.iterrows():
        print(f"{row['feature']:<18} {row['ls_std_space']:<12.4f} "
              f"{row['ls_raw_units']:<14.4g} {row['data_ranges']:<14.4g} "
              f"{row['ls_ratio']:<8.2f} {row['verdict']}")
    print(f"{'='*70}")

    flagged = df[df["ls_ratio"] > 2.0]
    if flagged.empty:
        print("All features informative.")
    else:
        for _, row in flagged.iterrows():
            lo = X_df[row["feature"]].min()
            hi = X_df[row["feature"]].max()
            print(f"\n  {row['feature']}: ls={row['ls_raw_units']:.4g} vs range {row['data_ranges']:.4g} "
                  f"({row['ls_ratio']:.1f}x) — GP surface nearly flat across [{lo:.4g}, {hi:.4g}]")

    logger.info("\n" + "\n".join(lines))
    return df
