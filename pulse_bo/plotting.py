"""Out-of-fold parity plotting for the selectivity GP surrogate."""

import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import N_SPLITS, N_REPEATS
from .data.features import extract_features, fit_scaler, scale
from .models.gp import make_gpr

logger = logging.getLogger(__name__)

# Fixed surrogate configuration used for the parity figure (matches the BO
# selectivity GP family).
KERNEL = "White + Matern"
ALPHA = 1e-6
N_RESTARTS = 8


def get_oof_predictions(X_raw, y, kernel=KERNEL, alpha=ALPHA, n_restarts=N_RESTARTS):
    """Repeated-k-fold out-of-fold predictions, averaged across repeats."""
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

    prediction_sum = np.zeros(len(y), dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_raw)):
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train = y[train_idx]

        mean, std = fit_scaler(X_train_raw)
        X_train = scale(X_train_raw, mean, std)
        X_test = scale(X_test_raw, mean, std)

        gp = make_gpr(kernel_name=kernel, n_features=X_train.shape[1],
                      alpha=alpha, n_restarts=n_restarts, seed=fold_idx)
        gp.fit(X_train, y_train)

        prediction_sum[test_idx] += gp.predict(X_test)
        prediction_count[test_idx] += 1

    return prediction_sum / prediction_count


def make_parity_plot(y_observed, y_predicted, outfile="results/selectivity_gp_parity.pdf"):
    """Render and save the observed-vs-predicted selectivity parity plot."""
    mae = mean_absolute_error(y_observed, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_observed, y_predicted))
    r2 = r2_score(y_observed, y_predicted)

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(y_observed, y_predicted, s=70, facecolor="#6688bb",
               edgecolor="black", linewidth=0.8, alpha=0.85, zorder=3)

    lower = min(y_observed.min(), y_predicted.min()) - 3
    upper = max(y_observed.max(), y_predicted.max()) + 3
    limits = [lower, upper]
    ax.plot(limits, limits, "k--", linewidth=1.5, zorder=2)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_xlabel("Observed selectivity", fontsize=18)
    ax.set_ylabel("Predicted selectivity", fontsize=18)
    ax.set_title("Selectivity GP surrogate parity", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)

    ax.text(0.05, 0.95,
            f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes, fontsize=14,
            verticalalignment="top", horizontalalignment="left",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                  "edgecolor": "gray"})

    ax.grid(alpha=0.3)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Parity metrics: MAE = %.3f | RMSE = %.3f | R2 = %.3f", mae, rmse, r2)
    logger.info("Saved %s", outfile)
    return {"mae": mae, "rmse": rmse, "r2": r2}
