"""End-to-end BO pipeline: CV -> fit -> length-scale audit -> candidates -> batch.

``run_bo`` ties the stages together and backs the ``pulse-bo-run`` console entry
point.
"""

import logging
import os

import pandas as pd

from .config import (
    X_COLS,
    BOUNDS,
    DEP_BAD_THRESH,
    DEFAULT_DATA_FILE,
    DEFAULT_RESULTS_DIR,
)
from .logging_utils import setup_logging
from .data.features import extract_features
from .models.evaluation import evaluate_hyperparameters, inspect_kernel_length_scales
from .models.gp import fit_final_models
from .optimization.candidates import generate_candidates
from .optimization.batch import select_batch

logger = logging.getLogger(__name__)


def log_cv_table(df: pd.DataFrame, label: str) -> None:
    cols = ["kernel", "n_restarts", "alpha",
            "mean_rmse", "std_rmse", "mean_r2",
            "mean_coverage_90", "mean_nlpd", "mean_lml"]
    logger.info("%s CV results (sorted by mean RMSE):\n%s",
                label, df[cols].to_string(index=False))


def log_batch(batch: pd.DataFrame) -> None:
    display_cols = list(BOUNDS.keys()) + [
        "pred_sel_mean", "pred_sel_std", "p_feasible", "cei", "selection_type"
    ]
    logger.info("Recommended batch (%d candidates):\n%s",
                len(batch), batch[display_cols].to_string(index=False))


def run_bo(data_file: str = DEFAULT_DATA_FILE,
           results_dir: str = DEFAULT_RESULTS_DIR,
           save_candidates: bool = True,
           write_log: bool = True):
    """Run the full workflow and write result CSVs into ``results_dir``.

    Parameters
    ----------
    data_file : str
        Path to the cleaned single-sheet workbook.
    results_dir : str
        Directory for output CSVs and the run log (created if needed).
    save_candidates : bool
        If False, skip writing the large ``candidates_full.csv``.
    write_log : bool
        If True, write a timestamped ``run.log`` into ``results_dir`` alongside
        the CSVs, giving each run a self-describing record.
    """
    os.makedirs(results_dir, exist_ok=True)
    logfile = os.path.join(results_dir, "run.log") if write_log else None
    setup_logging(logfile=logfile)

    X_df, y_sel, y_dep, feasible = extract_features(data_file)
    X_raw = X_df.to_numpy(dtype=float)

    logger.info("Loaded %d points  |  feasible: %d  |  infeasible: %d  (%.1f%%)",
                len(X_raw), feasible.sum(), (~feasible).sum(),
                100 * (~feasible).sum() / len(X_raw))
    logger.info("Feasibility threshold: y_dep >= %.4f ppm", DEP_BAD_THRESH)

    X_feas_raw = X_raw[feasible]
    y_feas = y_sel[feasible]

    logger.info("Searching selectivity GP hyperparameters (feasible points only)...")
    cv_sel_df, best_sel = evaluate_hyperparameters(X_feas_raw, y_feas, label="selectivity")
    log_cv_table(cv_sel_df, "Selectivity GP")
    logger.info("Best: %s, alpha=%s, n_restarts=%d, RMSE=%.4f, R2=%.3f, cov90=%.3f",
                best_sel["kernel"], best_sel["alpha"], int(best_sel["n_restarts"]),
                best_sel["mean_rmse"], best_sel["mean_r2"], best_sel["mean_coverage_90"])

    logger.info("Searching deposition GP hyperparameters (all points)...")
    cv_dep_df, best_dep = evaluate_hyperparameters(X_raw, y_dep, label="deposition")
    log_cv_table(cv_dep_df, "Deposition GP")
    logger.info("Best: %s, alpha=%s, n_restarts=%d, RMSE=%.4e, R2=%.3f, cov90=%.3f",
                best_dep["kernel"], best_dep["alpha"], int(best_dep["n_restarts"]),
                best_dep["mean_rmse"], best_dep["mean_r2"], best_dep["mean_coverage_90"])

    logger.info("Fitting final models on full dataset...")
    gp_sel, mean_sel, std_sel, gp_dep, mean_dep, std_dep = fit_final_models(
        X_raw, y_sel, y_dep, feasible, best_sel, best_dep
    )
    logger.info("Selectivity GP: %s", gp_sel.kernel_)
    logger.info("Deposition  GP: %s", gp_dep.kernel_)

    ls_sel_df = inspect_kernel_length_scales(gp_sel, X_COLS, BOUNDS, std_sel,
                                             label="Selectivity GP")
    ls_dep_df = inspect_kernel_length_scales(gp_dep, X_COLS, BOUNDS, std_dep,
                                             label="Deposition GP")
    ls_sel_df.to_csv(os.path.join(results_dir, "length_scales_selectivity.csv"), index=False)
    ls_dep_df.to_csv(os.path.join(results_dir, "length_scales_deposition.csv"), index=False)

    y_best = float(y_feas.max())
    logger.info("Generating candidates (Sobol + L-BFGS-B), y_best = %.4f...", y_best)
    candidates_df = generate_candidates(
        gp_sel, mean_sel, std_sel,
        gp_dep, mean_dep, std_dep,
        y_best=y_best,
    )

    batch = select_batch(candidates_df)
    log_batch(batch)

    cv_sel_df.to_csv(os.path.join(results_dir, "cv_selectivity_gp.csv"), index=False)
    cv_dep_df.to_csv(os.path.join(results_dir, "cv_deposition_gp.csv"), index=False)
    batch.to_csv(os.path.join(results_dir, "batch_recommendations.csv"), index=False)
    if save_candidates:
        candidates_df.to_csv(os.path.join(results_dir, "candidates_full.csv"), index=False)

    logger.info("Saved results to %s/: cv_selectivity_gp.csv, cv_deposition_gp.csv, "
                "length_scales_selectivity.csv, length_scales_deposition.csv, "
                "batch_recommendations.csv%s%s",
                results_dir,
                ", candidates_full.csv" if save_candidates else "",
                ", run.log" if write_log else "")

    return batch, candidates_df
