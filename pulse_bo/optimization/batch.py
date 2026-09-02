"""
Selects the recommended batch as an exploit set and an explore set under a diversity filter.
"""

# Library import
import logging

# Third party imports
import numpy as np
import pandas as pd

from ..config import (
    BOUNDS,
    EXPLOIT_K,
    EXPLORE_K,
    MIN_DIST_THRESHOLD,
    STD_SIMILARITY_TOL,
    EXPLORE_PRIOR_CAP,
)

logger = logging.getLogger(__name__)


def _detect_prior_dominated_explore(explore_pool: pd.DataFrame,
                                    tol: float = STD_SIMILARITY_TOL) -> tuple[bool, int]:
    stds = explore_pool["pred_sel_std"].to_numpy()
    if len(stds) < 2:
        return False, EXPLORE_K
    mean_std = stds.mean()
    if mean_std < 1e-12:
        return True, EXPLORE_PRIOR_CAP
    is_dominated = (stds.max() - stds.min()) / mean_std < tol
    return is_dominated, EXPLORE_PRIOR_CAP if is_dominated else EXPLORE_K


def select_batch(candidates_df: pd.DataFrame,
                 exploit_k: int = EXPLOIT_K,
                 explore_k: int = EXPLORE_K,
                 min_dist: float = MIN_DIST_THRESHOLD,
                 std_tol: float = STD_SIMILARITY_TOL) -> pd.DataFrame:
    """Returns the exploit and explore batch with a selection_type column."""
    feat_cols = list(BOUNDS.keys())
    lo = np.array([BOUNDS[c][0] for c in feat_cols])
    hi = np.array([BOUNDS[c][1] for c in feat_cols])

    def normalise(X):
        return (X - lo) / (hi - lo + 1e-12)

    selected = []
    selected_X_norm = []

    def add_if_diverse(row):
        x_norm = normalise(row[feat_cols].to_numpy())
        if selected_X_norm:
            if np.linalg.norm(np.array(selected_X_norm) - x_norm, axis=1).min() < min_dist:
                return False
        selected.append(row)
        selected_X_norm.append(x_norm)
        return True

    exploit_pool = candidates_df.sort_values("cei", ascending=False).head(500)
    for _, row in exploit_pool.iterrows():
        if add_if_diverse(row) and len(selected) == exploit_k:
            break

    explore_pool = candidates_df.sort_values("pred_sel_std", ascending=False).head(500)
    is_dominated, effective_explore_k = _detect_prior_dominated_explore(explore_pool, tol=std_tol)

    stds = explore_pool["pred_sel_std"].to_numpy()
    mean_std = stds.mean() if len(stds) else 0.0
    spread = (stds.max() - stds.min()) / mean_std if mean_std > 1e-12 else 0.0

    if is_dominated:
        logger.info("[select_batch] Prior-dominated explore pool "
                    "(std spread %.4f < tol %s). "
                    "Capping explore: %d -> %d, freed slots go to exploit.",
                    spread, std_tol, explore_k, effective_explore_k)
    else:
        logger.info("[select_batch] Explore pool ok (std spread %.4f), "
                    "using explore_k=%d.", spread, explore_k)

    freed = explore_k - effective_explore_k
    effective_exploit_k = exploit_k + freed

    if freed > 0 and len(selected) == exploit_k:
        for _, row in exploit_pool.iterrows():
            if any(row.equals(s) for s in selected):
                continue
            if add_if_diverse(row) and len(selected) == effective_exploit_k:
                break

    selected_ids = {id(r) for r in selected}
    for _, row in explore_pool.iterrows():
        if id(row) in selected_ids:
            continue
        if add_if_diverse(row) and len(selected) == effective_exploit_k + effective_explore_k:
            break

    batch = pd.DataFrame(selected).reset_index(drop=True)
    n_exploit_actual = min(effective_exploit_k, len(batch))
    batch["selection_type"] = (
        ["exploit"] * n_exploit_actual +
        ["explore"] * max(0, len(batch) - n_exploit_actual)
    )
    return batch
