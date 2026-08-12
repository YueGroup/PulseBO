"""
Generates Bayesian optimization candidate recommendations using the stacked deposition/selectivity model.
"""

# Library import
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from src.config import (
    EXCEL_FILE,
    SHEET_NAME,
    TARGET_COL,
    RAW_DEPOSITION_COL,
    RANDOM_SEED,
    CANDIDATE_RESULTS_DIR,
    N_GLOBAL,
    N_LOCAL,
    N_EXPLOIT,
    N_EXPLORE,
    TOP_K,
    LOCAL_SCALE_FRAC,
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

from src.BO.candidate_utils import (
    generate_random_candidates,
    enforce_physical_consistency_base,
    add_derived_features_to_matrix,
    build_bounds,
)

from src.BO.acquisition import (
    score_stacked,
)

from src.validated_gp_config import (
    BEST_DEPLOYED_DEPOSITION_GP,
    BEST_DEPLOYED_SELECTIVITY_GP,
)

def minmax_scale(
        X: np.ndarray,
        bounds: list[tuple[float, float]]
) -> np.ndarray:

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower

    if np.any(span <= 0):
        raise ValueError("Non-positive span in bounds.")

    return (X - lower) / span


def pick_diverse_top_indices(
    X: np.ndarray,
    scores: np.ndarray,
    k: int,
    bounds: list[tuple[float, float]],
    min_dist: float = 0.10,
) -> list[int]:
    """
    Select a high-scoring batch while enforcing spacing in scaled input space.

    If too few candidates satisfy the spacing threshold, the remaining slots are
    filled by score order so the requested batch size is preserved.
    """
    if len(X) == 0:
        return []

    scaled = minmax_scale(X, bounds)
    order = np.argsort(scores)[::-1]
    chosen: list[int] = []

    for idx in order:
        idx = int(idx)
        if not chosen:
            chosen.append(idx)
        else:
            d = np.sqrt(np.sum((scaled[idx] - scaled[chosen]) ** 2, axis=1))
            if np.min(d) >= min_dist:
                chosen.append(idx)
        if len(chosen) >= k:
            break

    if len(chosen) < k:
        for idx in order:
            idx = int(idx)
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= k:
                break

    return chosen


def run_bo_candidates(
        outdir: Path = CANDIDATE_RESULTS_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Generate candidate recommendations from the deployed stacked model.

    The candidate pool combines global random samples with local perturbations
    around the best observed condition, removes physically invalid or already
    observed settings, predicts deposition for remaining candidates, then scores
    selectivity with the deployed stack.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df = add_engineered_features(df)

    base_cols = make_base_controllable_input_cols(df)
    controllable_cols = make_model_input_cols(base_cols)
    dep_cols = make_deposition_input_cols(df)
    mechanistic_cols = controllable_cols + [RAW_DEPOSITION_COL]

    print("BASE COLS:", base_cols)
    print("MODEL INPUT COLS:", controllable_cols)
    print("DEPOSITION INPUT COLS:", dep_cols)
    print("MECHANISTIC COLS:", mechanistic_cols)

    needed_cols = list(dict.fromkeys(
        base_cols + controllable_cols + dep_cols + mechanistic_cols + [TARGET_COL]
    ))

    train_df = df.dropna(subset=needed_cols).copy()

    model_b_bundle, _ = train_selectivity_model(
        train_df,
        input_cols=mechanistic_cols,
        target_col=TARGET_COL,
        **BEST_DEPLOYED_SELECTIVITY_GP,
    )

    dep_transform = BEST_DEPLOYED_DEPOSITION_GP["transform"]
    dep_inverse_method = BEST_DEPLOYED_DEPOSITION_GP.get(
        "inverse_method",
        "median",
    )
    dep_gp_config = {
        key: value
        for key, value in BEST_DEPLOYED_DEPOSITION_GP.items()
        if key not in {"transform", "inverse_method"}
    }

    dep_bundle, _ = train_deposition_model(
        train_df,
        controllable_cols=dep_cols,
        deposition_col=RAW_DEPOSITION_COL,
        transform=dep_transform,
        **dep_gp_config,
    )

    y_best = float(train_df[TARGET_COL].max())

    bounds = build_bounds(train_df, base_cols)
    observed_base = train_df[base_cols].values.astype(float)

    best_idx = int(np.argmax(train_df[TARGET_COL].values.astype(float)))
    best_x = observed_base[best_idx]

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower

    global_cands = generate_random_candidates(bounds, N_GLOBAL, rng)
    local_cands = best_x + rng.normal(0.0, LOCAL_SCALE_FRAC, size=(N_LOCAL, len(bounds))) * span
    local_cands = np.clip(local_cands, lower, upper)

    candidates_base = np.vstack([global_cands, local_cands])
    candidates_base = np.unique(candidates_base, axis=0)
    candidates_base = enforce_physical_consistency_base(candidates_base, base_cols)

    # Exclude exact repeats of observed controllable settings before scoring.
    obs_set = {tuple(np.round(row, 10)) for row in observed_base}
    keep_mask = np.array([tuple(np.round(row, 10)) not in obs_set for row in candidates_base], dtype=bool)
    candidates_base = candidates_base[keep_mask]

    # Keep pulse counts near the historically observed range for deployability.
    total_von_idx = base_cols.index("Total Von (s)")
    von_idx = base_cols.index("Von (s)")
    observed_pc = observed_base[:, total_von_idx] / np.maximum(observed_base[:, von_idx], 1e-12)
    max_pc_allowed = min(float(np.max(observed_pc) * 1.05), 1000.0)

    candidate_pc = candidates_base[:, total_von_idx] / np.maximum(candidates_base[:, von_idx], 1e-12)
    candidates_base = candidates_base[candidate_pc > 0.0]
    candidates_base = candidates_base[candidate_pc[candidate_pc > 0.0] <= max_pc_allowed]

    candidates_ctrl = add_derived_features_to_matrix(
        candidates_base,
        base_cols,
        controllable_cols,
    )

    candidates_dep = add_derived_features_to_matrix(
        candidates_base,
        base_cols,
        dep_cols,
    )

    dep_mu, dep_sigma = predict_deposition_logspace(
        dep_bundle,
        candidates_dep,
    )
    dep_pred, dep_std_proxy = deposition_prediction_to_raw(
        dep_mu,
        dep_sigma,
        transform=dep_transform,
        inverse_method=dep_inverse_method,
    )

    mech_X = np.column_stack([candidates_ctrl, dep_pred])
    mu_stack, sigma_stack = predict_gp(model_b_bundle, mech_X)
    scores_stack = score_stacked(mu_stack, sigma_stack, dep_std_proxy, y_best)

    top_idx = np.argsort(scores_stack)[::-1][:TOP_K]
    batch_idx = pick_diverse_top_indices(
        candidates_base,
        scores_stack,
        N_EXPLOIT + N_EXPLORE,
        bounds,
        min_dist=0.10,
    )

    col_idx = {col: i for i, col in enumerate(base_cols)}

    top_rows = []

    for rank, idx in enumerate(top_idx, start=1):
        row = {
            "Rank": rank,
            "Applied V": float(candidates_base[idx, col_idx["Applied V"]]),
            "Von (s)": float(candidates_base[idx, col_idx["Von (s)"]]),
            "Voff (s)": float(candidates_base[idx, col_idx["Voff (s)"]]),
            "Total Von (s)": float(candidates_base[idx, col_idx["Total Von (s)"]]),
            "Pred_Deposition": float(dep_pred[idx]),
            "Pred_Deposition_Unc_Proxy": float(dep_std_proxy[idx]),
            "Pred_Selectivity": float(mu_stack[idx]),
            "Pred_Selectivity_Unc": float(sigma_stack[idx]),
            "BO_Score": float(scores_stack[idx]),
        }

        if "V_stripping" in col_idx:
            row["V_stripping"] = float(candidates_base[idx, col_idx["V_stripping"]])

        row.update({
            "Pred_Deposition": float(dep_pred[idx]),
            "Pred_Deposition_Unc_Proxy": float(dep_std_proxy[idx]),
            "Pred_Selectivity": float(mu_stack[idx]),
            "Pred_Selectivity_Unc": float(sigma_stack[idx]),
            "BO_Score": float(scores_stack[idx]),
        })

        top_rows.append(row)

    batch_rows = []

    for rank, idx in enumerate(batch_idx, start=1):
        row = {
            "Batch_Rank": rank,
            "Applied V": float(candidates_base[idx, col_idx["Applied V"]]),
            "Von (s)": float(candidates_base[idx, col_idx["Von (s)"]]),
            "Voff (s)": float(candidates_base[idx, col_idx["Voff (s)"]]),
            "Total Von (s)": float(candidates_base[idx, col_idx["Total Von (s)"]]),
            "Pred_Deposition": float(dep_pred[idx]),
            "Pred_Deposition_Unc_Proxy": float(dep_std_proxy[idx]),
            "Pred_Selectivity": float(mu_stack[idx]),
            "Pred_Selectivity_Unc": float(sigma_stack[idx]),
            "BO_Score": float(scores_stack[idx]),
            "Role": "exploit" if rank <= N_EXPLOIT else "explore",
        }

        if "V_stripping" in col_idx:
            row["V_stripping"] = float(candidates_base[idx, col_idx["V_stripping"]])

        row.update({
            "Pred_Deposition": float(dep_pred[idx]),
            "Pred_Deposition_Unc_Proxy": float(dep_std_proxy[idx]),
            "Pred_Selectivity": float(mu_stack[idx]),
            "Pred_Selectivity_Unc": float(sigma_stack[idx]),
            "BO_Score": float(scores_stack[idx]),
        })
        batch_rows.append(row)

    top_df = pd.DataFrame(top_rows)
    batch_df = pd.DataFrame(batch_rows)

    print("\nTop candidate conditions")
    print(top_df.to_string(index=False))

    print("\nRecommended next BO batch")
    print(batch_df.to_string(index=False))

    top_df.to_excel(outdir / "bo_top_candidates.xlsx", index=False)
    batch_df.to_excel(outdir / "bo_recommended_batch.xlsx", index=False)

    print("\nSaved: bo_top_candidates.xlsx")
    print("Saved: bo_recommended_batch.xlsx")

    return {
        "top_candidates": top_df,
        "recommended_batch": batch_df,
    }

