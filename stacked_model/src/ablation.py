"""
Runs ablation tests to compare scoring components within the Bayesian optimization workflow.
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
    ABLATION_RESULTS_DIR,
    TOP_K,
    LOCAL_SCALE_FRAC,
    N_CANDIDATES,
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
    expected_improvement,
    normalize_01,
    score_stacked,
)
from src.validated_gp_config import (
    BEST_DEPLOYED_DEPOSITION_GP,
    BEST_DEPLOYED_SELECTIVITY_GP,
)

def score_model_a(
        mu: np.ndarray,
        sigma: np.ndarray,
        y_best: float,
) -> np.ndarray:
    ei = expected_improvement(mu, sigma, y_best)
    ucb = mu + 0.6 * sigma

    return 0.65 * normalize_01(ei) + 0.35 * normalize_01(ucb)

def score_no_dep_unc(
        sel_mean: np.ndarray,
        sel_std: np.ndarray,
        dep_std: np.ndarray,
        y_best: float,
) -> np.ndarray:
    ei = expected_improvement(sel_mean, sel_std, y_best)
    ucb = sel_mean + 0.379175956 * sel_std

    return (
        0.60 * normalize_01(sel_mean - y_best)
        + 0.20 * normalize_01(ucb)
        + 0.20 * normalize_01(ei)
    )


def score_no_uncertainty_bonus(
        sel_mean: np.ndarray,
        sel_std: np.ndarray,
        dep_std: np.ndarray,
        y_best: float,
) -> np.ndarray:
    ei = expected_improvement(sel_mean, sel_std, y_best)

    return (
        0.70 * normalize_01(sel_mean - y_best)
        + 0.25 * normalize_01(ei)
        - 0.05 * normalize_01(dep_std)
    )


def score_mean_only(
        sel_mean: np.ndarray,
        sel_std: np.ndarray,
        dep_std: np.ndarray,
        y_best: float,
) -> np.ndarray:

    return normalize_01(sel_mean)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(scores)[::-1][:k]


def topk_summary(
        name: str,
        idx: np.ndarray,
        sel_mean: np.ndarray,
        sel_std: np.ndarray,
        dep_std: np.ndarray | None = None,
) -> dict:
    out = {
        "Method": name,
        "Mean_Top10_Pred_Selectivity": float(np.mean(sel_mean[idx])),
        "Mean_Top10_Pred_Uncertainty": float(np.mean(sel_std[idx])),
    }

    if dep_std is not None:
        out["Mean_Top10_Dep_Unc_Proxy"] = float(np.mean(dep_std[idx]))

    return out


def overlap_count(
        idx_a: np.ndarray,
        idx_b: np.ndarray,
) -> int:

    return len(set(idx_a.tolist()).intersection(set(idx_b.tolist())))


def run_ablation_analysis(
        outdir: Path = ABLATION_RESULTS_DIR,
) -> dict[str, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df = add_engineered_features(df)

    base_cols = make_base_controllable_input_cols(df)
    controllable_cols = make_model_input_cols(base_cols)
    dep_cols = make_deposition_input_cols(df)
    mechanistic_cols = controllable_cols + [RAW_DEPOSITION_COL]

    train_df = df.dropna(
        subset=list(dict.fromkeys(base_cols + dep_cols + mechanistic_cols + [TARGET_COL]))
    ).copy()

    model_a_bundle, _ = train_selectivity_model(
        train_df,
        input_cols=controllable_cols,
        target_col=TARGET_COL,
    )

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

    global_cands = generate_random_candidates(bounds, N_CANDIDATES, rng)
    local_cands = best_x + rng.normal(0.0, LOCAL_SCALE_FRAC, size=(N_CANDIDATES, len(bounds))) * span
    local_cands = np.clip(local_cands, lower, upper)

    candidates_base = np.vstack([global_cands, local_cands])
    candidates_base = np.unique(candidates_base, axis=0)
    candidates_base = enforce_physical_consistency_base(candidates_base, base_cols)

    obs_set = {tuple(np.round(row, 10)) for row in observed_base}
    keep_mask = np.array([tuple(np.round(row, 10)) not in obs_set for row in candidates_base], dtype=bool)
    candidates_base = candidates_base[keep_mask]

    candidates_ctrl = add_derived_features_to_matrix(candidates_base, base_cols, controllable_cols)
    candidates_dep = add_derived_features_to_matrix(candidates_base, base_cols, dep_cols)

    # Model A predictions
    mu_a, sigma_a = predict_gp(model_a_bundle, candidates_ctrl)

    # Stacked predictions
    dep_mu, dep_sigma = predict_deposition_logspace(dep_bundle, candidates_dep)
    dep_pred, dep_std_proxy = deposition_prediction_to_raw(
        dep_mu,
        dep_sigma,
        transform=dep_transform,
        inverse_method=dep_inverse_method,
    )

    mech_X = np.column_stack([candidates_ctrl, dep_pred])
    mu_stack, sigma_stack = predict_gp(model_b_bundle, mech_X)

    score_map = {
        "Full_Stacked": score_stacked(mu_stack, sigma_stack, dep_std_proxy, y_best),
        "No_Dep_Unc_Penalty": score_no_dep_unc(mu_stack, sigma_stack, dep_std_proxy, y_best),
        "No_Uncertainty_Bonus": score_no_uncertainty_bonus(mu_stack, sigma_stack, dep_std_proxy, y_best),
        "Mean_Only": score_mean_only(mu_stack, sigma_stack, dep_std_proxy, y_best),
        "Model_A_Baseline": score_model_a(mu_a, sigma_a, y_best),
    }

    top_map = {name: topk_indices(scores, TOP_K) for name, scores in score_map.items()}
    full_top = top_map["Full_Stacked"]

    summary_rows = []

    for name, idx in top_map.items():
        if name == "Model_A_Baseline":
            row = topk_summary(name, idx, mu_a, sigma_a, None)

        else:
            row = topk_summary(name, idx, mu_stack, sigma_stack, dep_std_proxy)

        row["Overlap_With_Full_Top10"] = overlap_count(full_top, idx)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    detail_rows = []

    for name, idx in top_map.items():

        for rank, cand_idx in enumerate(idx, start=1):
            row = {
                "Method": name,
                "Rank": rank,
                "Applied V": float(candidates_base[cand_idx, base_cols.index("Applied V")]),
                "Von (s)": float(candidates_base[cand_idx, base_cols.index("Von (s)")]),
                "Voff (s)": float(candidates_base[cand_idx, base_cols.index("Voff (s)")]),
            }
            if "Total Von (s)" in base_cols:
                row["Total Von (s)"] = float(candidates_base[cand_idx, base_cols.index("Total Von (s)")])

            if name == "Model_A_Baseline":
                row["Pred_Selectivity"] = float(mu_a[cand_idx])
                row["Pred_Selectivity_Unc"] = float(sigma_a[cand_idx])
                row["Score"] = float(score_map[name][cand_idx])
            else:
                row["Pred_Deposition"] = float(dep_pred[cand_idx])
                row["Pred_Deposition_Unc_Proxy"] = float(dep_std_proxy[cand_idx])
                row["Pred_Selectivity"] = float(mu_stack[cand_idx])
                row["Pred_Selectivity_Unc"] = float(sigma_stack[cand_idx])
                row["Score"] = float(score_map[name][cand_idx])

            detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)

    print("\n" + "=" * 80)
    print("ABLATION TEST SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\nTop candidates by method")
    print(detail_df.to_string(index=False))

    summary_df.to_excel(outdir / "ablation_summary.xlsx", index=False)
    detail_df.to_excel(outdir / "ablation_top_candidates.xlsx", index=False)

    print("\nSaved: ablation_summary.xlsx")
    print("Saved: ablation_top_candidates.xlsx")
