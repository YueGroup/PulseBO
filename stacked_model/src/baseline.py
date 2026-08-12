"""
Compares the controls only baseline model against the stacked deposition/selectivity pipeline.
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
    BASELINE_RESULTS_DIR,
    N_CANDIDATES,
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

def pick_top_unique(scores: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    return order[:k]

def generate_candidate_set(
        train_df: pd.DataFrame,
        base_cols: list[str],
        rng: np.random.Generator,
) -> tuple[np.array, list[tuple[float, float]]]:

    required_base_cols = ["Applied V", "Von (s)", "Voff (s)"]

    missing = [col for col in required_base_cols if col not in base_cols]

    if missing:
        raise ValueError(
            f"generate_candidate_set() requires base_cols with {required_base_cols}, "
            f"but missing {missing}. Received base_cols={base_cols}"
        )

    bounds = build_bounds(train_df, base_cols)

    observed_base = train_df[base_cols].values.astype(float)

    best_idx = int(np.argmax(train_df[TARGET_COL].values.astype(float)))
    best_x = observed_base[best_idx]

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower

    global_cands = generate_random_candidates(bounds, N_CANDIDATES, rng)

    local_cands = best_x + rng.normal(
        0.0,
        LOCAL_SCALE_FRAC,
        size = (N_CANDIDATES, len(bounds)),
    ) * span

    local_cands = np.clip(local_cands, lower, upper)

    candidates_base = np.vstack([global_cands, local_cands])
    candidates_base = np.unique(candidates_base, axis=0)

    candidates_base = enforce_physical_consistency_base(
        candidates_base,
        base_cols,
    )

    obs_set = {tuple(np.round(row, 10)) for row in observed_base}

    keep_mask = np.array(
        [tuple(np.round(row, 10)) not in obs_set for row in candidates_base],
        dtype=bool,
    )

    candidates_base = candidates_base[keep_mask]

    return candidates_base, bounds

def run_baseline_comparison(
        outdir: Path = BASELINE_RESULTS_DIR,
) -> dict[str, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df = add_engineered_features(df)

    # Match your current validated feature structure
    base_cols = make_base_controllable_input_cols(df)
    controllable_cols = make_model_input_cols(base_cols)
    dep_cols = make_deposition_input_cols(df)
    mechanistic_cols = controllable_cols + [RAW_DEPOSITION_COL]

    train_df = df.dropna(
        subset=list(dict.fromkeys(dep_cols + mechanistic_cols + [TARGET_COL]))
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

    # Build matrices
    candidates_ctrl = add_derived_features_to_matrix(candidates_base, base_cols, controllable_cols)
    candidates_dep = add_derived_features_to_matrix(candidates_base, base_cols, dep_cols)

    # Model A scoring
    mu_a, sigma_a = predict_gp(model_a_bundle, candidates_ctrl)
    scores_a = score_model_a(mu_a, sigma_a, y_best)

    # Stacked scoring
    dep_mu, dep_sigma = predict_deposition_logspace(dep_bundle, candidates_dep)
    dep_pred, dep_std_proxy = deposition_prediction_to_raw(
        dep_mu,
        dep_sigma,
        transform=dep_transform,
        inverse_method=dep_inverse_method,
    )

    mech_X = np.column_stack([candidates_ctrl, dep_pred])
    mu_stack, sigma_stack = predict_gp(model_b_bundle, mech_X)
    scores_stack = score_stacked(mu_stack, sigma_stack, dep_std_proxy, y_best)

    top_a = pick_top_unique(scores_a, TOP_K)
    top_stack = pick_top_unique(scores_stack, TOP_K)

    overlap = len(set(top_a.tolist()).intersection(set(top_stack.tolist())))

    rows_a = []
    for rank, idx in enumerate(top_a, start=1):
        row = {col: candidates_ctrl[idx, j] for j, col in enumerate(controllable_cols)}
        row["Rank"] = rank
        row["Method"] = "Model_A_Baseline"
        row["Pred_Selectivity"] = float(mu_a[idx])
        row["Pred_Selectivity_Unc"] = float(sigma_a[idx])
        row["Score"] = float(scores_a[idx])
        rows_a.append(row)

    rows_stack = []
    for rank, idx in enumerate(top_stack, start=1):
        row = {col: candidates_ctrl[idx, j] for j, col in enumerate(controllable_cols)}
        row["Rank"] = rank
        row["Method"] = "Stacked_Pipeline"
        row["Pred_Deposition"] = float(dep_pred[idx])
        row["Pred_Deposition_Unc_Proxy"] = float(dep_std_proxy[idx])
        row["Pred_Selectivity"] = float(mu_stack[idx])
        row["Pred_Selectivity_Unc"] = float(sigma_stack[idx])
        row["Score"] = float(scores_stack[idx])
        rows_stack.append(row)

    out_a = pd.DataFrame(rows_a)
    out_stack = pd.DataFrame(rows_stack)

    summary = pd.DataFrame([
        {
            "Metric": "Best observed selectivity",
            "Value": y_best,
        },
        {
            "Metric": "Mean predicted top-10 selectivity (Model A)",
            "Value": float(np.mean(mu_a[top_a])),
        },
        {
            "Metric": "Mean predicted top-10 selectivity (Stacked)",
            "Value": float(np.mean(mu_stack[top_stack])),
        },
        {
            "Metric": "Mean predicted top-10 uncertainty (Model A)",
            "Value": float(np.mean(sigma_a[top_a])),
        },
        {
            "Metric": "Mean predicted top-10 uncertainty (Stacked)",
            "Value": float(np.mean(sigma_stack[top_stack])),
        },
        {
            "Metric": "Top-10 index overlap count",
            "Value": overlap,
        },
    ])

    summary.to_excel(outdir / "baseline_comparison_summary.xlsx", index=False)
    out_a.to_excel(outdir / "baseline_controls_only_top10.xlsx", index=False)
    out_stack.to_excel(outdir / "baseline_stacked_top10.xlsx", index=False)

    print("\n" + "=" * 80)
    print("BASELINE_COMPARISON")
    print("=" * 80)
    print(summary.to_string(index=False))

    print(f"\nSaved results to: {outdir}")

    return {
        "summary": summary,
        "controls_only_top10": out_a,
        "stacked_top10": out_stack,
    }
