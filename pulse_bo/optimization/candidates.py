"""Two-phase candidate generation.

For each voltage on the discrete grid: (1) a Sobol quasi-random sweep over the
continuous dimensions scored by CEI, then (2) L-BFGS-B refinement seeded from the
best Sobol points. All points are pooled, scored once more, and returned sorted
by CEI.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from ..config import (
    BOUNDS,
    APPLIED_V_GRID,
    N_SOBOL_CANDIDATES,
    N_LBFGS_STARTS,
    N_LBFGS_CANDIDATES,
    XI,
)
from ..data.features import scale
from .acquisition import (
    expected_improvement,
    probability_of_feasibility,
    constrained_ei,
)


def generate_candidates(gp_sel, mean_sel, std_sel,
                        gp_dep, mean_dep, std_dep,
                        y_best: float,
                        bounds: dict = BOUNDS,
                        applied_v_grid: np.ndarray = None,
                        n_sobol: int = N_SOBOL_CANDIDATES,
                        n_lbfgs_starts: int = N_LBFGS_STARTS,
                        n_lbfgs_pool: int = N_LBFGS_CANDIDATES,
                        xi: float = XI) -> pd.DataFrame:
    """Return a DataFrame of scored candidates sorted by descending CEI."""
    if applied_v_grid is None:
        applied_v_grid = APPLIED_V_GRID

    names = list(bounds.keys())
    cont_names = [n for n in names if n != "Applied V"]
    lo_cont = np.array([bounds[n][0] for n in cont_names])
    hi_cont = np.array([bounds[n][1] for n in cont_names])
    scipy_bounds_cont = list(zip(lo_cont, hi_cont))

    all_X_rows = []

    for v in applied_v_grid:
        v = round(v, 10)  # avoid float drift like -0.9000000001

        sampler = qmc.Sobol(d=len(cont_names), scramble=True, seed=42)
        X_cont_sobol = qmc.scale(sampler.random(n_sobol), lo_cont, hi_cont)
        X_sobol_full = np.column_stack([np.full(len(X_cont_sobol), v), X_cont_sobol])

        cei_sobol = constrained_ei(scale(X_sobol_full, mean_sel, std_sel),
                                   scale(X_sobol_full, mean_dep, std_dep),
                                   gp_sel, gp_dep, y_best, xi)
        X_seeds_cont = X_cont_sobol[np.argsort(cei_sobol)[-n_lbfgs_pool:][::-1]]

        def neg_cei_cont(x_cont, v=v):
            x_full = np.atleast_2d(np.concatenate([[v], x_cont]))
            return -float(np.squeeze(constrained_ei(
                scale(x_full, mean_sel, std_sel),
                scale(x_full, mean_dep, std_dep),
                gp_sel, gp_dep, y_best, xi,
            )))

        lbfgs_cont = []
        for seed in X_seeds_cont[:n_lbfgs_starts]:
            res = minimize(neg_cei_cont, seed, method="L-BFGS-B",
                           bounds=scipy_bounds_cont)
            if res.success or res.fun < 0:
                lbfgs_cont.append(res.x)

        cont_pool = np.vstack(
            [X_cont_sobol] + ([np.array(lbfgs_cont)] if lbfgs_cont else [])
        )
        all_X_rows.append(np.column_stack([np.full(len(cont_pool), v), cont_pool]))

    all_X = np.vstack(all_X_rows)
    X_sc_sel = scale(all_X, mean_sel, std_sel)
    X_sc_dep = scale(all_X, mean_dep, std_dep)

    mu_sel, sig_sel = gp_sel.predict(X_sc_sel, return_std=True)
    mu_dep, sig_dep = gp_dep.predict(X_sc_dep, return_std=True)

    df = pd.DataFrame(all_X, columns=names)
    df["pred_sel_mean"] = mu_sel
    df["pred_sel_std"] = sig_sel
    df["pred_dep_mean"] = mu_dep
    df["pred_dep_std"] = sig_dep
    df["p_feasible"] = probability_of_feasibility(mu_dep, sig_dep)
    df["ei"] = expected_improvement(mu_sel, sig_sel, y_best, xi)
    df["cei"] = constrained_ei(X_sc_sel, X_sc_dep, gp_sel, gp_dep, y_best, xi)

    return df.sort_values("cei", ascending=False).reset_index(drop=True)
