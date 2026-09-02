"""
Builds the selectivity and deposition Gaussian processes and fits the final models.
"""

# Third party imports
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

# Local imports
from ..data.features import fit_scaler, scale


def build_kernel(kernel_name: str, n_features: int):
    ls = np.ones(n_features)
    if kernel_name == "White + RBF":
        return WhiteKernel(noise_level=1.0) + RBF(length_scale=ls)
    if kernel_name == "White + Matern":
        return WhiteKernel(noise_level=1.0) + Matern(length_scale=ls, nu=1.5)
    raise ValueError(f"Unknown kernel: {kernel_name}")


def make_gpr(kernel_name: str, n_features: int, alpha: float,
             n_restarts: int, seed: int) -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=build_kernel(kernel_name, n_features),
        alpha=alpha,
        normalize_y=True,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )


def fit_final_models(X_raw: np.ndarray, y_sel: np.ndarray,
                     y_dep: np.ndarray, feasible: np.ndarray,
                     best_sel: dict, best_dep: dict):
    """Fits both GPs on the full dataset using the selected hyperparameters."""
    # Selectivity GP: feasible runs only
    X_feas_raw = X_raw[feasible]
    y_feas = y_sel[feasible]
    mean_sel, std_sel = fit_scaler(X_feas_raw)
    gp_sel = make_gpr(best_sel["kernel"], X_feas_raw.shape[1],
                      best_sel["alpha"], int(best_sel["n_restarts"]), seed=42)
    gp_sel.fit(scale(X_feas_raw, mean_sel, std_sel), y_feas)

    # Deposition GP: all runs
    mean_dep, std_dep = fit_scaler(X_raw)
    gp_dep = make_gpr(best_dep["kernel"], X_raw.shape[1],
                      best_dep["alpha"], int(best_dep["n_restarts"]), seed=42)
    gp_dep.fit(scale(X_raw, mean_dep, std_dep), y_dep)

    return gp_sel, mean_sel, std_sel, gp_dep, mean_dep, std_dep
