from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

from src.config import (
    RANDOM_SEED,
    RAW_DEPOSITION_COL,
)



N_RESTARTS_OPTIMIZER = 3
LENGTH_SCALE_BOUNDS = (1e-3, 1e4)
NOISE_LEVEL_BOUNDS = (1e-3, 1.0)
DEFAULT_NOISE_LEVEL = 5e-2
BOUND_BUFFER = 0.005


@dataclass
class GPModelBundle:
    gp: GaussianProcessRegressor
    bounds: list[tuple[float, float]]
    y_mean: float
    y_std: float
    feature_names: list[str]


def minmax_scale(X: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower
    if np.any(span <= 0):
        raise ValueError(f"Non-positive span found in bounds: {bounds}")
    return (X - lower) / span


def build_bounds(
        df: pd.DataFrame,
        cols: list[str],
        bound_buffer: float = BOUND_BUFFER,
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []

    for col in cols:

        lower = float(df[col].min())
        upper = float(df[col].max())
        span = upper - lower

        if span <= 0:
            raise ValueError(f"Column '{col}' has zero/negative span; cannot scale safely.")

        lower_buffered = lower - bound_buffer * span
        upper_buffered = upper + bound_buffer * span

        if col in ["Von (s)", "Voff (s)", "CyclePeriod_s", "Total Von (s)"]:
            lower_buffered = max(0.0, lower_buffered)

        if col == "DutyCycle":
            lower_buffered = max(0.0, lower_buffered)
            upper_buffered = min(1.0, upper_buffered)

        if col in [
            "Frequency_Hz",
            "AbsVoltage",
            "AbsV_x_Von",
            "AbsV_x_Period",
            "AbsV_x_TotalVon",
            "PulseCount",
        ]:
            lower_buffered = max(0.0, lower_buffered)

        bounds.append((float(lower_buffered), float(upper_buffered)))

    return bounds


def build_gp_kernel(
    n_features: int,
    kernel_kind: str = "matern",
    matern_nu: float = 2.5,
):
    if kernel_kind == "matern":
        base_kernel = Matern(
            length_scale=[0.5] * n_features,
            length_scale_bounds=LENGTH_SCALE_BOUNDS,
            nu=matern_nu,
        )
    elif kernel_kind == "rbf":
        base_kernel = RBF(
            length_scale=[0.5] * n_features,
            length_scale_bounds=LENGTH_SCALE_BOUNDS,
        )
    else:
        raise ValueError(f"Unknown kernel_kind: {kernel_kind}")

    return (
        ConstantKernel(1.0, (1e-2, 1e2))
        * base_kernel
        + WhiteKernel(
            noise_level=DEFAULT_NOISE_LEVEL,
            noise_level_bounds=NOISE_LEVEL_BOUNDS,
        )
    )

def parse_kernel_type(
        kernel_type: Optional[str] = None,
        kernel_kind: str = "matern",
        matern_nu: float = 2.5,
) -> tuple[str, float]:
    """
    Convert validation kernel labels into the internal kernel settings.

    Accepted:
    - "rbf"
    - "matern"
    - "matern_1.5"
    - "matern_2.5"
    """

    if kernel_type is None:
        return kernel_kind, matern_nu

    kt = str(kernel_type).strip().lower()

    if kt == "rbf":
        return "rbf", matern_nu

    if kt == "matern":
        return "matern", matern_nu

    if kt == "matern_1.5":
        return "matern", 1.5

    if kt == "matern_2.5":
        return "matern", 2.5

    raise ValueError(f"Unknown kernel_type: {kernel_type}")

def fit_gp_model(
        X: np.ndarray,
        y: np.ndarray,
        bounds: list[tuple[float, float]],
        feature_names: list[str],
        kernel_kind: str = "matern",
        matern_nu: float = 2.5,
        n_restarts_optimizer: int = N_RESTARTS_OPTIMIZER,
        alpha: float = 1e-10,
) -> GPModelBundle:
    X_scaled = minmax_scale(X, bounds)

    y_mean = float(np.mean(y))
    y_std = float(np.std(y))

    if y_std <= 0:
        raise ValueError("Target has zero variance; GP cannot be fit meaningfully.")

    y_scaled = (y - y_mean) / y_std

    kernel = build_gp_kernel(
        X.shape[1],
        kernel_kind=kernel_kind,
        matern_nu=matern_nu,
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=RANDOM_SEED,
        normalize_y=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.fit(X_scaled, y_scaled)

        if caught:
            print("\n[GP fit warnings]")
            for w in caught:
                print(f"  {w.category.__name__}: {w.message}")

    return GPModelBundle(
        gp=gp,
        bounds=bounds,
        y_mean=y_mean,
        y_std=y_std,
        feature_names=feature_names,
    )


def predict_gp(bundle: GPModelBundle, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = minmax_scale(X, bundle.bounds)
    mu_scaled, sigma_scaled = bundle.gp.predict(X_scaled, return_std=True)

    mu = mu_scaled * bundle.y_std + bundle.y_mean
    sigma = sigma_scaled * bundle.y_std

    return mu, sigma


def transform_deposition_target(y_raw: np.ndarray, transform: str) -> np.ndarray:
    """
    Convert measured deposition into the target space used by the deposition GP.

    The active deployed workflow uses `raw`; `log1p` remains supported for
    historical comparisons and diagnostic checks.
    """
    if transform == "log1p":
        return np.log1p(y_raw)
    if transform == "raw":
        return y_raw
    raise ValueError(f"Unknown deposition transform: {transform}")


def train_selectivity_model(
        df: pd.DataFrame,
        input_cols: list[str],
        target_col: str = "Co selectivity",
        kernel_kind: str = "matern",
        matern_nu: float = 2.5,
        kernel_type: Optional[str] = None,
        n_restarts_optimizer: int = N_RESTARTS_OPTIMIZER,
        alpha: float = 1e-10,
) -> tuple[GPModelBundle, pd.DataFrame]:

    kernel_kind, matern_nu = parse_kernel_type(
        kernel_type=kernel_type,
        kernel_kind=kernel_kind,
        matern_nu=matern_nu,
    )

    sel_df = df.dropna(subset=input_cols + [target_col]).copy()

    X_sel = sel_df[input_cols].values.astype(float)
    y_sel = sel_df[target_col].values.astype(float)

    sel_bounds = build_bounds(sel_df, input_cols)

    bundle = fit_gp_model(
        X_sel,
        y_sel,
        sel_bounds,
        input_cols,
        kernel_kind=kernel_kind,
        matern_nu=matern_nu,
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=alpha,
    )

    return bundle, sel_df


def train_deposition_model(
        df: pd.DataFrame,
        controllable_cols: list[str],
        deposition_col: str = RAW_DEPOSITION_COL,
        transform: str = "log1p",
        kernel_kind: str = "matern",
        matern_nu: float = 2.5,
        kernel_type: Optional[str] = None,
        n_restarts_optimizer: int = N_RESTARTS_OPTIMIZER,
        alpha: float = 1e-10,
) -> tuple[GPModelBundle, pd.DataFrame]:
    """
    Fit the first-stage deposition surrogate.

    This model only uses controllable pulse features. The transform argument
    determines the GP target space; predictions are converted back to the raw
    deposition scale before entering the downstream selectivity GP.
    """

    kernel_kind, matern_nu = parse_kernel_type(
        kernel_type=kernel_type,
        kernel_kind=kernel_kind,
        matern_nu=matern_nu,
    )

    dep_df = df.dropna(subset=controllable_cols + [deposition_col]).copy()

    X_dep = dep_df[controllable_cols].values.astype(float)
    y_dep_raw = dep_df[deposition_col].values.astype(float)
    y_dep = transform_deposition_target(y_dep_raw, transform)

    dep_bounds = build_bounds(dep_df, controllable_cols)

    bundle = fit_gp_model(
        X_dep,
        y_dep,
        dep_bounds,
        controllable_cols,
        kernel_kind=kernel_kind,
        matern_nu=matern_nu,
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=alpha,
    )

    return bundle, dep_df


def predict_deposition_logspace(dep_bundle: GPModelBundle, X_ctrl_real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict deposition in the GP target space.

    The function name is retained for compatibility with older log-space runs;
    when the deposition model is trained with `transform="raw"`, the returned
    mean and standard deviation are already in raw-deposition target space.
    """
    return predict_gp(dep_bundle, X_ctrl_real)


def deposition_prediction_to_raw(
        mu: np.ndarray,
        sigma: np.ndarray,
        transform: str,
        inverse_method: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert deposition GP predictions to the raw deposition scale and a raw-space
    uncertainty proxy.

    For raw-target GPs, negative predictive means are clipped to zero for the
    point estimate. The uncertainty proxy remains the symmetric one-standard-
    deviation interval width in the target scale.
    """

    if transform == "log1p":
        if inverse_method == "median":
            pred = np.expm1(mu)
        elif inverse_method == "mean":
            pred = np.expm1(mu + 0.5 * np.maximum(sigma, 1e-12) ** 2)
        else:
            raise ValueError(f"Unknown inverse_method: {inverse_method}")

        upper = np.expm1(mu + sigma)
        lower = np.expm1(mu - sigma)

    elif transform == "raw":
        pred = np.maximum(mu, 0.0)
        upper = mu + sigma
        lower = mu - sigma

    else:
        raise ValueError(f"Unknown deposition transform: {transform}")

    std_proxy = np.maximum((upper - lower) / 2.0, 1e-12)

    return pred, std_proxy


def summarize_logspace_positive_prediction(
    mu_log: np.ndarray,
    sigma_log: np.ndarray,
    n_mc: int = 400,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    samples_log = rng.normal(
        loc=mu_log[:, None],
        scale=np.maximum(sigma_log[:, None], 1e-12),
        size=(len(mu_log), n_mc),
    )

    samples = np.expm1(samples_log)
    samples = np.maximum(samples, 0.0)

    return np.mean(samples, axis=1), np.std(samples, axis=1)
