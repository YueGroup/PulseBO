"""
Utility functions for candidate generation, filtering, and feature construction.
"""

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from src.config import (
    BOUND_BUFFER,
)

# Local helper functions
def generate_random_candidates(
        bounds: list[tuple[float, float]],
        n_samples: int,
        rng: np.random.Generator,
) -> np.ndarray:

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    samples = rng.random((n_samples, len(bounds)))

    return lower + samples * (upper - lower)

def enforce_physical_consistency_base(
        candidates_base: np.ndarray,
        base_cols: list[str]
) -> np.ndarray:

    out = candidates_base.copy()
    valid = np.ones(len(out), dtype=bool)

    col_idx = {col: i for i, col in enumerate(base_cols)}

    applied_v = out[:, col_idx["Applied V"]]
    von = out[:, col_idx["Von (s)"]]
    voff = out[:, col_idx["Voff (s)"]]

    valid &= np.isfinite(applied_v)
    valid &= np.isfinite(von)
    valid &= np.isfinite(voff)

    valid &= von >= 0.01
    valid &= voff >= 0.0
    valid &= (von + voff) > 0.0

    if "Total Von (s)" in col_idx:
        total_von = out[:, col_idx["Total Von (s)"]]
        valid &= np.isfinite(total_von)
        valid &= total_von > 0.0
        valid &= total_von >= von

    return out[valid]

def add_derived_features_to_matrix(
    X_base: np.ndarray,
    base_cols: list[str],
    full_cols: list[str],
) -> np.ndarray:

    col_idx = {c: i for i, c in enumerate(base_cols)}

    applied_v = X_base[:, col_idx["Applied V"]]
    von = X_base[:, col_idx["Von (s)"]]
    voff = X_base[:, col_idx["Voff (s)"]]

    out = []

    for col in full_cols:
        if col in col_idx:
            out.append(X_base[:, col_idx[col]])

        elif col == "AbsVoltage":
            out.append(np.abs(applied_v))

        elif col == "CyclePeriod_s":
            out.append(von + voff)

        elif col == "AbsV_x_Von":
            out.append(np.abs(applied_v) * von)

        elif col == "AbsV_x_TotalVon":
            total_von = X_base[:, col_idx["Total Von (s)"]]
            out.append(np.abs(applied_v) * total_von)

        elif col == "PulseCount":
            total_von = X_base[:, col_idx["Total Von (s)"]]
            out.append(total_von / np.maximum(von, 1e-12))

        else:
            raise ValueError(f"Unknown derived feature: {col}")

    return np.column_stack(out)

def build_bounds(
        df: pd.DataFrame,
        cols: list[str],
        bound_buffer: float = BOUND_BUFFER,
) -> list[tuple[float, float]]:

    bounds = []

    for col in cols:

        lower = float(df[col].min())
        upper = float(df[col].max())
        span = upper - lower

        if span <= 0:
            raise ValueError(f"Column '{col}' has zero/negative span.")

        lower_buffered = lower - bound_buffer * span
        upper_buffered = upper + bound_buffer * span

        if col in ["Von (s)", "Voff (s)", "CyclePeriod_s", "Total Von (s)"]:
            lower_buffered = max(0.0, lower_buffered)

        if col in ["AbsVoltage", "AbsV_x_Von", "AbsV_x_TotalVon", "PulseCount"]:
            lower_buffered = max(0.0, lower_buffered)

        bounds.append((float(lower_buffered), float(upper_buffered)))

    return bounds