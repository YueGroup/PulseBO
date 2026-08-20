"""Feature engineering, dataset extraction, and standard scaling.

The GPs work in engineered pulse-shape features (voltage, duty cycle, period,
total on-time) rather than the raw on/off timings, which are collinear. Scaling
is a plain per-feature standardisation whose statistics are always fit on the
training split only.
"""

import numpy as np
import pandas as pd

from ..config import (
    RAW_X_COLS,
    Y_SEL_COL,
    Y_DEP_COL,
    DEP_BAD_THRESH,
)


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Von/Voff timings to (V, duty_cycle, period, total_von).

    Duty cycle and period are less collinear than the raw on/off times and map
    more directly onto the physically meaningful shape of the pulse.
    """
    df = df_raw.copy()
    period = df["Von (ms)"] + df["Voff (ms)"]
    df["duty_cycle"] = df["Von (ms)"] / period
    df["period_ms"] = period
    df["total_von_ms"] = df["Total Von (ms)"]
    return df[["Applied V", "duty_cycle", "period_ms", "total_von_ms"]]


def extract_features(file: str):
    """Load every valid sheet from ``file`` and return X, y_sel, y_dep, feasible.

    Sheets missing any required column are skipped. Feasibility is defined by the
    deposition threshold in :data:`pulse_bo.config.DEP_BAD_THRESH`.

    Returns
    -------
    X_df : pandas.DataFrame
        Engineered features.
    y_sel : numpy.ndarray
        Co selectivity (%).
    y_dep : numpy.ndarray
        Total deposition (ppm).
    feasible : numpy.ndarray of bool
        ``y_dep >= DEP_BAD_THRESH``.
    """
    all_data = pd.read_excel(file, sheet_name=None)
    X_list, ysel_list, ydep_list = [], [], []

    for _sheet, df in all_data.items():
        needed = RAW_X_COLS + [Y_SEL_COL, Y_DEP_COL]
        if not all(c in df.columns for c in needed):
            continue
        if "Solution Label" in df.columns:
            key = df["Solution Label"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
            df = df.assign(_sort_key=key).sort_values("_sort_key", kind="stable").drop(columns="_sort_key")
        sub = df[needed].dropna()
        sub = df[needed].dropna()
        if sub.empty:
            continue
        X_list.append(engineer_features(sub))
        ysel_list.append(sub[Y_SEL_COL])
        ydep_list.append(sub[Y_DEP_COL])

    if not X_list:
        raise ValueError("No valid sheets found with the required columns.")

    X_df = pd.concat(X_list, ignore_index=True)
    y_sel = pd.concat(ysel_list, ignore_index=True).to_numpy(dtype=float)
    y_dep = pd.concat(ydep_list, ignore_index=True).to_numpy(dtype=float)

    return X_df, y_sel, y_dep, (y_dep >= DEP_BAD_THRESH)


def fit_scaler(X_raw: np.ndarray):
    """Return (mean, std) for per-feature standardisation; guard zero-variance."""
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return mean, std


def scale(X_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply standardisation with previously fit statistics."""
    return (X_raw - mean) / std
