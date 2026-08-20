"""
Functions for engineering feature inputs and input variable consistency.
"""

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from src.config import (
    RAW_V_COL,
    RAW_VON_MS_COL,
    RAW_VOFF_MS_COL,
)

# Local helper functions
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the pulse-derived variables used throughout validation and BO.

    These transformations define the scientific feature space for the current
    workflow, so downstream scripts rely on both the column names and the
    calculations remaining stable.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    df[RAW_V_COL] = pd.to_numeric(df[RAW_V_COL], errors="coerce")
    df["Von (s)"] = pd.to_numeric(df[RAW_VON_MS_COL], errors="coerce") / 1000.0
    df["Voff (s)"] = pd.to_numeric(df[RAW_VOFF_MS_COL], errors="coerce") / 1000.0
    df["Total Von (s)"] = pd.to_numeric(df["Total Von (s)"], errors="coerce")

    # Derived features
    df["AbsVoltage"] = np.abs(df[RAW_V_COL])
    df["CyclePeriod_s"] = df["Von (s)"] + df["Voff (s)"]
    df["AbsV_x_Von"] = np.abs(df[RAW_V_COL]) * df["Von (s)"]
    df["AbsV_x_TotalVon"] = np.abs(df[RAW_V_COL]) * df["Total Von (s)"]

    df["PulseCount"] = df["Total Von (s)"] / np.maximum(df["Von (s)"], 1e-12)

    return df


def make_base_controllable_input_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the experimentally controllable pulse variables before derivation.

    `Total Von (s)` is included only when present in the source table because
    historical input files may differ in available columns.
    """
    cols = ["Applied V", "Von (s)", "Voff (s)"]

    if "Total Von (s)" in df.columns:
        cols.append("Total Von (s)")

    return cols

def make_model_input_cols(base_cols: list[str]) -> list[str]:
    """
    Return the selectivity-model control features in the current fixed order.

    The selectivity GP uses derived pulse descriptors instead of raw `Von (s)`
    and `Total Von (s)` in this workflow. Preserve this ordering because saved
    validation artifacts and trained GP feature arrays depend on it.
    """
    cols = list(base_cols)

    if "AbsV_x_Von" not in cols:
        cols.append("AbsV_x_Von")

    if "PulseCount" not in cols:
        cols.append("PulseCount")

    if "Von (s)" in cols:
        cols.remove("Von (s)")

    if "Total Von (s)" in cols:
        cols.remove("Total Von (s)")

    return cols

def make_deposition_input_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the deposition-GP input features in the current fixed order.

    The deposition surrogate is intentionally trained on this controls-only
    subset so the deployed stack can predict deposition for new candidates
    before selectivity is evaluated.
    """
    cols = [
        "Applied V",
        "Voff (s)",
        "AbsV_x_Von",
        "PulseCount",
    ]

    return [col for col in cols if col in df.columns]
