"""
Dataset loading and feasibility filtering.

All entry points read the shared repository dataset through this module so the
row subset used for training is defined in exactly one place.
"""

# Third party imports
import pandas as pd

# Local imports
from src.config import (
    EXCEL_FILE,
    SHEET_NAME,
    RAW_DEPOSITION_COL,
    FEASIBILITY_THRESHOLD,
    FEASIBLE_ONLY,
)


def load_dataset(feasible_only: bool | None = None) -> pd.DataFrame:
    """
    Load the shared dataset and apply the deposition feasibility filter.

    Rows below the deposition threshold are dropped because selectivity is not
    physically meaningful when almost nothing has deposited. Filtering here
    rather than relying on missing target values keeps the feasible subset
    consistent with the threshold used elsewhere in the repository.
    """
    if feasible_only is None:
        feasible_only = FEASIBLE_ONLY

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip()

    if RAW_DEPOSITION_COL not in df.columns:
        raise KeyError(
            f"Expected deposition column {RAW_DEPOSITION_COL!r} in {EXCEL_FILE}. "
            f"Found: {list(df.columns)}"
        )

    if feasible_only:
        keep = df[RAW_DEPOSITION_COL] >= FEASIBILITY_THRESHOLD
        df = df.loc[keep].reset_index(drop=True)

    df["_sort_key"] = df["Solution Label"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    df = df.sort_values("_sort_key", kind="stable").drop(columns="_sort_key").reset_index(drop=True)

    return df
