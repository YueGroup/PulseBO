"""
Cleans a raw experiment workbook into a single sheet.

Sheets that use a non-zero stripping voltage are dropped because the models have
no stripping voltage feature.
"""

# Library import
import logging

# Third party imports
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

COL_ALIASES = {
    "Applied V": "Applied V",
    "V_deposition (V)": "Applied V",
    "V_deposition": "Applied V",
    "Von (ms)": "Von (ms)",
    "t_deposition (ms)": "Von (ms)",
    "Voff (ms)": "Voff (ms)",
    "t_stripping (ms)": "Voff (ms)",
    "Total Von (s)": "Total Von (s)",
    "Total t_deposition (s)": "Total Von (s)",
    "Total V_deposition (s)": "Total Von (s)",
    "Solution Label": "Solution Label",
    "Co selectivity": "Co selectivity",
    "Total amount of deposition (ppm)": "Total amount of deposition (ppm)",
    "Total amount of deposition (moles)": "Total amount of deposition (moles)",
}

REQUIRED = [
    "Applied V",
    "Von (ms)",
    "Co selectivity",
    "Total amount of deposition (ppm)",
]

OUTPUT_COLS = [
    "Solution Label",
    "Applied V",
    "Von (ms)",
    "Voff (ms)",
    "Total Von (s)",
    "Total Von (ms)",
    "Co selectivity",
    "Total amount of deposition (ppm)",
    "Total amount of deposition (moles)",
]


def _clean_sheet(df: pd.DataFrame, sheet_name: str) -> tuple[pd.DataFrame, list[str]]:
    warnings = []

    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()

    if "V_stripping (V)" in df.columns:
        nonzero = df["V_stripping (V)"].dropna()
        nonzero = nonzero[nonzero != 0]
        if len(nonzero) > 0:
            warnings.append(
                f"[EXCLUDED] '{sheet_name}' has non-zero V_stripping values: "
                f"{sorted(nonzero.unique())}. GP model has no V_stripping feature."
            )
            return pd.DataFrame(columns=OUTPUT_COLS), warnings

    df = df.rename(columns={c: COL_ALIASES[c] for c in df.columns if c in COL_ALIASES})

    if "Voff (ms)" not in df.columns:
        df["Voff (ms)"] = 0.0

    if "Total Von (ms)" not in df.columns or df["Total Von (ms)"].isna().all():
        if "Total Von (s)" in df.columns:
            df["Total Von (ms)"] = df["Total Von (s)"] * 1000.0
        else:
            df["Total Von (ms)"] = np.nan
            warnings.append(f"[warn] '{sheet_name}' has no Total Von column.")

    if "Total Von (s)" not in df.columns:
        df["Total Von (s)"] = df["Total Von (ms)"] / 1000.0

    for col in ["Solution Label", "Total amount of deposition (ppm)"]:
        if col not in df.columns:
            df[col] = np.nan

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        warnings.append(f"[SKIPPED] '{sheet_name}' missing columns: {missing}")
        return pd.DataFrame(columns=OUTPUT_COLS), warnings

    mask = pd.Series(True, index=df.index)
    for col in REQUIRED:
        mask &= df[col].notna()

    return df[mask], warnings


def parse(source: str, dest: str) -> None:
    """Cleans a raw workbook into a single sheet."""
    """Read a raw workbook and write a single cleaned sheet to ``dest``."""
    xl = pd.ExcelFile(source)
    logger.info("Reading %s  (%s)", source, xl.sheet_names)

    frames, all_warnings = [], []

    for sheet in xl.sheet_names:
        raw = pd.read_excel(source, sheet_name=sheet, header=2)
        cols = list(raw.columns)
        # Columns whose names only appear in a merged header row come through as
        # Unnamed. Assign them by position: 20=selectivity, 22=deposition ppm,
        # 24=deposition moles.
        name_map = {
            20: "Co selectivity",
            22: "Total amount of deposition (ppm)",
            24: "Total amount of deposition (moles)",
        }
        for i, name in name_map.items():
            if i < len(cols) and str(cols[i]).startswith("Unnamed"):
                cols[i] = name
        raw.columns = cols
        cleaned, warns = _clean_sheet(raw, sheet)
        all_warnings.extend(warns)
        status = "EXCLUDED" if cleaned.empty else f"{len(cleaned)} rows"
        logger.info("  %s: %d raw -> %s", sheet, len(raw), status)
        if not cleaned.empty:
            frames.append(cleaned)

    if not frames:
        logger.warning("No valid data found.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[[c for c in OUTPUT_COLS if c in combined.columns]]
    combined.to_excel(dest, index=False, sheet_name="Sheet1")
    logger.info("%d rows -> %s", len(combined), dest)

    missing_dep = combined["Total amount of deposition (ppm)"].isna().sum()
    if missing_dep:
        logger.warning("%d rows missing deposition (ppm)", missing_dep)

    for w in all_warnings:
        logger.warning(w)
