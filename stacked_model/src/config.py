from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

EXCEL_FILE = REPO_ROOT / "data" / "cleaned_data.xlsx"
SHEET_NAME = 0

RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
ABLATION_RESULTS_DIR = RESULTS_DIR / "ablation"
VALIDATION_RESULTS_DIR = RESULTS_DIR / "validation"
CANDIDATE_RESULTS_DIR = RESULTS_DIR / "candidates"
PROSPECTIVE_RESULTS_DIR = RESULTS_DIR / "prospective"
SCALAR_SWEEP_RESULTS_DIR = RESULTS_DIR / "scalar_sweep"

RAW_V_COL = "Applied V"
RAW_VON_MS_COL = "Von (ms)"
RAW_VOFF_MS_COL = "Voff (ms)"
RAW_DEPOSITION_COL = "Total amount of deposition (ppm)"
DEPOSITION_UNITS = "ppm"
TARGET_COL = "Co selectivity"

# Feasibility is defined by the deposition threshold, not by a missing
# selectivity value. The shared dataset reports a selectivity number for every
# run, including sub-threshold runs where that number is not physically
# meaningful, so a null-based filter would silently train on the full table.
FEASIBILITY_THRESHOLD = 0.2942  # ppm, equivalent to 5e-9 mol
FEASIBLE_ONLY = True

CV_FOLDS = 5
CV_REPEATS = 10

RANDOM_SEED = 30

# Scalars
BETA_MAIN = 0.379175956
BETA_EXTRAP = 1.878553762

DEP_UNC_PENALTY_MAIN = 0.197503624
DEP_UNC_PENALTY_EXTRAP = 0.042781584

PC_PENALTY_MAIN = 0.086154037
PC_PENALTY_EXTRAP = 0.091300147

N_GLOBAL = 3000
N_LOCAL = 3000
N_CANDIDATES = 3000
LOCAL_SCALE_FRAC = 0.03
TOP_K = 10
N_EXPLOIT = 3
N_EXPLORE = 1
BOUND_BUFFER = 0.005
