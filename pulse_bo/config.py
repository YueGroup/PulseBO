import numpy as np

# Columns
RAW_X_COLS = ["Applied V", "Von (ms)", "Voff (ms)", "Total Von (ms)"]

X_COLS = ["Applied V", "duty_cycle", "period_ms", "total_von_ms"]

Y_SEL_COL = "Co selectivity"
Y_DEP_COL = "Total amount of deposition (ppm)"

# Feasibility
DEP_BAD_THRESH = 0.2942  # ppm

# Search space
BOUNDS = {
    "Applied V": (-1.5, -0.9),
    "duty_cycle": (0.005, 0.995),
    "period_ms": (15.0, 5000.0),
    "total_von_ms": (10000.0, 1000000.0),
}
MIN_PULSE_MS = 5.0  # shortest Von/Voff realized experimentally

APPLIED_V_GRID = np.arange(-1.5, -0.9 + 1e-9, 0.1)

# Cross validation
N_SPLITS = 5
N_REPEATS = 10

# Candidate generation
N_SOBOL_CANDIDATES = 8192  # Sobol points per voltage
N_LBFGS_STARTS = 32  # local-optimizer restarts per voltage
N_LBFGS_CANDIDATES = 256  # top Sobol points kept as L-BFGS seeds

# Batch selection
EXPLOIT_K = 10  # highest-CEI candidates
EXPLORE_K = 5  # highest-uncertainty candidates
BATCH_SIZE = EXPLOIT_K + EXPLORE_K
MIN_DIST_THRESHOLD = 0.10  # min normalised distance between chosen candidates
EXPLORE_PRIOR_CAP = 2  # explore slots kept when the pool is prior-dominated
STD_SIMILARITY_TOL = 0.01  # relative std spread below which explore is "flat"

# Acquisition
XI = 0.05  # exploration margin in Expected Improvement
KAPPA = 2.0  # UCB trade-off (diagnostic only)

# Hyperparameter grid
ALPHA_GRID = [1e-10, 1e-6, 1e-4]
N_RESTART_GRID = [8, 16]
KERNEL_NAMES = ["White + RBF", "White + Matern"]

# Defaults
DEFAULT_DATA_FILE = "data/cleaned_data.xlsx"
DEFAULT_RESULTS_DIR = "results"
