"""
Best GP configurations selected from repeated-CV validation.

Generated automatically by scripts/02_select_best_gp.py
"""

BEST_MODEL_A_GP = {
    "kernel_type": "matern_1.5",
    "n_restarts_optimizer": 0,
    "alpha": 0.01,
}

BEST_MODEL_B_GP = {
    "kernel_type": "rbf",
    "n_restarts_optimizer": 0,
    "alpha": 1e-06,
}

BEST_DEPOSITION_GP = {
    "kernel_type": "matern_1.5",
    "n_restarts_optimizer": 0,
    "alpha": 0.01,
}

# Deployed stack configuration.
# The deposition GP is selected from raw-deposition validation rows.
# controls -> deposition GP -> predicted deposition -> selectivity GP.
BEST_DEPLOYED_DEPOSITION_GP = {
    **BEST_DEPOSITION_GP,
    "transform": "raw",
    "inverse_method": "mean",
}

BEST_DEPLOYED_SELECTIVITY_GP = BEST_MODEL_B_GP
