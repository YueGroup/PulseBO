"""pulse_bo: constrained Bayesian optimization of pulsed electrodeposition.

Two Gaussian-process surrogates (a feasibility-aware selectivity model and a
deposition model) drive a Constrained-Expected-Improvement batch-BO loop that
proposes the next round of pulse-electrodeposition experiments.
"""

import logging as _logging

from . import config
from .logging_utils import setup_logging
from .data import extract_features, parse, engineer_features, fit_scaler, scale

# Library best practice: attach a NullHandler so importing the package never
# emits "No handlers could be found" and never configures logging for the host
# application. The console/file handlers are added by setup_logging(), which the
# CLI entry points and run_bo() call for you.
_logging.getLogger("pulse_bo").addHandler(_logging.NullHandler())
from .models import (
    make_gpr,
    build_kernel,
    fit_final_models,
    evaluate_hyperparameters,
    calibration_metrics,
    inspect_kernel_length_scales,
)
from .optimization import (
    expected_improvement,
    probability_of_feasibility,
    constrained_ei,
    generate_candidates,
    select_batch,
)
from .pipeline import run_bo

__version__ = "0.1.0"

__all__ = [
    "config",
    "setup_logging",
    "extract_features",
    "parse",
    "engineer_features",
    "fit_scaler",
    "scale",
    "make_gpr",
    "build_kernel",
    "fit_final_models",
    "evaluate_hyperparameters",
    "calibration_metrics",
    "inspect_kernel_length_scales",
    "expected_improvement",
    "probability_of_feasibility",
    "constrained_ei",
    "generate_candidates",
    "select_batch",
    "run_bo",
]
