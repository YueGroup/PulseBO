"""
Constrained Bayesian optimization of pulsed electrodeposition.

A selectivity GP and a deposition GP drive a constrained expected improvement
batch loop that proposes the next round of experiments.
"""

import logging as _logging

from . import config
from .logging_utils import setup_logging
from .data import extract_features, parse, engineer_features, fit_scaler, scale

# NullHandler so importing the package does not configure logging
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
