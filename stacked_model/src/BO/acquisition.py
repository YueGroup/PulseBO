"""
Utility functions for scoring parameters/acquisition function.
"""

# Third party imports
import numpy as np
from scipy.stats import norm

# Local helper functions
def expected_improvement(
        mu: np.ndarray,
        sigma: np.ndarray,
        y_best: float,
        xi: float = 0.01
) -> np.ndarray:
    """
    Expected improvement for maximizing selectivity.

    `xi` is the fixed exploration offset used by the current scoring workflow.
    """

    sigma_safe = np.maximum(sigma, 1e-12)
    improve = mu - y_best - xi
    z = improve / sigma_safe

    ei = improve * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei[sigma <= 1e-12] = np.maximum(improve[sigma <= 1e-12], 0.0)

    return np.maximum(ei, 0.0)

def normalize_01(
        x: np.ndarray
) -> np.ndarray:

    x = np.asarray(x, dtype=float)

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    if xmax - xmin <= 1e-12:
        return np.zeros_like(x)

    return (x - xmin) / (xmax - xmin)

def score_stacked(
        sel_mean: np.ndarray,
        sel_std: np.ndarray,
        dep_std: np.ndarray,
        y_best: float,
) -> np.ndarray:
    """
    Composite deployed-stack acquisition score.

    The fixed weights combine predicted selectivity, an upper-confidence term,
    expected improvement, and a small penalty on deposition uncertainty. These
    weights are treated as part of the current scientific workflow.
    """

    ei = expected_improvement(sel_mean, sel_std, y_best)
    ucb = sel_mean + 0.379175956 * sel_std

    return (
            0.55 * normalize_01(sel_mean - y_best)
            + 0.20 * normalize_01(ucb)
            + 0.20 * normalize_01(ei)
            - 0.05 * normalize_01(dep_std)
    )
