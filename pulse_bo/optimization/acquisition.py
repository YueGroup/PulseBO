"""Acquisition functions for constrained Bayesian optimization.

The ranking score is Constrained Expected Improvement:

    CEI(x) = EI(x)  x  P(feasible | x)

where EI comes from the selectivity GP and the feasibility probability comes
from the deposition GP clearing the deposition threshold.
"""

import numpy as np
from scipy.stats import norm

from ..config import XI, KAPPA, DEP_BAD_THRESH


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         y_best: float, xi: float = XI) -> np.ndarray:
    """Expected Improvement over the current best, with exploration margin ``xi``."""
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    ei = np.zeros_like(mu)
    mask = sigma > 1e-12
    imp = mu[mask] - y_best - xi
    z = imp / sigma[mask]
    ei[mask] = imp * norm.cdf(z) + sigma[mask] * norm.pdf(z)
    return ei


def probability_of_feasibility(mu_dep: np.ndarray, sigma_dep: np.ndarray,
                               threshold: float = DEP_BAD_THRESH) -> np.ndarray:
    """P(deposition >= threshold) under the deposition GP posterior."""
    sigma_safe = np.where(sigma_dep < 1e-12, 1e-12, sigma_dep)
    return 1.0 - norm.cdf((threshold - mu_dep) / sigma_safe)


def constrained_ei(X_scaled_sel: np.ndarray, X_scaled_dep: np.ndarray,
                   gp_sel, gp_dep, y_best: float, xi: float = XI) -> np.ndarray:
    """CEI = EI(selectivity) x P(feasible from deposition)."""
    mu_sel, sig_sel = gp_sel.predict(X_scaled_sel, return_std=True)
    mu_dep, sig_dep = gp_dep.predict(X_scaled_dep, return_std=True)
    return (expected_improvement(mu_sel, sig_sel, y_best, xi) *
            probability_of_feasibility(mu_dep, sig_dep))


def ucb(mu: np.ndarray, sigma: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """Upper Confidence Bound (kept as a diagnostic acquisition)."""
    return mu + kappa * sigma
