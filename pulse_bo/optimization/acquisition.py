"""
Acquisition functions for constrained Bayesian optimization.

Candidates are ranked by constrained expected improvement, CEI(x) = EI(x) * P(feasible | x),
where the feasibility term comes from the deposition GP clearing the deposition threshold.
"""

# Third party imports
import numpy as np
from scipy.stats import norm

# Local imports
from ..config import XI, KAPPA, DEP_BAD_THRESH


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         y_best: float, xi: float = XI) -> np.ndarray:
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    ei = np.zeros_like(mu)
    mask = sigma > 1e-12
    imp = mu[mask] - y_best - xi
    z = imp / sigma[mask]
    ei[mask] = imp * norm.cdf(z) + sigma[mask] * norm.pdf(z)
    return ei


def probability_of_feasibility(mu_dep: np.ndarray, sigma_dep: np.ndarray,
                               threshold: float = DEP_BAD_THRESH) -> np.ndarray:
    sigma_safe = np.where(sigma_dep < 1e-12, 1e-12, sigma_dep)
    return 1.0 - norm.cdf((threshold - mu_dep) / sigma_safe)


def constrained_ei(X_scaled_sel: np.ndarray, X_scaled_dep: np.ndarray,
                   gp_sel, gp_dep, y_best: float, xi: float = XI) -> np.ndarray:
    """Constrained expected improvement, EI from selectivity times P(feasible) from deposition."""
    mu_sel, sig_sel = gp_sel.predict(X_scaled_sel, return_std=True)
    mu_dep, sig_dep = gp_dep.predict(X_scaled_dep, return_std=True)
    return (expected_improvement(mu_sel, sig_sel, y_best, xi) *
            probability_of_feasibility(mu_dep, sig_dep))


def ucb(mu: np.ndarray, sigma: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    return mu + kappa * sigma
