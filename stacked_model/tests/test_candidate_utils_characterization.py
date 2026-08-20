import unittest

import numpy as np
import pandas as pd

from src.BO.candidate_utils import (
    add_derived_features_to_matrix,
    build_bounds,
    enforce_physical_consistency_base,
    generate_random_candidates,
)


class CandidateUtilityCharacterizationTests(unittest.TestCase):
    def test_random_candidates_are_deterministic_for_fixed_rng(self):
        bounds = [(0.0, 1.0), (10.0, 12.0)]

        a = generate_random_candidates(
            bounds,
            n_samples=3,
            rng=np.random.default_rng(30),
        )
        b = generate_random_candidates(
            bounds,
            n_samples=3,
            rng=np.random.default_rng(30),
        )

        np.testing.assert_allclose(a, b)
        self.assertTrue(np.all(a[:, 0] >= 0.0))
        self.assertTrue(np.all(a[:, 0] <= 1.0))
        self.assertTrue(np.all(a[:, 1] >= 10.0))
        self.assertTrue(np.all(a[:, 1] <= 12.0))

    def test_physical_consistency_filter_keeps_current_valid_rows(self):
        base_cols = ["Applied V", "Von (s)", "Voff (s)", "Total Von (s)"]
        candidates = np.array(
            [
                [-2.0, 0.01, 0.0, 0.01],
                [-2.0, 0.009, 0.0, 1.0],
                [-2.0, 0.02, -0.1, 1.0],
                [-2.0, 0.02, 0.1, 0.01],
            ],
            dtype=float,
        )

        filtered = enforce_physical_consistency_base(candidates, base_cols)

        np.testing.assert_allclose(filtered, candidates[[0]])

    def test_derived_feature_matrix_preserves_requested_order(self):
        base_cols = ["Applied V", "Von (s)", "Voff (s)", "Total Von (s)"]
        full_cols = [
            "Applied V",
            "Voff (s)",
            "AbsV_x_Von",
            "PulseCount",
            "AbsV_x_TotalVon",
            "CyclePeriod_s",
        ]
        x_base = np.array([[-2.0, 0.5, 1.5, 10.0]])

        out = add_derived_features_to_matrix(x_base, base_cols, full_cols)

        np.testing.assert_allclose(
            out,
            [[-2.0, 1.5, 1.0, 20.0, 20.0, 2.0]],
        )

    def test_build_bounds_current_buffer_behavior(self):
        df = pd.DataFrame(
            {
                "Applied V": [-3.0, -1.0],
                "Von (s)": [0.5, 1.5],
                "PulseCount": [10.0, 20.0],
            }
        )

        bounds = build_bounds(
            df,
            ["Applied V", "Von (s)", "PulseCount"],
            bound_buffer=0.005,
        )

        self.assertEqual(bounds[0], (-3.01, -0.99))
        self.assertEqual(bounds[1], (0.495, 1.505))
        self.assertEqual(bounds[2], (9.95, 20.05))


if __name__ == "__main__":
    unittest.main()
