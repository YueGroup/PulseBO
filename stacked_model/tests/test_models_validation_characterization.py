import math
import unittest

import numpy as np
import pandas as pd

from src.models import (
    deposition_prediction_to_raw,
    minmax_scale,
    parse_kernel_type,
    transform_deposition_target,
)
from src.validation import (
    coverage_90_from_predictions,
    make_repeated_cv,
    metrics_from_predictions,
    nlpd_from_predictions,
    summarize_single_repeated_cv,
)


class ModelAndValidationCharacterizationTests(unittest.TestCase):
    def test_kernel_labels_map_to_current_internal_kernel_settings(self):
        self.assertEqual(parse_kernel_type("rbf"), ("rbf", 2.5))
        self.assertEqual(parse_kernel_type("matern"), ("matern", 2.5))
        self.assertEqual(parse_kernel_type("matern_1.5"), ("matern", 1.5))
        self.assertEqual(parse_kernel_type("matern_2.5"), ("matern", 2.5))

    def test_scaling_and_deposition_target_transforms(self):
        x = np.array([[0.0, 10.0], [5.0, 20.0]])
        scaled = minmax_scale(x, [(0.0, 10.0), (10.0, 30.0)])
        np.testing.assert_allclose(scaled, [[0.0, 0.0], [0.5, 0.5]])

        raw = np.array([0.0, 9.0])
        np.testing.assert_allclose(transform_deposition_target(raw, "raw"), raw)
        np.testing.assert_allclose(
            transform_deposition_target(raw, "log1p"),
            np.log1p(raw),
        )

    def test_deposition_prediction_conversion_current_behavior(self):
        mu = np.array([1.0, -2.0])
        sigma = np.array([0.2, 0.5])

        raw_pred, raw_std = deposition_prediction_to_raw(
            mu,
            sigma,
            transform="raw",
            inverse_method="mean",
        )

        np.testing.assert_allclose(raw_pred, [1.0, 0.0])
        np.testing.assert_allclose(raw_std, [0.2, 0.5])

        log_pred, log_std = deposition_prediction_to_raw(
            mu,
            sigma,
            transform="log1p",
            inverse_method="median",
        )

        np.testing.assert_allclose(log_pred, np.expm1(mu))
        np.testing.assert_allclose(
            log_std,
            (np.expm1(mu + sigma) - np.expm1(mu - sigma)) / 2.0,
        )

    def test_metrics_and_summary_schema_current_behavior(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        y_std = np.array([0.5, 0.5, 0.5])

        mae, rmse, r2 = metrics_from_predictions(y_true, y_pred)

        self.assertAlmostEqual(mae, 1.0 / 3.0)
        self.assertAlmostEqual(rmse, math.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(r2, 0.5)
        self.assertAlmostEqual(
            coverage_90_from_predictions(y_true, y_pred, y_std),
            2.0 / 3.0,
        )
        self.assertTrue(np.isfinite(nlpd_from_predictions(y_true, y_pred, y_std)))

        fold_results = pd.DataFrame(
            {
                "evaluation": [1, 2],
                "test_n": [3, 3],
                "mae": [1.0, 2.0],
                "rmse": [2.0, 4.0],
                "r2": [0.1, 0.3],
                "coverage_90": [0.5, 1.0],
                "nlpd": [1.5, 2.5],
                "lml": [-10.0, -12.0],
            }
        )

        summary = summarize_single_repeated_cv(
            fold_results_df=fold_results,
            model_name="synthetic",
            n=6,
            kernel_type="rbf",
            n_restarts_optimizer=0,
            alpha=1e-4,
        )

        self.assertEqual(summary.loc[0, "n_evaluations"], 2)
        self.assertEqual(summary.loc[0, "n_total_test_points"], 6)
        self.assertAlmostEqual(summary.loc[0, "rmse_mean"], 3.0)

    def test_repeated_cv_current_fold_count_and_determinism(self):
        x = np.arange(20).reshape(-1, 1)

        first = list(make_repeated_cv().split(x))
        second = list(make_repeated_cv().split(x))

        self.assertEqual(len(first), 50)
        self.assertEqual(len(second), 50)
        for (train_a, test_a), (train_b, test_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)


if __name__ == "__main__":
    unittest.main()
