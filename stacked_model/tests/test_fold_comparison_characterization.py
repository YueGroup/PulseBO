import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLD_COMPARISON_PATH = PROJECT_ROOT / "scripts" / "04_compare_validation_folds.py"

spec = importlib.util.spec_from_file_location(
    "compare_validation_folds",
    FOLD_COMPARISON_PATH,
)
fold_comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fold_comparison)


class FoldComparisonCharacterizationTests(unittest.TestCase):
    def test_select_model_rows_uses_current_config_and_expected_fold_count(self):
        config = {
            "kernel_type": "rbf",
            "n_restarts_optimizer": 0,
            "alpha": 1e-4,
        }
        rows = [
            {
                "model_name": "synthetic",
                "kernel_type": "rbf",
                "n_restarts_optimizer": 0,
                "alpha": 1e-4,
                "evaluation": evaluation,
            }
            for evaluation in range(1, fold_comparison.EXPECTED_EVALUATIONS + 1)
        ]
        rows.append(
            {
                "model_name": "synthetic",
                "kernel_type": "matern_1.5",
                "n_restarts_optimizer": 0,
                "alpha": 1e-4,
                "evaluation": 1,
            }
        )

        selected = fold_comparison.select_model_rows(
            pd.DataFrame(rows),
            "synthetic",
            config,
        )

        self.assertEqual(len(selected), fold_comparison.EXPECTED_EVALUATIONS)
        np.testing.assert_array_equal(
            selected["evaluation"].to_numpy(),
            np.arange(1, fold_comparison.EXPECTED_EVALUATIONS + 1),
        )

    def test_select_deployed_stack_rows_requires_current_deployed_config(self):
        dep_config = fold_comparison.BEST_DEPLOYED_DEPOSITION_GP
        sel_config = fold_comparison.BEST_DEPLOYED_SELECTIVITY_GP
        rows = [
            {
                "model_name": fold_comparison.STACKED_NAME,
                "kernel_type": sel_config["kernel_type"],
                "n_restarts_optimizer": sel_config["n_restarts_optimizer"],
                "alpha": sel_config["alpha"],
                "deposition_kernel_type": dep_config["kernel_type"],
                "deposition_n_restarts_optimizer": dep_config[
                    "n_restarts_optimizer"
                ],
                "deposition_alpha": dep_config["alpha"],
                "deposition_transform": dep_config["transform"],
                "deposition_inverse_method": dep_config["inverse_method"],
                "selectivity_kernel_type": sel_config["kernel_type"],
                "selectivity_n_restarts_optimizer": sel_config[
                    "n_restarts_optimizer"
                ],
                "selectivity_alpha": sel_config["alpha"],
                "evaluation": evaluation,
            }
            for evaluation in range(1, fold_comparison.EXPECTED_EVALUATIONS + 1)
        ]

        selected = fold_comparison.select_deployed_stack_rows(pd.DataFrame(rows))

        self.assertEqual(len(selected), fold_comparison.EXPECTED_EVALUATIONS)

        bad_rows = pd.DataFrame(rows)
        bad_rows.loc[0, "deposition_transform"] = "log1p"

        with self.assertRaises(ValueError):
            fold_comparison.select_deployed_stack_rows(bad_rows)


if __name__ == "__main__":
    unittest.main()
