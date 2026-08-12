import unittest

import numpy as np
import pandas as pd

from src.features import (
    add_engineered_features,
    make_base_controllable_input_cols,
    make_deposition_input_cols,
    make_model_input_cols,
)


class FeatureEngineeringCharacterizationTests(unittest.TestCase):
    def test_engineered_features_and_current_column_order(self):
        raw = pd.DataFrame(
            {
                " Applied V ": [-2.0, -3.0],
                "Von (ms)": [1000.0, 500.0],
                "Voff (ms)": [2000.0, 1500.0],
                "Total Von (s)": [10.0, 8.0],
            }
        )

        out = add_engineered_features(raw)

        np.testing.assert_allclose(out["Von (s)"], [1.0, 0.5])
        np.testing.assert_allclose(out["Voff (s)"], [2.0, 1.5])
        np.testing.assert_allclose(out["AbsVoltage"], [2.0, 3.0])
        np.testing.assert_allclose(out["CyclePeriod_s"], [3.0, 2.0])
        np.testing.assert_allclose(out["AbsV_x_Von"], [2.0, 1.5])
        np.testing.assert_allclose(out["AbsV_x_TotalVon"], [20.0, 24.0])
        np.testing.assert_allclose(out["PulseCount"], [10.0, 16.0])

        base_cols = make_base_controllable_input_cols(out)
        self.assertEqual(
            base_cols,
            ["Applied V", "Von (s)", "Voff (s)", "Total Von (s)"],
        )

        self.assertEqual(
            make_model_input_cols(base_cols),
            ["Applied V", "Voff (s)", "AbsV_x_Von", "PulseCount"],
        )
        self.assertEqual(
            make_deposition_input_cols(out),
            ["Applied V", "Voff (s)", "AbsV_x_Von", "PulseCount"],
        )


if __name__ == "__main__":
    unittest.main()
