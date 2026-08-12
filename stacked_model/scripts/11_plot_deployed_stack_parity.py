from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, VALIDATION_RESULTS_DIR
from src.plotting import parity_plot


PREDICTIONS_FILE = (
    VALIDATION_RESULTS_DIR
    / "deployed_stack_config"
    / "deployed_stack_predictions.xlsx"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "plots"
    / "validation"
    / "parity_deployed_stack_config.png"
)


def main() -> None:
    predictions = pd.read_excel(PREDICTIONS_FILE)

    parity_plot(
        predictions,
        observed_col="Observed",
        predicted_col="Predicted",
        output_path=OUTPUT_FILE,
        title="Deployed stacked pipeline selectivity parity",
    )

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
