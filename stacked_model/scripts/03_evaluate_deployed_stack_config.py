from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXCEL_FILE,
    SHEET_NAME,
    VALIDATION_RESULTS_DIR,
)
from src.data import load_dataset
from src.features import (
    add_engineered_features,
    make_base_controllable_input_cols,
    make_model_input_cols,
    make_deposition_input_cols,
)
from src.validated_gp_config import (
    BEST_DEPLOYED_DEPOSITION_GP,
    BEST_DEPLOYED_SELECTIVITY_GP,
)
from src.validation import cross_validate_stacked_deployed_pipeline, print_summary


OUTDIR = VALIDATION_RESULTS_DIR / "deployed_stack_config"


def main() -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df = add_engineered_features(df)

    base_cols = make_base_controllable_input_cols(df)
    controllable_cols = make_model_input_cols(base_cols)
    dep_cols = make_deposition_input_cols(df)

    dep_gp_config = {
        key: value
        for key, value in BEST_DEPLOYED_DEPOSITION_GP.items()
        if key not in {"transform", "inverse_method"}
    }

    result = cross_validate_stacked_deployed_pipeline(
        df=df,
        controllable_cols=controllable_cols,
        dep_cols=dep_cols,
        kernel_type=BEST_DEPLOYED_SELECTIVITY_GP["kernel_type"],
        n_restarts_optimizer=BEST_DEPLOYED_SELECTIVITY_GP[
            "n_restarts_optimizer"
        ],
        alpha=BEST_DEPLOYED_SELECTIVITY_GP["alpha"],
        deposition_kernel_type=dep_gp_config["kernel_type"],
        deposition_n_restarts_optimizer=dep_gp_config[
            "n_restarts_optimizer"
        ],
        deposition_alpha=dep_gp_config["alpha"],
        deposition_transform=BEST_DEPLOYED_DEPOSITION_GP["transform"],
        deposition_inverse_method=BEST_DEPLOYED_DEPOSITION_GP[
            "inverse_method"
        ],
        selectivity_kernel_type=BEST_DEPLOYED_SELECTIVITY_GP["kernel_type"],
        selectivity_n_restarts_optimizer=BEST_DEPLOYED_SELECTIVITY_GP[
            "n_restarts_optimizer"
        ],
        selectivity_alpha=BEST_DEPLOYED_SELECTIVITY_GP["alpha"],
    )

    result["fold_results"].to_excel(
        OUTDIR / "deployed_stack_fold_results.xlsx",
        index=False,
    )
    result["predictions"].to_excel(
        OUTDIR / "deployed_stack_predictions.xlsx",
        index=False,
    )
    result["summary"].to_excel(
        OUTDIR / "deployed_stack_summary.xlsx",
        index=False,
    )

    print_summary(result)
    print("\nSaved:")
    print(OUTDIR / "deployed_stack_summary.xlsx")
    print(OUTDIR / "deployed_stack_fold_results.xlsx")
    print(OUTDIR / "deployed_stack_predictions.xlsx")

    return result


if __name__ == "__main__":
    main()
