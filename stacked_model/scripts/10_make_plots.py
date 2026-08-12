from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, EXCEL_FILE, SHEET_NAME, RAW_DEPOSITION_COL
from src.features import add_engineered_features
from src.plotting import (
    parity_plot,
    correlation_heat_map_plot,
    set_plot_style,
    plot_bo_experimental_improvement,
)

PLOTS_DIR = RESULTS_DIR / "plots"

if __name__ == "__main__":
    set_plot_style()

    print("Font family:", plt.rcParams["font.family"])
    print("Sans-serif fonts:", plt.rcParams["font.sans-serif"])

if __name__ == "__main__":
    validation_dir = RESULTS_DIR / "validation"

    parity_plot(
        pd.read_excel(validation_dir / "model_a_controls_only_predictions.xlsx"),
        observed_col = "Observed",
        predicted_col = "Predicted",
        output_path = PLOTS_DIR / "validation" / "parity_model_a.png",
        title = "Controls only selectivity parity",
    )

    parity_plot(
        pd.read_excel(validation_dir / "deployed_stack_config" / "deployed_stack_predictions.xlsx"),
        observed_col="Observed",
        predicted_col="Predicted",
        output_path=PLOTS_DIR / "validation" / "parity_stacked_pipeline.png",
        title="Stacked deployed pipeline selectivity parity",
    )


    raw_df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    raw_df = add_engineered_features(raw_df)

    feature_cols = [
        "Applied V",
        "Von (s)",
        "Voff (s)",
        "Total von (s)",
        "V_stripping",
        "AbsV_x_Von",
        "PulseCount",
        RAW_DEPOSITION_COL,
        "Co selectivity",
    ]

    correlation_heat_map_plot(
        raw_df,
        cols = [c for c in feature_cols if c in raw_df.columns],
        output_path=PLOTS_DIR / "features" / "feature_correlation_heatmap.png",
    )

    candidate_dir = RESULTS_DIR / "candidates"

    batch_df = pd.read_excel(candidate_dir / "bo_recommended_batch.xlsx")
    top_df = pd.read_excel(candidate_dir / "bo_top_candidates.xlsx")


    plot_bo_experimental_improvement(
        df=raw_df,
        output_path=PLOTS_DIR / "bo_improvement_plot.png",
        x_col="Experiment number",
        y_col="Co selectivity",
        stage_col="Iteration",
        stage_order=["Initial", "BO1", "BO2"],
        initial_stage="Initial",
        bo_stages=["BO1", "BO2"],
        title="",
        initial_focus_y_min=80.0,
        y_axis_min=82.0,
        y_axis_max=91.5,
        zoom_start=100.0,
        pre_scale=0.35,
        post_scale=2.2,
        gap=3.0,
    )
