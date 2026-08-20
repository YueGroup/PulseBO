from pathlib import Path
import json
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_DIR / "results" / "validation"

SUMMARY_FILE = VALIDATION_DIR / "validation_summary.xlsx"
OUT_JSON = VALIDATION_DIR / "best_gp_configs.json"
OUT_PY = PROJECT_DIR / "src" / "validated_gp_config.py"


MODEL_NAME_MAP = {
    "Model A: controls only": "BEST_MODEL_A_GP",
    "Model B: controls + measured deposition": "BEST_MODEL_B_GP",
    "Deposition model: controls -> deposition": "BEST_DEPOSITION_GP",
}

DEPLOYED_DEPOSITION_TRANSFORM = "raw"


def main() -> None:
    summary = pd.read_excel(SUMMARY_FILE)

    required_cols = [
        "model_name",
        "kernel_type",
        "n_restarts_optimizer",
        "alpha",
        "rmse_mean",
    ]

    missing = [col for col in required_cols if col not in summary.columns]

    if missing:
        raise ValueError(f"Missing required columns in validation summary: {missing}")

    best_configs = {}

    for model_name, config_name in MODEL_NAME_MAP.items():
        model_df = summary[summary["model_name"] == model_name].copy()

        if model_df.empty:
            raise ValueError(f"No validation rows found for model: {model_name}")

        if config_name == "BEST_DEPOSITION_GP":
            if "deposition_transform" not in model_df.columns:
                raise ValueError(
                    "Deposition validation summary is missing "
                    "'deposition_transform'. Rerun scripts/01_validate_models.py "
                    "so BEST_DEPOSITION_GP is selected from raw-deposition "
                    "validation rows."
                )

            model_df = model_df[
                model_df["deposition_transform"] == DEPLOYED_DEPOSITION_TRANSFORM
            ].copy()

            if model_df.empty:
                raise ValueError(
                    "No raw-deposition validation rows found for the deposition "
                    "model. Rerun scripts/01_validate_models.py after the raw "
                    "deposition validation changes."
                )

        best_row = (
            model_df
            .sort_values("rmse_mean", ascending=True)
            .iloc[0]
        )

        best_configs[config_name] = {
            "kernel_type": str(best_row["kernel_type"]),
            "n_restarts_optimizer": int(best_row["n_restarts_optimizer"]),
            "alpha": float(best_row["alpha"]),
            "rmse_mean": float(best_row["rmse_mean"]),
        }

    with open(OUT_JSON, "w") as f:
        json.dump(best_configs, f, indent=4)

    py_lines = [
        '"""',
        "Best GP configurations selected from repeated-CV validation.",
        "",
        "Generated automatically by scripts/02_select_best_gp.py",
        '"""',
        "",
    ]

    for config_name, values in best_configs.items():
        py_lines.append(f"{config_name} = {{")
        py_lines.append(f'    "kernel_type": "{values["kernel_type"]}",')
        py_lines.append(f'    "n_restarts_optimizer": {values["n_restarts_optimizer"]},')
        py_lines.append(f'    "alpha": {values["alpha"]},')
        py_lines.append("}")
        py_lines.append("")

    py_lines.extend(
        [
            "# Deployed stack configuration.",
            "# The deposition GP is selected from raw-deposition validation rows.",
            "# controls -> deposition GP -> predicted deposition -> selectivity GP.",
            "BEST_DEPLOYED_DEPOSITION_GP = {",
            "    **BEST_DEPOSITION_GP,",
            '    "transform": "raw",',
            '    "inverse_method": "mean",',
            "}",
            "",
            "BEST_DEPLOYED_SELECTIVITY_GP = BEST_MODEL_B_GP",
            "",
        ]
    )

    OUT_PY.write_text("\n".join(py_lines))

    print("\nSelected best GP configurations by lowest RMSE:")
    for config_name, values in best_configs.items():
        print(
            f"{config_name}: "
            f"kernel={values['kernel_type']}, "
            f"restarts={values['n_restarts_optimizer']}, "
            f"alpha={values['alpha']}, "
            f"rmse_mean={values['rmse_mean']:.4f}"
        )

    print(f"\nSaved JSON: {OUT_JSON}")
    print(f"Saved Python config: {OUT_PY}")


if __name__ == "__main__":
    main()
