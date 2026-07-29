"""Reproduce the v6-dataset batch recommendation.

Runs the full constrained-BO workflow on the cleaned v6 dataset and writes all
result CSVs into this directory. Equivalent to `pulse-bo-run`, kept as a
standalone script so the experiment is self-contained.

Usage:
    cd experiments/v6_batch
    python script.py
"""

from pathlib import Path

from pulse_bo.pipeline import run_bo

# Repo root is three levels up from this file: experiments/v6_batch/script.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "data" / "cleaned_data.xlsx"
RESULTS_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_bo(
        data_file=str(DATA_FILE),
        results_dir=str(RESULTS_DIR),
        save_candidates=True,
    )
