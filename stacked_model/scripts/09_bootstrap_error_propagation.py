from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import run_bootstrap_error_propagation_best_stacked_model


if __name__ == "__main__":
    run_bootstrap_error_propagation_best_stacked_model()
