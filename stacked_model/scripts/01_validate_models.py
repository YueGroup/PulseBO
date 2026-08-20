from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import run_validation_analysis


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--include-stack-grid",
        action="store_true",
        help=(
            "Also run the diagnostic stacked-pipeline kernel grid. "
            "The deployed stack is evaluated separately by "
            "scripts/03_evaluate_deployed_stack_config.py."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation_analysis(include_stack_grid=args.include_stack_grid)
