"""
Command line entry points registered as console scripts in pyproject.toml.
"""

# Library import
import argparse
import logging

# Local imports
from .config import DEFAULT_DATA_FILE, DEFAULT_RESULTS_DIR
from .logging_utils import setup_logging
from .data.parse import parse
from .pipeline import run_bo

logger = logging.getLogger(__name__)


def parse_main(argv=None):
    p = argparse.ArgumentParser(
        prog="pulse-bo-parse",
        description="Clean a raw experiment workbook into a single sheet.")
    p.add_argument("source", help="raw .xlsx workbook (e.g. ED_QDPM_v7.xlsx)")
    p.add_argument("dest", help="output cleaned .xlsx path")
    args = p.parse_args(argv)
    setup_logging()  # console only
    parse(args.source, args.dest)


def run_main(argv=None):
    p = argparse.ArgumentParser(
        prog="pulse-bo-run",
        description="Run the full constrained-BO workflow and write result CSVs.")
    p.add_argument("--data", default=DEFAULT_DATA_FILE,
                   help=f"cleaned data workbook (default: {DEFAULT_DATA_FILE})")
    p.add_argument("--results", default=DEFAULT_RESULTS_DIR,
                   help=f"output directory (default: {DEFAULT_RESULTS_DIR})")
    p.add_argument("--no-candidates", action="store_true",
                   help="skip writing the large candidates_full.csv")
    p.add_argument("--no-log", action="store_true",
                   help="do not write run.log into the results directory")
    args = p.parse_args(argv)
    # run_bo configures logging (console + results/run.log) itself.
    run_bo(data_file=args.data, results_dir=args.results,
           save_candidates=not args.no_candidates,
           write_log=not args.no_log)


def parity_main(argv=None):
    # Imported here so that `pulse-bo-run` does not require matplotlib/seaborn.
    from .data.features import extract_features
    from .plotting import get_oof_predictions, make_parity_plot

    p = argparse.ArgumentParser(
        prog="pulse-bo-parity",
        description="Generate the selectivity GP out-of-fold parity plot.")
    p.add_argument("--data", default=DEFAULT_DATA_FILE,
                   help=f"cleaned data workbook (default: {DEFAULT_DATA_FILE})")
    p.add_argument("--out", default="results/selectivity_gp_parity.pdf",
                   help="output figure path")
    args = p.parse_args(argv)
    setup_logging()  # console only

    X_df, y_sel, _y_dep, feasible = extract_features(args.data)
    X_raw = X_df.to_numpy(dtype=float)
    X_feas, y_feas = X_raw[feasible], y_sel[feasible]
    logger.info("Feasible samples: %d", len(y_feas))
    logger.info("Generating out-of-fold predictions...")
    y_oof = get_oof_predictions(X_feas, y_feas)
    make_parity_plot(y_feas, y_oof, outfile=args.out)


if __name__ == "__main__":
    run_main()
