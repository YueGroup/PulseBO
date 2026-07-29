"""Logging configuration for the pulse_bo workflow.

Console output uses a bare, unadorned format so interactive runs stay readable;
the optional file handler is timestamped so a completed run leaves a
self-describing log for provenance.
"""

import logging
import os
import sys

PACKAGE_LOGGER = "pulse_bo"


def setup_logging(logfile: str | None = None,
                  level: int = logging.INFO,
                  console: bool = True) -> logging.Logger:
    """Configure the package logger. Safe to call more than once.

    Parameters
    ----------
    logfile : str or None
        If given, also write a timestamped log to this path (mode ``w``).
    level : int
        Logging level (default ``logging.INFO``).
    console : bool
        If True, echo records to stdout.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)
    logger.propagate = False

    # Reset handlers so repeated calls don't duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    if logfile:
        directory = os.path.dirname(logfile)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    return logger
