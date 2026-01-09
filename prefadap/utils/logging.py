"""Logging utilities for scripts.

This module provides simple helpers used by CLI entry points.  It
exposes two lightweight ``logging.Formatter`` implementations—one for
human readable console output and another that emits structured JSON
records suitable for log files.  In addition, a small helper disables
``tqdm`` progress bars when the current process is not attached to a
TTY, ensuring that log files are not polluted with control characters.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover -
        data = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


class ConsoleFormatter(logging.Formatter):
    """Simple formatter for console output."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__("%(asctime)s [%(levelname)s] %(message)s")


def setup_logging(*, log_file: Optional[Path] = None, quiet: bool = False) -> None:
    """Configure global logging handlers.

    This function sets up global logging configuration with optional file and
    console output. It does not return a logger instance.

    Parameters
    ----------
    log_file:
        When provided, a :class:`logging.FileHandler` writing JSON records is
        installed.
    quiet:
        If ``True`` console logging is suppressed.
    """
    handlers: list[logging.Handler] = []
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        handlers.append(fh)

    if not quiet:
        sh = logging.StreamHandler()
        sh.setFormatter(ConsoleFormatter())
        handlers.append(sh)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def create_logger(log_dir: str | Path, filename: str = "training.log") -> logging.Logger:
    """Create a logger that writes to both console and a file in the specified directory.

    This function creates and returns a configured logger instance that writes
    to both the console and a file within the given directory.

    Parameters
    ----------
    log_dir:
        Directory where the log file will be created. The directory is created
        if it doesn't exist.
    filename:
        Name of the log file. Defaults to "training.log".

    Returns
    -------
    logging.Logger:
        A configured logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs in interactive sessions
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(log_dir / filename),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


def disable_tqdm_if_not_tty() -> None:
    """Disable :mod:`tqdm` progress bars when not running in a TTY."""

    if not sys.stderr.isatty() or not sys.stdout.isatty():
        os.environ.setdefault("TQDM_DISABLE", "1")


__all__ = ["JsonFormatter", "ConsoleFormatter", "setup_logging", "create_logger", "disable_tqdm_if_not_tty"]

