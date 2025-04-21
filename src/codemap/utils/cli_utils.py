"""Utility functions for CLI operations in CodeMap."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from pathlib import Path

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(*, is_verbose: bool) -> None:
    """Configure logging based on verbosity.

    Args:
        is_verbose: Whether to enable debug logging.
    """
    # Override LOG_LEVEL environment variable if verbose flag is set
    log_level = "DEBUG" if is_verbose else os.environ.get("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=True)],
    )


def create_spinner_progress() -> Progress:
    """Create a spinner progress bar.

    Returns:
        A Progress instance with a spinner
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    )


def ensure_directory_exists(directory_path: Path) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to ensure exists
    """
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        console.print(f"[red]Unable to create directory {directory_path}: {e!s}")
        raise
