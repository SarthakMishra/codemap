"""Utility functions for CLI operations in CodeMap."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Iterator, Self

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from pathlib import Path

console = Console()
logger = logging.getLogger(__name__)


# Singleton class to track spinner state
class SpinnerState:
    """Singleton class to track spinner state."""

    _instance = None
    is_active = False

    def __new__(cls) -> Self:
        """Create or return the singleton instance.

        Returns:
            The singleton instance of SpinnerState
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


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


@contextlib.contextmanager
def loading_spinner(message: str = "Processing...") -> Iterator[None]:
    """Display a loading spinner while executing a task.

    Args:
        message: Message to display alongside the spinner

    Yields:
        None
    """
    # In test environments, don't display a spinner
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
        yield
        return

    # Check if a spinner is already active
    spinner_state = SpinnerState()
    if spinner_state.is_active:
        # If there's already an active spinner, don't create a new one
        yield
        return

    # Only use spinner in interactive environments
    try:
        spinner_state.is_active = True
        # Use rich.console.Console.status which is designed for this purpose
        # and provides the spinner animation
        with console.status(message):
            yield
    finally:
        spinner_state.is_active = False


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
