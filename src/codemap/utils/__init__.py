"""Utility modules for the CodeMap package."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.spinner import Spinner

from codemap.git.utils.git_utils import GitError, get_repo_root


def validate_repo_path(path: Path | None = None) -> Path | None:
    """Validate and return the repository path.

    Args:
        path: Optional path to validate (defaults to current directory)

    Returns:
        Path to the repository root if valid, None otherwise
    """
    try:
        # If no path provided, use current directory
        if path is None:
            path = Path.cwd()

        # Get the repository root
        return get_repo_root(path)
    except GitError:
        return None


@contextlib.contextmanager
def loading_spinner(message: str = "Processing...") -> Iterator[None]:
    """Display a loading spinner while executing a task.

    Args:
        message: Message to display alongside the spinner

    Yields:
        None
    """
    console = Console()
    spinner = Spinner("dots", text=message)
    with console.status(spinner):
        yield


__all__ = ["loading_spinner", "validate_repo_path"]
