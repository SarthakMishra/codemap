"""Utility modules for the CodeMap package."""

from __future__ import annotations

from pathlib import Path

from codemap.utils.cli_utils import (
    console,
    create_spinner_progress,
    ensure_directory_exists,
    loading_spinner,
    setup_logging,
)
from codemap.utils.file_utils import count_tokens, get_output_path
from codemap.utils.git_utils import GitError, get_repo_root


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


# Export commonly used utilities
__all__ = [
    "GitError",
    "console",
    "count_tokens",
    "create_spinner_progress",
    "ensure_directory_exists",
    "get_output_path",
    "get_repo_root",
    "loading_spinner",
    "setup_logging",
    "validate_repo_path",
]
