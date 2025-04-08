"""Utility modules for the CodeMap package."""

from __future__ import annotations

import os
from pathlib import Path

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


__all__ = ["validate_repo_path"]
