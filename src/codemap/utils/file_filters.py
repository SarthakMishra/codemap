"""File filtering utilities for CodeMap."""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound

logger = logging.getLogger(__name__)


class FileFilter:
    """Handles file filtering logic for CodeMap."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the file filter.

        Args:
            config: Configuration dictionary with use_gitignore setting.
        """
        self.config = config or {}
        self.gitignore_patterns: list[str] = []

        # Only load gitignore patterns if explicitly enabled in config
        if self.config.get("use_gitignore", False):
            self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Load patterns from .gitignore file if it exists."""
        gitignore_path = Path(".gitignore")
        self.gitignore_patterns = []  # Always start with empty list

        if gitignore_path.exists():
            with gitignore_path.open() as f:
                # Only include non-empty lines that don't start with # (comments)
                patterns = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith("#")]
                self.gitignore_patterns.extend(patterns)
                logger.debug("Loaded %d patterns from .gitignore", len(patterns))

    def _matches_pattern(self, file_path: Path, pattern: str) -> bool:
        """Check if a file path matches a glob pattern.

        Args:
            file_path: Path to check
            pattern: Glob pattern to match against

        Returns:
            True if the path matches the pattern
        """
        # Convert pattern and path to forward slashes for consistency
        pattern = pattern.replace(os.sep, "/")
        path_str = str(file_path).replace(os.sep, "/")

        # Handle directory patterns (ending with /)
        if pattern.endswith("/"):
            pattern = pattern.rstrip("/")
            path_parts = Path(path_str).parts
            return any(part == pattern for part in path_parts)

        # For .dot patterns (like .venv), match against both name and any part of the path
        if pattern.startswith("."):
            path_parts = Path(path_str).parts
            if any(part == pattern for part in path_parts):
                return True
            if any(part == pattern[1:] for part in path_parts):
                return True
            return (
                fnmatch.fnmatch(file_path.name, pattern)
                or fnmatch.fnmatch(path_str, pattern)
                or fnmatch.fnmatch(path_str, f"*/{pattern}")
            )

        # For patterns with directory separators (like **/temp/*), match the full path
        if "/" in pattern:
            return fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, f"**/{pattern}")

        # For simple patterns (like *.py), match against any part of the path
        path_parts = Path(path_str).parts
        return any(fnmatch.fnmatch(part, pattern) for part in path_parts)

    def matches_pattern(self, file_path: Path, pattern: str) -> bool:
        """Public method to check if a file path matches a glob pattern.

        Args:
            file_path: Path to check
            pattern: Glob pattern to match against

        Returns:
            True if the path matches the pattern
        """
        return self._matches_pattern(file_path, pattern)

    def should_parse(self, file_path: Path) -> bool:
        """Check if a file should be parsed.

        Args:
            file_path: Path to check.

        Returns:
            True if the file should be parsed, False otherwise.
        """
        # Default excluded directories and files
        default_excluded = ["__pycache__", ".git", ".env", ".venv", "venv", "build", "dist"]
        for excluded in default_excluded:
            if excluded in str(file_path):
                return False

        # Check gitignore patterns if enabled
        if self.config.get("use_gitignore", True):
            for pattern in self.gitignore_patterns:
                if self._matches_pattern(file_path, pattern):
                    return False

        # Try to get a lexer for the file - if successful, we can parse it
        try:
            get_lexer_for_filename(file_path.name)
        except ClassNotFound:
            return False
        else:
            return True
