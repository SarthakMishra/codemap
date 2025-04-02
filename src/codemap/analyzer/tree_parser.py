"""Simple file parsing for the CodeMap tool."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any

from pygments.lexers import get_lexer_for_filename, guess_lexer
from pygments.util import ClassNotFound

logger = logging.getLogger(__name__)


class CodeParser:
    """Parses source code files to extract basic file information."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the file parser.

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
                patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
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

    def parse_file(self, file_path: Path) -> dict[str, Any]:
        """Parse a file and extract basic information.

        Args:
            file_path: Path to the file to parse.

        Returns:
            Dictionary containing the file information.

        Raises:
            OSError: If file cannot be read.
        """
        logger.debug("Parsing file: %s", file_path)

        # Initialize empty info dictionary
        file_info: dict[str, Any] = {
            "imports": [],
            "classes": [],
            "references": [],
            "content": "",
            "language": "unknown",
        }

        if not self.should_parse(file_path):
            logger.debug("Skipping file (excluded by filters): %s", file_path)
            return file_info

        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
                file_info["content"] = content

                # Try to detect language using Pygments
                try:
                    # First try by filename
                    lexer = get_lexer_for_filename(file_path.name)
                except ClassNotFound:
                    try:
                        # If that fails, try to guess from content
                        lexer = guess_lexer(content)
                    except ClassNotFound:
                        lexer = None

                if lexer:
                    file_info["language"] = lexer.name.lower()

                # Only extract imports and classes for Python files
                if file_path.suffix == ".py":
                    # Extract basic imports using regex
                    import_pattern = re.compile(r"^(?:from|import)\s+([^\s]+)", re.MULTILINE)
                    imports = import_pattern.findall(content)
                    file_info["imports"] = [imp.strip() for imp in imports]

                    # Extract class names using regex
                    class_pattern = re.compile(r"^\s*class\s+([^\s(:]+)", re.MULTILINE)
                    classes = class_pattern.findall(content)
                    file_info["classes"] = [cls.strip() for cls in classes]

                return file_info
        except (OSError, UnicodeDecodeError):
            logger.exception("Failed to parse file %s", file_path)
            return file_info
