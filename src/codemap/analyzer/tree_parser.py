"""Simple file parsing for the CodeMap tool."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from pygments.lexers import get_lexer_for_filename, guess_lexer
from pygments.util import ClassNotFound

from codemap.utils.file_filters import FileFilter

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CodeParser:
    """Parses source code files to extract basic file information."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the file parser.

        Args:
            config: Configuration dictionary with use_gitignore setting.

        """
        self.config = config or {}
        self.file_filter = FileFilter(config)

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

        if not self.file_filter.should_parse(file_path):
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
