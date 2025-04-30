"""Utility functions for file operations in CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def count_tokens(file_path: Path) -> int:
	"""
	Rough estimation of tokens in a file.

	Args:
	    file_path: Path to the file to count tokens in.

	Returns:
	    Estimated number of tokens in the file.

	"""
	try:
		with file_path.open(encoding="utf-8") as f:
			content = f.read()
			# Simple tokenization by whitespace
			return len(content.split())
	except (OSError, UnicodeDecodeError):
		return 0


def read_file_content(file_path: Path | str) -> str:
	"""
	Read content from a file with proper error handling.

	Args:
	    file_path: Path to the file to read

	Returns:
	    Content of the file as string

	Raises:
	    OSError: If the file cannot be read
	    UnicodeDecodeError: If the file content cannot be decoded as UTF-8

	"""
	path_obj = Path(file_path)
	try:
		with path_obj.open("r", encoding="utf-8") as f:
			return f.read()
	except UnicodeDecodeError:
		# Try to read as binary and then decode with error handling
		logger.warning("File %s contains non-UTF-8 characters, attempting to decode with errors='replace'", path_obj)
		with path_obj.open("rb") as f:
			content = f.read()
			return content.decode("utf-8", errors="replace")
