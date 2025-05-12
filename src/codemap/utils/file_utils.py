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


def read_file_content(file_path: Path | str) -> str | None:
	"""
	Read content from a file with proper error handling.

	Args:
	    file_path: Path to the file to read

	Returns:
	    Content of the file as string, or None if the file cannot be read

	"""
	path_obj = Path(file_path)
	try:
		with path_obj.open("r", encoding="utf-8") as f:
			return f.read()
	except FileNotFoundError:
		# Handle case where file was tracked but has been deleted
		logger.debug(f"File not found: {path_obj} - possibly deleted since last tracked")
		return None
	except UnicodeDecodeError:
		# Try to read as binary and then decode with error handling
		logger.warning("File %s contains non-UTF-8 characters, attempting to decode with errors='replace'", path_obj)
		try:
			with path_obj.open("rb") as f:
				content = f.read()
				return content.decode("utf-8", errors="replace")
		except (OSError, FileNotFoundError):
			logger.debug(f"Unable to read file as binary: {path_obj}")
			return None
	except OSError as e:
		# Handle other file access errors
		logger.debug(f"Error reading file {path_obj}: {e}")
		return None


def ensure_directory_exists(dir_path: Path) -> None:
	"""
	Ensure that a directory exists, creating it if necessary.

	Args:
	    dir_path (Path): The path to the directory.

	"""
	if not dir_path.exists():
		logger.info(f"Creating directory: {dir_path}")
		try:
			dir_path.mkdir(parents=True, exist_ok=True)
		except OSError:
			logger.exception(f"Failed to create directory {dir_path}")
			raise
	elif not dir_path.is_dir():
		logger.error(f"Path exists but is not a directory: {dir_path}")
		msg = f"Path exists but is not a directory: {dir_path}"
		raise NotADirectoryError(msg)


# Utility functions for file type checking
def is_binary_file(file_path: Path) -> bool:
	"""
	Check if a file is binary.

	Args:
	        file_path: Path to the file

	Returns:
	        True if the file is binary, False otherwise

	"""
	# Skip files larger than 10 MB
	try:
		if file_path.stat().st_size > 10 * 1024 * 1024:
			return True

		# Try to read as text
		with file_path.open(encoding="utf-8") as f:
			chunk = f.read(1024)
			return "\0" in chunk
	except UnicodeDecodeError:
		return True
	except (OSError, PermissionError):
		return True
