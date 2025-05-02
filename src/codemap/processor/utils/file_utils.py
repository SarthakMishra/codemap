"""Utilities for file system operations."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


def read_file_content(file_path: Path) -> tuple[str, bool]:
	"""
	Read file content with basic error handling and encoding detection.

	Attempts to read with UTF-8, falling back to latin-1 if needed.

	Args:
	    file_path (Path): The path to the file.

	Returns:
	    tuple[str, bool]: A tuple containing:
	        - The file content as a string.
	        - A boolean indicating success (True) or failure (False).
	"""
	try:
		content = file_path.read_text(encoding="utf-8")
		return content, True
	except UnicodeDecodeError:
		logger.warning(f"UTF-8 decoding failed for {file_path}. Attempting latin-1.")
		try:
			content = file_path.read_text(encoding="latin-1")
			return content, True
		except Exception:
			logger.exception(f"Failed to read file {file_path} with latin-1 encoding")
			return "", False
	except FileNotFoundError:
		logger.exception(f"File not found: {file_path}")
		return "", False
	except OSError:
		logger.exception(f"Error reading file {file_path}")
		return "", False
	except Exception:
		logger.exception(f"An unexpected error occurred while reading {file_path}")
		return "", False
