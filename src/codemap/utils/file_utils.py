"""Utility functions for file operations in CodeMap."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
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


def get_output_path(repo_root: Path, output_path: Path | None, config: dict) -> Path:
	"""
	Get the output path for documentation.

	Args:
	    repo_root: Root directory of the project
	    output_path: Optional output path from command line
	    config: Configuration dictionary

	Returns:
	    Output path

	"""
	if output_path:
		return output_path

	# Get output directory from config
	output_dir = config.get("output_dir", "documentation")

	# If output_dir is absolute, use it directly
	output_dir_path = Path(output_dir)
	if not output_dir_path.is_absolute():
		# Otherwise, create the output directory in the project root
		output_dir_path = repo_root / output_dir

	output_dir_path.mkdir(parents=True, exist_ok=True)

	# Generate a filename with timestamp
	timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
	filename = f"documentation_{timestamp}.md"

	return output_dir_path / filename
