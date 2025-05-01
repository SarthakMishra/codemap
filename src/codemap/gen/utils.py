"""Utility functions for the gen command."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence

logger = logging.getLogger(__name__)


def generate_tree(target_path: Path, filtered_paths: Sequence[Path]) -> str:
	"""
	Generate a directory tree representation.

	Args:
	    target_path: Root path
	    filtered_paths: List of filtered **absolute** paths within target_path

	Returns:
	    Tree representation as string

	"""
	# Build a nested dictionary representing the file structure
	tree = {}
	for abs_path in filtered_paths:
		# Ensure we only process paths within the target_path
		try:
			rel_path = abs_path.relative_to(target_path)
		except ValueError:
			continue  # Skip paths not under target_path

		parts = rel_path.parts
		current_level = tree
		for i, part in enumerate(parts):
			if i == len(parts) - 1:  # Last part (file or final directory)
				current_level[part] = "file" if abs_path.is_file() else "dir"
			else:
				if part not in current_level:
					current_level[part] = {}
				current_level = current_level[part]
				# Handle case where a file might exist with the same name as a directory part
				if not isinstance(current_level, dict):
					break  # Stop processing this path if structure is inconsistent

	# Recursive function to generate formatted tree lines
	tree_lines = []

	def format_level(level: dict, prefix: str = "") -> None:
		items = sorted(level.keys())
		for i, name in enumerate(items):
			connector = "└── " if i == len(items) - 1 else "├── "
			item_type = level[name]

			if isinstance(item_type, dict):  # It's a directory
				tree_lines.append(f"{prefix}{connector}{name}/")
				new_prefix = prefix + ("    " if i == len(items) - 1 else "│   ")
				format_level(item_type, new_prefix)
			else:  # It's a file
				tree_lines.append(f"{prefix}{connector}{name}")

	# Start formatting from the root
	format_level(tree)

	return "\n".join(tree_lines)


def determine_output_path(project_root: Path, output: Path | None, config_data: dict) -> Path:
	"""
	Determine the output path for documentation.

	Args:
	    project_root: Root directory of the project
	    output: Optional output path from command line
	    config_data: Gen-specific configuration data

	Returns:
	    The determined output path

	"""
	from datetime import UTC, datetime

	# If output is provided, use it directly
	if output:
		return output.resolve()

	# Check for output file in config
	if "output_file" in config_data:
		output_file = Path(config_data["output_file"])
		if output_file.is_absolute():
			return output_file
		return project_root / output_file

	# Get output directory from config
	output_dir = config_data.get("output_dir", "documentation")

	# If output_dir is absolute, use it directly
	output_dir_path = Path(output_dir)
	if not output_dir_path.is_absolute():
		# Otherwise, create the output directory in the project root
		output_dir_path = project_root / output_dir

	output_dir_path.mkdir(parents=True, exist_ok=True)

	# Generate a filename with timestamp
	timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
	filename = f"documentation_{timestamp}.md"

	return output_dir_path / filename


def write_documentation(output_path: Path, documentation: str) -> None:
	"""
	Write documentation to the specified output path.

	Args:
	    output_path: Path to write documentation to
	    documentation: Documentation content to write

	"""
	from codemap.utils.cli_utils import console, ensure_directory_exists, show_error

	try:
		# Ensure parent directory exists
		ensure_directory_exists(output_path.parent)
		output_path.write_text(documentation)
		console.print(f"[green]Documentation written to {output_path}")
	except (PermissionError, OSError) as e:
		show_error(f"Error writing documentation to {output_path}: {e!s}")
		raise
