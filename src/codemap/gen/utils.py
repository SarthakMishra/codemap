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
	    filtered_paths: List of filtered paths

	Returns:
	    Tree representation as string

	"""
	tree_lines = []

	# Get relative paths sorted by directory/file
	rel_paths = [p.relative_to(target_path) for p in filtered_paths]
	sorted_paths = sorted(rel_paths, key=lambda p: (p.parent, not p.parent.is_dir(), p.name))

	# Track current directory level
	current_dirs = []

	for path in sorted_paths:
		# Get path components
		parts = path.parts

		# Determine depth of common parent
		common_depth = 0
		for depth, (current, part) in enumerate(zip(current_dirs, parts[:-1], strict=False)):
			if current != part:
				break
			common_depth = depth + 1

		# Remove directories not in current path
		current_dirs = current_dirs[:common_depth]

		# Add new directories
		for new_dir in parts[common_depth:-1]:
			indent = "  " * len(current_dirs)
			tree_lines.append(f"{indent}└── {new_dir}/")
			current_dirs.append(new_dir)

		# Add file
		if path.is_file():
			indent = "  " * len(current_dirs)
			tree_lines.append(f"{indent}└── {parts[-1]}")

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
