"""Utility functions for the gen command."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codemap.config import ConfigLoader
from codemap.config.config_schema import GenSchema
from codemap.processor.lod import LODEntity, LODGenerator, LODLevel
from codemap.utils.cli_utils import progress_indicator
from codemap.utils.file_utils import is_binary_file
from codemap.utils.path_utils import filter_paths_by_gitignore

if TYPE_CHECKING:
	from collections.abc import Sequence


logger = logging.getLogger(__name__)


def _process_single_file_lod(file_path: Path, lod_level: LODLevel, lod_generator: LODGenerator) -> LODEntity | None:
	"""
	Worker function to process a single file for LOD generation.

	Args:
	        file_path: Path to the file to process.
	        lod_level: The level of detail required.
	        lod_generator: An instance of LODGenerator.

	Returns:
	        LODEntity if successful, None otherwise.

	"""
	if is_binary_file(file_path):
		logger.debug("Skipping non-text file for LOD generation: %s", file_path)
		return None
	try:
		logger.debug("Generating LOD for file: %s at level %s", file_path, lod_level.name)
		entity = lod_generator.generate_lod(file_path, lod_level)
		if not entity:
			logger.warning("LODGenerator returned None for %s (unsupported or empty?)", file_path)
		return entity
	except Exception:
		logger.exception("Error generating LOD for file %s", file_path)
		return None


def process_files_for_lod(
	paths: Sequence[Path],
	lod_level: LODLevel,
	max_workers: int = 4,
) -> list[LODEntity]:
	"""
	Process a list of file paths to generate LOD entities in parallel.

	Bypasses the main ProcessingPipeline and uses LODGenerator directly.

	Args:
	        paths: Sequence of file paths to process.
	        lod_level: The level of detail required.
	        max_workers: Maximum number of parallel worker threads.

	Returns:
	        A list of successfully generated LODEntity objects.

	"""
	lod_generator = LODGenerator()
	lod_entities = []
	futures: list[Future[LODEntity | None]] = []
	files_to_process = [p for p in paths if p.is_file()]
	task_description = ""

	task_description = f"Processing {len(files_to_process)} files for LOD..."
	current_progress = 0

	with (
		progress_indicator(task_description, style="progress", total=len(files_to_process)) as update_progress,
		ThreadPoolExecutor(max_workers=max_workers) as executor,
	):
		for file_path in files_to_process:
			future = executor.submit(_process_single_file_lod, file_path, lod_level, lod_generator)
			futures.append(future)

		for future in as_completed(futures):
			result = future.result()
			if result:
				lod_entities.append(result)
			current_progress += 1
			update_progress(task_description, current_progress, len(files_to_process))

	logger.info("Finished LOD processing. Generated %d entities.", len(lod_entities))
	return lod_entities


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
	tree: dict[str, Any] = {}

	# Process directories first to ensure complete structure
	# Sort paths to process directories first, then files, all in alphabetical order
	sorted_paths = sorted(filtered_paths, key=lambda p: (p.is_file(), str(p).lower()))

	# Ensure target_path itself is in the structure if it's not already
	dir_paths = [p for p in sorted_paths if p.is_dir()]

	# Add all directories first to ensure complete structure
	for abs_path in dir_paths:
		# Ensure we only process paths within the target_path
		try:
			rel_path = abs_path.relative_to(target_path)
			dir_parts = rel_path.parts

			current_level: dict[str, Any] = tree
			for _i, part in enumerate(dir_parts):
				if part not in current_level:
					current_level[part] = {}
				current_level = current_level[part]
		except ValueError:
			continue  # Skip paths not under target_path

	# Then add files to the structure
	file_paths = [p for p in sorted_paths if p.is_file()]
	for abs_path in file_paths:
		try:
			rel_path = abs_path.relative_to(target_path)
			parts = rel_path.parts

			current_level = tree
			for i, part in enumerate(parts):
				if i == len(parts) - 1:  # Last part (file)
					current_level[part] = "file"
				else:
					# Create directory levels if they don't exist
					if part not in current_level:
						current_level[part] = {}

					# Get reference to the next level
					next_level = current_level[part]

					# Handle case where a file might exist with the same name as a directory part
					if not isinstance(next_level, dict):
						# This shouldn't happen with proper directory structure, but handle just in case
						logger.warning(f"Name conflict: {part} is both a file and a directory in path {rel_path}")
						# Convert to dictionary with special file marker
						current_level[part] = {"__file__": True}

					current_level = current_level[part]
		except ValueError:
			continue  # Skip paths not under target_path

	# Get just the target directory name to display at the root of the tree
	# rather than the full absolute path
	target_dir_name = target_path.name

	# Initialize tree_lines with just the directory name at the root
	tree_lines = [target_dir_name]

	# Recursive function to generate formatted tree lines
	def format_level(level: dict[str, Any], prefix: str = "") -> None:
		"""Recursively formats a directory tree level into ASCII tree representation.

		Args:
		    level: Dictionary representing the current directory level, where keys are names
		        and values are either subdirectories (dicts) or files (strings).
		    prefix: String used for indentation and tree connectors in the output. Defaults to "".

		Returns:
		    None: Modifies the tree_lines list in the closure by appending formatted lines.

		Note:
		    - Directories are sorted before files
		    - Items are sorted alphabetically within their type group
		    - Special markers (like "__file__") are skipped
		"""
		# Sort items: directories first (dictionaries), then files (strings)
		sorted_items = sorted(level.items(), key=lambda x: (not isinstance(x[1], dict), x[0].lower()))

		for i, (name, item_type) in enumerate(sorted_items):
			is_last_item = i == len(sorted_items) - 1
			connector = "└── " if is_last_item else "├── "

			if name == "__file__":
				# Skip special markers
				continue

			if isinstance(item_type, dict):  # It's a directory
				tree_lines.append(f"{prefix}{connector}{name}/")
				new_prefix = prefix + ("    " if is_last_item else "│   ")
				format_level(item_type, new_prefix)
			else:  # It's a file
				tree_lines.append(f"{prefix}{connector}{name}")

	# Start formatting from the root
	format_level(tree)

	# If tree only contains the target path (no files/directories under it)
	if len(tree_lines) == 1:
		return tree_lines[0] + "/"

	return "\n".join(tree_lines)


def determine_output_path(project_root: Path, config_loader: ConfigLoader, output: Path | None) -> Path:
	"""
	Determine the output path for documentation.

	Args:
	    project_root: Root directory of the project
	    config_loader: ConfigLoader instance
	    output: Optional output path from command line

	Returns:
	    The determined output path

	"""
	from datetime import UTC, datetime

	# If output is provided, use it directly
	if output:
		return output.resolve()

	# Get output directory from config
	output_dir = config_loader.get.gen.output_dir

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


def process_codebase(
	target_path: Path,
	config: GenSchema,
	config_loader: ConfigLoader | None = None,
) -> tuple[list[LODEntity], dict]:
	"""
	Process a codebase using the LOD pipeline architecture.

	Args:
	    target_path: Path to the target codebase
	    config: Generation configuration
	    config_loader: Optional ConfigLoader instance to use

	Returns:
	    Tuple of (list of LOD entities, metadata dict)

	Raises:
	    RuntimeError: If processing fails

	"""
	logger.info("Starting codebase processing for: %s", target_path)

	# Get processor configuration from ConfigLoader
	if config_loader is None:
		config_loader = ConfigLoader().get_instance()
		logger.debug("Created new ConfigLoader instance in process_codebase")

	processor_config = config_loader.get.processor
	max_workers = processor_config.max_workers
	logger.debug(f"Using max_workers: {max_workers} from configuration")

	try:
		# Handle both directories and individual files
		if target_path.is_file():
			# Single file processing
			if not is_binary_file(target_path):
				processable_paths = [target_path]
			else:
				logger.debug(f"Skipping binary file: {target_path}")
				processable_paths = []
		elif target_path.is_dir():
			# Directory processing (existing logic)
			project_root = Path.cwd()  # Assuming CWD is project root
			all_paths = list(target_path.rglob("*"))

			# Filter paths based on .gitignore patterns found in project_root
			filtered_paths: Sequence[Path] = filter_paths_by_gitignore(all_paths, project_root)

			# Filter out binary files
			processable_paths = []
			for path in filtered_paths:
				if path.is_file():
					if not is_binary_file(path):
						processable_paths.append(path)
					else:
						logger.debug(f"Skipping binary file: {path}")
		else:
			logger.warning(f"Target path does not exist or is not a file/directory: {target_path}")
			processable_paths = []

		# Use the new utility function to process files and generate LOD entities
		# The utility function will handle parallel processing and progress updates
		entities = process_files_for_lod(
			paths=processable_paths,
			lod_level=config.lod_level,
			max_workers=max_workers,  # Get from configuration
		)
	except Exception as e:
		logger.exception("Error during LOD file processing")
		error_msg = f"LOD processing failed: {e}"
		raise RuntimeError(error_msg) from e

	# Update counts based on actual processed entities
	processed_files = len(entities)
	logger.info(f"LOD processing complete. Generated {processed_files} entities.")
	# total_files count is now handled within process_files_for_lod for progress

	# Generate repository metadata
	languages = {entity.language for entity in entities if entity.language}
	# Get total file count accurately from the filtered list *before* processing
	total_files_scanned = len(processable_paths)

	metadata: dict[str, Any] = {
		"name": target_path.name,
		"target_path": str(target_path.resolve()),  # Keep absolute target path for file path resolution
		"original_path": str(target_path),  # Store original path as provided in the command (could be relative)
		"stats": {
			"total_files_scanned": total_files_scanned,  # Total files scanned matching criteria
			"processed_files": processed_files,  # Files successfully processed for LOD
			"total_lines": sum(entity.end_line - entity.start_line + 1 for entity in entities),
			"languages": list(languages),
		},
	}

	# Generate directory tree if requested
	if config.include_tree:
		metadata["tree"] = generate_tree(target_path, filtered_paths)

	return entities, metadata


def write_documentation(output_path: Path, documentation: str) -> None:
	"""
	Write documentation to the specified output path.

	Args:
	    output_path: Path to write documentation to
	    documentation: Documentation content to write

	"""
	from codemap.utils.cli_utils import console
	from codemap.utils.file_utils import ensure_directory_exists

	try:
		# Ensure parent directory exists
		ensure_directory_exists(output_path.parent)
		output_path.write_text(documentation)
		console.print(f"[green]Documentation written to {output_path}")
	except (PermissionError, OSError):
		logger.exception(f"Error writing documentation to {output_path}")
		raise
