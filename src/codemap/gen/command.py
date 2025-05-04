"""Command implementation for code documentation generation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codemap.processor.lod import LODEntity
from codemap.utils.cli_utils import console, show_error
from codemap.utils.config_loader import ConfigLoader
from codemap.utils.path_utils import filter_paths_by_gitignore

from .models import GenConfig
from .utils import generate_tree, process_files_for_lod

if TYPE_CHECKING:
	from collections.abc import Sequence

	from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)


def process_codebase(
	target_path: Path,
	config: GenConfig,
	progress: Progress,
	task_id: TaskID,
	config_loader: ConfigLoader | None = None,
) -> tuple[list[LODEntity], dict]:
	"""
	Process a codebase using the LOD pipeline architecture.

	Args:
	    target_path: Path to the target codebase
	    config: Generation configuration
	    progress: Progress indicator
	    task_id: Task ID for progress reporting
	    config_loader: Optional ConfigLoader instance to use

	Returns:
	    Tuple of (list of LOD entities, metadata dict)

	Raises:
	    RuntimeError: If processing fails

	"""
	logger.info("Starting codebase processing for: %s", target_path)
	progress.update(task_id, description="Scanning files...")

	# Get processor configuration from ConfigLoader
	if config_loader is None:
		config_loader = ConfigLoader()
		logger.debug("Created new ConfigLoader instance in process_codebase")

	processor_config = config_loader.get("processor", {})
	max_workers = processor_config.get("max_workers", 4)
	logger.debug(f"Using max_workers: {max_workers} from configuration")

	try:
		# Need project root to correctly locate .gitignore
		project_root = Path.cwd()  # Assuming CWD is project root
		all_paths = list(target_path.rglob("*"))

		# Filter paths based on .gitignore patterns found in project_root
		filtered_paths: Sequence[Path] = filter_paths_by_gitignore(all_paths, project_root)

		# Use the new utility function to process files and generate LOD entities
		# The utility function will handle parallel processing and progress updates
		entities = process_files_for_lod(
			paths=filtered_paths,
			lod_level=config.lod_level,
			max_workers=max_workers,  # Get from configuration
			progress=progress,
			task_id=task_id,
		)
	except Exception as e:
		logger.exception("Error during LOD file processing")
		error_msg = f"LOD processing failed: {e}"
		show_error(error_msg)
		raise RuntimeError(error_msg) from e

	# Update counts based on actual processed entities
	processed_files = len(entities)
	logger.info(f"LOD processing complete. Generated {processed_files} entities.")
	# total_files count is now handled within process_files_for_lod for progress

	# Generate repository metadata
	languages = {entity.language for entity in entities if entity.language}
	# Get total file count accurately from the filtered list *before* processing
	total_files_scanned = sum(1 for p in filtered_paths if p.is_file())

	metadata: dict[str, Any] = {
		"name": target_path.name,
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


class GenCommand:
	"""Main implementation of the gen command."""

	def __init__(self, config: GenConfig, config_loader: ConfigLoader | None = None) -> None:
		"""
		Initialize the gen command.

		Args:
		    config: Generation configuration
		    config_loader: Optional ConfigLoader instance to use

		"""
		self.config = config
		self.config_loader = config_loader or ConfigLoader()
		logger.debug("GenCommand initialized with ConfigLoader")

	def execute(self, target_path: Path, output_path: Path) -> bool:
		"""
		Execute the gen command.

		Args:
		    target_path: Path to the target codebase
		    output_path: Path to write the output

		Returns:
		    True if successful, False otherwise

		"""
		from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

		from .generator import CodeMapGenerator
		from .utils import write_documentation

		start_time = time.time()

		try:
			# Create generator
			generator = CodeMapGenerator(self.config, output_path)

			# Process codebase with progress tracking
			with Progress(
				TextColumn("[progress.description]{task.description}"),
				BarColumn(),
				TextColumn("{task.completed}/{task.total}"),
				TimeElapsedColumn(),
			) as progress:
				task_id = progress.add_task("Processing codebase...", total=None)
				entities, metadata = process_codebase(
					target_path, self.config, progress, task_id, config_loader=self.config_loader
				)

			# Generate documentation
			console.print("[green]Processing complete. Generating documentation...")
			content = generator.generate_documentation(entities, metadata)

			# Write documentation to output file
			write_documentation(output_path, content)

			# Show completion message with timing
			elapsed = time.time() - start_time
			console.print(f"[green]Generation completed in {elapsed:.2f} seconds.")

			return True

		except Exception as e:
			logger.exception("Error during gen command execution")
			# Show a clean error message to the user
			error_msg = f"Generation failed: {e!s}"
			show_error(error_msg)
			return False
