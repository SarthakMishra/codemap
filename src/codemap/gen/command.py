"""Command implementation for code documentation generation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codemap.processor import LODEntity, create_processor
from codemap.processor.pipeline import is_text_file
from codemap.utils.cli_utils import console, show_error
from codemap.utils.path_utils import filter_paths_by_gitignore

from .models import GenConfig
from .utils import generate_tree

if TYPE_CHECKING:
	from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)


def process_codebase(
	target_path: Path,
	config: GenConfig,
	progress: Progress,
	task_id: TaskID,
) -> tuple[list[LODEntity], dict]:
	"""
	Process a codebase using the LOD pipeline architecture.

	Args:
	    target_path: Path to the target codebase
	    config: Generation configuration
	    progress: Progress indicator
	    task_id: Task ID for progress reporting

	Returns:
	    Tuple of (list of LOD entities, metadata dict)

	Raises:
	    RuntimeError: If initialization fails

	"""
	logger.info("Starting codebase processing for: %s", target_path)
	progress.update(task_id, description="Initializing processor...")

	try:
		# Initialize processor pipeline with LOD support
		pipeline = create_processor(
			repo_path=target_path,
			default_lod_level=config.lod_level,
		)
	except RuntimeError as e:
		logger.exception("Error initializing processor")
		error_msg = f"Processor initialization failed: {e}"
		show_error(error_msg)
		raise RuntimeError(error_msg) from e

	# Scan target path for files to process
	progress.update(task_id, description="Scanning files...")
	# Need project root to correctly locate .gitignore
	project_root = Path.cwd()  # Assuming CWD is project root
	all_paths = list(target_path.rglob("*"))

	# Filter paths based on .gitignore patterns found in project_root
	filtered_paths = filter_paths_by_gitignore(all_paths, project_root)
	# Count only non-binary files that will actually be processed
	total_files = sum(1 for p in filtered_paths if p.is_file() and is_text_file(p))

	# Update progress information
	progress.update(task_id, total=total_files)
	progress.update(task_id, description=f"Processing {total_files} files...")

	# Process files asynchronously
	for path in filtered_paths:
		if path.is_file():
			# Submit async processing task
			pipeline.process_file(path, config.lod_level)

	# Wait for all submitted tasks to complete
	progress.update(task_id, description="Waiting for processing tasks to finish...")
	if not pipeline.wait_for_completion():
		logger.warning("Processing tasks did not complete within the expected time.")
		# Decide if we should continue or raise an error
		# For now, let's continue with potentially partial results

	# Retrieve results directly from the pipeline's cache after waiting
	# The cache should be populated by the worker threads
	entities = [pipeline.processed_files[p] for p in filtered_paths if p.is_file() and p in pipeline.processed_files]
	processed_files = len(entities)
	logger.info(f"Retrieved {processed_files} processed entities from pipeline cache.")

	# Update progress one last time after collecting all results
	progress.update(task_id, completed=processed_files, description=f"Collected {processed_files} results.")

	# Clean up
	pipeline.stop()

	# Generate repository metadata
	languages = {entity.language for entity in entities if entity.language}

	metadata: dict[str, Any] = {
		"name": target_path.name,
		"stats": {
			"total_files": total_files,
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

	def __init__(self, config: GenConfig) -> None:
		"""
		Initialize the gen command.

		Args:
		    config: Generation configuration

		"""
		self.config = config

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
				entities, metadata = process_codebase(target_path, self.config, progress, task_id)

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
