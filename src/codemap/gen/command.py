"""Command implementation for code documentation generation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codemap.config import ConfigLoader
from codemap.processor.lod import LODEntity
from codemap.utils.cli_utils import console, progress_indicator
from codemap.utils.file_utils import is_binary_file
from codemap.utils.path_utils import filter_paths_by_gitignore

from .models import GenConfig
from .utils import generate_tree, process_files_for_lod

if TYPE_CHECKING:
	from collections.abc import Sequence

logger = logging.getLogger(__name__)


def show_error(message: str) -> None:
	"""
	Display an error message using the console.

	Args:
		message: The error message to display
	"""
	console.print(f"[red]Error:[/red] {message}")


def process_codebase(
	target_path: Path,
	config: GenConfig,
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
		# Need project root to correctly locate .gitignore
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
		from .generator import CodeMapGenerator
		from .utils import write_documentation

		start_time = time.time()

		try:
			# Create generator
			generator = CodeMapGenerator(self.config, output_path)

			# Process codebase with progress tracking
			with progress_indicator("Processing codebase..."):
				# Process the codebase
				entities, metadata = process_codebase(target_path, self.config, config_loader=self.config_loader)

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
			show_error(f"Generation failed: {e}")
			return False
