"""Command implementation for code documentation generation."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, cast

from codemap.processor import initialize_processor
from codemap.processor.chunking.base import Chunk
from codemap.processor.embedding.models import EmbeddingConfig
from codemap.utils.cli_utils import console, show_error
from codemap.utils.path_utils import filter_paths_by_gitignore

from .models import GenConfig
from .utils import generate_tree

if TYPE_CHECKING:
	from pathlib import Path

	from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)


def process_codebase(
	target_path: Path,
	config: GenConfig,
	progress: Progress,
	task_id: TaskID,
) -> tuple[list[Chunk], dict]:
	"""
	Process a codebase using the pipeline architecture.

	Args:
	    target_path: Path to the target codebase
	    config: Generation configuration
	    progress: Progress indicator
	    task_id: Task ID for progress reporting

	Returns:
	    Tuple of (list of chunks, metadata dict)

	Raises:
	    RuntimeError: If initialization fails or daemon connectivity issues occur

	"""
	logger.info("Starting codebase processing for: %s", target_path)
	progress.update(task_id, description="Initializing processor...")

	# Configure pipeline settings based on generation config
	# Fixed constructor parameters
	embedding_config = EmbeddingConfig(dimensions=384)

	try:
		# Initialize processor pipeline
		pipeline = initialize_processor(
			repo_path=target_path,
			embedding_config=embedding_config,
			enable_lsp=config.semantic_analysis,
		)
	except ConnectionError as e:
		# Use show_error for user-friendly error messages
		error_msg = (
			"Failed to connect to the CodeMap daemon service. "
			"Please ensure the daemon is running with 'codemap daemon start' or "
			"use the --auto-start-daemon flag to start it automatically."
		)
		show_error(error_msg)
		logger.exception("Failed to connect to daemon service")
		raise RuntimeError(error_msg) from e
	except RuntimeError as e:
		logger.exception("Error initializing processor")
		if "daemon" in str(e).lower() or "connection" in str(e).lower():
			error_msg = (
				f"Daemon connection error: {e}. Please ensure the daemon is running with 'codemap daemon start'."
			)
			show_error(error_msg)
			raise RuntimeError(error_msg) from e
		error_msg = f"Processor initialization failed: {e}"
		show_error(error_msg)
		raise RuntimeError(error_msg) from e

	# Scan target path for files to process
	progress.update(task_id, description="Scanning files...")
	all_paths = list(target_path.rglob("*"))

	# Filter paths based on .gitignore patterns
	filtered_paths = filter_paths_by_gitignore(all_paths, target_path)
	total_files = sum(1 for p in filtered_paths if p.is_file())

	# Update progress information
	progress.update(task_id, total=total_files)
	progress.update(task_id, description=f"Processing {total_files} files...")

	# Process files in batches - convert to Sequence to address type issue
	pipeline.batch_process(cast("list[str | Path]", filtered_paths))

	# Collect processed chunks and metadata
	processed_chunks = []
	processed_count = 0

	for path in filtered_paths:
		if not path.is_file():
			continue

		job = pipeline.get_job_status(path)
		if job and job.completed_at and not job.error:
			processed_chunks.extend(job.chunks)
			processed_count += 1
			progress.update(task_id, completed=processed_count)

	# Generate repository metadata
	metadata = {
		"name": target_path.name,
		"stats": {
			"total_files": total_files,
			"total_lines": sum(
				chunk.metadata.location.end_line - chunk.metadata.location.start_line + 1 for chunk in processed_chunks
			),
			"languages": list({chunk.metadata.language for chunk in processed_chunks if chunk.metadata.language}),
		},
	}

	# Generate directory tree if requested
	if config.include_tree:
		metadata["tree"] = generate_tree(target_path, filtered_paths)

	return processed_chunks, metadata


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
				chunks, metadata = process_codebase(target_path, self.config, progress, task_id)

			# Generate documentation based on mode
			from .models import GenerationMode

			console.print("[green]Processing complete. Generating documentation...")

			if self.config.mode == GenerationMode.LLM:
				content = generator.generate_llm_context(chunks, metadata)
			else:
				content = generator.generate_human_docs(chunks, metadata)

			# Write documentation to output file
			write_documentation(output_path, content)

			# Show completion message with timing
			elapsed = time.time() - start_time
			console.print(f"[green]Generation completed in {elapsed:.2f} seconds.")

			return True

		except Exception as e:
			logger.exception("Error during gen command execution")
			from codemap.utils.cli_utils import show_error

			# Show a clean error message to the user
			error_msg = f"Generation failed: {e!s}"
			show_error(error_msg)
			return False
