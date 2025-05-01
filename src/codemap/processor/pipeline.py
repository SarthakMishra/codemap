"""
Pipeline orchestration for code processing.

This module defines the main processing pipeline that orchestrates:
1. Generating code metadata with tree-sitter at different LOD levels
2. Providing results to client modules

"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from codemap.processor.lod import LODEntity, LODGenerator, LODLevel

logger = logging.getLogger(__name__)


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


def is_text_file(file_path: Path) -> bool:
	"""
	Check if a file is a text file.

	Args:
	        file_path: Path to the file

	Returns:
	        True if the file is a text file, False otherwise

	"""
	return not is_binary_file(file_path)


class ProcessingPipeline:
	"""
	Main pipeline for code processing and analysis.

	This class orchestrates the generation of LOD data for code files and
	directories using tree-sitter, with parallel processing capabilities.

	"""

	def __init__(
		self,
		repo_path: Path,
		ignored_patterns: set[str] | None = None,
		max_workers: int = 4,
		default_lod_level: LODLevel = LODLevel.SIGNATURES,
	) -> None:
		"""
		Initialize the processing pipeline.

		Args:
		        repo_path: Path to the repository root
		        ignored_patterns: Set of glob patterns to ignore when processing
		        max_workers: Maximum number of worker threads for processing
		        default_lod_level: Default Level of Detail to use for processing

		"""
		self.repo_path = repo_path
		self.ignored_patterns = ignored_patterns or {
			"**/.git/**",
			"**/__pycache__/**",
			"**/.venv/**",
			"**/node_modules/**",
			"**/.DS_Store",
			"**/*.pyc",
			"**/*.pyo",
			"**/*.pyd",
			"**/~*",
		}

		# Setup processing components
		self.lod_generator = LODGenerator()
		self.default_lod_level = default_lod_level

		# Threading setup
		self.max_workers = max_workers
		self.executor = ThreadPoolExecutor(max_workers=max_workers)
		self._futures: list[Future] = []

		# In-memory cache for processed files
		self.processed_files: dict[Path, LODEntity] = {}

	def stop(self) -> None:
		"""Stop the processing pipeline and clean up resources."""
		logger.info("Stopping processing pipeline")

		# Shutdown thread pool
		self.executor.shutdown(wait=True)
		self._futures.clear()

		logger.info("Processing pipeline stopped")

	def process_file(self, file_path: str | Path, level: LODLevel | None = None) -> None:
		"""
		Process a single file asynchronously.

		Args:
		        file_path: Path to the file to process
		        level: Level of Detail to use, defaults to pipeline's default level

		"""
		path_obj = Path(file_path)

		# Skip binary files
		if is_binary_file(path_obj):
			logger.debug("Skipping binary file: %s", path_obj)
			return

		# Submit to thread pool
		future = self.executor.submit(self._process_file_worker, path_obj, level or self.default_lod_level)
		self._futures.append(future)

	def process_file_sync(self, file_path: str | Path, level: LODLevel | None = None) -> LODEntity | None:
		"""
		Process a single file synchronously and return the result.

		Args:
		        file_path: Path to the file to process
		        level: Level of Detail to use, defaults to pipeline's default level

		Returns:
		        LOD entity or None if processing failed or file is binary

		"""
		path_obj = Path(file_path)

		# Skip binary files
		if is_binary_file(path_obj):
			logger.debug("Skipping binary file: %s", path_obj)
			return None

		# Process directly
		return self._process_file_worker(path_obj, level or self.default_lod_level)

	def _process_file_worker(self, file_path: Path, level: LODLevel) -> LODEntity | None:
		"""
		Worker function to process a file.

		Args:
		        file_path: Path to the file to process
		        level: Level of Detail to use

		Returns:
		        LOD entity or None if processing failed

		"""
		try:
			logger.debug("Processing file: %s at LOD level %s", file_path, level.name)

			# Generate LOD entity
			lod_entity = self.lod_generator.generate_lod(file_path, level)

			# Store in cache
			if lod_entity:
				self.processed_files[file_path] = lod_entity
				logger.debug("Successfully generated LOD entity for %s", file_path)
			else:
				logger.warning("LODGenerator returned None for %s (file might be unsupported or empty)", file_path)

			return lod_entity

		except Exception:
			logger.exception("Error processing file %s", file_path)
			return None
		finally:
			# Log final status (redundant with above but good for overview)
			status = (
				"succeeded (added to cache)"
				if file_path in self.processed_files
				else "failed or skipped (not in cache)"
			)
			logger.debug("Final processing status for %s: %s", file_path, status)

	def batch_process(self, paths: list[str | Path], level: LODLevel | None = None) -> None:
		"""
		Process multiple files in batch asynchronously.

		Args:
		        paths: List of file paths to process
		        level: Level of Detail to use, defaults to pipeline's default level

		"""
		for path in paths:
			self.process_file(path, level)

	def wait_for_completion(self, timeout: float | None = None) -> bool:
		"""
		Wait for all pending file processing tasks to complete.

		Args:
		        timeout: Maximum time to wait in seconds, or None to wait indefinitely

		Returns:
		        True if all tasks completed, False if timeout occurred

		"""
		from concurrent.futures import ALL_COMPLETED, wait

		if not self._futures:
			return True

		_, not_done = wait(self._futures, timeout=timeout, return_when=ALL_COMPLETED)

		# Clean up completed futures
		self._futures = list(not_done)

		return len(not_done) == 0

	def get_lod_entity(
		self, file_path: Path, level: LODLevel | None = None, force_refresh: bool = False
	) -> LODEntity | None:
		"""
		Get the LOD entity for a file.

		If the file has already been processed at the requested level or higher,
		returns the cached entity. Otherwise, processes the file at the requested level.

		Args:
		        file_path: Path to the file
		        level: Level of Detail requested, defaults to pipeline's default level
		        force_refresh: Whether to force reprocessing even if cached

		Returns:
		        LOD entity for the file, or None if processing failed

		"""
		requested_level = level or self.default_lod_level

		# Check if we have a cached entity at the same or higher level
		if not force_refresh and file_path in self.processed_files:
			existing_entity = self.processed_files[file_path]

			# If we have content in the entity and level is FULL, we know it was processed at FULL level
			if requested_level == LODLevel.FULL and existing_entity.content:
				return existing_entity

			# If we have signature and level is SIGNATURES, we know it was processed at SIGNATURES level
			if requested_level == LODLevel.SIGNATURES and existing_entity.signature:
				return existing_entity

			# If we have docstring and level is DOCS, we know it was processed at DOCS level
			if requested_level == LODLevel.DOCS and existing_entity.docstring:
				return existing_entity

		# Process synchronously in this case
		return self.process_file_sync(file_path, requested_level)

	def process_repository(self, repo_path: Path | None = None) -> int:
		"""
		Process an entire repository asynchronously.

		Args:
		        repo_path: Repository path, uses the pipeline's repo path if None

		Returns:
		        Number of files submitted for processing

		"""
		target_repo = repo_path or self.repo_path

		# Get all files in the repository
		all_files = []
		for path_obj in target_repo.rglob("*"):
			if path_obj.is_file() and is_text_file(path_obj):
				# Check against ignored patterns
				should_ignore = False
				for pattern in self.ignored_patterns:
					if path_obj.match(pattern):
						should_ignore = True
						break

				if not should_ignore:
					all_files.append(path_obj)

		# Process all files
		self.batch_process(all_files)

		return len(all_files)

	def get_repository_structure(
		self, root_path: Path | None = None, level: LODLevel = LODLevel.STRUCTURE
	) -> dict[str, Any]:
		"""
		Get a structured representation of the repository.

		Args:
		        root_path: Root path to start from, uses pipeline's repo path if None
		        level: Level of Detail to use for files

		Returns:
		        Hierarchical structure of the repository with LOD entities

		"""
		target_path = root_path or self.repo_path

		# Start with the basic directory structure
		result = {"type": "directory", "name": target_path.name, "path": str(target_path), "children": []}

		# List all items in the directory
		try:
			items = list(target_path.iterdir())

			# Process directories first
			for item in sorted([i for i in items if i.is_dir()], key=lambda x: x.name):
				# Skip ignored directories
				should_ignore = False
				for pattern in self.ignored_patterns:
					if item.match(pattern):
						should_ignore = True
						break

				if should_ignore:
					continue

				# Recursively process this directory
				child_structure = self.get_repository_structure(item, level)
				result["children"].append(child_structure)

			# Then process files
			for item in sorted([i for i in items if i.is_file()], key=lambda x: x.name):
				# Skip ignored files
				should_ignore = False
				for pattern in self.ignored_patterns:
					if item.match(pattern):
						should_ignore = True
						break

				if should_ignore:
					continue

				# Get LOD entity for this file
				file_entity = self.get_lod_entity(item, level)

				file_result = {"type": "file", "name": item.name, "path": str(item), "entity": file_entity}

				result["children"].append(file_result)

		except (PermissionError, FileNotFoundError) as e:
			logger.warning(f"Error processing directory {target_path}: {e}")

		return result
