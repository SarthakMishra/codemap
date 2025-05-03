"""
Unified pipeline for CodeMap data processing, synchronization, and retrieval.

This module defines the `ProcessingPipeline`, which acts as the central orchestrator
for managing and interacting with the HNSW vector database. It handles initialization,
synchronization with the Git repository, and provides semantic search capabilities.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rich.progress import Progress, TaskID

from codemap.config import DEFAULT_CONFIG
from codemap.db.client import DatabaseClient
from codemap.processor.tree_sitter import TreeSitterAnalyzer
from codemap.processor.utils.embedding_utils import generate_embedding

# Added Vector specific imports
from codemap.processor.vector.chunking import TreeSitterChunker
from codemap.processor.vector.hnsw_manager import HNSWManager
from codemap.processor.vector.synchronizer import VectorSynchronizer
from codemap.utils.config_loader import ConfigError, ConfigLoader

# from codemap.processor.utils.sync_utils import compare_states # Moved to VectorSynchronizer
from codemap.utils.path_utils import find_project_root, get_cache_path

if TYPE_CHECKING:
	from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)

EXPECTED_METADATA_COLUMNS = 9  # Placeholder, not used in HNSW search


class ProcessingPipeline:
	"""
	Orchestrates data processing, synchronization, and retrieval for CodeMap using HNSW.

	Manages connections and interactions with HNSW vector database,
	ensuring it is synchronized with the Git repository state. Provides
	methods for semantic search.

	"""

	def __init__(
		self,
		repo_path: Path | None = None,
		config_loader: ConfigLoader | None = None,
		sync_on_init: bool = True,  # Add sync_on_init parameter
		progress: Progress | None = None,
		task_id: TaskID | None = None,
	) -> None:
		"""
		Initialize the unified processing pipeline.

		Args:
		    repo_path: Path to the repository root. If None, it will be determined.
		    config_loader: Application configuration loader. If None, a default one is created.
		    sync_on_init: If True, run database synchronization during initialization.
		    progress: Optional rich Progress instance for unified status display.
		    task_id: Optional rich TaskID for the main initialization/sync task.

		"""
		self.repo_path = repo_path or find_project_root()
		if not self.repo_path:
			msg = "Repository path could not be determined."
			raise ValueError(msg)

		self.config_loader = config_loader or ConfigLoader()
		self.config = self.config_loader.load_config()

		if not isinstance(self.config, dict):
			# This case should ideally not happen if load_config works correctly
			logger.error(f"Config loading failed or returned unexpected type: {type(self.config)}")
			msg = "Failed to load a valid Config object."
			raise ConfigError(msg)

		self.db_client = DatabaseClient(db_path=self.config.get("database_path"))

		logger.info("Initializing Processing Pipeline for repository: %s", self.repo_path)

		# --- Initialize Shared Components --- #
		self.analyzer = TreeSitterAnalyzer()

		# Get embedding config from loaded config with fallback to DEFAULT_CONFIG
		embedding_config = self.config.get("embedding", {})
		self.embedding_model_name = embedding_config.get("model_name", DEFAULT_CONFIG["embedding"]["model_name"])
		embedding_dimension = embedding_config.get("dimension", DEFAULT_CONFIG["embedding"]["dimension"])
		logger.info(f"Using embedding model: {self.embedding_model_name} with dimension: {embedding_dimension}")

		# --- Initialize Vector Components --- #
		vector_config = self.config.get("vector", {})
		# Default index path within cache
		default_index_dir = get_cache_path() / "vector_index"
		index_dir_path = Path(vector_config.get("index_path", default_index_dir))

		# Use higher default values to address the "ef too small" errors
		self.hnsw_manager = HNSWManager(
			index_dir_path=index_dir_path,
			space="cosine",
			dim=embedding_dimension,
			max_elements=vector_config.get("max_elements", 100000),  # Increased to 100K
			m=vector_config.get("m", 128),  # Increased to 128 for high-dim vectors
			ef_construction=vector_config.get("ef_construction", 2000),  # Increased to 2000
			ef_query=vector_config.get("ef_query", 4000),  # Increased to 4000
			allow_replace_deleted=vector_config.get("allow_replace_deleted", False),
		)
		# Initialize chunker with its default LODGenerator
		self.chunker = TreeSitterChunker()
		self.vector_synchronizer = VectorSynchronizer(
			self.repo_path, self.hnsw_manager, self.chunker, self.embedding_model_name
		)
		logger.info("Vector components initialized.")
		self.has_vector_db = True  # Vector capability exists

		# --- Check if initial sync is needed --- #
		needs_sync = False
		if sync_on_init:
			needs_sync = True
			logger.info("`sync_on_init` is True. Performing index synchronization...")
		else:
			try:
				# Check if index has any items
				if self.hnsw_manager.get_current_count() == 0:
					logger.info("No existing data in HNSW index. Initial sync is needed.")
					needs_sync = True
				else:
					logger.info(
						f"Found {self.hnsw_manager.get_current_count()} items in HNSW index. "
						"Skipping initial sync triggered by emptiness check."
					)
					needs_sync = False
			except Exception:
				logger.exception("Error checking HNSW state for initial sync. Assuming sync is needed.")
				needs_sync = True

		if needs_sync:
			self.sync_databases(progress=progress, task_id=task_id)
			if progress and task_id and not needs_sync:
				progress.update(
					task_id, description="[green]✓[/green] Pipeline initialized (sync skipped).", completed=100
				)
		elif progress and task_id:
			# Ensure progress completes even if sync fails or isn't needed but was checked
			if not needs_sync:
				progress.update(
					task_id, description="[green]✓[/green] Pipeline initialized (sync skipped).", completed=100
				)
			# If sync was needed but failed, sync_databases handles final progress state

		logger.info(f"ProcessingPipeline initialized for repo: {self.repo_path}")

	def stop(self) -> None:
		"""Stops the pipeline and releases resources."""
		logger.info("Stopping ProcessingPipeline...")
		# HNSWManager doesn't require explicit closing, but we save on stop
		if self.hnsw_manager:
			try:
				self.hnsw_manager.save()
				logger.info("HNSW index and metadata saved.")
			except Exception:
				logger.exception("Error saving HNSW index/metadata during stop.")
			self.hnsw_manager = None

		logger.info("ProcessingPipeline stopped.")

	# --- Synchronization --- #

	def sync_databases(self, progress: Progress | None = None, task_id: TaskID | None = None) -> None:
		"""
		Synchronize the HNSW index with the Git repository state.

		Args:
		    progress: Optional rich Progress instance for status updates.
		    task_id: Optional rich TaskID for the main sync task.

		"""
		logger.info("Starting vector index synchronization...")
		if progress and task_id is not None:
			# Reset task for sync operation
			progress.update(task_id, description="Starting index synchronization...", completed=0, total=100)

		try:
			# Pass progress context down to the vector synchronizer
			sync_success = self.vector_synchronizer.sync_index(
				progress=progress,
				task_id=task_id,
			)

			if sync_success:
				logger.info("Vector index synchronization completed successfully.")
				# Final update is handled by vector_synchronizer
			else:
				logger.error("Vector index synchronization failed or had issues.")
				# Final update is handled by vector_synchronizer
		except Exception:
			logger.exception("Error during vector index synchronization.")
			if progress and task_id:
				progress.update(task_id, description="[red]Error:[/red] Index sync failed.", completed=100)

	# Removed sync_databases_simple as it was graph-focused

	# --- Retrieval Methods --- #

	def semantic_search(self, query: str, k: int = 5) -> list[dict[str, Any]] | None:
		"""
		Perform semantic search for code chunks similar to the query using HNSW index.

		Args:
		    query: The search query string.
		    k: The number of top similar results to retrieve.

		Returns:
		    A list of search result dictionaries (including metadata and distance),
		    or None if an error occurs.

		"""
		if not self.hnsw_manager:
			logger.error("HNSWManager not available for semantic search.")
			return None

		logger.debug("Performing semantic search for query: '%s', k=%d", query, k)

		try:
			# 1. Generate query embedding
			query_embedding = generate_embedding(query)
			if query_embedding is None:
				logger.error("Failed to generate embedding for query.")
				return None

			# Convert to numpy array for HNSW query
			query_vector = np.array(query_embedding, dtype=np.float32)

			# 2. Query HNSW index
			chunk_ids, distances = self.hnsw_manager.knn_query(query_vector, k)

			if not chunk_ids:
				logger.debug("HNSW index query returned no results.")
				return []

			# 3. Fetch metadata for each result ID
			formatted_results = []
			for chunk_id, distance in zip(chunk_ids, distances, strict=False):
				metadata = self.hnsw_manager.get_metadata(chunk_id)
				if metadata:
					formatted_results.append(
						{
							"id": chunk_id,
							"distance": distance,
							"metadata": metadata,
						}
					)
				else:
					logger.warning(f"Metadata not found for vector hit with chunk_id: {chunk_id}")

			logger.debug("Semantic search found %d results.", len(formatted_results))
			return formatted_results

		except Exception:
			logger.exception("Error during semantic search.")
			return None

	# Removed graph_query
	# Removed graph_enhanced_search
	# Removed get_repository_structure
