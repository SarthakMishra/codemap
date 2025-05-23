"""
Unified pipeline for CodeMap data processing, synchronization, and retrieval.

This module defines the `ProcessingPipeline`, which acts as the central orchestrator
for managing and interacting with the HNSW vector database. It handles initialization,
synchronization with the Git repository, and provides semantic search capabilities.

"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Self

from qdrant_client import models as qdrant_models

from codemap.processor.hash_calculation import RepoChecksumCalculator

# Use async embedding utils
from codemap.processor.utils.embedding_utils import (
	generate_embedding,
)
from codemap.processor.vector.qdrant_manager import QdrantManager
from codemap.utils.cli_utils import progress_indicator
from codemap.utils.docker_utils import ensure_qdrant_running

# Import Qdrant specific classes
from codemap.utils.git_utils import GitRepoContext
from codemap.watcher.file_watcher import Watcher

if TYPE_CHECKING:
	from types import TracebackType

	from codemap.config import ConfigLoader
	from codemap.db.client import DatabaseClient
	from codemap.processor.tree_sitter import TreeSitterAnalyzer
	from codemap.processor.vector.chunking import TreeSitterChunker
	from codemap.processor.vector.synchronizer import VectorSynchronizer

logger = logging.getLogger(__name__)


class ProcessingPipeline:
	"""
	Orchestrates data processing, synchronization, and retrieval for CodeMap using Qdrant.

	Manages connections and interactions with the Qdrant vector database,
	ensuring it is synchronized with the Git repository state. Provides
	methods for semantic search. Uses asyncio for database and embedding
	operations.

	"""

	_instance: ProcessingPipeline | None = None
	_lock = asyncio.Lock()

	@classmethod
	async def get_instance(
		cls,
		config_loader: ConfigLoader | None = None,
	) -> ProcessingPipeline:
		"""
		Get or create a singleton instance of ProcessingPipeline.

		Args:
		    config_loader: Application configuration loader. If None, a default one is created.

		Returns:
		    The singleton ProcessingPipeline instance.
		"""
		async with cls._lock:
			if cls._instance is None:
				cls._instance = cls(config_loader=config_loader)
				await cls._instance.async_init()
			return cls._instance

	@classmethod
	def reset_instance(cls) -> None:
		"""Reset the singleton instance. Useful for testing."""
		cls._instance = None

	def __init__(
		self,
		config_loader: ConfigLoader | None = None,
	) -> None:
		"""
		Initialize the processing pipeline synchronously.

		Core async initialization is done via `async_init`.

		Args:
		    config_loader: Application configuration loader. If None, a default one is created.
		"""
		# Import ConfigLoader at the beginning to ensure it's always available
		from codemap.config import ConfigLoader

		if config_loader:
			self.config_loader = config_loader
		else:
			self.config_loader = ConfigLoader.get_instance()

		self.git_context = GitRepoContext.get_instance()

		self.repo_path = self.config_loader.get.repo_root

		if not self.repo_path:
			self.repo_path = self.git_context.repo_root

		if not self.repo_path:
			self.repo_path = self.git_context.get_repo_root()

		if not self.repo_path:
			msg = "Repository path could not be determined. Please ensure it's a git repository or set in config."
			logger.critical(msg)

		if self.repo_path:
			from pathlib import Path

			self.repo_path = Path(self.repo_path)
		else:
			logger.error("Critical: repo_path is None, RepoChecksumCalculator cannot be initialized.")

		if not isinstance(self.config_loader, ConfigLoader):
			from codemap.config import ConfigError

			logger.error(f"Config loading failed or returned unexpected type: {type(self.config_loader)}")
			msg = "Failed to load a valid Config object."
			raise ConfigError(msg)

		self.repo_checksum_calculator: RepoChecksumCalculator | None = None
		if self.repo_path and self.repo_path.is_dir():
			self.repo_checksum_calculator = RepoChecksumCalculator.get_instance(
				repo_path=self.repo_path, git_context=self.git_context, config_loader=self.config_loader
			)
			logger.info(f"RepoChecksumCalculator initialized for {self.repo_path}")
		else:
			logger.warning(
				"RepoChecksumCalculator could not be initialized because repo_path is invalid or not set. "
				"Checksum-based quick sync will be skipped."
			)

		# --- Defer Shared Components Initialization --- #
		self._analyzer: TreeSitterAnalyzer | None = None
		self._chunker: TreeSitterChunker | None = None
		self._db_client: DatabaseClient | None = None

		# --- Load Configuration --- #
		embedding_config = self.config_loader.get.embedding
		embedding_model = embedding_config.model_name
		qdrant_dimension = embedding_config.dimension
		distance_metric = embedding_config.dimension_metric

		self.embedding_model_name: str = "minishlab/potion-base-8M"
		if embedding_model and isinstance(embedding_model, str):
			self.embedding_model_name = embedding_model

		if not qdrant_dimension:
			logger.warning("Missing qdrant dimension in configuration, using default 256")
			qdrant_dimension = 256

		logger.info(f"Using embedding model: {self.embedding_model_name} with dimension: {qdrant_dimension}")

		vector_config = self.config_loader.get.embedding

		if self.repo_path:
			qdrant_location = self.repo_path / ".codemap_cache" / "qdrant"
			qdrant_location.mkdir(parents=True, exist_ok=True)

		qdrant_url = vector_config.url
		qdrant_api_key = vector_config.api_key

		distance_enum = qdrant_models.Distance.COSINE
		if distance_metric and distance_metric.upper() in ["COSINE", "EUCLID", "DOT", "MANHATTAN"]:
			distance_enum = getattr(qdrant_models.Distance, distance_metric.upper())

		str(self.repo_path) if self.repo_path else "no_repo_path"
		branch_str = self.git_context.branch or "no_branch"

		stable_repo_id = str(self.repo_path.resolve()) if self.repo_path else "unknown_repo"
		collection_base_name = hashlib.sha256(stable_repo_id.encode()).hexdigest()[:16]
		collection_name = f"codemap_{collection_base_name}_{branch_str}"

		import re

		safe_branch_str = re.sub(r"[^a-zA-Z0-9_-]", "_", branch_str)
		collection_name = f"codemap_{collection_base_name}_{safe_branch_str}"

		logger.info(f"Configuring Qdrant client for URL: {qdrant_url}, Collection: {collection_name}")

		self.qdrant_manager = QdrantManager(
			config_loader=self.config_loader,
			collection_name=collection_name,
			dim=qdrant_dimension,
			distance=distance_enum,
			url=qdrant_url,
			api_key=qdrant_api_key,
		)
		self._vector_synchronizer: VectorSynchronizer | None = None

		logger.info(f"ProcessingPipeline synchronous initialization complete for repo: {self.repo_path}")
		self.is_async_initialized = False
		self.watcher: Watcher | None = None
		self._watcher_task: asyncio.Task | None = None
		self._sync_lock = asyncio.Lock()

	@property
	def analyzer(self) -> TreeSitterAnalyzer:
		"""
		Lazily initialize and return a shared TreeSitterAnalyzer instance.

		Returns:
			TreeSitterAnalyzer: The shared analyzer instance.
		"""
		if self._analyzer is None:
			from codemap.processor.tree_sitter import TreeSitterAnalyzer

			self._analyzer = TreeSitterAnalyzer()
		return self._analyzer

	@property
	def chunker(self) -> TreeSitterChunker:
		"""
		Lazily initialize and return a TreeSitterChunker.

		Returns:
			TreeSitterChunker: The chunker instance.
		"""
		if self._chunker is None:
			from codemap.processor.lod import LODGenerator
			from codemap.processor.vector.chunking import TreeSitterChunker

			lod_generator = LODGenerator(analyzer=self.analyzer)
			self._chunker = TreeSitterChunker(
				lod_generator=lod_generator,
				config_loader=self.config_loader,
				git_context=self.git_context,
				repo_checksum_calculator=self.repo_checksum_calculator,
			)
		return self._chunker

	@property
	def db_client(self) -> DatabaseClient:
		"""
		Lazily initialize and return a DatabaseClient instance.

		Returns:
			DatabaseClient: The database client instance.

		Raises:
			RuntimeError: If the DatabaseClient cannot be initialized.
		"""
		if self._db_client is None:  # Only attempt initialization if not already done
			try:
				from codemap.db.client import DatabaseClient

				self._db_client = DatabaseClient()  # Add necessary args if any
			except ImportError:
				logger.exception(
					"DatabaseClient could not be imported. DB features will be unavailable. "
					"Ensure database dependencies are installed if needed."
				)
				# We will raise a RuntimeError below if _db_client is still None.
				# Allow to proceed to the check below
			except Exception:
				# Catch other potential errors during DatabaseClient instantiation
				logger.exception("Error initializing DatabaseClient")
				# We will raise a RuntimeError below if _db_client is still None.

		# After attempting initialization, check if it was successful.
		if self._db_client is None:
			msg = (
				"Failed to initialize DatabaseClient. It remains None after attempting import and instantiation. "
				"Check logs for import errors or instantiation issues."
			)
			logger.critical(msg)  # Use critical for such a failure
			raise RuntimeError(msg)

		return self._db_client

	@property
	def vector_synchronizer(self) -> VectorSynchronizer:
		"""
		Lazily initialize and return a VectorSynchronizer.

		Returns:
			VectorSynchronizer: The synchronizer instance.
		"""
		if self._vector_synchronizer is None:
			from codemap.processor.vector.synchronizer import VectorSynchronizer

			if self.repo_path is None:
				msg = "repo_path must not be None for VectorSynchronizer"
				logger.error(msg)
				raise RuntimeError(msg)
			if self.qdrant_manager is None:
				msg = "qdrant_manager must not be None for VectorSynchronizer"
				logger.error(msg)
				raise RuntimeError(msg)

			self._vector_synchronizer = VectorSynchronizer(
				repo_path=self.repo_path,
				qdrant_manager=self.qdrant_manager,
				chunker=self.chunker,
				embedding_model_name=self.embedding_model_name,
				analyzer=self.analyzer,
				config_loader=self.config_loader,
				repo_checksum_calculator=self.repo_checksum_calculator,
			)
		return self._vector_synchronizer

	async def async_init(self, sync_on_init: bool = True) -> None:
		"""
		Perform asynchronous initialization steps, including Qdrant connection and initial sync.

		Args:
		    sync_on_init: If True, run database synchronization during initialization.
		    update_progress: Optional ProgressUpdater instance for progress updates.

		"""
		if self.is_async_initialized:
			logger.info("Pipeline already async initialized.")
			return

		with progress_indicator("Initializing pipeline components..."):
			try:
				# Get embedding configuration for Qdrant URL
				embedding_config = self.config_loader.get.embedding
				qdrant_url = embedding_config.url

				# Check for Docker containers
				if qdrant_url:
					with progress_indicator("Checking Docker containers..."):
						# Only check Docker if we're using a URL that looks like localhost/127.0.0.1
						if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
							logger.info("Ensuring Qdrant container is running")
							success, message = await ensure_qdrant_running(wait_for_health=True, qdrant_url=qdrant_url)

							if not success:
								logger.warning(f"Docker check failed: {message}")

							else:
								logger.info(f"Docker container check: {message}")

				# Initialize Qdrant client (connects, creates collection if needed)
				if self.qdrant_manager:
					with progress_indicator("Initializing Qdrant manager..."):
						await self.qdrant_manager.initialize()
						logger.info("Qdrant manager initialized asynchronously.")
				else:
					# This case should theoretically not happen if __init__ succeeded
					msg = "QdrantManager was not initialized in __init__."
					logger.error(msg)
					raise RuntimeError(msg)

				needs_sync = False
				if sync_on_init:
					needs_sync = True
					logger.info("`sync_on_init` is True. Performing index synchronization...")
				else:
					# Optional: Could add a check here if Qdrant collection is empty
					# requires another call to qdrant_manager, e.g., get_count()
					logger.info("Skipping sync on init as requested.")
					needs_sync = False

				# Set initialized flag *before* potentially long sync operation
				self.is_async_initialized = True
				logger.info("ProcessingPipeline async core components initialized.")

				if needs_sync:
					await self.sync_databases()

			except Exception:
				logger.exception("Failed during async initialization")
				# Optionally re-raise or handle specific exceptions
				raise

	async def stop(self) -> None:
		"""Stops the pipeline and releases resources, including closing Qdrant connection."""
		logger.info("Stopping ProcessingPipeline asynchronously...")
		if self.qdrant_manager:
			await self.qdrant_manager.close()
			self.qdrant_manager = None  # type: ignore[assignment]
		else:
			logger.warning("Qdrant Manager already None during stop.")

		# Stop the watcher if it's running
		if self._watcher_task and not self._watcher_task.done():
			logger.info("Stopping file watcher...")
			self._watcher_task.cancel()
			try:
				await self._watcher_task  # Allow cancellation to propagate
			except asyncio.CancelledError:
				logger.info("File watcher task cancelled.")
			if self.watcher:
				self.watcher.stop()
				logger.info("File watcher stopped.")
			self.watcher = None
			self._watcher_task = None

		# Other cleanup if needed
		self.is_async_initialized = False
		logger.info("ProcessingPipeline stopped.")

	# --- Synchronization --- #

	async def _sync_callback_wrapper(self) -> None:
		"""Async wrapper for the sync callback to handle locking."""
		if self._sync_lock.locked():
			logger.info("Sync already in progress, skipping watcher-triggered sync.")
			return

		async with self._sync_lock:
			logger.info("Watcher triggered sync starting...")
			# Run sync without progress bars from watcher
			await self.sync_databases()
			logger.info("Watcher triggered sync finished.")

	async def sync_databases(self) -> None:
		"""
		Asynchronously synchronize the Qdrant index with the Git repository state.

		Args:
		    update_progress: Optional ProgressUpdater instance for progress updates.

		"""
		if not self.is_async_initialized:
			logger.error("Cannot sync databases, async initialization not complete.")
			return

		# Acquire lock only if not already held (for watcher calls)
		if not self._sync_lock.locked():
			async with self._sync_lock:
				logger.info("Starting vector index synchronization using VectorSynchronizer...")
				# VectorSynchronizer handles its own progress updates internally now
				await self.vector_synchronizer.sync_index()
				# Final status message/logging is handled by sync_index
		else:
			# If lock is already held (likely by watcher call), just run it
			logger.info("Starting vector index synchronization (lock already held)...")
			await self.vector_synchronizer.sync_index()

	# --- Watcher Methods --- #

	def initialize_watcher(self, debounce_delay: float = 2.0) -> None:
		"""
		Initialize the file watcher.

		Args:
		    debounce_delay: Delay in seconds before triggering sync after a file change.

		"""
		if not self.repo_path:
			logger.error("Cannot initialize watcher without a repository path.")
			return

		if self.watcher:
			logger.warning("Watcher already initialized.")
			return

		logger.info(f"Initializing file watcher for path: {self.repo_path}")
		try:
			self.watcher = Watcher(
				path_to_watch=self.repo_path,
				on_change_callback=self._sync_callback_wrapper,  # Use the lock wrapper
				debounce_delay=debounce_delay,
			)
			logger.info("File watcher initialized.")
		except ValueError:
			logger.exception("Failed to initialize watcher")
			self.watcher = None
		except Exception:
			logger.exception("Unexpected error initializing watcher.")
			self.watcher = None

	async def start_watcher(self) -> None:
		"""
		Start the file watcher in the background.

		`initialize_watcher` must be called first.

		"""
		if not self.watcher:
			logger.error("Watcher not initialized. Call initialize_watcher() first.")
			return

		if self._watcher_task and not self._watcher_task.done():
			logger.warning("Watcher task is already running.")
			return

		logger.info("Starting file watcher task in the background...")
		# Create a task to run the watcher's start method asynchronously
		self._watcher_task = asyncio.create_task(self.watcher.start())
		# We don't await the task here; it runs independently.
		# Error handling within the watcher's start method logs issues.

	# --- Retrieval Methods --- #

	async def semantic_search(
		self,
		query: str,
		k: int = 5,
		filter_params: dict[str, Any] | None = None,
	) -> list[dict[str, Any]] | None:
		"""
		Perform semantic search for code chunks similar to the query using Qdrant.

		Args:
		    query: The search query string.
		    k: The number of top similar results to retrieve.
		    filter_params: Optional dictionary for filtering results. Supports:
		        - exact match: {"field": "value"} or {"match": {"field": "value"}}
		        - multiple values: {"match_any": {"field": ["value1", "value2"]}}
		        - range: {"range": {"field": {"gt": value, "lt": value}}}
		        - complex: {"must": [...], "should": [...], "must_not": [...]}

		Returns:
		    A list of search result dictionaries (Qdrant ScoredPoint converted to dict),
		    or None if an error occurs.

		"""
		if not self.is_async_initialized or not self.qdrant_manager:
			logger.error("QdrantManager not available for semantic search.")
			return None

		logger.debug("Performing semantic search for query: '%s', k=%d", query, k)

		try:
			# 1. Generate query embedding (must be async)
			query_embedding = generate_embedding([query], self.config_loader)
			if query_embedding is None:
				logger.error("Failed to generate embedding for query.")
				return None

			# Convert to numpy array if needed by Qdrant client, though list is often fine
			# query_vector = np.array(query_embedding, dtype=np.float32)
			query_vector = query_embedding[0]  # Qdrant client typically accepts list[float]

			# 2. Process filter parameters to Qdrant filter format
			query_filter = None
			if filter_params:
				query_filter = self._build_qdrant_filter(filter_params)
				logger.debug("Using filter for search: %s", query_filter)

			# 3. Query Qdrant index (must be async)
			search_results: list[qdrant_models.ScoredPoint] = await self.qdrant_manager.search(
				query_vector, k, query_filter=query_filter
			)

			if not search_results:
				logger.debug("Qdrant search returned no results.")
				return []

			# 4. Format results (convert ScoredPoint to dictionary)
			formatted_results = []
			for scored_point in search_results:
				# Convert Qdrant model to dict for consistent output
				# Include score (similarity) and payload
				from codemap.processor.vector.schema import ChunkMetadataSchema

				payload = ChunkMetadataSchema.model_validate(scored_point.payload)

				result_dict = {
					"id": str(scored_point.id),  # Ensure ID is string
					"score": scored_point.score,
					"payload": payload,
				}
				formatted_results.append(result_dict)

			logger.debug("Semantic search found %d results.", len(formatted_results))
			return formatted_results

		except Exception:
			logger.exception("Error during semantic search.")
			return None

	def _build_qdrant_filter(self, filter_params: dict[str, Any]) -> qdrant_models.Filter:
		"""
		Convert filter parameters to Qdrant filter format.

		Args:
		    filter_params: Dictionary of filter parameters

		Returns:
		    Qdrant filter object

		"""
		# If already a proper Qdrant filter, return as is
		if isinstance(filter_params, qdrant_models.Filter):
			return filter_params

		# Check for clause-based filter (must, should, must_not)
		if any(key in filter_params for key in ["must", "should", "must_not"]):
			filter_obj = {}

			# Process must conditions (AND)
			if "must" in filter_params:
				filter_obj["must"] = [self._build_qdrant_filter(cond) for cond in filter_params["must"]]

			# Process should conditions (OR)
			if "should" in filter_params:
				filter_obj["should"] = [self._build_qdrant_filter(cond) for cond in filter_params["should"]]

			# Process must_not conditions (NOT)
			if "must_not" in filter_params:
				filter_obj["must_not"] = [self._build_qdrant_filter(cond) for cond in filter_params["must_not"]]

			return qdrant_models.Filter(**filter_obj)

		# Check for condition-based filter (match, range, etc.)
		if "match" in filter_params:
			field, value = next(iter(filter_params["match"].items()))
			return qdrant_models.Filter(
				must=[qdrant_models.FieldCondition(key=field, match=qdrant_models.MatchValue(value=value))]
			)

		if "match_any" in filter_params:
			field, values = next(iter(filter_params["match_any"].items()))
			# For string values
			if (values and isinstance(values[0], str)) or (values and isinstance(values[0], (int, float))):
				return qdrant_models.Filter(
					should=[
						qdrant_models.FieldCondition(key=field, match=qdrant_models.MatchValue(value=value))
						for value in values
					]
				)
			# Default case
			return qdrant_models.Filter(
				should=[
					qdrant_models.FieldCondition(key=field, match=qdrant_models.MatchValue(value=value))
					for value in values
				]
			)

		if "range" in filter_params:
			field, range_values = next(iter(filter_params["range"].items()))
			return qdrant_models.Filter(
				must=[qdrant_models.FieldCondition(key=field, range=qdrant_models.Range(**range_values))]
			)

		# Default: treat as simple field-value pairs (exact match)
		must_conditions = []
		for field, value in filter_params.items():
			must_conditions.append(qdrant_models.FieldCondition(key=field, match=qdrant_models.MatchValue(value=value)))

		return qdrant_models.Filter(must=must_conditions)

	# Context manager support for async operations
	async def __aenter__(self) -> Self:
		"""Return self for use as async context manager."""
		# Basic initialization is sync, async init must be called separately
		# Consider if automatic async_init here is desired, or keep it explicit
		# await self.async_init() # Example if auto-init is desired
		return self

	async def __aexit__(
		self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
	) -> None:
		"""Clean up resources when exiting the async context manager."""
		await self.stop()
