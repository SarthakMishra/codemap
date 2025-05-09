"""
Unified pipeline for CodeMap data processing, synchronization, and retrieval.

This module defines the `ProcessingPipeline`, which acts as the central orchestrator
for managing and interacting with the HNSW vector database. It handles initialization,
synchronization with the Git repository, and provides semantic search capabilities.

"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Self

from qdrant_client import models as qdrant_models  # Use alias to avoid name clash

from codemap.db.client import DatabaseClient
from codemap.git.utils import get_repo_root
from codemap.processor.tree_sitter import TreeSitterAnalyzer

# Use async embedding utils
from codemap.processor.utils.embedding_utils import (
	generate_embedding,
)

# Import Qdrant specific classes
from codemap.processor.vector.chunking import TreeSitterChunker
from codemap.processor.vector.qdrant_manager import QdrantManager
from codemap.processor.vector.synchronizer import VectorSynchronizer
from codemap.utils.cli_utils import progress_indicator
from codemap.utils.docker_utils import ensure_qdrant_running
from codemap.watcher.file_watcher import Watcher

if TYPE_CHECKING:
	from types import TracebackType

	from codemap.config import ConfigLoader

logger = logging.getLogger(__name__)


class ProcessingPipeline:
	"""
	Orchestrates data processing, synchronization, and retrieval for CodeMap using Qdrant.

	Manages connections and interactions with the Qdrant vector database,
	ensuring it is synchronized with the Git repository state. Provides
	methods for semantic search. Uses asyncio for database and embedding
	operations.

	"""

	# Note: __init__ cannot be async directly. Initialization happens in an async method.
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
		if config_loader:
			self.config_loader = config_loader
		else:
			from codemap.config import ConfigLoader

			self.config_loader = ConfigLoader.get_instance()

		self.repo_path = self.config_loader.get.repo_root

		if not self.repo_path:
			self.repo_path = get_repo_root()

		_config_loader = self.config_loader.__class__
		if not config_loader:
			from codemap.config import ConfigLoader as _ActualConfigLoader

			_config_loader = _ActualConfigLoader

		if not isinstance(self.config_loader, _config_loader):
			from codemap.config import ConfigError

			logger.error(f"Config loading failed or returned unexpected type: {type(self.config_loader)}")
			msg = "Failed to load a valid Config object."
			raise ConfigError(msg)

		# --- Initialize Shared Components (Synchronous) --- #
		self.analyzer = TreeSitterAnalyzer()
		self.chunker = TreeSitterChunker(config_loader=self.config_loader)
		self.db_client = DatabaseClient()

		# --- Load Configuration --- #
		# Get embedding configuration
		embedding_config = self.config_loader.get.embedding
		embedding_model = embedding_config.model_name
		qdrant_dimension = embedding_config.dimension
		distance_metric = embedding_config.dimension_metric

		# Make sure embedding_model_name is always a string
		self.embedding_model_name: str = "voyage-code-3"  # Default
		if embedding_model and isinstance(embedding_model, str):
			self.embedding_model_name = embedding_model

		if not qdrant_dimension:
			logger.warning("Missing qdrant dimension in configuration, using default 1024")
			qdrant_dimension = 1024

		logger.info(f"Using embedding model: {self.embedding_model_name} with dimension: {qdrant_dimension}")

		# Get Qdrant configuration
		vector_config = self.config_loader.get.embedding
		qdrant_location = vector_config.qdrant_location
		qdrant_collection = vector_config.qdrant_collection_name
		qdrant_url = vector_config.url
		qdrant_api_key = vector_config.api_key

		# Convert distance metric string to enum
		distance_enum = qdrant_models.Distance.COSINE
		if distance_metric and distance_metric.upper() in ["COSINE", "EUCLID", "DOT"]:
			distance_enum = getattr(qdrant_models.Distance, distance_metric.upper())

		# Use URL if provided, otherwise use location (defaults to :memory: in QdrantManager)
		qdrant_init_args = {
			"config_loader": self.config_loader,  # Pass ConfigLoader to QdrantManager
			"collection_name": qdrant_collection,
			"dim": qdrant_dimension,
			"distance": distance_enum,
		}

		if qdrant_url:
			qdrant_init_args["url"] = qdrant_url
			if qdrant_api_key:
				qdrant_init_args["api_key"] = qdrant_api_key
			logger.info(f"Configuring Qdrant client for URL: {qdrant_url}")
		elif qdrant_location:
			qdrant_init_args["location"] = qdrant_location
			logger.info(f"Configuring Qdrant client for local path/memory: {qdrant_location}")
		else:
			# Let QdrantManager use its default (:memory:)
			logger.info("Configuring Qdrant client for default location (:memory:)")

		# --- Initialize Managers (Synchronous) --- #
		self.qdrant_manager = QdrantManager(**qdrant_init_args)

		# Initialize VectorSynchronizer with the embedding model name and config_loader
		self.vector_synchronizer = VectorSynchronizer(
			self.repo_path,
			self.qdrant_manager,
			self.chunker,
			self.embedding_model_name,
			self.analyzer,
			config_loader=self.config_loader,  # Pass ConfigLoader to VectorSynchronizer
		)

		logger.info(f"ProcessingPipeline synchronous initialization complete for repo: {self.repo_path}")
		self.is_async_initialized = False
		self.watcher: Watcher | None = None
		self._watcher_task: asyncio.Task | None = None
		self._sync_lock = asyncio.Lock()

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

		init_description = "Initializing pipeline components..."

		with progress_indicator(init_description):
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

				# Initialize the database client asynchronously
				with progress_indicator("Initializing database client..."):
					try:
						await self.db_client.initialize()
						logger.info("Database client initialized successfully.")

					except RuntimeError as e:
						logger.warning(
							f"Database initialization failed (RuntimeError): {e}. Some features may not work properly."
						)
					except (ConnectionError, OSError) as e:
						logger.warning(f"Database connection failed: {e}. Some features may not work properly.")

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

		# Cleanup database client
		if hasattr(self, "db_client") and self.db_client:
			try:
				await self.db_client.cleanup()
				logger.info("Database client cleaned up.")
			except RuntimeError:
				logger.exception("Error during database client cleanup")
			except (ConnectionError, OSError):
				logger.exception("Connection error during database client cleanup")

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
			query_embedding = await generate_embedding(
				query,
				model=self.embedding_model_name,
				config_loader=self.config_loader,  # Pass ConfigLoader to generate_embedding
			)
			if query_embedding is None:
				logger.error("Failed to generate embedding for query.")
				return None

			# Convert to numpy array if needed by Qdrant client, though list is often fine
			# query_vector = np.array(query_embedding, dtype=np.float32)
			query_vector = query_embedding  # Qdrant client typically accepts list[float]

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
				result_dict = {
					"id": str(scored_point.id),  # Ensure ID is string
					"score": scored_point.score,
					"payload": scored_point.payload,
					# Optionally include version if needed
					# "version": scored_point.version,
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
