"""Module for managing Qdrant vector database collections."""

import logging
import uuid
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ValidationError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, ExtendedPointId, PointStruct, VectorParams

from codemap.processor.vector.schema import ChunkMetadataSchema

if TYPE_CHECKING:
	from codemap.config import ConfigLoader

logger = logging.getLogger(__name__)

# HTTP status code for Not Found
HTTP_NOT_FOUND = 404


class QdrantManager:
	"""
	Manages interactions with a Qdrant vector database collection.

	Handles initialization, adding/upserting points (vectors + payload),
	searching, and retrieving points based on IDs or filters. Uses an
	AsyncQdrantClient for non-blocking operations.

	"""

	value_error_msg = "Filter condition cannot be None if point_ids are not provided"

	def __init__(
		self,
		config_loader: "ConfigLoader",
		collection_name: str | None = None,
		dim: int | None = None,
		distance: Distance | None = None,
		api_key: str | None = None,  # For Qdrant Cloud
		url: str | None = None,  # For self-hosted or Cloud URL
		prefer_grpc: bool | None = None,
		timeout: float | None = None,
	) -> None:
		"""
		Initialize the QdrantManager.

		Args:
		    config_loader: Configuration loader instance.
		    location: Path for local storage or ":memory:". Ignored if url is provided.
		    collection_name: Name of the Qdrant collection to use.
		    dim: Dimension of the vectors.
		    distance: Distance metric for vector comparison.
		    api_key: API key for Qdrant Cloud authentication.
		    url: URL for connecting to a remote Qdrant instance (overrides location).
		    prefer_grpc: Whether to prefer gRPC over REST.
		    timeout: Connection timeout in seconds.

		"""
		# Get embedding configuration
		self.config_loader = config_loader
		embedding_config = self.config_loader.get.embedding

		# Get distance metric from config or parameter
		distance_metric_str = embedding_config.dimension_metric.upper()
		default_distance = (
			getattr(Distance, distance_metric_str) if hasattr(Distance, distance_metric_str) else Distance.COSINE
		)

		# Load values from parameters or fall back to config
		self.collection_name = collection_name or "codemap_vectors"
		self.dim = dim or embedding_config.dimension
		self.distance = distance or default_distance

		# Build client args
		self.client_args = {
			"api_key": api_key or embedding_config.api_key,
			"url": url or embedding_config.url,
			"prefer_grpc": prefer_grpc if prefer_grpc is not None else embedding_config.prefer_grpc,
			"timeout": timeout or embedding_config.timeout,
		}
		# Remove None values from args
		self.client_args = {k: v for k, v in self.client_args.items() if v is not None}

		# Initialize client later in an async context
		self.client: AsyncQdrantClient | None = None
		self.is_initialized = False

	async def initialize(self) -> None:
		"""Asynchronously initialize the Qdrant client and ensure the collection exists."""
		if self.is_initialized:
			return

		try:
			logger.info("Initializing Qdrant client with args: %s", self.client_args)
			# Create client with explicit type casting to avoid type issues
			client_kwargs: dict[str, Any] = {}
			if "api_key" in self.client_args and self.client_args["api_key"] is not None:
				client_kwargs["api_key"] = str(self.client_args["api_key"])
			if "url" in self.client_args and self.client_args["url"] is not None:
				client_kwargs["url"] = str(self.client_args["url"])
			if "prefer_grpc" in self.client_args and self.client_args["prefer_grpc"] is not None:
				client_kwargs["prefer_grpc"] = bool(self.client_args["prefer_grpc"])
			if "timeout" in self.client_args and self.client_args["timeout"] is not None:
				client_kwargs["timeout"] = int(self.client_args["timeout"])

			self.client = AsyncQdrantClient(**client_kwargs)

			# Check if collection exists
			collections_response = await self.client.get_collections()
			collection_names = {col.name for col in collections_response.collections}

			if self.collection_name not in collection_names:
				logger.info(f"Collection '{self.collection_name}' not found. Creating...")
				await self.client.create_collection(
					collection_name=self.collection_name,
					vectors_config=VectorParams(size=self.dim, distance=self.distance),
				)
				logger.info(f"Collection '{self.collection_name}' created successfully.")

				# Create payload indexes for commonly filtered fields
				logger.info(f"Creating payload indexes for collection '{self.collection_name}'")
				# Common fields used in filters from ChunkMetadata
				await self._create_payload_indexes()
			else:
				logger.info(f"Using existing Qdrant collection: '{self.collection_name}'")
				# Check if indexes exist, create if missing
				await self._ensure_payload_indexes()

			self.is_initialized = True
			logger.info("QdrantManager initialized successfully.")

		except Exception:
			logger.exception("Failed to initialize QdrantManager or collection")
			self.client = None  # Ensure client is None if init fails
			raise  # Re-raise the exception to signal failure

	async def _create_payload_indexes(self) -> None:
		"""Create payload indexes for all fields in ChunkMetadataSchema and GitMetadataSchema."""
		if self.client is None:
			msg = "Client should be initialized before creating indexes"
			raise RuntimeError(msg)
		client: AsyncQdrantClient = self.client

		try:
			# Index fields for ChunkMetadataSchema
			index_fields = [
				("chunk_id", models.PayloadSchemaType.KEYWORD),
				("start_line", models.PayloadSchemaType.INTEGER),
				("end_line", models.PayloadSchemaType.INTEGER),
				("entity_type", models.PayloadSchemaType.KEYWORD),
				("entity_name", models.PayloadSchemaType.KEYWORD),
				("hierarchy_path", models.PayloadSchemaType.KEYWORD),
				# FileMetadataSchema subfields (nested under file_metadata)
				("file_metadata.file_path", models.PayloadSchemaType.KEYWORD),
				("file_metadata.language", models.PayloadSchemaType.KEYWORD),
				("file_metadata.last_modified_time", models.PayloadSchemaType.FLOAT),
				("file_metadata.file_content_hash", models.PayloadSchemaType.KEYWORD),
				# GitMetadataSchema subfields (nested under git_metadata)
				("git_metadata.git_hash", models.PayloadSchemaType.KEYWORD),
				("git_metadata.tracked", models.PayloadSchemaType.BOOL),
				("git_metadata.branch", models.PayloadSchemaType.KEYWORD),
				# ("git_metadata.blame", models.PayloadSchemaType.KEYWORD),  # Blame is a list of objects, not indexed
			]
			# Add indexes for all fields
			for field_name, field_type in index_fields:
				logger.info(f"Creating index for field: {field_name} ({field_type})")
				await client.create_payload_index(
					collection_name=self.collection_name, field_name=field_name, field_schema=field_type
				)
			logger.info(f"Created {len(index_fields)} payload indexes successfully")
		except Exception as e:  # noqa: BLE001
			logger.warning(f"Error creating payload indexes: {e}")
			# Continue even if index creation fails - collection will still work

	async def _ensure_payload_indexes(self) -> None:
		"""Check if payload indexes exist and create any missing ones to match the schema."""
		if self.client is None:
			msg = "Client should be initialized before checking indexes"
			raise RuntimeError(msg)
		client: AsyncQdrantClient = self.client

		try:
			# Get existing collection info
			collection_info = await client.get_collection(collection_name=self.collection_name)
			existing_schema = collection_info.payload_schema

			# List of fields that should be indexed (same as in _create_payload_indexes)
			index_fields = [
				("chunk_id", models.PayloadSchemaType.KEYWORD),
				("file_path", models.PayloadSchemaType.KEYWORD),
				("start_line", models.PayloadSchemaType.INTEGER),
				("end_line", models.PayloadSchemaType.INTEGER),
				("entity_type", models.PayloadSchemaType.KEYWORD),
				("entity_name", models.PayloadSchemaType.KEYWORD),
				("language", models.PayloadSchemaType.KEYWORD),
				("hierarchy_path", models.PayloadSchemaType.KEYWORD),
				# FileMetadataSchema subfields
				("file_metadata.file_path", models.PayloadSchemaType.KEYWORD),
				("file_metadata.language", models.PayloadSchemaType.KEYWORD),
				("file_metadata.last_modified_time", models.PayloadSchemaType.FLOAT),
				("file_metadata.file_content_hash", models.PayloadSchemaType.KEYWORD),
				# GitMetadataSchema subfields
				("git_metadata.git_hash", models.PayloadSchemaType.KEYWORD),
				("git_metadata.tracked", models.PayloadSchemaType.BOOL),
				("git_metadata.branch", models.PayloadSchemaType.KEYWORD),
				# ("git_metadata.blame", models.PayloadSchemaType.KEYWORD),  # Not indexed
			]
			# Create any missing indexes
			for field_name, field_type in index_fields:
				if field_name not in existing_schema:
					logger.info(f"Creating missing index for field: {field_name} ({field_type})")
					await client.create_payload_index(
						collection_name=self.collection_name, field_name=field_name, field_schema=field_type
					)
		except Exception as e:  # noqa: BLE001
			logger.warning(f"Error checking or creating payload indexes: {e}")
			# Continue even if index check fails

	async def _ensure_initialized(self) -> None:
		"""Ensure the client is initialized before performing operations."""
		if not self.is_initialized or self.client is None:
			await self.initialize()
		if not self.client:
			# Should not happen if initialize didn't raise, but check anyway
			msg = "Qdrant client is not available after initialization attempt."
			raise RuntimeError(msg)

	async def upsert_points(self, points: list[PointStruct]) -> None:
		"""
		Add or update points (vectors and payloads) in the collection.

		Args:
		    points: A list of Qdrant PointStruct objects. Each payload should be a dict matching ChunkMetadataSchema.

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)
		client: AsyncQdrantClient = self.client
		if not points:
			logger.warning("upsert_points called with an empty list.")
			return
		# Ensure all payloads are dicts (convert from Pydantic BaseModel if needed)
		for point in points:
			if hasattr(point, "payload") and isinstance(point.payload, BaseModel):
				point.payload = point.payload.model_dump()
		try:
			logger.info(f"Upserting {len(points)} points into '{self.collection_name}'")
			await client.upsert(
				collection_name=self.collection_name,
				points=points,
				wait=True,  # Wait for operation to complete
			)
			logger.debug(f"Successfully upserted {len(points)} points.")
		except Exception:
			logger.exception("Error upserting points into Qdrant")
			# Decide if partial failure needs specific handling or re-raising

	async def delete_points(self, point_ids: list[str]) -> None:
		"""
		Delete points from the collection by their IDs.

		Args:
		    point_ids: A list of point IDs to delete.

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)
		if not point_ids:
			logger.warning("delete_points called with an empty list.")
			return

		try:
			logger.info(f"Deleting {len(point_ids)} points from '{self.collection_name}'")
			qdrant_ids: list[ExtendedPointId] = [cast("ExtendedPointId", pid) for pid in point_ids]

			await self.client.delete(
				collection_name=self.collection_name,
				points_selector=models.PointIdsList(points=qdrant_ids),
				wait=True,
			)
			logger.debug(f"Successfully deleted {len(point_ids)} points.")
		except Exception:
			logger.exception("Error deleting points from Qdrant")
			# Consider error handling strategy

	async def delete_points_by_filter(self, qdrant_filter: models.Filter) -> None:
		"""
		Delete points from the collection based on a filter condition.

		Args:
			qdrant_filter: A Qdrant Filter object specifying the points to delete.
		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)
		try:
			logger.info(f"Deleting points from '{self.collection_name}' based on filter")
			await self.client.delete(
				collection_name=self.collection_name,
				points_selector=models.FilterSelector(filter=qdrant_filter),
				wait=True,
			)
			logger.debug("Successfully deleted points.")
		except Exception:
			logger.exception("Error deleting points from Qdrant")

	async def search(
		self,
		query_vector: list[float],
		k: int = 5,
		query_filter: models.Filter | None = None,
	) -> list[models.ScoredPoint]:
		"""
		Perform a vector search with optional filtering.

		Args:
		    query_vector: The vector to search for.
		    k: The number of nearest neighbors to return.
		    query_filter: Optional Qdrant filter conditions.

		Returns:
		    A list of ScoredPoint objects, including ID, score, and payload.

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)
		if not query_vector:
			logger.error("Search called with empty query vector.")
			return []

		try:
			search_result = await self.client.search(
				collection_name=self.collection_name,
				query_vector=query_vector,
				query_filter=query_filter,
				limit=k,
				with_payload=True,  # Always include payload
				with_vectors=False,  # Usually not needed in results
			)
			logger.debug(f"Search returned {len(search_result)} results.")
			return search_result
		except Exception:
			logger.exception("Error during Qdrant search")
			return []

	async def get_all_point_ids_with_filter(
		self, query_filter: models.Filter | None = None
	) -> list[str | int | uuid.UUID]:
		"""
		Retrieves all point IDs currently in the collection, optionally filtered.

		Uses scrolling API to handle potentially large collections.

		Args:
		    query_filter: Optional Qdrant filter to apply.

		Returns:
		    A list of all point IDs matching the filter.

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)
		client: AsyncQdrantClient = self.client
		all_ids: list[str | int | uuid.UUID] = []
		# Use Any for offset type hint due to persistent linter issues
		next_offset: Any | None = None
		limit_per_scroll = 1000

		# Add logging for parameters
		logger.info(
			f"[QdrantManager Get IDs] Fetching all point IDs from collection '{self.collection_name}'%s...",
			f" with filter: {query_filter}" if query_filter else "",
		)

		while True:
			try:
				logger.debug(f"[QdrantManager Get IDs] Scrolling with offset: {next_offset}")
				scroll_response, next_offset_id = await client.scroll(
					collection_name=self.collection_name,
					scroll_filter=query_filter,
					limit=limit_per_scroll,
					offset=next_offset,
					with_payload=False,
					with_vectors=False,
				)
				batch_ids: list[ExtendedPointId] = [point.id for point in scroll_response]
				logger.debug(f"[QdrantManager Get IDs] Scroll returned {len(batch_ids)} IDs in this batch.")
				if not batch_ids:
					logger.debug("[QdrantManager Get IDs] No more IDs returned by scroll. Stopping.")
					break
				all_ids.extend([cast("str | int | uuid.UUID", point_id) for point_id in batch_ids])
				# Assign the returned offset ID - type is likely PointId but linter struggles
				next_offset = next_offset_id  # No ignore needed if next_offset is Any
				if next_offset is None:
					break

			except UnexpectedResponse as e:
				# Qdrant might return 404 if offset points to non-existent ID after deletions
				if e.status_code == HTTP_NOT_FOUND and "Point with id" in str(e.content):
					logger.warning(
						f"Scroll encountered potentially deleted point ID at offset: {next_offset}. Stopping scroll."
					)
					break
				logger.exception("Error scrolling through Qdrant points")
				raise  # Reraise other unexpected errors
			except Exception:
				logger.exception("Error scrolling through Qdrant points")
				raise

		logger.info(f"Retrieved {len(all_ids)} point IDs from collection '{self.collection_name}'.")
		return all_ids

	async def get_payloads_by_ids(self, point_ids: list[str | int | uuid.UUID]) -> dict[str, dict[str, Any]]:
		"""
		Retrieves payloads for specific point IDs.

		Args:
		    point_ids: List of point IDs to fetch payloads for.

		Returns:
		    A dictionary mapping point IDs (as strings) to their payloads (as dicts matching ChunkMetadataSchema).

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)
		if not point_ids:
			return {}

		try:
			qdrant_ids: list[ExtendedPointId] = [cast("ExtendedPointId", pid) for pid in point_ids]
			points_data = await self.client.retrieve(
				collection_name=self.collection_name,
				# Ignore linter complaint about list type compatibility
				ids=qdrant_ids,  # type: ignore[arg-type]
				with_payload=True,
				with_vectors=False,
			)
			payloads = {}
			for point in points_data:
				if point.payload:
					# Try to parse as ChunkMetadataSchema, fallback to dict
					try:
						payloads[str(point.id)] = ChunkMetadataSchema.model_validate(point.payload).model_dump()
					except (ValidationError, TypeError, ValueError):
						payloads[str(point.id)] = point.payload
			logger.debug(f"Retrieved payloads for {len(payloads)} points.")
			return payloads
		except Exception:
			logger.exception("Error retrieving payloads from Qdrant")
			return {}

	async def close(self) -> None:
		"""Close the AsyncQdrantClient connection."""
		if self.client:
			logger.info("Closing Qdrant client connection.")
			try:
				await self.client.close()
			except Exception:
				logger.exception("Error closing Qdrant client")
			finally:
				self.client = None
				self.is_initialized = False

	async def __aenter__(self) -> "QdrantManager":
		"""Enter the async context manager, initializing the Qdrant client."""
		await self.initialize()
		return self

	async def __aexit__(
		self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
	) -> None:
		"""Exit the async context manager, closing the Qdrant client."""
		await self.close()

	async def set_payload(
		self,
		filter_condition: models.Filter,
		payload: dict[str, Any],
		point_ids: list[str | int | uuid.UUID] | None = None,
		key: str | None = None,
	) -> None:
		"""
		Set specific payload fields for points without overwriting existing fields.

		Args:
		    payload: Dictionary of payload fields to set
		    point_ids: Optional list of point IDs to update
		    filter_condition: Optional filter to select points
		    key: Optional specific payload key path to modify

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)

		if not payload:
			logger.warning("set_payload called with empty payload.")
			return

		if not point_ids and not filter_condition:
			logger.warning("set_payload called without point_ids or filter.")
			return

		try:
			logger.info(f"Setting payload fields: {list(payload.keys())} for points in '{self.collection_name}'")

			if point_ids:
				# Convert to list of Qdrant point IDs
				qdrant_ids: list[ExtendedPointId] = [cast("ExtendedPointId", pid) for pid in point_ids]

				# Create points selector with IDs
				points_selector = models.PointIdsList(points=qdrant_ids)

				await self.client.set_payload(
					collection_name=self.collection_name,
					payload=payload,
					points=points_selector,
					key=key,
					wait=True,
				)
			else:
				# We know filter_condition is not None here because of the initial check
				points_selector = models.FilterSelector(filter=filter_condition)

				await self.client.set_payload(
					collection_name=self.collection_name,
					payload=payload,
					points=points_selector,
					key=key,
					wait=True,
				)

			logger.debug(f"Successfully set payload fields: {list(payload.keys())}")
		except Exception:
			logger.exception("Error setting payload in Qdrant")

	async def overwrite_payload(
		self,
		payload: dict[str, Any],
		point_ids: list[str | int | uuid.UUID] | None = None,
		filter_condition: models.Filter | None = None,
	) -> None:
		"""
		Completely replace the payload for points with the new payload.

		Args:
		    payload: Dictionary of payload fields to set
		    point_ids: Optional list of point IDs to update
		    filter_condition: Optional filter to select points

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)

		if not payload:
			logger.warning("overwrite_payload called with empty payload.")
			return

		if not point_ids and not filter_condition:
			logger.warning("overwrite_payload called without point_ids or filter.")
			return

		try:
			logger.info(f"Overwriting payload for points in '{self.collection_name}'")

			if point_ids:
				# Convert to list of Qdrant point IDs
				qdrant_ids: list[ExtendedPointId] = [cast("ExtendedPointId", pid) for pid in point_ids]

				# Create points selector with IDs
				points_selector = models.PointIdsList(points=qdrant_ids)

				await self.client.overwrite_payload(
					collection_name=self.collection_name,
					payload=payload,
					points=points_selector,
					wait=True,
				)
			else:
				# We know filter_condition is not None here
				if filter_condition is None:
					raise ValueError(self.value_error_msg)
				points_selector = models.FilterSelector(filter=filter_condition)

				await self.client.overwrite_payload(
					collection_name=self.collection_name,
					payload=payload,
					points=points_selector,
					wait=True,
				)

			logger.debug("Successfully overwrote payload")
		except Exception:
			logger.exception("Error overwriting payload in Qdrant")

	async def clear_payload(
		self, point_ids: list[str | int | uuid.UUID] | None = None, filter_condition: models.Filter | None = None
	) -> None:
		"""
		Remove all payload fields from points.

		Args:
		    point_ids: Optional list of point IDs to update
		    filter_condition: Optional filter to select points

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)

		if not point_ids and not filter_condition:
			logger.warning("clear_payload called without point_ids or filter.")
			return

		try:
			logger.info(f"Clearing payload for points in '{self.collection_name}'")

			if point_ids:
				# Convert to list of Qdrant point IDs
				qdrant_ids: list[ExtendedPointId] = [cast("ExtendedPointId", pid) for pid in point_ids]

				# Create points selector with IDs
				points_selector = models.PointIdsList(points=qdrant_ids)  # type: ignore[arg-type]

				await self.client.clear_payload(
					collection_name=self.collection_name,
					points_selector=points_selector,
					wait=True,
				)
			else:
				# We know filter_condition is not None here
				if filter_condition is None:
					raise ValueError(self.value_error_msg)
				points_selector = models.FilterSelector(filter=filter_condition)

				await self.client.clear_payload(
					collection_name=self.collection_name,
					points_selector=points_selector,  # type: ignore[arg-type]
					wait=True,
				)

			logger.debug("Successfully cleared payload")
		except Exception:
			logger.exception("Error clearing payload in Qdrant")

	async def delete_payload_keys(
		self,
		keys: list[str],
		point_ids: list[str | int | uuid.UUID] | None = None,
		filter_condition: models.Filter | None = None,
	) -> None:
		"""
		Delete specific payload fields from points.

		Args:
		    keys: List of payload field keys to delete
		    point_ids: Optional list of point IDs to update
		    filter_condition: Optional filter to select points

		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)

		if not keys:
			logger.warning("delete_payload_keys called with empty keys list.")
			return

		if not point_ids and not filter_condition:
			logger.warning("delete_payload_keys called without point_ids or filter.")
			return

		try:
			logger.info(f"Deleting payload keys {keys} for points in '{self.collection_name}'")

			if point_ids:
				# Convert to list of Qdrant point IDs
				qdrant_ids: list[ExtendedPointId] = [cast("ExtendedPointId", pid) for pid in point_ids]

				# Create points selector with IDs
				points_selector = models.PointIdsList(points=qdrant_ids)

				await self.client.delete_payload(
					collection_name=self.collection_name,
					keys=keys,
					points=points_selector,
					wait=True,
				)
			else:
				# We know filter_condition is not None here
				if filter_condition is None:
					raise ValueError(self.value_error_msg)
				points_selector = models.FilterSelector(filter=filter_condition)

				await self.client.delete_payload(
					collection_name=self.collection_name,
					keys=keys,
					points=points_selector,
					wait=True,
				)

			logger.debug(f"Successfully deleted payload keys: {keys}")
		except Exception:
			logger.exception("Error deleting payload keys in Qdrant")

	async def get_all_chunks_content_hashes(self) -> set[str]:
		"""
		Retrieves all content hashes of chunks in the collection.

		Used for deduplication when syncing.

		Returns:
			A set of combined content_hash:file_content_hash strings for all chunks in the collection.
		"""
		await self._ensure_initialized()
		if self.client is None:
			msg = "Client should be initialized here"
			raise RuntimeError(msg)

		content_hashes = set()

		try:
			# Get all payloads for deduplication check
			all_ids = await self.get_all_point_ids_with_filter()

			if not all_ids:
				return content_hashes

			# Process in batches to avoid memory issues
			for i in range(0, len(all_ids), 100):
				batch_ids = all_ids[i : i + 100]
				payloads = await self.get_payloads_by_ids(batch_ids)

				for payload in payloads.values():
					if not payload:
						continue

					content_hash = payload.get("content_hash", "")
					file_metadata = payload.get("file_metadata", {})

					if isinstance(file_metadata, dict):
						file_content_hash = file_metadata.get("file_content_hash", "")
					else:
						file_content_hash = ""

					if content_hash and file_content_hash:
						# Create a composite key for both the chunk content and file content
						dedup_key = f"{content_hash}:{file_content_hash}"
						content_hashes.add(dedup_key)

			logger.info(f"Retrieved {len(content_hashes)} unique content hash combinations from Qdrant.")
			return content_hashes

		except Exception:
			logger.exception("Error retrieving content hashes from Qdrant")
			return set()


# Utility function to create PointStruct easily
def create_qdrant_point(chunk_id: str, vector: list[float], payload: dict[str, Any]) -> PointStruct:
	"""
	Helper function to create a Qdrant PointStruct.

	Ensures the ID is treated appropriately (UUIDs if possible).

	"""
	point_id: ExtendedPointId
	try:
		# Validate it's a UUID but use string representation for ExtendedPointId
		uuid_obj = uuid.UUID(chunk_id)
		point_id = str(uuid_obj)
	except ValueError:
		point_id = chunk_id

	return PointStruct(id=point_id, vector=vector, payload=payload)
