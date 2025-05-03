"""Manages HNSWLib index and associated metadata."""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import hnswlib
import numpy as np

from codemap.config import DEFAULT_CONFIG
from codemap.utils.file_utils import ensure_directory_exists

logger = logging.getLogger(__name__)

# Constants for magic values
EXPECTED_LINE_RANGE_PARTS = 2
MIN_CHUNK_ID_PARTS = 2
VECTOR_SHAPE_2D = 2  # For checking vector.shape dimensions
VECTOR_FIRST_DIM = 1  # Expected first dimension size for 2D vector

# Significantly increased defaults to handle larger repositories
DEFAULT_SPACE = "cosine"
DEFAULT_DIMENSION = DEFAULT_CONFIG["embedding"]["dimension"]  # Use dimension from config
DEFAULT_INDEX_FILENAME = "hnsw_index.bin"
DEFAULT_METADATA_FILENAME = "hnsw_metadata.json"
DEFAULT_MAX_ELEMENTS = 100000  # Increased from 20000
DEFAULT_RESIZE_FACTOR = 1.5
DEFAULT_M = 128  # Increased from 64 for high-dim vectors
DEFAULT_EF_CONSTRUCTION = 2000  # Increased from 800 for better index quality
DEFAULT_EF_QUERY = 4000  # Increased from 800 for better search accuracy


class HNSWManager:
	"""Handles HNSWLib index and metadata persistence."""

	def __init__(
		self,
		index_dir_path: str | Path,
		space: Literal["l2", "ip", "cosine"] = DEFAULT_SPACE,
		dim: int = DEFAULT_DIMENSION,
		max_elements: int = DEFAULT_MAX_ELEMENTS,
		m: int = DEFAULT_M,
		ef_construction: int = DEFAULT_EF_CONSTRUCTION,
		ef_query: int = DEFAULT_EF_QUERY,
		index_filename: str = DEFAULT_INDEX_FILENAME,
		metadata_filename: str = DEFAULT_METADATA_FILENAME,
		allow_replace_deleted: bool = False,  # Consider enabling if deletes are frequent
		resize_factor: float = DEFAULT_RESIZE_FACTOR,
	) -> None:
		"""
		Initialize the HNSWManager.

		Args:
		    index_dir_path: Directory to store index and metadata files.
		    space: Vector space type ('l2', 'ip', 'cosine').
		    dim: Dimension of vectors.
		    max_elements: Initial maximum capacity of the index.
		    m: HNSW M parameter (connections per node).
		    ef_construction: HNSW efConstruction parameter.
		    ef_query: HNSW ef parameter for querying.
		    index_filename: Filename for the HNSW index.
		    metadata_filename: Filename for the metadata JSON.
		    allow_replace_deleted: Allow reusing IDs of deleted items.
		    resize_factor: Factor to increase index size by when resizing.

		"""
		self.index_dir_path = Path(index_dir_path)
		self.space = space
		self.dim = dim
		self.max_elements = max_elements
		self.m = m
		self.ef_construction = ef_construction
		self.ef_query = ef_query
		self.index_file_path = self.index_dir_path / index_filename
		self.metadata_file_path = self.index_dir_path / metadata_filename
		self.allow_replace_deleted = allow_replace_deleted
		self.resize_factor = resize_factor

		self.index: hnswlib.Index | None = None
		self.metadata_store: dict[str, Any] = {"chunks": {}, "file_hashes": {}}
		self.next_internal_id = 0  # HNSW uses sequential internal IDs
		self.chunk_id_to_internal_id: dict[str, int] = {}
		self.internal_id_to_chunk_id: dict[int, str] = {}

		ensure_directory_exists(self.index_dir_path)
		self._initialize_or_load()

	def _initialize_or_load(self) -> None:
		"""Load existing index and metadata or initialize new ones."""
		if self.index_file_path.exists() and self.metadata_file_path.exists():
			try:
				self._load_index()
				self._load_metadata()
				logger.info(f"Loaded existing index ({self.get_current_count()} items) and metadata.")
			except Exception as e:  # noqa: BLE001
				logger.warning(
					f"Failed to load existing index/metadata ({e}). Initializing new ones.",
					exc_info=False,  # Log only warning, not full stack trace here
				)
				self._initialize_new_index()
				self.metadata_store = {"chunks": {}, "file_hashes": {}}  # Reset metadata
		else:
			logger.info("No existing index found. Initializing new index and metadata.")
			self._initialize_new_index()

	def _initialize_new_index(self) -> None:
		"""Initialize a new HNSW index."""
		# Assign first
		index = hnswlib.Index(space=self.space, dim=self.dim)  # type: ignore[arg-type]
		# Then call methods on the local variable
		index.init_index(
			max_elements=self.max_elements,
			ef_construction=self.ef_construction,
			M=self.m,
		)
		index.set_ef(self.ef_query)
		# Assign to self.index at the end
		self.index = index
		self.next_internal_id = 0
		self.chunk_id_to_internal_id = {}
		self.internal_id_to_chunk_id = {}

	def reset_index(self) -> None:
		"""Reset the index to an empty state, clearing all data."""
		self._initialize_new_index()
		self.metadata_store = {"chunks": {}, "file_hashes": {}}
		logger.info("Reset HNSW index to empty state")

	def _load_index(self) -> None:
		"""Load HNSW index from file."""
		try:
			self.index = hnswlib.Index(space=self.space, dim=self.dim)  # type: ignore[arg-type]
			# We'll rely on the saved metadata for the correct max_elements if available
			try:
				# Use Path.open()
				with self.metadata_file_path.open(encoding="utf-8") as f:
					loaded_meta = json.load(f)
					self.max_elements = loaded_meta.get("index_capacity", self.max_elements)

					# Check if the index was created with a different dimension
					stored_dim = loaded_meta.get("dimension")
					if stored_dim is not None and stored_dim != self.dim:
						logger.warning(
							f"Dimension mismatch: index was created with dim={stored_dim}, "
							f"but current configuration is dim={self.dim}. Rebuilding index."
						)
						self._initialize_new_index()
						return

			except (FileNotFoundError, json.JSONDecodeError) as e:  # Catch specific errors
				logger.warning(f"Could not load index capacity from metadata ({e}), using default for index load.")

			# Ensure index is not None after potential load failure path
			if self.index is None:
				# This case should ideally not happen if _initialize_or_load logic is correct
				# but handles potential edge cases or future refactoring issues.
				logger.error("Index object is unexpectedly None before loading. Re-initializing.")
				self._initialize_new_index()
				if self.index is None:  # If still None after re-init, something is seriously wrong
					msg = "Failed to initialize HNSW index object."
					raise RuntimeError(msg)

			if not self.index_file_path.exists():
				logger.warning(f"Index file {self.index_file_path} not found. Initializing new index.")
				self._initialize_new_index()
				return

			# Removed assert: self.index is not None
			self.index.load_index(str(self.index_file_path), max_elements=self.max_elements)
			self.index.set_ef(self.ef_query)
			logger.info(f"Loaded HNSW index from {self.index_file_path}")
		except FileNotFoundError:
			logger.warning(f"Index file {self.index_file_path} not found. Initializing new index.")
			self._initialize_new_index()
		except (OSError, RuntimeError, ValueError, AttributeError) as e:
			logger.warning(f"Failed to load index: {e}. Initializing new index.")
			self._initialize_new_index()

	def _parse_line_range(self, line_range: str) -> tuple[int, int]:
		"""
		Parse line range from string format like '12-15'.

		Args:
		    line_range: String in format 'start_line-end_line'

		Returns:
		    Tuple of (start_line, end_line) as integers

		"""
		try:
			parts = line_range.split("-")
			if len(parts) == EXPECTED_LINE_RANGE_PARTS:
				return int(parts[0]), int(parts[1])
		except (ValueError, IndexError):
			pass

		# Default fallback if parsing fails
		return 0, 0

	def _load_metadata(self) -> None:
		"""Load metadata from JSON file."""
		try:
			# Use Path.open()
			if self.metadata_file_path.exists():
				with self.metadata_file_path.open(encoding="utf-8") as f:
					self.metadata_store = json.load(f)
			else:
				logger.warning(f"Metadata file {self.metadata_file_path} does not exist. Initializing empty metadata.")
				self.metadata_store = {"chunks": {}, "file_hashes": {}}
				return

			# Initialize critical fields if they don't exist
			if "chunks" not in self.metadata_store:
				logger.warning("No 'chunks' key in metadata file. Initializing empty chunks dictionary.")
				self.metadata_store["chunks"] = {}

			if "file_hashes" not in self.metadata_store:
				logger.warning("No 'file_hashes' key in metadata file. Initializing empty file_hashes dictionary.")
				self.metadata_store["file_hashes"] = {}

			# Rebuild internal ID mappings from loaded metadata
			chunk_id_map = self.metadata_store.get("_chunk_id_map", {})
			self.chunk_id_to_internal_id = chunk_id_map
			self.internal_id_to_chunk_id = {int(v): k for k, v in chunk_id_map.items()}
			self.next_internal_id = self.metadata_store.get("_next_internal_id", 0)

			logger.info(f"Loaded metadata with {len(self.metadata_store['chunks'])} chunks")
		except FileNotFoundError:
			logger.warning(f"Metadata file {self.metadata_file_path} not found. Initializing empty metadata.")
			self.metadata_store = {"chunks": {}, "file_hashes": {}}
			self._reset_id_mappings()
		except json.JSONDecodeError:
			logger.warning("Invalid JSON in metadata file. Initializing empty metadata.")
			self.metadata_store = {"chunks": {}, "file_hashes": {}}
			self._reset_id_mappings()
		except (KeyError, TypeError, ValueError, AttributeError, OSError) as e:
			logger.warning(f"Error loading metadata: {e}. Initializing empty metadata.")
			self.metadata_store = {"chunks": {}, "file_hashes": {}}
			self._reset_id_mappings()

	def _reset_id_mappings(self) -> None:
		"""Reset internal ID mappings."""
		self.next_internal_id = 0
		self.chunk_id_to_internal_id = {}
		self.internal_id_to_chunk_id = {}

	def save(self) -> None:
		"""Save the index and metadata."""
		self._save_index()
		self._save_metadata()

	def _save_index(self) -> None:
		"""Save HNSW index to file."""
		if self.index is None:
			logger.error("Cannot save index: Index is not initialized.")
			return
		try:
			logger.info(f"Saving HNSW index to {self.index_file_path}...")
			self.index.save_index(str(self.index_file_path))
			logger.info("HNSW index saved successfully.")
		except Exception:
			# Remove exception from log message
			logger.exception("Error saving HNSW index")

	def _save_metadata(self) -> None:
		"""Save metadata to JSON file."""
		try:
			logger.info(f"Saving metadata to {self.metadata_file_path}...")
			# Add internal state needed for reloading
			persist_data = self.metadata_store.copy()
			persist_data["_next_internal_id"] = self.next_internal_id
			persist_data["_chunk_id_map"] = self.chunk_id_to_internal_id
			persist_data["index_capacity"] = self.get_max_elements()
			persist_data["dimension"] = self.dim  # Store current dimension to detect changes

			# Ensure chunks metadata exists and is initialized if empty
			if "chunks" not in persist_data or not persist_data["chunks"]:
				logger.warning("Chunks metadata is empty, initializing to empty dict")
				persist_data["chunks"] = {}

			# Ensure file_hashes metadata exists
			if "file_hashes" not in persist_data or not persist_data["file_hashes"]:
				logger.warning("File hashes metadata is empty, initializing to empty dict")
				persist_data["file_hashes"] = {}

			# Rebuild file_hashes from chunks if needed
			if not persist_data["file_hashes"] and persist_data["chunks"]:
				logger.info("Rebuilding file_hashes from chunk metadata")
				for metadata in persist_data["chunks"].values():
					if "file_path" in metadata and "git_hash" in metadata and metadata["git_hash"]:
						persist_data["file_hashes"][metadata["file_path"]] = metadata["git_hash"]

			# Use Path.open()
			with self.metadata_file_path.open("w", encoding="utf-8") as f:
				json.dump(persist_data, f, indent=4)
			logger.info("Metadata saved successfully.")
		except Exception:
			# Remove exception from log message
			logger.exception("Error saving metadata file")

	def _ensure_capacity(self, num_new_items: int) -> None:
		"""Resize the index if needed."""
		if self.index is None:
			return

		current_count = self.get_current_count()
		required_capacity = current_count + num_new_items
		if required_capacity > self.get_max_elements():
			new_capacity = max(required_capacity, int(self.get_max_elements() * self.resize_factor))
			logger.warning(
				f"Index capacity ({self.get_max_elements()}) exceeded. "
				f"Resizing to {new_capacity} (current count: {current_count}, adding: {num_new_items})."
			)
			try:
				# HNSWlib resize needs to be called on the index object itself
				self.index.resize_index(new_capacity)
				self.max_elements = new_capacity  # Update the manager's tracking
				logger.info(f"Index resized successfully to {new_capacity}.")
			except Exception as e:
				# Remove exception from log message
				logger.exception("Failed to resize HNSW index. Cannot add items.")
				# Should probably raise an error here to stop the add process
				msg = "Failed to resize HNSW index"
				raise RuntimeError(msg) from e

	def add_items(self, vectors: np.ndarray, chunk_ids: list[str], metadatas: list[dict[str, Any]]) -> None:
		"""
		Add vectors and their metadata.

		Args:
		    vectors: Numpy array of shape (N, dim).
		    chunk_ids: List of N unique string IDs for the chunks.
		    metadatas: List of N metadata dictionaries corresponding to the chunks.

		"""
		if self.index is None:
			logger.error("Cannot add items: Index is not initialized.")
			return
		if not len(vectors) == len(chunk_ids) == len(metadatas):
			logger.error("Mismatch between number of vectors, chunk_ids, and metadatas.")
			return
		if len(vectors) == 0:
			return  # Nothing to add

		# Ensure metadata collections exist
		if "chunks" not in self.metadata_store:
			self.metadata_store["chunks"] = {}
		if "file_hashes" not in self.metadata_store:
			self.metadata_store["file_hashes"] = {}

		try:
			self._ensure_capacity(len(vectors))
		except RuntimeError:
			return  # Stop if resizing failed

		# Process all items as new additions
		internal_ids = []
		vectors_to_add = []

		# Process each vector/chunk
		for _i, (vector, chunk_id, metadata) in enumerate(zip(vectors, chunk_ids, metadatas, strict=True)):
			# Store metadata
			self.metadata_store["chunks"][chunk_id] = metadata

			# Update file hashes
			if "file_path" in metadata and "git_hash" in metadata and metadata["git_hash"]:
				self.metadata_store["file_hashes"][metadata["file_path"]] = metadata["git_hash"]

			# Assign a new internal ID regardless of whether the chunk ID exists
			internal_id = self.next_internal_id
			internal_ids.append(internal_id)
			vectors_to_add.append(vector)

			# Update mappings
			self.chunk_id_to_internal_id[chunk_id] = internal_id
			self.internal_id_to_chunk_id[internal_id] = chunk_id
			self.next_internal_id += 1

		# Perform batch add
		if vectors_to_add:
			try:
				# Add to HNSW index
				internal_ids_np = np.array(internal_ids, dtype=np.int64)
				vectors_np = np.array(vectors_to_add, dtype=np.float32)
				logger.info(f"Adding {len(vectors_np)} items to HNSW index...")
				self.index.add_items(vectors_np, internal_ids_np)
				logger.info(f"Successfully added {len(vectors_np)} items.")
			except Exception:
				logger.exception("Error adding items to HNSW index")

		# Save metadata after bulk operations to ensure consistency
		if vectors_to_add:
			try:
				self._save_metadata()
			except Exception:
				logger.exception("Error saving metadata after adding items")

	def knn_query(self, vector: np.ndarray, k: int = 5) -> tuple[list[str], list[float]]:
		"""
		Query for k nearest neighbors.

		Args:
		    vector: The query vector (1D numpy array).
		    k: Number of neighbors to retrieve.

		Returns:
		    A tuple containing: (list of chunk IDs, list of distances).
		    Returns empty lists if index is empty or query fails.

		"""
		if self.index is None or self.get_current_count() == 0:
			logger.warning("Cannot perform search: Index is empty or not initialized.")
			return [], []

		# Check if k is greater than the number of elements in the index
		current_count = self.get_current_count()
		if k > current_count:
			logger.warning(
				f"Requested k={k} is greater than the number of elements in the index ({current_count}). "
				f"Reducing k to {current_count}."
			)
			k = current_count

		# If k is still 0, return empty results
		if k == 0:
			logger.warning("No elements in index, returning empty results.")
			return [], []

		# Ensure vector has the correct shape and type
		if len(vector.shape) == 1:
			vector = vector.reshape(1, -1).astype(np.float32)
		elif len(vector.shape) == VECTOR_SHAPE_2D and vector.shape[0] == VECTOR_FIRST_DIM:
			vector = vector.astype(np.float32)
		else:
			logger.error(f"Invalid vector shape: {vector.shape}. Expected 1D array or 2D array with shape (1, dim).")
			return [], []

		# Make sure the vector has the correct dimension
		if vector.shape[1] != self.dim:
			logger.error(f"Vector dimension mismatch: got {vector.shape[1]}, expected {self.dim}.")
			return [], []

		try:
			# Try with a simpler query - just get 1 neighbor first to confirm it works
			logger.debug(f"Attempting simple query with k=1, ef={self.ef_query}")
			self.index.set_ef(self.ef_query)

			# Try to get just one result first to check if the index is functional
			k_test = 1
			ids, distances = self.index.knn_query(vector, k=k_test)

			# Now try the full query with the requested k
			logger.debug(f"Simple query worked, now trying full query with k={k}")

			# If we got here, we can try with the original k
			self.index.set_ef(max(self.ef_query, k * 2))  # Set ef to at least 2*k
			ids, distances = self.index.knn_query(vector, k=k)

			# Convert to flat lists
			id_list = ids[0].tolist()
			distance_list = distances[0].tolist()

			# Map internal IDs back to chunk IDs
			chunk_ids = []
			valid_distances = []
			for i, internal_id in enumerate(id_list):
				chunk_id = self.internal_id_to_chunk_id.get(internal_id)
				if chunk_id:
					chunk_ids.append(chunk_id)
					valid_distances.append(distance_list[i])
				else:
					logger.debug(f"No chunk ID found for internal ID {internal_id}")

			logger.debug(f"Search successful, returning {len(chunk_ids)} results")
			return chunk_ids, valid_distances

		except Exception:
			logger.exception("Search failed")
			# Try one more time with k=1 and very high ef as a last resort
			try:
				logger.info("Trying fallback search with k=1 and very high ef=100000")
				self.index.set_ef(100000)
				ids, distances = self.index.knn_query(vector, k=1)
				id_list = ids[0].tolist()
				distance_list = distances[0].tolist()

				# Map internal IDs back to chunk IDs
				chunk_ids = []
				valid_distances = []
				for i, internal_id in enumerate(id_list):
					chunk_id = self.internal_id_to_chunk_id.get(internal_id)
					if chunk_id:
						chunk_ids.append(chunk_id)
						valid_distances.append(distance_list[i])

				logger.info(f"Fallback search returned {len(chunk_ids)} results")
				return chunk_ids, valid_distances
			except Exception:
				logger.exception("Fallback search also failed")
				return [], []

	def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
		"""
		Search for similar chunks and return with metadata.

		Args:
		    query_vector: The query vector.
		    k: Number of results to return.

		Returns:
		    List of dictionaries with chunk data and metadata.

		"""
		# Ensure vector is correctly normalized for cosine similarity if using cosine space
		if self.space == "cosine":
			query_norm = np.linalg.norm(query_vector)
			if query_norm > 0:
				query_vector = query_vector / query_norm

		# Perform the search
		chunk_ids, distances = self.knn_query(query_vector, k=k)

		# Compute similarity scores (1 = identical, 0 = completely different)
		if self.space == "cosine":
			# Convert cosine distance to similarity score
			scores = [1 - (d / 2) for d in distances]  # Cosine distance ranges from 0-2
		else:
			# For L2 (Euclidean) normalize to 0-1 range (approximate)
			max_dist = max(distances) if distances else 1.0
			scores = [1 - (d / max_dist) for d in distances] if max_dist > 0 else []

		results = []
		for i, chunk_id in enumerate(chunk_ids):
			# Get metadata for this chunk
			metadata = self._get_metadata_for_chunk(chunk_id)

			if metadata:
				# Create result with metadata and score
				result = {
					"chunk_id": chunk_id,
					"score": round(scores[i], 4) if i < len(scores) else 0,
					"distance": round(distances[i], 4) if i < len(distances) else 0,
					"metadata": metadata,
				}
				results.append(result)
			else:
				logger.warning(f"No metadata found for chunk_id: {chunk_id}")

		logger.info(f"Found {len(results)} results for query vector")
		return results

	def _get_metadata_for_chunk(self, chunk_id: str) -> dict[str, Any] | None:
		"""Get metadata for a specific chunk ID."""
		return self.metadata_store["chunks"].get(chunk_id)

	def get_metadata(self, chunk_id: str) -> dict[str, Any] | None:
		"""
		Get metadata for a specific chunk ID.

		Args:
		    chunk_id: The chunk ID to get metadata for.

		Returns:
		    The metadata dictionary or None if not found.

		"""
		return self._get_metadata_for_chunk(chunk_id)

	def get_all_metadata(self) -> dict[str, Any]:
		"""Get all chunk metadata."""
		return self.metadata_store["chunks"].copy()

	def get_file_hashes(self) -> dict[str, str]:
		"""Get the dictionary mapping file paths to their tracked git hashes."""
		return self.metadata_store.get("file_hashes", {}).copy()

	def get_max_elements(self) -> int:
		"""Return the current maximum capacity of the index."""
		if self.index is None:
			return 0
		try:
			# Check if max_elements property exists (might vary across versions)
			if hasattr(self.index, "max_elements"):
				return self.index.max_elements
			# If not, rely on the value we tracked during init/resize
			return self.max_elements
		except AttributeError:
			logger.warning("Could not get max_elements directly from index, using stored value.")
			return self.max_elements

	def get_current_count(self) -> int:
		"""Return the number of items currently in the index."""
		if self.index is None:
			return 0
		try:
			return self.index.get_current_count()
		except Exception:
			# Remove exception from log message
			logger.exception("Error getting current count from index")
			return 0
