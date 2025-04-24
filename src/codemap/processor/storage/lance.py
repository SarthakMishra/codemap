"""LanceDB storage backend implementation for CodeMap."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import lancedb
import pandas as pd

from codemap.processor.chunking.base import Chunk
from codemap.processor.embedding.models import EmbeddingResult
from codemap.processor.storage.base import StorageBackend, StorageConfig
from codemap.processor.storage.utils import (
    chunk_to_dict,
    create_pyarrow_schema_for_chunks,
    create_pyarrow_schema_for_embeddings,
    dict_to_chunk,
    embedding_to_dict,
)

if TYPE_CHECKING:
    from lancedb.table import Table

logger = logging.getLogger(__name__)


class LanceDBStorage(StorageBackend):
    """LanceDB storage backend for CodeMap.

    This backend uses LanceDB for storing and retrieving code chunks,
    metadata, and embeddings.
    """

    # Table names
    CHUNKS_TABLE = "chunks"
    EMBEDDINGS_TABLE = "embeddings"
    FILE_HISTORY_TABLE = "file_history"

    def __init__(self, config: StorageConfig) -> None:
        """Initialize the LanceDB storage backend.

        Args:
            config: Configuration for the storage backend
        """
        super().__init__(config)
        self._db = None
        self._connection_initialized = False

    def initialize(self) -> None:
        """Initialize the LanceDB storage backend.

        This method connects to LanceDB and creates necessary tables if they don't exist.
        """
        if self._connection_initialized:
            return

        logger.info("Initializing LanceDB storage with URI: %s", self.config.uri)

        # Get connection parameters
        uri = self.config.uri
        api_key = self.config.api_key
        region = self.config.region
        kwargs = {}

        # Determine connection type (local file vs cloud)
        if uri.startswith("db://"):
            # Cloud LanceDB
            if not api_key:
                msg = "API key is required for cloud LanceDB"
                raise ValueError(msg)

            kwargs["api_key"] = api_key
            if region:
                kwargs["region"] = region
        else:
            # Local LanceDB
            # Ensure directory exists
            Path(uri).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        try:
            self._db = lancedb.connect(uri, **kwargs)
            logger.info("Connected to LanceDB at %s", uri)
            self._connection_initialized = True

            # Create tables if they don't exist
            self._create_tables_if_needed()
        except Exception as e:
            logger.exception("Failed to connect to LanceDB")
            msg = f"Failed to connect to LanceDB: {e}"
            raise RuntimeError(msg) from e

    def _create_tables_if_needed(self) -> None:
        """Create necessary tables if they don't exist."""
        if not self._db:
            msg = "Database connection not initialized"
            raise RuntimeError(msg)

        existing_tables = self._db.table_names()

        # Create chunks table if it doesn't exist
        if self.CHUNKS_TABLE not in existing_tables:
            logger.info("Creating chunks table")
            schema = create_pyarrow_schema_for_chunks()
            # Create empty table with schema
            self._db.create_table(self.CHUNKS_TABLE, schema=schema, mode="create")
            # Create indices for efficient queries
            chunks_table = self._db.open_table(self.CHUNKS_TABLE)
            # Create indices for common query fields
            self._create_indices(chunks_table, ["file_path", "language", "entity_type"])

        # Create embeddings table if it doesn't exist
        if self.EMBEDDINGS_TABLE not in existing_tables:
            logger.info("Creating embeddings table")
            schema = create_pyarrow_schema_for_embeddings()
            self._db.create_table(self.EMBEDDINGS_TABLE, schema=schema, mode="create")
            # Create vector index for embeddings
            embeddings_table = self._db.open_table(self.EMBEDDINGS_TABLE)
            self._create_vector_index(embeddings_table)

        # Create file history table if it doesn't exist
        if self.FILE_HISTORY_TABLE not in existing_tables:
            logger.info("Creating file history table")
            # Create file history table with schema
            # We skip using pyarrow directly since it's causing linter issues
            history_data = {
                "file_path": [],
                "commit_id": [],
                "timestamp": [],
                "is_deleted": [],
            }
            history_df = pd.DataFrame(history_data)
            self._db.create_table(self.FILE_HISTORY_TABLE, data=history_df, mode="create")
            # Create index on file_path
            file_history_table = self._db.open_table(self.FILE_HISTORY_TABLE)
            self._create_indices(file_history_table, ["file_path"])

    def _create_indices(self, table: Table, columns: list[str]) -> None:
        """Create indices on specified columns.

        Args:
            table: LanceDB table
            columns: List of columns to create indices on
        """
        # Create indices for all columns without having try-except in the loop
        for column in columns:
            try_create_index(table, column)

    def _create_vector_index(self, table: Table) -> None:
        """Create vector index on embedding column.

        Args:
            table: LanceDB table containing embeddings
        """
        try:
            # Create vector index for fast similarity search
            # Using string literals for compatibility
            table.create_index(
                "embedding",
                index_type="IVF_FLAT",  # Use LanceDB compatible index types
                vector_column_name="embedding",
            )
        except (ValueError, RuntimeError, TypeError) as e:
            # Use specific error types instead of broad Exception
            logger.warning("Failed to create vector index: %s", e)

    def close(self) -> None:
        """Close the LanceDB storage backend."""
        logger.info("Closing LanceDB storage")
        self._db = None
        self._connection_initialized = False

    def store_chunks(self, chunks: Sequence[Chunk], commit_id: str | None = None) -> None:
        """Store code chunks.

        Args:
            chunks: Sequence of code chunks to store
            commit_id: Optional Git commit ID to associate with the chunks
        """
        if not chunks:
            return

        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot store chunks")
            return

        chunks_table = self._db.open_table(self.CHUNKS_TABLE)

        # Convert chunks to dictionaries
        chunk_dicts = [chunk_to_dict(chunk, commit_id) for chunk in chunks]

        # Store chunks
        chunks_df = pd.DataFrame(chunk_dicts)

        # Upsert chunks (insert or update based on id)
        try:
            # LanceDB supports merging via 'mode' parameter
            chunks_table.add(chunks_df, mode="overwrite")
            logger.info("Stored %d chunks", len(chunks))

            # Also update file history
            self._update_file_history(chunks, commit_id)
        except Exception:
            logger.exception("Failed to store chunks")
            raise

    def _update_file_history(self, chunks: Sequence[Chunk], commit_id: str | None = None) -> None:
        """Update file history tracking.

        Args:
            chunks: Sequence of code chunks
            commit_id: Optional Git commit ID
        """
        if not commit_id:
            return  # Skip history tracking if no commit_id provided

        if not self._db:
            return

        # Get unique file paths
        file_paths = set()
        for chunk in chunks:
            file_path = str(chunk.metadata.location.file_path)
            file_paths.add(file_path)

        # Create history records
        timestamp = datetime.now(timezone.utc).isoformat()

        # Use list comprehension for better performance
        history_records = [
            {
                "file_path": file_path,
                "commit_id": commit_id,
                "timestamp": timestamp,
                "is_deleted": False,
            }
            for file_path in file_paths
        ]

        # Store history records
        if history_records:
            history_table = self._db.open_table(self.FILE_HISTORY_TABLE)
            history_df = pd.DataFrame(history_records)
            history_table.add(history_df)

    def store_embeddings(self, embeddings: Sequence[EmbeddingResult]) -> None:
        """Store embeddings for code chunks.

        Args:
            embeddings: Sequence of embedding results to store
        """
        if not embeddings:
            return

        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot store embeddings")
            return

        embeddings_table = self._db.open_table(self.EMBEDDINGS_TABLE)
        chunks_table = self._db.open_table(self.CHUNKS_TABLE)

        # Convert embeddings to dictionaries
        embedding_dicts = []
        chunk_updates = []

        for embedding in embeddings:
            # Get chunk ID from chunk object or from embedding
            chunk_id = embedding.chunk_id
            if not chunk_id:
                logger.warning("Embedding without chunk ID, skipping")
                continue

            # Store embedding
            embedding_dict = embedding_to_dict(embedding, chunk_id)
            embedding_dicts.append(embedding_dict)

            # Prepare chunk update with vector
            chunk_updates.append(
                {
                    "id": chunk_id,
                    "vector": embedding_dict["embedding"],
                }
            )

        # Store embeddings
        if embedding_dicts:
            embeddings_df = pd.DataFrame(embedding_dicts)
            embeddings_table.add(embeddings_df, mode="overwrite")
            logger.info("Stored %d embeddings", len(embedding_dicts))

        # Update chunks with vectors
        if chunk_updates:
            chunks_df = pd.DataFrame(chunk_updates)
            chunks_table.add(chunks_df, mode="overwrite")
            logger.info("Updated %d chunks with vectors", len(chunk_updates))

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            The chunk if found, None otherwise
        """
        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot retrieve chunk")
            return None

        chunks_table = self._db.open_table(self.CHUNKS_TABLE)

        # Query for the chunk
        results = chunks_table.search().where(f"id = '{chunk_id}'").to_pandas()

        if results.empty:
            return None

        # Convert to Chunk object
        chunk_data = results.iloc[0].to_dict()
        return dict_to_chunk(chunk_data)

    def get_chunks_by_file(self, file_path: str, commit_id: str | None = None) -> list[Chunk]:
        """Retrieve all chunks for a file.

        Args:
            file_path: Path to the file
            commit_id: Optional Git commit ID to filter by

        Returns:
            List of chunks for the file
        """
        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot retrieve chunks")
            return []

        chunks_table = self._db.open_table(self.CHUNKS_TABLE)

        # Build query
        query = f"file_path = '{file_path}'"
        if commit_id:
            # This works because we store commit_id in chunk_id
            query += f" AND id LIKE '%:{commit_id}'"

        # Query for chunks
        results = chunks_table.search().where(query).to_pandas()

        if results.empty:
            return []

        # Convert to Chunk objects
        chunks = []
        for _, row in results.iterrows():
            chunk_data = row.to_dict()
            chunk = dict_to_chunk(chunk_data)
            chunks.append(chunk)

        # Reconstruct parent-child relationships
        self._restore_chunk_hierarchy(chunks)

        return chunks

    def _restore_chunk_hierarchy(self, chunks: list[Chunk]) -> None:
        """Restore parent-child relationships between chunks.

        Args:
            chunks: List of chunks to process
        """
        # Create mapping of full_name to chunk
        chunk_map = {chunk.full_name: chunk for chunk in chunks}

        # Set parent-child relationships
        for chunk in chunks:
            parent_id = getattr(chunk, "_parent_id", None)
            if parent_id and parent_id in chunk_map:
                # Chunk objects are frozen, so we need to recreate them
                parent = chunk_map[parent_id]
                # This is a workaround since Chunk is frozen
                object.__setattr__(chunk, "parent", parent)

                # Add chunk to parent's children
                children = list(parent.children)
                children.append(chunk)
                object.__setattr__(parent, "children", tuple(children))

    def search_by_content(self, query: str, limit: int = 10) -> list[tuple[Chunk, float]]:
        """Search for chunks by content.

        This is a text-based search, not a semantic search.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of (chunk, score) tuples, sorted by score (highest first)
        """
        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot search content")
            return []

        chunks_table = self._db.open_table(self.CHUNKS_TABLE)

        # Perform full-text search
        # Note: This depends on LanceDB's capabilities; we'll use a fallback approach
        try:
            # Try direct search if supported
            results = chunks_table.search(query).limit(limit).to_pandas()
        except (ValueError, NotImplementedError, RuntimeError) as e:
            # Use specific exception types instead of broad Exception
            logger.warning("Full-text search failed, falling back to LIKE query: %s", e)
            # Fallback to LIKE query
            results = chunks_table.search().where(f"content LIKE '%{query}%'").limit(limit).to_pandas()

        if results.empty:
            return []

        # Convert to Chunk objects with scores
        chunks_with_scores = []
        for _, row in results.iterrows():
            chunk_data = row.to_dict()
            chunk = dict_to_chunk(chunk_data)

            # Get score from results if available, otherwise use 1.0
            score = row.get("_score", 1.0)
            chunks_with_scores.append((chunk, score))

        return chunks_with_scores

    def search_by_vector(self, vector: list[float], limit: int = 10) -> list[tuple[Chunk, float]]:
        """Search for chunks by vector similarity.

        Args:
            vector: Query vector
            limit: Maximum number of results to return

        Returns:
            List of (chunk, score) tuples, sorted by score (highest first)
        """
        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot search by vector")
            return []

        chunks_table = self._db.open_table(self.CHUNKS_TABLE)

        # Ensure vector is in the correct format
        query_vector = vector

        # Perform vector search
        results = chunks_table.search(query_vector, vector_column_name="vector").limit(limit).to_pandas()

        if results.empty:
            return []

        # Convert to Chunk objects with scores
        chunks_with_scores = []
        for _, row in results.iterrows():
            chunk_data = row.to_dict()
            chunk = dict_to_chunk(chunk_data)

            # Get score from results
            score = row.get("_distance", 0.5)
            # Convert distance to similarity score (1.0 - distance)
            # This assumes cosine distance; adjust for other metrics
            similarity = 1.0 - score
            chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def search_hybrid(
        self, query: str, vector: list[float], limit: int = 10, weight: float = 0.5
    ) -> list[tuple[Chunk, float]]:
        """Hybrid search combining text and vector similarity.

        Args:
            query: Text search query
            vector: Vector search query
            limit: Maximum number of results to return
            weight: Weight between text (0.0) and vector (1.0) search

        Returns:
            List of (chunk, score) tuples, sorted by score (highest first)
        """
        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot perform hybrid search")
            return []

        # For now, we'll implement a simplified hybrid search approach
        # by performing vector search and filtering results
        try:
            # Get vector search results
            vector_results = self.search_by_vector(vector, limit * 2)  # Get more results to filter

            if not vector_results:
                # Fall back to text search if vector search returns nothing
                return self.search_by_content(query, limit)

            # Filter results based on the text query
            filtered_results = []
            for chunk, score in vector_results:
                if query.lower() in chunk.content.lower():
                    # Boost score for text matches
                    adjusted_score = score * (1 - weight) + weight
                    filtered_results.append((chunk, adjusted_score))
                else:
                    # Keep vector score for non-text matches
                    filtered_results.append((chunk, score * (1 - weight)))

            # Sort by score and limit results
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            return filtered_results[:limit]

        except (ValueError, RuntimeError) as e:
            # Use specific exception types instead of broad Exception
            logger.warning("Hybrid search failed, falling back to vector search: %s", e)
            # Fallback to vector-only search
            return self.search_by_vector(vector, limit)

    def delete_file(self, file_path: str) -> None:
        """Delete all chunks for a file.

        Args:
            file_path: Path to the file
        """
        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot delete file")
            return

        chunks_table = self._db.open_table(self.CHUNKS_TABLE)
        embeddings_table = self._db.open_table(self.EMBEDDINGS_TABLE)

        # Get chunk IDs for this file
        results = chunks_table.search().where(f"file_path = '{file_path}'").select(["id"]).to_pandas()

        if results.empty:
            logger.info("No chunks found for file %s", file_path)
            return

        chunk_ids = results["id"].tolist()

        # Delete chunks
        for chunk_id in chunk_ids:
            chunks_table.delete(f"id = '{chunk_id}'")
            embeddings_table.delete(f"chunk_id = '{chunk_id}'")

        logger.info("Deleted %d chunks for file %s", len(chunk_ids), file_path)

        # Update file history to mark the file as deleted
        self._mark_file_deleted(file_path)

    def _mark_file_deleted(self, file_path: str) -> None:
        """Mark a file as deleted in file history.

        Args:
            file_path: Path to the file
        """
        if not self._db:
            return

        history_table = self._db.open_table(self.FILE_HISTORY_TABLE)

        # Add deletion record
        history_table.add(
            pd.DataFrame(
                [
                    {
                        "file_path": file_path,
                        "commit_id": "",  # No commit ID for manual deletions
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "is_deleted": True,
                    }
                ]
            )
        )

    def get_file_history(self, file_path: str) -> list[tuple[datetime, str]]:
        """Get history of a file.

        Args:
            file_path: Path to the file

        Returns:
            List of (timestamp, commit_id) tuples, sorted by timestamp (newest first)
        """
        if not self._connection_initialized:
            self.initialize()

        if not self._db:
            logger.warning("No database connection, cannot retrieve file history")
            return []

        history_table = self._db.open_table(self.FILE_HISTORY_TABLE)

        # Query for file history
        results = (
            history_table.search().where(f"file_path = '{file_path}'").sort("timestamp", ascending=False).to_pandas()
        )

        if results.empty:
            return []

        # Convert to list of (timestamp, commit_id) tuples
        history = []
        for _, row in results.iterrows():
            timestamp = datetime.fromisoformat(row["timestamp"])
            commit_id = row["commit_id"]
            history.append((timestamp, commit_id))

        return history


def try_create_index(table: Table, column: str) -> None:
    """Create index on a column, handling errors outside the loop.

    Args:
        table: LanceDB table
        column: Column name to create index on
    """
    try:
        table.create_index(column)
    except (ValueError, RuntimeError, TypeError) as e:
        # Use specific error types instead of broad Exception
        logger.warning("Failed to create index on %s: %s", column, e)
