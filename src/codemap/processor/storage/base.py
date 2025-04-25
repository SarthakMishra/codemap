"""Base storage interfaces and models for the CodeMap application."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from codemap.processor.chunking.base import Chunk
from codemap.processor.embedding.models import EmbeddingResult

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for storage backends.

    Contains settings for connecting to and configuring storage
    backends. Different backends may use different subset of these
    settings.

    """

    uri: str
    """URI for the storage backend (e.g., file path, DB connection string)."""

    create_if_missing: bool = True
    """Whether to create storage if it doesn't exist."""

    read_only: bool = False
    """Whether the storage is read-only."""

    cache_dir: Path | None = None
    """Directory for storage cache, if supported."""

    additional_config: dict[str, Any] = field(default_factory=dict)
    """Additional backend-specific configuration."""

    api_key: str | None = None
    """API key for cloud storage services, if needed."""

    region: str | None = None
    """Cloud region for cloud storage services, if needed."""


class StorageBackend(abc.ABC):
    """Abstract base class for storage backends.

    This class defines the interface that all storage backends must implement.
    It provides methods for storing and retrieving:
    - Code chunks and their metadata
    - Vector embeddings for semantic search
    - Historical versions of code chunks

    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize the storage backend.

        Args:
            config: Configuration for the storage backend

        """
        self.config = config

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend.

        This method should set up any necessary tables, indices, etc.

        """

    @abc.abstractmethod
    def close(self) -> None:
        """Close the storage backend.

        This method should release any resources held by the backend.

        """

    @abc.abstractmethod
    def store_chunks(self, chunks: Sequence[Chunk], commit_id: str | None = None) -> None:
        """Store code chunks.

        Args:
            chunks: Sequence of code chunks to store
            commit_id: Optional Git commit ID to associate with the chunks

        """

    @abc.abstractmethod
    def store_embeddings(self, embeddings: Sequence[EmbeddingResult]) -> None:
        """Store embeddings for code chunks.

        Args:
            embeddings: Sequence of embedding results to store

        """

    @abc.abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            The chunk if found, None otherwise

        """

    @abc.abstractmethod
    def get_chunks_by_file(self, file_path: str, commit_id: str | None = None) -> list[Chunk]:
        """Retrieve all chunks for a file.

        Args:
            file_path: Path to the file
            commit_id: Optional Git commit ID to filter by

        Returns:
            List of chunks for the file

        """

    @abc.abstractmethod
    def search_by_content(self, query: str, limit: int = 10) -> list[tuple[Chunk, float]]:
        """Search for chunks by content.

        This is a text-based search, not a semantic search.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of (chunk, score) tuples, sorted by score (highest first)

        """

    @abc.abstractmethod
    def search_by_vector(self, vector: list[float], limit: int = 10) -> list[tuple[Chunk, float]]:
        """Search for chunks by vector similarity.

        Args:
            vector: Query vector
            limit: Maximum number of results to return

        Returns:
            List of (chunk, score) tuples, sorted by score (highest first)

        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def delete_file(self, file_path: str) -> None:
        """Delete all chunks for a file.

        Args:
            file_path: Path to the file

        """

    @abc.abstractmethod
    def get_file_history(self, file_path: str) -> list[tuple[datetime, str]]:
        """Get history of a file.

        Args:
            file_path: Path to the file

        Returns:
            List of (timestamp, commit_id) tuples, sorted by timestamp (newest first)

        """
