"""Pipeline orchestration for code processing.

This module defines the main processing pipeline that orchestrates:
1. File watching (via watcher module)
2. Parsing/chunking files on change
3. Analyzing code and extracting metadata
4. Generating embeddings
5. Storing processed data
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from codemap.processor.analysis.git.analyzer import GitMetadataAnalyzer
from codemap.processor.analysis.git.models import GitMetadata
from codemap.processor.analysis.lsp.analyzer import LSPAnalyzer
from codemap.processor.analysis.lsp.models import LSPMetadata
from codemap.processor.chunking.base import Chunk
from codemap.processor.chunking.tree_sitter import TreeSitterChunker
from codemap.processor.embedding.generator import EmbeddingGenerator
from codemap.processor.embedding.models import EmbeddingConfig
from codemap.processor.storage.base import StorageConfig
from codemap.processor.storage.lance import LanceDBStorage
from codemap.processor.watcher import FileWatcher
from codemap.utils.file_utils import read_file_content

logger = logging.getLogger(__name__)


@dataclass
class ProcessingJob:
    """Represents a file processing job in the pipeline."""

    file_path: Path
    """Path to the file being processed."""

    is_deletion: bool = False
    """Whether this job is for a file deletion."""

    chunks: list[Chunk] = field(default_factory=list)
    """Chunks extracted from the file (if not a deletion)."""

    lsp_metadata: dict[str, LSPMetadata] = field(default_factory=dict)
    """LSP metadata for chunks, keyed by chunk full name."""

    started_at: float = field(default_factory=time.time)
    """Time when processing started."""

    completed_at: float | None = None
    """Time when processing completed, or None if still in progress."""

    error: Exception | None = None
    """Error that occurred during processing, if any."""


class ProcessingPipeline:
    """Main pipeline for code processing and analysis.

    This class orchestrates the complete flow from file changes to processed data,
    connecting file watching with code analysis, chunking, embedding, and storage.
    """

    def __init__(
        self,
        repo_path: Path,
        storage_config: StorageConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
        ignored_patterns: set[str] | None = None,
        max_workers: int = 4,
        enable_lsp: bool = True,
        on_chunks_processed: Callable[[list[Chunk], Path], None] | None = None,
        on_file_deleted: Callable[[Path], None] | None = None,
    ) -> None:
        """Initialize the processing pipeline.

        Args:
            repo_path: Path to the repository root
            storage_config: Configuration for storage backend, if None a default local LanceDB config will be used
            embedding_config: Configuration for embedding generation, if None a default config will be used
            ignored_patterns: Set of glob patterns to ignore when watching for changes
            max_workers: Maximum number of worker threads for processing
            enable_lsp: Whether to enable LSP analysis (True by default)
            on_chunks_processed: Callback when chunks are processed for a file
            on_file_deleted: Callback when a file is deleted
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
        self.chunker = TreeSitterChunker()
        self.git_analyzer = GitMetadataAnalyzer(repo_path)

        # Initialize LSP analyzer if enabled
        self.enable_lsp = enable_lsp
        self.lsp_analyzer = LSPAnalyzer(repo_path) if enable_lsp else None

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(embedding_config)

        # Initialize storage
        if storage_config is None:
            # Default to a local LanceDB database in the repo
            storage_dir = repo_path / ".codemap" / "storage"
            storage_config = StorageConfig(uri=str(storage_dir))

        self.storage = LanceDBStorage(storage_config)

        # Try to initialize storage early to catch any issues
        try:
            self.storage.initialize()
        except (RuntimeError, ValueError, ConnectionError) as e:
            logger.warning("Failed to initialize storage: %s", e)
            logger.warning("Storage operations will be attempted at runtime")

        # Callback handlers
        self.on_chunks_processed = on_chunks_processed
        self.on_file_deleted = on_file_deleted

        # Threading setup
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Processing state
        self.active_jobs: dict[Path, ProcessingJob] = {}
        self.watcher: FileWatcher | None = None

    def start(self) -> None:
        """Start the processing pipeline, including file watching."""
        logger.info("Starting processing pipeline for repository: %s", self.repo_path)

        # Initialize file watcher with callbacks
        self.watcher = FileWatcher(
            paths=self.repo_path,
            on_created=self._handle_file_created,
            on_modified=self._handle_file_modified,
            on_deleted=self._handle_file_deleted,
            ignored_patterns=self.ignored_patterns,
            recursive=True,
        )

        # Start file watching
        self.watcher.start()
        logger.info("File watcher started")

    def stop(self) -> None:
        """Stop the processing pipeline and clean up resources."""
        logger.info("Stopping processing pipeline")

        # Stop file watcher
        if self.watcher:
            self.watcher.stop()

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        # Close LSP analyzer if enabled
        if self.lsp_analyzer:
            try:
                self.lsp_analyzer.close()
            except Exception:
                logger.exception("Error closing LSP analyzer")

        # Close storage
        if hasattr(self, "storage"):
            try:
                self.storage.close()
            except Exception:
                logger.exception("Error closing storage")

        logger.info("Processing pipeline stopped")

    def process_file(self, file_path: str | Path) -> None:
        """Process a single file.

        Args:
            file_path: Path to the file to process
        """
        path_obj = Path(file_path)

        # Create a job record
        job = ProcessingJob(file_path=path_obj)
        self.active_jobs[path_obj] = job

        # Submit to thread pool
        self.executor.submit(self._process_file_worker, path_obj)

    def _process_file_worker(self, file_path: Path) -> None:
        """Worker function to process a file in a separate thread.

        Args:
            file_path: Path to the file to process
        """
        job = self.active_jobs.get(file_path)
        if not job:
            return

        try:
            logger.debug("Processing file: %s", file_path)

            # Read the file content
            content = read_file_content(file_path)

            # Get Git metadata for the file
            git_metadata = GitMetadata(
                is_committed=True,  # Assuming the file is committed
                commit_id=self.git_analyzer.get_current_commit(),
                commit_message="",  # Simplified - not fetching commit message
                timestamp=datetime.now(UTC),  # Simplified
                branch=[self.git_analyzer.get_current_branch()],
            )

            # Chunk the file
            chunks = list(self.chunker.chunk(content, file_path, git_metadata))

            # Update job with results
            job.chunks = chunks

            # Enrich with LSP metadata if enabled
            lsp_metadata = {}
            if self.enable_lsp and self.lsp_analyzer and chunks:
                try:
                    logger.debug("Enriching chunks with LSP metadata for %s", file_path)
                    lsp_metadata = self.lsp_analyzer.enrich_chunks(chunks)
                    job.lsp_metadata = lsp_metadata
                    logger.debug("Enriched %d chunks with LSP metadata for %s", len(lsp_metadata), file_path)
                except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                    logger.warning("Error enriching chunks with LSP for %s: %s", file_path, e)

            # Generate embeddings for the chunks
            # Note: We could potentially enhance the content with LSP info for better embeddings
            embeddings = self.embedding_generator.generate_embeddings(chunks)

            # Store chunks and embeddings
            try:
                commit_id = git_metadata.commit_id
                self.storage.store_chunks(chunks, commit_id)

                # Store LSP metadata if available
                if lsp_metadata:
                    self.storage.store_lsp_metadata(lsp_metadata, chunks, commit_id)

                if embeddings:
                    self.storage.store_embeddings(embeddings)

                logger.debug(
                    "Stored %d chunks, %d LSP metadata, and %d embeddings for %s",
                    len(chunks),
                    len(lsp_metadata),
                    len(embeddings),
                    file_path,
                )
            except Exception:
                logger.exception("Error storing data for %s", file_path)

            job.completed_at = time.time()

            # Call callback if provided
            if self.on_chunks_processed:
                self.on_chunks_processed(chunks, file_path)

            logger.debug("Completed processing file: %s", file_path)

        except Exception as e:
            logger.exception("Error processing file %s", file_path)
            job.error = e
            job.completed_at = time.time()

        finally:
            # Clean up finished job after a delay
            # (keeping it around briefly for status checks)
            self.executor.submit(self._cleanup_job, file_path, delay=60)

    def _cleanup_job(self, file_path: Path, delay: float = 0) -> None:
        """Clean up a completed job after an optional delay.

        Args:
            file_path: Path to the file/job to clean up
            delay: Time in seconds to wait before cleanup
        """
        if delay > 0:
            time.sleep(delay)
        self.active_jobs.pop(file_path, None)

    def _handle_file_created(self, file_path: str) -> None:
        """Handle file creation events from the watcher.

        Args:
            file_path: Path to the created file
        """
        logger.debug("File created: %s", file_path)
        self.process_file(file_path)

    def _handle_file_modified(self, file_path: str) -> None:
        """Handle file modification events from the watcher.

        Args:
            file_path: Path to the modified file
        """
        logger.debug("File modified: %s", file_path)
        self.process_file(file_path)

    def _handle_file_deleted(self, file_path: str) -> None:
        """Handle file deletion events from the watcher.

        Args:
            file_path: Path to the deleted file
        """
        path_obj = Path(file_path)
        logger.debug("File deleted: %s", file_path)

        # Create a deletion job record
        job = ProcessingJob(file_path=path_obj, is_deletion=True)
        self.active_jobs[path_obj] = job

        # Delete from storage
        try:
            self.storage.delete_file(file_path)
            logger.debug("Deleted %s from storage", file_path)
        except Exception as deletion_error:
            logger.exception("Error deleting %s from storage", file_path)
            job.error = deletion_error

        # Call callback if provided
        if self.on_file_deleted:
            try:
                self.on_file_deleted(path_obj)
            except Exception as e:
                logger.exception("Error handling file deletion for %s", file_path)
                job.error = e

        job.completed_at = time.time()

    def get_job_status(self, file_path: str | Path) -> ProcessingJob | None:
        """Get the status of a processing job.

        Args:
            file_path: Path to the file

        Returns:
            Processing job status if available, None otherwise
        """
        path_obj = Path(file_path)
        return self.active_jobs.get(path_obj)

    def batch_process(self, paths: list[str | Path]) -> None:
        """Process multiple files in batch.

        Args:
            paths: List of file paths to process
        """
        for path in paths:
            self.process_file(path)

    def search(self, query: str, limit: int = 10, use_vector: bool = True) -> list[tuple[Chunk, float]]:
        """Search the codebase using the storage backend.

        If use_vector is True, the query will be transformed into a vector
        and used for semantic search. Otherwise, plain text search will be used.

        Args:
            query: Search query
            limit: Maximum number of results to return
            use_vector: Whether to use vector search

        Returns:
            List of (chunk, score) tuples, sorted by score (highest first)
        """
        if use_vector:
            # Generate a vector for the query using the embedding generator
            try:
                # Create a dummy chunk for the query
                from codemap.processor.chunking.base import ChunkMetadata, EntityType, Location

                dummy_location = Location(
                    file_path=Path("query"),
                    start_line=1,
                    end_line=1,
                )

                dummy_metadata = ChunkMetadata(
                    entity_type=EntityType.UNKNOWN,
                    name="query",
                    location=dummy_location,
                    language="text",
                )

                query_chunk = Chunk(content=query, metadata=dummy_metadata)

                # Generate embedding
                embedding = self.embedding_generator.generate_embedding(query_chunk)

                if embedding and embedding.embedding is not None:
                    # Convert to list if needed
                    vector = embedding.embedding
                    if hasattr(vector, "tolist"):
                        vector = vector.tolist()

                    # Ensure we have a list of floats as required by the API
                    if isinstance(vector, list):
                        # Search by vector
                        return self.storage.search_by_vector(vector, limit)
            except Exception:
                logger.exception("Error during vector search")

        # Fallback to text search or if use_vector is False
        return self.storage.search_by_text(query, limit)
