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
from codemap.processor.chunking.base import Chunk
from codemap.processor.chunking.tree_sitter import TreeSitterChunker
from codemap.utils.file_utils import read_file_content
from codemap.watcher.watcher import FileWatcher

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
        ignored_patterns: set[str] | None = None,
        max_workers: int = 4,
        on_chunks_processed: Callable[[list[Chunk], Path], None] | None = None,
        on_file_deleted: Callable[[Path], None] | None = None,
    ) -> None:
        """Initialize the processing pipeline.

        Args:
            repo_path: Path to the repository root
            ignored_patterns: Set of glob patterns to ignore when watching for changes
            max_workers: Maximum number of worker threads for processing
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

        # Call callback if provided
        if self.on_file_deleted:
            try:
                self.on_file_deleted(path_obj)
            except Exception as e:
                logger.exception("Error handling file deletion for %s", file_path)
                job.error = e

        job.completed_at = time.time()

        # Clean up deletion job after a delay
        self.executor.submit(self._cleanup_job, path_obj, delay=60)

    def get_job_status(self, file_path: str | Path) -> ProcessingJob | None:
        """Get the status of a processing job.

        Args:
            file_path: Path to the file to check

        Returns:
            The processing job if found, None otherwise
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
