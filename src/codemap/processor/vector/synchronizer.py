"""Module for synchronizing HNSW index with Git state."""

import logging
from pathlib import Path

import numpy as np
from rich.progress import Progress, TaskID

from codemap.processor.utils.embedding_utils import generate_embedding
from codemap.processor.utils.git_utils import get_git_tracked_files
from codemap.processor.vector.chunking import TreeSitterChunker
from codemap.processor.vector.hnsw_manager import HNSWManager

logger = logging.getLogger(__name__)


class VectorSynchronizer:
	"""Handles synchronization between Git repository and HNSW vector index."""

	def __init__(
		self,
		repo_path: Path,
		hnsw_manager: HNSWManager,
		chunker: TreeSitterChunker,
		embedding_model_name: str = "multilingual-e5-small",
	) -> None:
		"""
		Initialize the vector synchronizer.

		Args:
		    repo_path: Path to the git repository root.
		    hnsw_manager: Instance of HNSWManager to handle vector storage.
		    chunker: Instance of chunker used to create code chunks.
		    embedding_model_name: Name of the embedding model to use.

		"""
		self.repo_path = repo_path
		self.hnsw_manager = hnsw_manager
		self.chunker = chunker
		self.embedding_model_name = embedding_model_name

	def sync_index(self, progress: Progress | None = None, task_id: TaskID | None = None) -> bool:
		"""
		Synchronize the HNSW index with the current repository state.

		Args:
		    progress: Optional rich Progress instance for UI updates.
		    task_id: Optional rich TaskID for progress tracking.

		Returns:
		    True if synchronization completed successfully, False otherwise.

		"""
		try:
			logger.info("Starting vector index synchronization...")

			# Initialize a new index instead of updating
			try:
				# Create a new index object
				self.hnsw_manager.reset_index()
				logger.info("Initialized new HNSW index for complete rebuild")
			except Exception:
				logger.exception("Failed to initialize new index")
				return False

			if progress and task_id is not None:
				progress.update(task_id, completed=10, description="Analyzing repository files...")

			# Get all current files from Git
			git_files = get_git_tracked_files(self.repo_path) or {}

			if not git_files:
				logger.warning("No tracked files found in the repository.")
				if progress and task_id is not None:
					progress.update(task_id, completed=100, description="No files to process.")
				return True

			logger.info(f"Found {len(git_files)} tracked files in the repository.")

			# Process all files
			file_paths = [self.repo_path / file_path for file_path in git_files]

			# Setup progress tracking
			total_files = len(file_paths)
			if progress and task_id is not None:
				progress.update(task_id, completed=20, description=f"Processing {total_files} files...")

			# Process files in batches to avoid memory issues
			batch_size = 10
			progress_increment = 60 / (total_files // batch_size + 1)  # Spread 20-80% progress range
			current_progress = 20

			# Process all files in batches
			for batch_idx in range(0, len(file_paths), batch_size):
				batch_files = file_paths[batch_idx : batch_idx + batch_size]

				# Update progress
				if progress and task_id is not None:
					current_progress += progress_increment
					progress.update(
						task_id,
						completed=min(80, int(current_progress)),
						description=(
							f"Processing files {batch_idx + 1}-{min(batch_idx + batch_size, total_files)} "
							f"of {total_files}..."
						),
					)

				# Get chunks and embeddings for this batch
				all_chunks = []

				for file_path in batch_files:
					try:
						file_chunks = list(
							self.chunker.chunk_file(
								file_path,
								git_hash=git_files.get(str(file_path.relative_to(self.repo_path)), None),
							)
						)
						all_chunks.extend(file_chunks)
					except Exception:
						logger.exception(f"Error processing file {file_path}")

				# Skip if no chunks were generated
				if not all_chunks:
					continue

				# Generate embeddings for all chunks in this batch
				texts = [chunk["content"] for chunk in all_chunks]
				try:
					embeddings = np.array([generate_embedding(text) for text in texts], dtype=np.float32)
				except Exception:
					logger.exception("Error generating embeddings")
					continue

				if embeddings.size == 0:
					continue

				# Add to the index
				chunk_ids = []
				metadatas = []

				for chunk in all_chunks:
					file_path = chunk["metadata"]["file_path"]
					start_line = chunk["metadata"]["start_line"]
					end_line = chunk["metadata"]["end_line"]
					chunk_id = f"{file_path}:{start_line}-{end_line}"
					metadatas.append(chunk["metadata"])
					chunk_ids.append(chunk_id)

				# Add items to the index
				try:
					self.hnsw_manager.add_items(embeddings, chunk_ids, metadatas)
					logger.info(f"Added {len(chunk_ids)} chunks to the index")
				except Exception:
					logger.exception("Error adding chunks to index")

			# Save the index and metadata
			try:
				self.hnsw_manager.save()
				logger.info("HNSW index and metadata saved")
			except Exception:
				logger.exception("Error saving HNSW index and metadata")
				return False

			if progress and task_id is not None:
				progress.update(
					task_id,
					completed=100,
					description=f"[green]✓[/green] Synchronized {self.hnsw_manager.get_current_count()} chunks.",
				)

			logger.info(f"Vector index synchronization completed with {self.hnsw_manager.get_current_count()} chunks.")
			return True

		except Exception:
			logger.exception("Unhandled error during vector index synchronization")
			if progress and task_id is not None:
				progress.update(task_id, completed=100, description="[red]✗[/red] Synchronization failed.")
			return False
