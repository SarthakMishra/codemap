"""Module for synchronizing HNSW index with Git state."""

import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import voyageai
from rich.progress import Progress, TaskID

from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer
from codemap.processor.utils.embedding_utils import generate_embeddings_batch
from codemap.processor.utils.git_utils import _should_exclude_path, get_git_tracked_files
from codemap.processor.vector.chunking import CodeChunk, TreeSitterChunker
from codemap.processor.vector.qdrant_manager import QdrantManager, create_qdrant_point
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Define a default maximum token limit per batch for Voyage AI
# This can be overridden in config if needed. Add a safety margin.
# Reduced token limit to attempt to avoid timeouts
DEFAULT_VOYAGE_TOKEN_LIMIT = 80000  # Max is 120k, reduced from 110k


class VectorSynchronizer:
	"""Handles asynchronous synchronization between Git repository and Qdrant vector index."""

	def __init__(
		self,
		repo_path: Path,
		qdrant_manager: QdrantManager,
		chunker: TreeSitterChunker,
		embedding_model_name: str,
		analyzer: TreeSitterAnalyzer | None = None,
		config_loader: ConfigLoader | None = None,
	) -> None:
		"""
		Initialize the vector synchronizer.

		Args:
		    repo_path: Path to the git repository root.
		    qdrant_manager: Instance of QdrantManager to handle vector storage.
		    chunker: Instance of chunker used to create code chunks.
		    embedding_model_name: Name of the embedding model to use.
		    analyzer: Optional TreeSitterAnalyzer instance.
		    config_loader: Configuration loader instance.

		"""
		self.repo_path = repo_path
		self.qdrant_manager = qdrant_manager
		self.chunker = chunker
		self.embedding_model_name = embedding_model_name
		self.analyzer = analyzer or TreeSitterAnalyzer()
		self.config_loader = config_loader or ConfigLoader()

		# Get configuration values
		embedding_config = self.config_loader.get("embedding", {})
		self.batch_size = embedding_config.get("batch_size", 32)
		self.qdrant_batch_size = embedding_config.get("qdrant_batch_size", 100)
		self.voyage_token_limit = embedding_config.get("voyage_token_limit", DEFAULT_VOYAGE_TOKEN_LIMIT)

		# Instantiate synchronous VoyageAI client for token counting
		try:
			# API key is picked up from environment automatically
			# type: ignore for linter error on Client export
			self.voyage_client = voyageai.Client()  # type: ignore[arg-type]
			logger.info("VoyageAI client initialized for token counting.")
		except Exception:
			logger.exception("Failed to initialize VoyageAI client for token counting. Ensure VOYAGE_API_KEY is set.")
			# Depending on requirements, might want to raise here or handle gracefully
			self.voyage_client = None

		logger.info(
			f"VectorSynchronizer initialized for repo: {repo_path} "
			f"using Qdrant collection: '{qdrant_manager.collection_name}' "
			f"and embedding model: {embedding_model_name}"
		)

	async def _get_qdrant_state(self) -> dict[str, set[tuple[str, str]]]:
		"""
		Retrieves the current state from Qdrant, mapping file paths to sets of (chunk_id, git_hash).

		Returns:
		    A dictionary where keys are file paths relative to the repo root,
		    and values are sets containing tuples of (chunk_id, git_hash)
		    for each chunk associated with that file path.

		"""
		await self.qdrant_manager.initialize()
		logger.info("Retrieving current state from Qdrant collection...")
		qdrant_state: dict[str, set[tuple[str, str]]] = defaultdict(set)
		all_ids = await self.qdrant_manager.get_all_point_ids_with_filter()
		logger.info(f"[State Check] Retrieved {len(all_ids)} point IDs from Qdrant.")

		# Fetch payloads in batches to avoid overloading retrieve
		payloads = {}
		if all_ids:
			for i in range(0, len(all_ids), self.qdrant_batch_size):
				batch_ids = all_ids[i : i + self.qdrant_batch_size]
				batch_payloads = await self.qdrant_manager.get_payloads_by_ids(batch_ids)
				payloads.update(batch_payloads)
		logger.info(f"[State Check] Retrieved {len(payloads)} payloads from Qdrant.")

		processed_count = 0
		for point_id, payload in payloads.items():
			if payload and "file_path" in payload and "git_hash" in payload:
				file_path = payload["file_path"]
				git_hash = payload["git_hash"]
				# Store chunk_id as string, as Qdrant might return UUID objects
				qdrant_state[file_path].add((str(point_id), git_hash))
				processed_count += 1
			else:
				logger.warning(f"Point ID {point_id} is missing file_path or git_hash in payload.")

		logger.info(f"Retrieved state for {len(qdrant_state)} files ({processed_count} chunks) from Qdrant.")
		return qdrant_state

	async def _compare_states(
		self,
		current_git_files: dict[str, str],
		qdrant_state: dict[str, set[tuple[str, str]]],
	) -> tuple[set[str], set[str], set[str]]:
		"""
		Compare current Git state with Qdrant state to find differences.

		Args:
		    current_git_files: Dictionary mapping file paths to their current Git hash.
		    qdrant_state: Dictionary mapping file paths to sets of (chunk_id, git_hash).

		Returns:
		    tuple[set[str], set[str], set[str]]: A tuple containing:
		        - files_to_process: Files that are new or have changed hash.
		        - files_to_delete_chunks_for: Files that no longer exist in Git.
		        - chunks_to_delete: Specific chunk IDs to delete (e.g., from updated files).

		"""
		# current_git_files uses RELATIVE paths
		# qdrant_state uses ABSOLUTE paths as keys (based on stored metadata)
		logger.info(
			f"[Compare Check] Comparing {len(current_git_files)} Git files with "
			f"{len(qdrant_state)} files in Qdrant state."
		)

		git_relative_paths = set(current_git_files.keys())
		# qdrant_absolute_paths = set(qdrant_state.keys()) # Removed unused variable F841

		files_to_process: set[str] = set()  # Store RELATIVE paths
		chunks_to_delete: set[str] = set()
		processed_relative_paths: set[str] = set()  # Keep track of relative paths found in Qdrant

		# Iterate through Qdrant state (absolute paths)
		for abs_path_str, qdrant_chunks_set in qdrant_state.items():
			relative_path_str = abs_path_str  # Default if conversion fails
			try:
				abs_path = Path(abs_path_str)
				if abs_path.is_absolute():
					relative_path = abs_path.relative_to(self.repo_path)
					relative_path_str = str(relative_path)
				else:
					# If it's somehow already relative, log a warning but use it
					logger.warning(
						f"[Compare Check] Could not make Qdrant path {abs_path_str} "
						f"relative to {self.repo_path}. Skipping comparison for this path."
					)
					continue  # Skip to next item in qdrant_state

			except (ValueError, TypeError):
				logger.warning(
					f"[Compare Check] Could not make Qdrant path {abs_path_str} "
					f"relative to {self.repo_path}. Skipping comparison for this path."
				)
				continue  # Skip to next item in qdrant_state

			processed_relative_paths.add(relative_path_str)

			# Check if this RELATIVE path exists in the current Git state
			if relative_path_str in git_relative_paths:
				# File exists in both: Compare Hashes
				current_hash = current_git_files[relative_path_str]
				db_hashes = {git_hash for _, git_hash in qdrant_chunks_set}
				db_chunk_ids = {chunk_id for chunk_id, _ in qdrant_chunks_set}

				if current_hash not in db_hashes:
					# File content changed
					logger.info(
						f"[Compare Hash] Mismatch for {relative_path_str}. "
						f"Current: {current_hash}, DB: {db_hashes}. Marking for reprocessing and deletion."
					)
					files_to_process.add(relative_path_str)
					chunks_to_delete.update(db_chunk_ids)
				elif len(db_hashes) > 1:
					# File hash matches, but stale entries exist
					logger.warning(
						f"[Compare Hash] File '{relative_path_str}' has matching hash '{current_hash}' "
						f"but also stale entries in Qdrant. Cleaning up."
					)
					stale_chunk_ids = {chunk_id for chunk_id, git_hash in qdrant_chunks_set if git_hash != current_hash}
					chunks_to_delete.update(stale_chunk_ids)
				# Else: Hash matches and no stale entries -> Do nothing for this file
			else:
				# File exists in Git but not in Qdrant state (handled below)
				logger.info(f"[Compare Check] File deleted from Git: {relative_path_str}. Marking chunks for deletion.")
				chunks_to_delete.update(chunk_id for chunk_id, _ in qdrant_chunks_set)

		# Now find files that are ONLY in Git (new files)
		new_git_files = git_relative_paths - processed_relative_paths
		files_to_process.update(new_git_files)
		logger.info(f"[Compare Check] Found {len(new_git_files)} new files in Git to process.")

		# We don't use files_to_delete_chunks_for currently, assign to _
		files_to_delete_chunks_for = {rp for rp in processed_relative_paths if rp not in git_relative_paths}

		logger.info(
			f"[Compare Check] Result: {len(files_to_process)} relative files to process, "
			# f"{len(files_to_delete_chunks_for)} files with all chunks to delete, "
			f"{len(chunks_to_delete)} specific chunks to delete."
		)
		# Return relative paths for processing, and chunk IDs to delete
		return files_to_process, files_to_delete_chunks_for, chunks_to_delete

	async def _process_and_upsert_batch(self, chunk_batch: list[CodeChunk]) -> int:
		"""Processes a batch of chunks: gets embeddings and upserts to Qdrant."""
		if not chunk_batch:
			return 0

		logger.info(f"Processing batch of {len(chunk_batch)} chunks for embedding and upsert.")
		texts_to_embed = [chunk["content"] for chunk in chunk_batch]
		embeddings = await generate_embeddings_batch(
			texts_to_embed, model=self.embedding_model_name, config_loader=self.config_loader
		)

		if embeddings is None or len(embeddings) != len(chunk_batch):
			logger.error(
				"Embed batch failed: "
				f"got {len(embeddings) if embeddings else 0}, "
				f"expected {len(chunk_batch)}. Skipping."
			)
			# Log details of the failed batch chunks if possible
			failed_files = {chunk["metadata"].get("file_path", "unknown") for chunk in chunk_batch}
			logger.error(f"Failed batch involved files: {failed_files}")
			return 0

		points_to_upsert = []
		for chunk, embedding in zip(chunk_batch, embeddings, strict=True):  # Use strict=True
			# Get the original file path from metadata (likely absolute)
			original_file_path_str = chunk["metadata"].get("file_path", "unknown")

			# No longer converting to relative path here. Let the original path be stored.
			# relative_path_str = original_file_path_str ...

			chunk_id = str(uuid.uuid4())
			chunk["metadata"]["chunk_id"] = chunk_id
			# Ensure the file_path in metadata remains as it came from the chunker
			chunk["metadata"]["file_path"] = original_file_path_str
			payload: dict[str, Any] = cast("dict[str, Any]", chunk["metadata"])
			point = create_qdrant_point(chunk_id, embedding, payload)
			points_to_upsert.append(point)

		if points_to_upsert:
			await self.qdrant_manager.upsert_points(points_to_upsert)
			logger.info(f"Successfully upserted {len(points_to_upsert)} points from batch.")
			return len(points_to_upsert)
		logger.warning("No points generated from batch to upsert.")
		return 0

	async def sync_index(self, progress: Progress | None = None, task_id: TaskID | None = None) -> bool:
		"""
		Asynchronously synchronize the Qdrant index with the current repository state.

		Args:
		    progress: Optional rich Progress instance for UI updates.
		    task_id: Optional rich TaskID for progress tracking.

		Returns:
		    True if synchronization completed successfully, False otherwise.

		"""
		sync_success = False
		final_message = "[red]Error:[/red] Index sync failed (initialization error)."
		total_steps = 5  # Git state, Qdrant state, Compare, Delete, Process
		current_step = 0
		processed_chunk_count = 0

		def update_progress(description: str, step_increment: int = 1, processed_chunks: int = 0) -> None:
			nonlocal current_step, processed_chunk_count
			current_step += step_increment
			processed_chunk_count += processed_chunks
			if progress and task_id is not None:
				# Add processed chunk count to description if relevant
				desc_with_count = description
				if processed_chunk_count > 0:
					desc_with_count += f" ({processed_chunk_count} chunks processed)"
				progress.update(
					task_id,
					description=desc_with_count,
					completed=(current_step / total_steps) * 100,
				)

		try:
			await self.qdrant_manager.initialize()

			# Ensure VoyageAI client for token counting is ready
			if not self.voyage_client:
				logger.error("VoyageAI client for token counting is not initialized. Cannot proceed.")
				final_message = "[red]Error:[/red] VoyageAI client init failed."
				if progress and task_id:
					progress.update(task_id, description=final_message, completed=100)
				return False

			# 1. Get current Git state (tracked files and hashes)
			update_progress("[1/5] Reading Git tracked files...", step_increment=0)
			git_files_raw = get_git_tracked_files(self.repo_path)
			if git_files_raw is None:
				logger.error("Failed to retrieve Git tracked files.")
				# Use the initialized final_message
				if progress and task_id:
					progress.update(task_id, description=final_message, completed=100)
				return False

			# Filter out excluded files
			git_files = {fp: h for fp, h in git_files_raw.items() if not _should_exclude_path(fp)}
			logger.info(f"Found {len(git_files)} tracked files in Git (after exclusion).")
			update_progress(f"[1/5] Found {len(git_files)} tracked files.")

			# 2. Get current Qdrant state (file paths -> chunk IDs and hashes)
			update_progress("[2/5] Retrieving existing vector state...")
			qdrant_state = await self._get_qdrant_state()

			# 3. Compare states
			update_progress("[3/5] Comparing Git state with vector state...")
			# Use _ for the unused 'files_to_delete_chunks_for' variable
			files_to_process, _, chunks_to_delete = await self._compare_states(git_files, qdrant_state)

			# 4. Delete outdated chunks
			update_progress(f"[4/5] Deleting {len(chunks_to_delete)} outdated vectors...")
			if chunks_to_delete:
				# Delete in batches
				delete_ids_list = list(chunks_to_delete)
				for i in range(0, len(delete_ids_list), self.qdrant_batch_size):
					batch_ids = delete_ids_list[i : i + self.qdrant_batch_size]
					# Use cast to handle type checking
					await self.qdrant_manager.delete_points(cast("list[str | int | uuid.UUID]", batch_ids))
					logger.info(f"Deleted batch of {len(batch_ids)} vectors.")
				logger.info(f"Finished deleting {len(chunks_to_delete)} vectors.")
			else:
				logger.info("No vectors to delete.")

			# 5. Process new/updated files - **REFACTORED BATCHING LOGIC**
			update_progress(f"[5/5] Processing {len(files_to_process)} new/updated files...")
			files_processed_count = 0

			# Batching variables
			current_chunk_batch: list[CodeChunk] = []
			current_batch_token_count = 0

			for file_path in files_to_process:
				git_hash = git_files.get(file_path)
				if git_hash:
					absolute_path = self.repo_path / file_path
					try:
						logger.debug(f"Chunking file: {file_path}")
						# Use iter_chunk_file if available, otherwise list()
						file_chunks = list(self.chunker.chunk_file(absolute_path, git_hash))

						if not file_chunks:
							logger.debug(f"No chunks generated for file: {file_path}")
							continue  # Skip to next file

						for chunk in file_chunks:
							# Estimate token count for the chunk content
							chunk_text = chunk["content"]
							try:
								# Note: count_tokens expects a list
								chunk_token_count = self.voyage_client.count_tokens(
									[chunk_text], model=self.embedding_model_name
								)
							except Exception:
								logger.exception(f"Failed to count tokens for chunk in {file_path}. Skipping chunk.")
								continue  # Skip this chunk

							# Check if adding this chunk exceeds the limit
							if (
								current_batch_token_count + chunk_token_count > self.voyage_token_limit
								and current_chunk_batch
							):
								# Process the current batch before adding the new chunk
								logger.info(
									f"Token limit ({self.voyage_token_limit}) reached. "
									f"Processing batch ({len(current_chunk_batch)} chunks)."
								)
								upserted_count = await self._process_and_upsert_batch(current_chunk_batch)
								update_progress(
									f"[5/5] Processing {len(files_to_process)} files...",
									step_increment=0,
									processed_chunks=upserted_count,
								)

								# Reset batch
								current_chunk_batch = []
								current_batch_token_count = 0

							# Add the current chunk to the (potentially new) batch
							# Ensure chunk_token_count itself isn't over the limit (handle large chunks)
							if chunk_token_count > self.voyage_token_limit:
								logger.warning(
									f"Chunk in {file_path} ({chunk['metadata'].get('start_line', '?')}-...) "
									f"exceeds token limit ({chunk_token_count} > {self.voyage_token_limit}). Skipping."
								)
								continue  # Skip this large chunk

							current_chunk_batch.append(chunk)
							current_batch_token_count += chunk_token_count

					except Exception:
						logger.exception(f"Error processing file {file_path} during sync")
						# Decide if we should continue with other files or stop
						continue  # Continue processing other files

				files_processed_count += 1

			# Process any remaining chunks in the last batch
			if current_chunk_batch:
				logger.info(
					f"Processing final batch of {len(current_chunk_batch)} chunks ({current_batch_token_count} tokens)."
				)
				upserted_count = await self._process_and_upsert_batch(current_chunk_batch)
				update_progress("[5/5] Final batch processed...", step_increment=0, processed_chunks=upserted_count)

			sync_success = True
			final_message = (
				f"[green]✓[/green] Vector index synchronized. {processed_chunk_count} chunks processed/upserted."
			)
			logger.info("Vector index synchronization completed successfully.")

		except Exception:
			logger.exception("An unexpected error occurred during index synchronization")
			final_message = "[red]Error:[/red] Index sync failed unexpectedly."
			sync_success = False  # Ensure success is False on exception
		finally:
			# Final progress update
			if progress and task_id is not None:
				progress.update(task_id, description=final_message, completed=100)

		return sync_success
