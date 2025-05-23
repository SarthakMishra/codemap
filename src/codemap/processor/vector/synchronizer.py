"""Module for synchronizing HNSW index with Git state."""

import asyncio
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from qdrant_client import models

from codemap.git.utils import ExtendedGitRepoContext
from codemap.processor.hash_calculation import RepoChecksumCalculator
from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer
from codemap.processor.utils.embedding_utils import generate_embedding
from codemap.processor.vector.chunking import TreeSitterChunker
from codemap.processor.vector.qdrant_manager import QdrantManager, create_qdrant_point
from codemap.utils.cli_utils import progress_indicator

if TYPE_CHECKING:
	from codemap.config import ConfigLoader

logger = logging.getLogger(__name__)


class VectorSynchronizer:
	"""Handles asynchronous synchronization between Git repository and Qdrant vector index."""

	def __init__(
		self,
		repo_path: Path,
		qdrant_manager: QdrantManager,
		chunker: TreeSitterChunker | None,
		embedding_model_name: str,
		analyzer: TreeSitterAnalyzer | None = None,
		config_loader: "ConfigLoader | None" = None,
		repo_checksum_calculator: RepoChecksumCalculator | None = None,
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
		    repo_checksum_calculator: Optional RepoChecksumCalculator instance.

		"""
		self.repo_path = repo_path
		self.qdrant_manager = qdrant_manager
		self.git_context = ExtendedGitRepoContext.get_instance()
		self.embedding_model_name = embedding_model_name
		self.analyzer = analyzer or TreeSitterAnalyzer()

		# Ensure RepoChecksumCalculator is instantiated with git_context
		if repo_checksum_calculator is None:
			self.repo_checksum_calculator = RepoChecksumCalculator.get_instance(
				repo_path=self.repo_path, git_context=self.git_context, config_loader=config_loader
			)
		else:
			self.repo_checksum_calculator = repo_checksum_calculator
			# Ensure existing calculator also has git_context, as it might be crucial for branch logic
			if self.repo_checksum_calculator.git_context is None and self.git_context:
				self.repo_checksum_calculator.git_context = self.git_context

		if config_loader:
			self.config_loader = config_loader
		else:
			from codemap.config import ConfigLoader

			self.config_loader = ConfigLoader.get_instance()

		embedding_config = self.config_loader.get.embedding
		self.qdrant_batch_size = embedding_config.qdrant_batch_size

		if chunker is None:
			self.chunker = TreeSitterChunker(
				git_context=self.git_context,
				config_loader=self.config_loader,
				repo_checksum_calculator=self.repo_checksum_calculator,
			)
		else:
			if getattr(chunker, "git_context", None) is None:
				chunker.git_context = self.git_context
			if (
				hasattr(chunker, "repo_checksum_calculator")
				and getattr(chunker, "repo_checksum_calculator", None) is None
				and self.repo_checksum_calculator
			):
				chunker.repo_checksum_calculator = self.repo_checksum_calculator
			self.chunker = chunker

		logger.info(
			f"VectorSynchronizer initialized for repo: {repo_path} "
			f"using Qdrant collection: '{qdrant_manager.collection_name}' "
			f"and embedding model: {embedding_model_name}"
		)
		if not self.repo_checksum_calculator:
			logger.warning("RepoChecksumCalculator could not be initialized. Checksum-based sync will be skipped.")

		# Initialize checksum map attribute
		self.all_nodes_map_from_checksum: dict[str, dict[str, str]] = {}

	def _get_checksum_cache_path(self) -> Path:
		"""Gets the path to the checksum cache file within .codemap_cache."""
		# Ensure .codemap_cache directory is at the root of the repo_path passed to VectorSynchronizer
		cache_dir = self.repo_path / ".codemap_cache"
		return cache_dir / "last_sync_checksum.txt"

	def _read_last_sync_checksum(self) -> str | None:
		"""Reads the last successful sync checksum from the cache file."""
		cache_file = self._get_checksum_cache_path()
		try:
			if cache_file.exists():
				return cache_file.read_text().strip()
		except OSError as e:
			logger.warning(f"Error reading checksum cache file {cache_file}: {e}")
		return None

	def _write_last_sync_checksum(self, checksum: str) -> None:
		"""Writes the given checksum to the cache file."""
		cache_file = self._get_checksum_cache_path()
		try:
			cache_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure .codemap_cache exists
			cache_file.write_text(checksum)
			logger.info(f"Updated checksum cache file {cache_file} with checksum {checksum[:8]}...")
		except OSError:
			logger.exception(f"Error writing checksum cache file {cache_file}")

	async def _generate_chunks_for_file(self, file_path_str: str, file_hash: str) -> list[dict[str, Any]]:
		"""Helper coroutine to generate chunks for a single file.

		File hash is passed for context (e.g. git hash or content hash).
		"""
		chunks_for_file = []
		absolute_path = self.repo_path / file_path_str
		try:
			# Pass the file_hash (which could be git_hash for tracked or content_hash for untracked) to chunker
			file_chunks_generator = self.chunker.chunk_file(absolute_path, git_hash=file_hash)
			if file_chunks_generator:
				chunks_for_file.extend([chunk.model_dump() for chunk in file_chunks_generator])
				logger.debug(f"Generated {len(chunks_for_file)} chunks for {file_path_str}")
			else:
				logger.debug(f"No chunks generated for file: {file_path_str}")
		except Exception:
			logger.exception(f"Error processing file {file_path_str} during chunk generation")
		return chunks_for_file

	async def _get_qdrant_state(self) -> dict[str, set[tuple[str, str]]]:
		r"""
		Retrieves the current state from Qdrant.

		Maps file paths to sets of (chunk_id, git_or_content_hash).
		The hash stored in Qdrant should be the file\'s content hash for
		untracked or git_hash for tracked files.
		"""
		await self.qdrant_manager.initialize()
		logger.info("Retrieving current state from Qdrant collection...")
		qdrant_state: dict[str, set[tuple[str, str]]] = defaultdict(set)
		all_ids = await self.qdrant_manager.get_all_point_ids_with_filter()
		logger.info(f"[State Check] Retrieved {len(all_ids)} point IDs from Qdrant.")

		payloads = {}
		if all_ids:
			for i in range(0, len(all_ids), self.qdrant_batch_size):
				batch_ids = all_ids[i : i + self.qdrant_batch_size]
				batch_payloads = await self.qdrant_manager.get_payloads_by_ids(batch_ids)
				payloads.update(batch_payloads)
		logger.info(f"[State Check] Retrieved {len(payloads)} payloads from Qdrant.")

		processed_count = 0
		for point_id, payload_dict in payloads.items():
			if payload_dict:
				file_metadata = payload_dict.get("file_metadata")
				file_path_val: str | None = None
				comparison_hash_for_state: str | None = None

				if isinstance(file_metadata, dict):
					file_path_val = file_metadata.get("file_path")
					file_content_hash_from_meta = file_metadata.get("file_content_hash")

					git_metadata = payload_dict.get("git_metadata")
					if isinstance(git_metadata, dict) and git_metadata.get("tracked") is True:
						git_hash_from_meta = git_metadata.get("git_hash")
						if isinstance(git_hash_from_meta, str):
							comparison_hash_for_state = git_hash_from_meta
					elif isinstance(file_content_hash_from_meta, str):
						comparison_hash_for_state = file_content_hash_from_meta

				if (
					isinstance(file_path_val, str)
					and file_path_val.strip()
					and isinstance(comparison_hash_for_state, str)
				):
					qdrant_state[file_path_val].add((str(point_id), comparison_hash_for_state))
					processed_count += 1
				else:
					logger.warning(
						f"Point {point_id} in Qdrant is missing file_path or comparison_hash. "
						f"Payload components: file_path_val={file_path_val}, "
						f"comparison_hash_for_state={comparison_hash_for_state}"
					)
					continue
			else:
				logger.warning(f"Point ID {point_id} has an empty or None payload. Skipping.")

		logger.info(f"Retrieved state for {len(qdrant_state)} files ({processed_count} chunks) from Qdrant.")
		return qdrant_state

	async def _compare_states(
		self,
		current_file_hashes: dict[str, str],  # relative_path -> content_hash (from all_repo_files)
		previous_nodes_map: dict[str, dict[str, str]] | None,  # path -> {"type": "file"|"dir", "hash": hash_val}
		qdrant_state: dict[str, set[tuple[str, str]]],  # file_path -> set of (chunk_id, git_or_content_hash_in_db)
	) -> tuple[set[str], set[str]]:
		"""
		Compare current file hashes with previous checksum map and Qdrant state.

		Args:
		    current_file_hashes: Current state of files in repo (path -> content/git hash).
		    previous_nodes_map: Previously stored checksum map (path -> {type, hash}).
		    qdrant_state: Current state of chunks in Qdrant.

		Returns:
		    tuple[set[str], set[str]]:
		        - files_to_process: Relative paths of files that are new or changed.
		        - chunks_to_delete_from_qdrant: Specific Qdrant chunk_ids to delete.
		"""
		files_to_process: set[str] = set()
		chunks_to_delete_from_qdrant: set[str] = set()
		processed_qdrant_paths_this_cycle: set[str] = set()

		# 1. Determine files to process based on current vs. previous checksums
		for file_path, current_hash in current_file_hashes.items():
			previous_file_hash: str | None = None
			if previous_nodes_map and file_path in previous_nodes_map:
				node_info = previous_nodes_map[file_path]
				if node_info.get("type") == "file":
					previous_file_hash = node_info.get("hash")

			if previous_file_hash is None or previous_file_hash != current_hash:
				logger.info(
					f"[Compare] File '{file_path}' is new or changed. "
					f"Old hash: {previous_file_hash}, New hash: {current_hash}"
				)
				files_to_process.add(file_path)
				# If the file was present and its hash matches, it can be skipped
				if file_path in qdrant_state:
					logger.info(
						f"[Compare] Marking existing Qdrant chunks for changed file '{file_path}' for deletion."
					)
					chunks_to_delete_from_qdrant.update(cid for cid, _ in qdrant_state[file_path])
			processed_qdrant_paths_this_cycle.add(file_path)  # Mark as seen from current repo state

		# 2. Determine files/chunks to delete based on previous checksums vs. current
		if previous_nodes_map:
			for old_path, node_info in previous_nodes_map.items():
				if node_info.get("type") == "file" and old_path not in current_file_hashes:
					logger.info(
						f"[Compare] File '{old_path}' was in previous checksum but not current. Deleting from Qdrant."
					)
					if old_path in qdrant_state:
						for chunk_id, _ in qdrant_state[old_path]:
							chunks_to_delete_from_qdrant.add(chunk_id)
					processed_qdrant_paths_this_cycle.add(old_path)  # Mark as seen from previous repo state

		# 3. Clean up any orphaned Qdrant entries not covered by current or previous valid states
		# These might be from very old states or errors.
		all_known_valid_paths = set(current_file_hashes.keys())
		if previous_nodes_map:
			all_known_valid_paths.update(p for p, info in previous_nodes_map.items() if info.get("type") == "file")

		for qdrant_file_path, qdrant_chunks_set in qdrant_state.items():
			if qdrant_file_path not in all_known_valid_paths:
				logger.warning(
					f"Orphaned file_path '{qdrant_file_path}' in Qdrant not found in current "
					"or previous valid checksums. Deleting its chunks."
				)
				for chunk_id, _ in qdrant_chunks_set:
					chunks_to_delete_from_qdrant.add(chunk_id)

		logger.info(
			f"[Compare States] Result: {len(files_to_process)} files to process/reprocess, "  # noqa: S608
			f"{len(chunks_to_delete_from_qdrant)} specific chunks to delete from Qdrant."
		)
		# The second element of the tuple (files_whose_chunks_are_all_deleted) is no longer
		# explicitly needed with this logic.
		return files_to_process, chunks_to_delete_from_qdrant

	async def _process_and_upsert_batch(self, chunk_batch: list[dict[str, Any]]) -> int:
		"""Process a batch of chunks by generating embeddings and upserting to Qdrant."""
		if not chunk_batch:
			return 0

		deduplicated_batch = []
		seen_content_hashes = set()
		for chunk in chunk_batch:
			content_hash = chunk["metadata"].get("content_hash", "")
			file_metadata_dict = chunk["metadata"].get("file_metadata", {})
			file_content_hash = (
				file_metadata_dict.get("file_content_hash", "") if isinstance(file_metadata_dict, dict) else ""
			)
			dedup_key = f"{content_hash}:{file_content_hash}"
			if dedup_key not in seen_content_hashes:
				seen_content_hashes.add(dedup_key)
				deduplicated_batch.append(chunk)

		if len(deduplicated_batch) < len(chunk_batch):
			logger.info(
				f"Removed {len(chunk_batch) - len(deduplicated_batch)} "
				f"duplicate chunks, processing {len(deduplicated_batch)} unique chunks"
			)

		if not deduplicated_batch:  # If all chunks were duplicates
			logger.info("All chunks in the batch were duplicates. Nothing to process.")
			return 0

		logger.info(f"Processing batch of {len(deduplicated_batch)} unique chunks for embedding and upsert.")
		texts_to_embed = [chunk["content"] for chunk in deduplicated_batch]
		embeddings = generate_embedding(texts_to_embed, self.config_loader)

		if embeddings is None or len(embeddings) != len(deduplicated_batch):
			logger.error(
				"Embed batch failed: "
				f"got {len(embeddings) if embeddings else 0}, expected {len(deduplicated_batch)}. Skipping."
			)
			failed_files = {chunk["metadata"].get("file_path", "unknown") for chunk in deduplicated_batch}
			logger.error(f"Failed batch involved files: {failed_files}")
			return 0

		points_to_upsert = []
		for chunk, embedding in zip(deduplicated_batch, embeddings, strict=True):
			original_file_path_str = chunk["metadata"].get("file_path", "unknown")
			chunk_id = str(uuid.uuid4())
			chunk["metadata"]["chunk_id"] = chunk_id
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

	async def sync_index(self) -> bool:
		"""
		Asynchronously synchronize the Qdrant index with the current repository state.

		Returns True if synchronization completed successfully, False otherwise.
		"""
		sync_success = False
		current_repo_root_checksum: str | None = None
		# This local variable will hold the map for the current sync operation.
		current_nodes_map: dict[str, dict[str, str]] = {}  # Initialize as empty

		previous_root_hash: str | None = None
		previous_nodes_map: dict[str, dict[str, str]] | None = None

		if self.repo_checksum_calculator:
			# Attempt to read the checksum from the last successful sync for the current branch
			logger.info("Attempting to read latest checksum data for current branch...")
			prev_hash, prev_map = self.repo_checksum_calculator.read_latest_checksum_data_for_current_branch()
			if prev_hash:
				previous_root_hash = prev_hash
			if prev_map:
				previous_nodes_map = prev_map

			try:
				# calculate_repo_checksum returns: tuple[str, dict[str, dict[str, str]]]
				# Renamed local var to avoid confusion if self.all_nodes_map_from_checksum is used elsewhere
				(
					calculated_root_hash,
					calculated_nodes_map,
				) = await self.repo_checksum_calculator.calculate_repo_checksum()
				current_repo_root_checksum = calculated_root_hash
				self.all_nodes_map_from_checksum = calculated_nodes_map  # Store the fresh map on self
				current_nodes_map = self.all_nodes_map_from_checksum  # Use this fresh map for the current sync

				# Quick sync: If root checksums match, assume no changes and skip detailed comparison.
				if previous_root_hash and current_repo_root_checksum == previous_root_hash:
					branch_name = (
						self.repo_checksum_calculator.git_context.get_current_branch()
						if self.repo_checksum_calculator.git_context
						else "unknown"
					)
					logger.info(
						f"Root checksum ({current_repo_root_checksum}) matches "
						f"previous state for branch '{branch_name}'. "
						"Quick sync indicates no changes needed."
					)
					# Consider updating a 'last_synced_timestamp' or similar marker here if needed.
					return True  # Successfully synced (no changes)
				logger.info(
					"Root checksum mismatch or no previous checksum. Proceeding with detailed comparison and sync."
				)
			except Exception:  # pylint: disable=broad-except
				logger.exception(
					"Error calculating repository checksum. "
					"Proceeding with full comparison using potentially stale or no current checksum data."
				)
				# current_nodes_map remains {}, signifying we couldn't get a fresh current state.
				# This will be handled by the check below.
		else:
			logger.warning(
				"RepoChecksumCalculator not available. Cannot perform checksum-based "
				"quick sync, read previous checksum, or get fresh current state. "
				"Proceeding with comparison based on Qdrant state only if necessary, "
				"but sync will likely be incomplete."
			)
			# previous_root_hash and previous_nodes_map remain None.
			# current_nodes_map remains {}, signifying we couldn't get a fresh current state.
			# This will be handled by the check below.

		# Populate current_file_hashes from the local current_nodes_map.
		# current_nodes_map will be populated if checksum calculation succeeded, otherwise it's {}.
		current_file_hashes: dict[str, str] = {}
		if not current_nodes_map:  # Checks if the map is empty
			# This means checksum calculation failed or RepoChecksumCalculator was not available.
			# We cannot reliably determine the current state of files.
			logger.error(
				"Current repository file map is empty (failed to calculate checksums "
				"or RepoChecksumCalculator missing). Cannot proceed with accurate sync "
				"as current file states are unknown."
			)
			return False  # Cannot sync without knowing current file states.

		# If current_nodes_map is not empty, proceed to populate current_file_hashes
		for path, node_info in current_nodes_map.items():
			if node_info.get("type") == "file" and "hash" in node_info:  # Ensure hash key exists
				current_file_hashes[path] = node_info["hash"]

		# If current_nodes_map was valid (not empty) but contained no files (e.g. empty repo),
		# current_file_hashes will be empty. This is a valid state for _compare_states.

		# Get the current state from Qdrant
		qdrant_state = await self._get_qdrant_state()

		with progress_indicator("Comparing repository state with vector state..."):
			files_to_process, chunks_to_delete = await self._compare_states(
				current_file_hashes, previous_nodes_map, qdrant_state
			)

		with progress_indicator(f"Deleting {len(chunks_to_delete)} outdated vectors..."):
			if chunks_to_delete:
				delete_ids_list = list(chunks_to_delete)
				for i in range(0, len(delete_ids_list), self.qdrant_batch_size):
					batch_ids_to_delete = delete_ids_list[i : i + self.qdrant_batch_size]
					await self.qdrant_manager.delete_points(batch_ids_to_delete)
					logger.info(f"Deleted batch of {len(batch_ids_to_delete)} vectors.")
				logger.info(f"Finished deleting {len(chunks_to_delete)} vectors.")
			else:
				logger.info("No vectors to delete.")

		# Step: Update git_metadata for files whose content hasn't changed but Git status might have
		logger.info("Checking for Git metadata updates for unchanged files...")

		# Candidate files: in current repo, content hash same as previous, so not in files_to_process
		files_to_check_for_git_metadata_update = set(current_file_hashes.keys()) - files_to_process

		chunk_ids_to_fetch_payloads_for_meta_check: list[str] = []
		# Maps file_path_str to list of its chunk_ids that are candidates for metadata update
		candidate_file_to_chunks_map: dict[str, list[str]] = defaultdict(list)

		for file_path_str in files_to_check_for_git_metadata_update:
			if file_path_str not in qdrant_state:  # No existing chunks for this file in Qdrant
				continue

			# Consider only chunks that are not already marked for deletion
			chunks_for_this_file = [cid for cid, _ in qdrant_state[file_path_str] if cid not in chunks_to_delete]
			if chunks_for_this_file:
				# qdrant_state stores chunk_ids as strings (derived from ExtendedPointId)
				chunk_ids_to_fetch_payloads_for_meta_check.extend(chunks_for_this_file)
				candidate_file_to_chunks_map[file_path_str] = chunks_for_this_file

		# Batch fetch payloads for all potentially affected chunks
		fetched_payloads_for_meta_check: dict[str, dict[str, Any]] = {}
		if chunk_ids_to_fetch_payloads_for_meta_check:
			logger.info(
				f"Fetching payloads for {len(chunk_ids_to_fetch_payloads_for_meta_check)} chunks to check Git metadata."
			)
			for i in range(0, len(chunk_ids_to_fetch_payloads_for_meta_check), self.qdrant_batch_size):
				batch_ids = chunk_ids_to_fetch_payloads_for_meta_check[i : i + self.qdrant_batch_size]
				# Cast to satisfy linter for QdrantManager's expected type
				typed_batch_ids = cast("list[str | int | uuid.UUID]", batch_ids)
				batch_payloads = await self.qdrant_manager.get_payloads_by_ids(typed_batch_ids)
				fetched_payloads_for_meta_check.update(batch_payloads)

		# Dictionary to group chunk_ids by the required new git_metadata
		# Key: frozenset of new_git_metadata.items() to make it hashable
		# Value: list of chunk_ids (strings)
		git_metadata_update_groups: dict[frozenset, list[str]] = defaultdict(list)

		for file_path_str, chunk_ids_in_file in candidate_file_to_chunks_map.items():
			current_is_tracked = self.git_context.is_file_tracked(file_path_str)
			current_branch = self.git_context.get_current_branch()
			current_git_hash_for_file: str | None = None
			if current_is_tracked:
				try:
					# This should be the blob OID for the file
					current_git_hash_for_file = self.git_context.get_file_git_hash(file_path_str)
				except Exception:  # noqa: BLE001
					logger.warning(
						f"Could not get git hash for tracked file {file_path_str} during metadata update check.",
						exc_info=True,
					)

			required_new_git_metadata = {
				"tracked": current_is_tracked,
				"branch": current_branch,
				"git_hash": current_git_hash_for_file,  # Will be None if untracked or error getting hash
			}

			for chunk_id in chunk_ids_in_file:  # chunk_id is already a string
				chunk_payload = fetched_payloads_for_meta_check.get(chunk_id)
				if not chunk_payload:
					logger.warning(
						f"Payload not found for chunk {chunk_id} of file {file_path_str} "
						"during metadata check. Skipping this chunk."
					)
					continue

				old_git_metadata = chunk_payload.get("git_metadata")
				update_needed = False
				if not isinstance(old_git_metadata, dict):
					update_needed = True  # If no proper old metadata, or it's missing, update to current
				elif (
					old_git_metadata.get("tracked") != required_new_git_metadata["tracked"]
					or old_git_metadata.get("branch") != required_new_git_metadata["branch"]
					or old_git_metadata.get("git_hash") != required_new_git_metadata["git_hash"]
				):
					update_needed = True

				if update_needed:
					key = frozenset(required_new_git_metadata.items())
					git_metadata_update_groups[key].append(chunk_id)

		if git_metadata_update_groups:
			num_chunks_to_update = sum(len(ids) for ids in git_metadata_update_groups.values())
			logger.info(
				f"Found {num_chunks_to_update} chunks requiring Git metadata updates, "
				f"grouped into {len(git_metadata_update_groups)} unique metadata sets."
			)

			total_update_batches = sum(
				(len(chunk_ids_group) + self.qdrant_batch_size - 1) // self.qdrant_batch_size
				for chunk_ids_group in git_metadata_update_groups.values()
			)
			# Ensure total is at least 1 if there are groups, for progress bar logic
			progress_total = (
				total_update_batches if total_update_batches > 0 else (1 if git_metadata_update_groups else 0)
			)

			if progress_total > 0:  # Only show progress if there's something to do
				with progress_indicator(
					"Applying Git metadata updates to chunks...",
					total=progress_total,
					style="progress",
					transient=True,
				) as update_meta_progress_bar:
					applied_batches_count = 0
					for new_meta_fset, chunk_ids_to_update_with_this_meta in git_metadata_update_groups.items():
						new_meta_dict = dict(new_meta_fset)
						payload_to_set = {"git_metadata": new_meta_dict}

						for i in range(0, len(chunk_ids_to_update_with_this_meta), self.qdrant_batch_size):
							batch_chunk_ids = chunk_ids_to_update_with_this_meta[i : i + self.qdrant_batch_size]
							if batch_chunk_ids:  # Ensure batch is not empty
								# Cast to satisfy linter for QdrantManager's expected type
								typed_point_ids = cast("list[str | int | uuid.UUID]", batch_chunk_ids)
								await self.qdrant_manager.set_payload(
									payload=payload_to_set, point_ids=typed_point_ids, filter_condition=models.Filter()
								)
								logger.info(
									f"Updated git_metadata for {len(batch_chunk_ids)} chunks with: {new_meta_dict}"
								)
							applied_batches_count += 1
							update_meta_progress_bar(None, applied_batches_count, None)  # Update progress

					# Ensure progress bar completes if all batches were empty but groups existed
					if applied_batches_count == 0 and git_metadata_update_groups:
						update_meta_progress_bar(None, progress_total, None)  # Force completion
			elif num_chunks_to_update > 0:  # Log if groups existed but somehow total_progress was 0
				logger.info(f"Updating {num_chunks_to_update} chunks' Git metadata without progress bar (zero batches)")
				for new_meta_fset, chunk_ids_to_update_with_this_meta in git_metadata_update_groups.items():
					new_meta_dict = dict(new_meta_fset)
					payload_to_set = {"git_metadata": new_meta_dict}
					if chunk_ids_to_update_with_this_meta:  # Check if list is not empty
						# Cast to satisfy linter for QdrantManager's expected type
						typed_point_ids_single_batch = cast(
							"list[str | int | uuid.UUID]", chunk_ids_to_update_with_this_meta
						)
						await self.qdrant_manager.set_payload(
							payload=payload_to_set,
							point_ids=typed_point_ids_single_batch,
							filter_condition=models.Filter(),
						)
						logger.info(
							f"Updated git_metadata for {len(chunk_ids_to_update_with_this_meta)} "
							f"chunks (in a single batch) with: {new_meta_dict}"
						)

		else:
			logger.info("No Git metadata updates required for existing chunks of unchanged files.")

		num_files_to_process = len(files_to_process)
		all_chunks: list[dict[str, Any]] = []  # Ensure all_chunks is initialized
		msg = "Processing new/updated files..."

		with progress_indicator(
			msg,
			style="progress",
			total=num_files_to_process if num_files_to_process > 0 else 1,  # total must be > 0
			transient=True,
		) as update_file_progress:
			if num_files_to_process > 0:
				processed_files_counter = [0]  # Use list to make it mutable from inner function

				# Wrapper coroutine to update progress as tasks complete
				async def wrapped_generate_chunks(file_path: str, f_hash: str) -> list[dict[str, Any]]:
					result: list[dict[str, Any]] = []
					try:
						result = await self._generate_chunks_for_file(file_path, f_hash)
					finally:
						processed_files_counter[0] += 1
						update_file_progress(None, processed_files_counter[0], None)
					return result

				tasks_to_gather = []
				for file_path_to_proc in files_to_process:
					file_current_hash = current_file_hashes.get(file_path_to_proc)
					if file_current_hash:
						tasks_to_gather.append(wrapped_generate_chunks(file_path_to_proc, file_current_hash))
					else:
						logger.warning(
							f"File '{file_path_to_proc}' marked to process but its current hash not found. Skipping."
						)
						# If a file is skipped, increment progress here as it won't be wrapped
						processed_files_counter[0] += 1
						update_file_progress(None, processed_files_counter[0], None)

				if tasks_to_gather:
					logger.info(f"Concurrently generating chunks for {len(tasks_to_gather)} files...")
					list_of_chunk_lists = await asyncio.gather(*tasks_to_gather)
					all_chunks = [chunk for sublist in list_of_chunk_lists for chunk in sublist]
					logger.info(f"Total chunks generated: {len(all_chunks)}.")
				else:
					logger.info("No files eligible for concurrent chunk generation.")

				# Final update to ensure the progress bar completes to 100% if some files were skipped
				# and caused processed_files_count to not reach num_files_to_process via the finally blocks alone.
				if processed_files_counter[0] < num_files_to_process:
					update_file_progress(None, num_files_to_process, None)

			else:  # num_files_to_process == 0
				logger.info("No new/updated files to process.")
				update_file_progress(None, 1, None)  # Complete the dummy task if total was 1

		with progress_indicator("Processing chunks..."):
			await self._process_and_upsert_batch(all_chunks)

		sync_success = True
		logger.info("Vector index synchronization completed successfully.")

		return sync_success
