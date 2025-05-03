"""Orchestrates the vector database synchronization process."""

import logging
import uuid
from pathlib import Path
from typing import Any, Never

from pymilvus import MilvusClient, exceptions

from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.utils.git_utils import get_git_tracked_files
from codemap.processor.utils.sync_utils import compare_states
from codemap.processor.vector import chunker, config
from codemap.processor.vector.client import get_milvus_client
from codemap.processor.vector.embedder import generate_embeddings

# Import ConfigLoader
from codemap.utils.config_loader import ConfigLoader
from codemap.utils.file_utils import read_file_content
from codemap.utils.path_utils import find_project_root

logger = logging.getLogger(__name__)
TARGET_FILE_LOG = "src/codemap/utils/config_loader.py"  # Target file for detailed logging

# Type alias for dictionary mapping file path to git hash
GitFileInfo = dict[str, str]
# Type alias for dictionary mapping file path to a set of git hashes found in Milvus
MilvusFileInfo = dict[str, set[str]]

# Removed MIN_GIT_LS_FILES_PARTS, now in git_utils


# Helper for TRY301
def _raise_embedding_error(message: str) -> Never:
	raise ValueError(message)


def synchronize_vectors(
	repo_path: Path | None = None,
	current_git_files: GitFileInfo | None = None,
	kuzu_manager: KuzuManager | None = None,
) -> None:
	"""
	Synchronizes the Milvus vector database with the provided Git state.

	1. Gets existing file paths and hashes from Milvus.
	2. Determines files to add, update, or delete based on provided Git state.
	3. Processes changes: chunks, embeds, and updates Milvus.

	Args:
	        repo_path (Path | None, optional): The path to the repository root.
	            If None, attempts to find it automatically. Defaults to None.
	        current_git_files (GitFileInfo | None, optional):
	            A dictionary mapping file paths to their Git blob hashes.
	            If None, it will be fetched from the repository.
	        kuzu_manager (KuzuManager | None, optional): An initialized KuzuManager instance.
	            If None, vector sync requiring Kuzu lookups might fail.

	"""
	if repo_path is None:
		try:
			repo_path = find_project_root()
		except FileNotFoundError:
			logger.exception("Synchronization failed: Could not determine repository path.")
			return

	milvus_client = get_milvus_client()
	if not milvus_client:
		logger.error("Synchronization failed: Milvus client not available.")
		return

	# --- Load Configuration --- #
	# Configuration loading is now implemented
	try:
		config_loader = ConfigLoader(repo_root=repo_path)
		app_config = config_loader.config  # Get the loaded config dictionary
		logger.info("Successfully loaded configuration.")
	except Exception:
		logger.exception("Failed to load configuration")
		return
	# --- End Load Configuration --- #

	logger.info(f"Starting vector synchronization for repository: {repo_path}")

	# 1. Get Git state (use provided or fetch)
	if current_git_files is None:
		logger.debug("Fetching current Git state...")
		current_git_files = get_git_tracked_files(repo_path)
		if current_git_files is None:
			logger.error("Synchronization failed: Could not get Git tracked files.")
			return
		logger.debug(f"Found {len(current_git_files)} files in Git.")
	else:
		logger.debug(f"Using provided Git state with {len(current_git_files)} files.")

	# 2. Get Milvus state
	logger.debug("Fetching current Milvus state...")
	existing_milvus_files = _get_milvus_file_hashes(milvus_client)
	if existing_milvus_files is None:
		logger.error("Synchronization failed: Could not get file hashes from Milvus.")
		return
	logger.debug(f"Found {len(existing_milvus_files)} files represented in Milvus.")

	# 3. Determine changes
	logger.debug("Comparing Git state with Milvus state...")
	files_to_add, files_to_update, files_to_delete = compare_states(current_git_files, existing_milvus_files)

	total_changes = len(files_to_add) + len(files_to_update) + len(files_to_delete)
	if total_changes == 0:
		logger.info("Vector database is already up-to-date.")
		return

	# Format long log message
	log_message = (
		f"Synchronization required: {len(files_to_add)} to add, "
		f"{len(files_to_update)} to update, {len(files_to_delete)} to delete."
	)
	logger.info(log_message)

	# 4. Process changes
	# Process deletions first
	if files_to_delete:
		logger.debug(f"Deleting {len(files_to_delete)} files from Milvus...")
		_delete_files_from_milvus(milvus_client, files_to_delete)

	# Process additions and updates (involve reading, chunking, embedding, inserting)
	files_to_process = files_to_add.union(files_to_update)
	if files_to_process:
		logger.debug(f"Processing {len(files_to_process)} files for addition/update...")
		_process_files(
			milvus_client,
			repo_path,
			files_to_process,
			current_git_files,  # Pass the dict containing current hashes
			files_to_update,  # Pass the set of files needing deletion first
			app_config,
			kuzu_manager=kuzu_manager,
		)

	logger.info("Vector synchronization finished.")


# Removed _get_git_tracked_files function (now in git_utils)


def _get_milvus_file_hashes(client: MilvusClient) -> MilvusFileInfo | None:
	"""Queries Milvus to get all unique file_path and git_hash combinations."""
	try:
		# Check if collection exists first
		if not client.has_collection(config.COLLECTION_NAME):
			logger.warning(f"Collection {config.COLLECTION_NAME} does not exist. Assuming no existing files.")
			return {}

		# Use query iterator for potentially large results
		# Fetch file_path and git_hash for all entries
		results_iterator = client.query_iterator(
			collection_name=config.COLLECTION_NAME,
			filter="",  # Get all
			output_fields=[config.FIELD_FILE_PATH, config.FIELD_GIT_HASH],
			batch_size=1000,  # Adjust batch size as needed
		)

		milvus_files: MilvusFileInfo = {}
		processed_count = 0
		batch = None  # Initialize batch to prevent UnboundLocalError
		# Correctly iterate over QueryIterator
		while True:
			try:
				batch = results_iterator.next()
				if not batch:  # Iterator exhausted
					break
				for result in batch:
					file_path = result.get(config.FIELD_FILE_PATH)
					git_hash = result.get(config.FIELD_GIT_HASH)
					if file_path and git_hash:
						# Store the *last seen* hash for each file path
						milvus_files[file_path] = git_hash
					processed_count += 1
			except StopIteration:
				break  # Explicitly handle StopIteration just in case
			except Exception:
				# Log batch info only if it was successfully retrieved in this iteration
				batch_info = f"batch: {batch}" if batch is not None else "error occurred before batch retrieval"
				logger.exception(f"Error processing Milvus query result {batch_info}")
				return None  # Indicate failure

		logger.debug(f"Processed {processed_count} entries from Milvus.")
		return milvus_files

	except exceptions.MilvusException:
		logger.exception("Milvus error querying file hashes")
		return None
	except Exception:
		logger.exception("Unexpected error getting Milvus file hashes")
		return None


# Removed _compare_states function (now in sync_utils)


def _delete_files_from_milvus(client: MilvusClient, files_to_delete: set[str]) -> None:
	"""Deletes all records associated with the given file paths from Milvus."""
	if not files_to_delete:
		return

	logger.info(f"Deleting {len(files_to_delete)} files from Milvus...")
	# Build a large OR filter expression for efficiency
	# Ensure file paths are properly escaped if they contain special chars
	# Milvus VARCHAR equality needs single quotes
	delete_expr_parts = [f"{config.FIELD_FILE_PATH} == '{path.replace("'", "\\'")}'" for path in files_to_delete]
	delete_expr = " or ".join(delete_expr_parts)

	try:
		logger.debug(f"Delete expression (first 500 chars): {delete_expr[:500]}")
		# Use delete operation
		res = client.delete(collection_name=config.COLLECTION_NAME, filter=delete_expr)
		# Corrected logging for Milvus delete result - Use getattr for safer access
		deleted_pk_count = len(getattr(res, "primary_keys", []))
		logger.info(f"Milvus deletion result PK count: {deleted_pk_count}")
		logger.info(f"Deletion operation completed for {len(files_to_delete)} files.")
	except exceptions.MilvusException:
		logger.exception("Milvus error deleting files")
	except Exception:
		logger.exception("Unexpected error deleting files from Milvus")


def _get_kuzu_ids_for_chunks(
	kuzu_manager: KuzuManager | None,
	chunk_identifiers: list[tuple[str, int, int, str | None]],
) -> dict[tuple[str, int, int, str | None], str] | None:
	"""
	Queries Kuzu to find entity IDs for a list of chunk identifiers.

	Args:
	        kuzu_manager: The KuzuManager instance.
	        chunk_identifiers: A list of tuples, where each tuple uniquely identifies
	                                           a chunk (e.g., (file_path, start_line, end_line, entity_name)).

	Returns:
	        A dictionary mapping the identifier tuple to the Kuzu entity_id string,
	        or None if the query fails.

	"""
	if not kuzu_manager:
		logger.warning("KuzuManager not provided, cannot perform Kuzu ID lookup.")
		return None
	if not chunk_identifiers:
		return {}

	logger.debug(f"Performing Kuzu lookup for {len(chunk_identifiers)} chunk identifiers.")

	# Prepare parameters for the Kuzu query. We need to match based on the identifier parts.
	file_paths = {ident[0] for ident in chunk_identifiers}
	# Corrected Query: Match CodeFile by path, traverse CONTAINS_ENTITY to get CodeEntity nodes
	cypher_query = """
	MATCH (f:CodeFile {file_path: $target_path})-[:CONTAINS_ENTITY]->(n:CodeEntity)
	RETURN f.file_path AS file_path, n.start_line AS start_line,
		   n.end_line AS end_line, n.name AS name, n.entity_id AS entity_id
	"""

	# We need to run the query per file path as Kuzu might not handle UNWIND + parameterised MATCH efficiently yet
	all_query_results = []
	for path in file_paths:
		params = {"target_path": path}
		try:
			query_results = kuzu_manager.execute_query(cypher_query, params)
			if query_results is not None:
				all_query_results.extend(query_results)
			else:
				logger.warning(f"Kuzu query for entity IDs returned None for path: {path}")
		except Exception:
			logger.exception(f"Error during Kuzu ID lookup query for path: {path}")
			# Decide if we should return None or continue with partial results
			# For now, let's return None to indicate a significant failure
			return None

	# Build a map from the Kuzu results using the same identifier structure
	result_map: dict[tuple[str, int, int, str | None], str] = {}
	for row in all_query_results:
		fpath, start, end, k_name, k_id = row
		# Ensure start/end are integers if they come back otherwise
		try:
			start = int(start)
			end = int(end)
		except (ValueError, TypeError):
			logger.warning(f"Skipping Kuzu result due to non-integer lines: {row}")
			continue

		# Use the same identifier format as input for matching
		kuzu_identifier = (fpath, start, end, k_name)
		result_map[kuzu_identifier] = k_id

	# Now, map the original chunk identifiers to the found Kuzu IDs
	final_id_map: dict[tuple[str, int, int, str | None], str] = {}
	match_count = 0
	for identifier in chunk_identifiers:
		# Try exact match first (including name)
		if identifier in result_map:
			final_id_map[identifier] = result_map[identifier]
			match_count += 1
		else:
			# Fallback: try matching without name if name was None or different
			# This requires iterating through result_map keys - potentially slow
			ident_fpath, ident_start, ident_end, _ = identifier
			found_fallback = False
			for kuzu_ident, kuzu_id in result_map.items():
				k_fpath, k_start, k_end, _ = kuzu_ident
				if k_fpath == ident_fpath and k_start == ident_start and k_end == ident_end:
					final_id_map[identifier] = kuzu_id
					match_count += 1
					found_fallback = True
					break  # Found a match based on location
			if not found_fallback:
				logger.debug(f"No Kuzu match found for identifier: {identifier}")

	logger.debug(f"Kuzu lookup mapped {match_count}/{len(chunk_identifiers)} identifiers.")
	return final_id_map


def _process_files(
	client: MilvusClient,
	repo_path: Path,
	files_to_process: set[str],
	current_git_files: GitFileInfo,  # Get hashes from here
	files_to_update: set[str],
	app_config: dict[str, Any],
	kuzu_manager: KuzuManager | None,
) -> None:
	"""
	Reads, chunks, embeds, and inserts/updates data for the given files.

	Handles deleting old entries for updated files before inserting new
	ones.

	"""
	# First, delete existing entries for files that need updating
	if files_to_update:
		logger.info(f"Deleting existing Milvus entries for {len(files_to_update)} updated files...")
		_delete_files_from_milvus(client, files_to_update)
		logger.info("Deletion of outdated entries complete.")

	processed_count = 0
	error_count = 0
	all_chunk_data_to_insert: list[dict[str, Any]] = []  # Accumulate data

	logger.info(f"Processing {len(files_to_process)} files...")
	for i, file_path_str in enumerate(files_to_process):
		is_target_file = file_path_str.endswith(TARGET_FILE_LOG)
		if is_target_file:
			logger.debug(
				f"VECTOR_MANAGER: Starting processing for target file: {file_path_str}"
				f" ({i + 1}/{len(files_to_process)})"
			)
		# else: # Optional: log non-target files
		# logger.debug(f"VECTOR_MANAGER: Processing file {file_path_str} ({i+1}/{len(files_to_process)})")

		file_path = Path(file_path_str)
		file_full_path = repo_path / file_path
		git_hash = current_git_files.get(file_path_str)

		if not git_hash:
			logger.warning(f"Skipping file {file_path_str}: Could not find its Git hash.")
			error_count += 1
			continue

		logger.debug(f"Processing file {i + 1}/{len(files_to_process)}: {file_path_str}")

		# 1. Read file content using utility
		content = read_file_content(file_full_path)
		if not content:
			logger.error(f"Skipping file {file_path_str}: Failed to read content.")
			error_count += 1
			continue

		# 2. Chunk file
		try:
			# Pass app_config if chunker needs it
			chunks = chunker.chunk_file(file_path, content, git_hash, app_config)
		except Exception:
			logger.exception(f"Failed to chunk file {file_path_str}")
			error_count += 1
			continue

		if not chunks:
			logger.warning(f"No chunks generated for file {file_path_str}. Skipping.")
			continue

		logger.debug(f"Generated {len(chunks)} chunks for {file_path_str}")

		# 3. Generate embeddings for all chunks in the file
		chunk_texts = [chunk[config.FIELD_CHUNK_TEXT] for chunk in chunks]
		embeddings = None  # Initialize embeddings to None
		try:
			embeddings = generate_embeddings(chunk_texts)
			if embeddings is None or len(embeddings) != len(chunks):
				# Use helper for TRY301
				_raise_embedding_error("Embedding generation failed or returned wrong number.")
		except Exception:
			logger.exception(f"Failed to generate embeddings for chunks in {file_path_str}")
			error_count += 1
			continue  # Skip this file if embeddings fail

		# Ensure embeddings is not None before proceeding to zip
		if embeddings is None:
			logger.error(f"Skipping file {file_path_str} due to failed embedding generation (embeddings is None).")
			error_count += 1
			continue

		# --- Kuzu ID Lookup Preparation ---
		# Collect identifiers for all chunks in the current file
		chunk_identifiers = []
		for chunk in chunks:
			# Use a tuple that uniquely identifies the chunk for Kuzu lookup
			# (file_path, start_line, end_line, entity_name) seems robust
			identifier = (
				file_path_str,
				chunk[config.FIELD_START_LINE],
				chunk[config.FIELD_END_LINE],
				chunk.get(config.FIELD_ENTITY_NAME),
			)
			chunk_identifiers.append(identifier)

		# --- Perform Kuzu Lookup for the file --- #
		kuzu_id_map = _get_kuzu_ids_for_chunks(kuzu_manager, chunk_identifiers)
		if kuzu_id_map is None:
			logger.warning(f"Failed to perform Kuzu ID lookup for file {file_path_str}. Skipping Kuzu IDs.")
			kuzu_id_map = {}

		# 4. Prepare data for Milvus insertion (batching)
		# Now it's safe to zip
		for chunk, embedding, identifier in zip(chunks, embeddings, chunk_identifiers, strict=True):
			# Generate UUID for the chunk
			chunk_uuid = str(uuid.uuid4())

			# Get the corresponding Kuzu ID from the map
			kuzu_entity_id = kuzu_id_map.get(identifier)
			if kuzu_entity_id is None:
				# Log if an ID wasn't found for a specific chunk identifier
				logger.debug(f"Could not find matching Kuzu node for chunk: {identifier}")

			chunk_data = {
				config.FIELD_ID: chunk_uuid,
				config.FIELD_EMBEDDING: embedding.tolist(),
				config.FIELD_FILE_PATH: file_path_str,
				config.FIELD_GIT_HASH: git_hash,
				config.FIELD_CHUNK_TEXT: chunk[config.FIELD_CHUNK_TEXT],
				config.FIELD_START_LINE: chunk[config.FIELD_START_LINE],
				config.FIELD_END_LINE: chunk[config.FIELD_END_LINE],
				config.FIELD_CHUNK_TYPE: chunk[config.FIELD_CHUNK_TYPE],
				config.FIELD_ENTITY_NAME: chunk.get(config.FIELD_ENTITY_NAME),
				"kuzu_entity_id": kuzu_entity_id,
			}
			all_chunk_data_to_insert.append(chunk_data)

		processed_count += 1
		if (i + 1) % 50 == 0:  # Log progress every 50 files
			logger.info(f"Processed {i + 1}/{len(files_to_process)} files...")

	# Batch insert accumulated data into Milvus
	if all_chunk_data_to_insert:
		logger.info(f"Inserting data for {len(all_chunk_data_to_insert)} chunks into Milvus...")
		try:
			# Assuming client.insert can handle a list of dictionaries
			logger.debug("Calling Milvus client.insert...")
			insert_result = client.insert(collection_name=config.COLLECTION_NAME, data=all_chunk_data_to_insert)
			logger.debug(f"Milvus client.insert call returned: {insert_result}")

			# Check insert result for success/failures
			# Insert result check is now implemented
			insert_count = getattr(insert_result, "insert_count", 0)
			err_count = getattr(insert_result, "err_count", 0)
			succ_count = getattr(insert_result, "succ_count", 0)
			primary_keys = getattr(insert_result, "primary_keys", [])

			expected_count = len(all_chunk_data_to_insert)

			if insert_count == expected_count and err_count == 0:
				logger.info(f"Successfully inserted {insert_count} entities.")
			else:
				logger.warning(
					f"Milvus insert result mismatch or errors. Expected: {expected_count}, "
					f"Inserted: {insert_count}, Success: {succ_count}, Errors: {err_count}. "
					f"Primary Keys ({len(primary_keys)}): {primary_keys[:10]}..."
				)
				# Potentially mark as error or handle partially failed batch?
				# For now, just log warning.

		except Exception:
			logger.exception("Failed to batch insert data into Milvus")
			# Consider adding partial failure handling if possible
			error_count += len(files_to_process) - processed_count  # Mark remaining as errors

	logger.info(f"File processing complete. Processed: {processed_count}, Errors: {error_count}")
