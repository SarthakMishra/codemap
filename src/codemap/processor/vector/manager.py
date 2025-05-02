"""Orchestrates the vector database synchronization process."""

import logging
import subprocess
from pathlib import Path

from pymilvus import MilvusClient, exceptions

from codemap.utils.config_loader import ConfigLoader

from . import chunker, client, config, embedder

logger = logging.getLogger(__name__)

GitFileInfo = dict[str, str]  # Maps file_path (str) to git_hash (str)
MilvusFileInfo = dict[str, set[str]]  # Maps file_path (str) to set of git_hash (str)

MIN_GIT_LS_FILES_PARTS = 4  # Constant for magic number 4


def synchronize_vectors(repo_path: Path) -> None:
	"""
	Synchronizes the Milvus vector database with the current Git state.

	1. Gets current tracked files and hashes from Git.
	2. Gets existing file paths and hashes from Milvus.
	3. Determines files to add, update, or delete.
	4. Processes changes: chunks, embeds, and updates Milvus.

	"""
	logger.info(f"Starting vector database synchronization for repo: {repo_path}")

	milvus_client = client.get_milvus_client()
	if not milvus_client:
		logger.error("Cannot synchronize: Failed to get Milvus client.")
		return

	# --- Load Configuration --- #
	try:
		config_loader = ConfigLoader(repo_root=repo_path)
		config_data = config_loader.config
		logger.info("Successfully loaded configuration.")
	except Exception:
		logger.exception(f"Cannot synchronize: Failed to load configuration from {repo_path}")
		return
	# --- End Load Configuration --- #

	# 1. Get Git state
	current_git_files = _get_git_tracked_files(repo_path)
	if current_git_files is None:
		logger.error("Cannot synchronize: Failed to get Git tracked files.")
		return
	logger.info(f"Found {len(current_git_files)} tracked files in Git.")

	# 2. Get Milvus state
	existing_milvus_files = _get_milvus_file_hashes(milvus_client)
	if existing_milvus_files is None:
		logger.error("Cannot synchronize: Failed to get existing files from Milvus.")
		return
	logger.info(f"Found {len(existing_milvus_files)} files represented in Milvus.")

	# 3. Determine changes
	files_to_add, files_to_update, files_to_delete = _compare_states(current_git_files, existing_milvus_files)

	total_changes = len(files_to_add) + len(files_to_update) + len(files_to_delete)
	if total_changes == 0:
		logger.info("Vector database is already synchronized. No changes needed.")
		return

	# Format long log message
	logger.info(
		f"Changes identified: {len(files_to_add)} to add, "
		f"{len(files_to_update)} to update, {len(files_to_delete)} to delete."
	)

	# 4. Process changes
	# Process deletions first
	_delete_files_from_milvus(milvus_client, files_to_delete)

	# Process additions and updates (involve reading, chunking, embedding, inserting)
	files_to_process = files_to_add.union(files_to_update)
	_process_files(milvus_client, repo_path, files_to_process, current_git_files, config_data)

	logger.info("Vector database synchronization finished.")


def _get_git_tracked_files(repo_path: Path) -> GitFileInfo | None:
	"""Uses 'git ls-files -s' to get tracked files and their blob hashes."""
	try:
		# Run git command
		# -s shows staged contents' mode bits, object name and stage number
		# --full-name ensures paths are relative to repo root
		# S603/S607: Calling git is considered safe here as it's a standard tool
		# and we are not constructing commands from user input.
		result = subprocess.run(  # noqa: S603
			["git", "ls-files", "-s", "--full-name"],  # noqa: S607
			cwd=repo_path,
			capture_output=True,
			text=True,
			check=True,
			encoding="utf-8",
		)

		files_info: GitFileInfo = {}
		for line in result.stdout.strip().split("\n"):
			if not line:
				continue
			# Format: <mode> <object> <stage>\t<file>
			parts = line.split()
			# Use constant instead of magic number
			if len(parts) >= MIN_GIT_LS_FILES_PARTS:
				git_hash = parts[1]
				file_path = parts[3]
				# For simplicity, ignore different stages for now, take the first seen hash
				if file_path not in files_info:
					files_info[file_path] = git_hash
			else:
				logger.warning(f"Could not parse git ls-files line: {line}")

		return files_info

	except FileNotFoundError:
		logger.exception("'git' command not found. Is Git installed and in PATH?")
		return None
	except subprocess.CalledProcessError as e:
		logger.exception("Git command failed")
		logger.exception(f"Stderr: {e.stderr}")
		return None
	except Exception:
		logger.exception("Error getting Git tracked files")
		return None


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
		for batch in results_iterator:
			for result in batch:
				file_path = result.get(config.FIELD_FILE_PATH)
				git_hash = result.get(config.FIELD_GIT_HASH)
				if file_path and git_hash:
					if file_path not in milvus_files:
						milvus_files[file_path] = set()
					milvus_files[file_path].add(git_hash)
					processed_count += 1

		logger.debug(f"Processed {processed_count} entries from Milvus.")
		return milvus_files

	except exceptions.MilvusException:
		logger.exception("Milvus error querying file hashes")
		return None
	except Exception:
		logger.exception("Unexpected error getting Milvus file hashes")
		return None


def _compare_states(git_files: GitFileInfo, milvus_files: MilvusFileInfo) -> tuple[set[str], set[str], set[str]]:
	"""Compares Git and Milvus states to find differences."""
	git_paths = set(git_files.keys())
	milvus_paths = set(milvus_files.keys())

	# Files in Git but not Milvus
	files_to_add = git_paths - milvus_paths

	# Files in Milvus but not Git (deleted)
	files_to_delete = milvus_paths - git_paths

	# Files in both - check hash
	files_to_update = set()
	common_paths = git_paths.intersection(milvus_paths)
	for path in common_paths:
		current_git_hash = git_files[path]
		# Check if the current hash is NOT the ONLY hash present in Milvus for this path
		# This handles cases where old versions might still be in Milvus due to previous errors
		if current_git_hash not in milvus_files[path] or len(milvus_files[path]) > 1:
			files_to_update.add(path)

	return files_to_add, files_to_update, files_to_delete


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


def _process_files(
	milvus_client: MilvusClient,
	repo_path: Path,
	files_to_process: set[str],
	git_file_info: GitFileInfo,
	config_data: dict,
) -> None:
	"""
	Reads, chunks, embeds, and inserts/updates data for the given files.

	Handles deleting old entries for updated files before inserting new
	ones.

	"""
	if not files_to_process:
		return

	logger.info(f"Processing {len(files_to_process)} files for addition/update...")

	# First, delete existing entries for files that need updating
	_delete_files_from_milvus(milvus_client, files_to_process)  # Safe to call even if file wasn't in Milvus (additions)

	processed_count = 0
	error_count = 0
	all_chunk_data_to_insert = []  # Accumulate data for batch insertion

	for file_path_str in files_to_process:
		file_path = repo_path / file_path_str
		current_git_hash = git_file_info.get(file_path_str)

		if not current_git_hash:
			logger.warning(f"Skipping {file_path_str}: Could not find its current git hash.")
			continue

		try:
			# Read file content - Use Path.open()
			# with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
			with file_path.open("r", encoding="utf-8", errors="ignore") as f:
				content = f.read()

			# Chunk the file - pass config_data
			chunks = chunker.chunk_file(Path(file_path_str), content, current_git_hash, config_data)
			if chunks is None:
				logger.warning(f"Skipping {file_path_str}: Chunking failed.")
				error_count += 1
				continue

			if not chunks:
				logger.debug(f"No chunks generated for {file_path_str}. Skipping insertion.")
				continue

			# Extract text for embedding
			texts_to_embed = [chunk[config.FIELD_CHUNK_TEXT] for chunk in chunks]

			# Generate embeddings
			embeddings = embedder.generate_embeddings(texts_to_embed)
			if embeddings is None:
				logger.warning(f"Skipping {file_path_str}: Embedding failed.")
				error_count += 1
				continue

			# Prepare data for Milvus insertion (list of dicts)
			data_batch = []
			for i, chunk in enumerate(chunks):
				# Ensure embedding is a list for Milvus client
				embedding_list = embeddings[i].tolist()
				chunk_data = {**chunk, config.FIELD_EMBEDDING: embedding_list}
				# Remove FIELD_CHUNK_TEXT if schema doesn't store full text (optional)
				# if not STORE_FULL_TEXT_IN_SCHEMA:
				#    del chunk_data[config.FIELD_CHUNK_TEXT]
				data_batch.append(chunk_data)

			all_chunk_data_to_insert.extend(data_batch)
			processed_count += 1
			logger.debug(f"Prepared {len(data_batch)} chunks for {file_path_str}.")

		except FileNotFoundError:
			logger.warning(f"Skipping {file_path_str}: File not found during processing (might be race condition?).")
			error_count += 1
		except Exception:
			logger.exception(f"Error processing file {file_path_str}")
			error_count += 1

	# Batch insert accumulated data into Milvus
	if all_chunk_data_to_insert:
		logger.info(f"Inserting {len(all_chunk_data_to_insert)} chunks into Milvus...")
		try:
			# MilvusClient insert takes list of dictionaries
			res = milvus_client.insert(collection_name=config.COLLECTION_NAME, data=all_chunk_data_to_insert)
			# Corrected logging for Milvus insert result - Use getattr for safer access
			insert_count = getattr(res, "insert_count", "N/A")
			logger.info(f"Milvus insertion result: PKs inserted - {insert_count}")  # Shows count
		except exceptions.MilvusException:
			logger.exception("Milvus error during batch insertion")
			error_count += len(all_chunk_data_to_insert)  # Count all as errors if batch fails
		except Exception:
			logger.exception("Unexpected error during batch insertion")
			error_count += len(all_chunk_data_to_insert)

	logger.info(f"File processing finished. Successfully processed: {processed_count}, Errors: {error_count}")


# Example Usage
if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)
	# Assumes running from the project root
	repo_directory = Path()
	logger.info("Running vector synchronization...")
	synchronize_vectors(repo_directory)
	logger.info("Synchronization process complete.")
