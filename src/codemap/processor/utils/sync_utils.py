"""Utilities for synchronization logic."""

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def compare_states(
	current_files: Mapping[str, str],
	db_files: Mapping[str, str | set[str]],
) -> tuple[set[str], set[str], set[str]]:
	"""
	Compare current file state with database state to find differences.

	Handles cases where the database might store multiple hashes per file path
	(e.g., if different chunks of the same file have different source hashes,
	although typically it should be one hash per file path in the DB state dict).

	Args:
	    current_files (Mapping[str, str]): Dictionary mapping file paths to their
	                                     current hash (e.g., from Git).
	    db_files (Mapping[str, str | set[str]]): Dictionary mapping file paths to their
	                                           hash(es) stored in the database.
	                                           Values can be single hashes (str) or
	                                           sets of hashes (set[str]).

	Returns:
	    tuple[set[str], set[str], set[str]]: A tuple containing:
	        - files_to_add: Set of file paths present in current_files but not db_files.
	        - files_to_update: Set of file paths present in both, but with different hashes.
	        - files_to_delete: Set of file paths present in db_files but not current_files.

	"""
	current_paths = set(current_files.keys())
	db_paths = set(db_files.keys())

	# Files in current state but not in DB -> Add
	files_to_add = current_paths - db_paths

	# Files in DB but not in current state -> Delete
	files_to_delete = db_paths - current_paths

	# Files in both -> Check hash for updates
	files_to_update: set[str] = set()
	common_paths = current_paths.intersection(db_paths)

	for path in common_paths:
		current_hash = current_files[path]
		db_hash_or_hashes = db_files[path]

		needs_update = False
		if isinstance(db_hash_or_hashes, str):
			# DB stores a single hash for the file
			if current_hash != db_hash_or_hashes:
				needs_update = True
		elif isinstance(db_hash_or_hashes, set):
			# DB stores multiple hashes (e.g., different versions/chunks)
			# Update if the current hash is not among the DB hashes
			if current_hash not in db_hash_or_hashes:
				needs_update = True
			# Optional: Consider updating if the set doesn't *exactly* match
			# (e.g., if DB has extra hashes not in current state -> cleanup?)
			# For now, just check if the current hash exists.
		else:
			logger.warning(f"Unexpected hash type in db_files for path '{path}': {type(db_hash_or_hashes)}")
			# Treat as needing update to be safe
			needs_update = True

		if needs_update:
			files_to_update.add(path)

	logger.debug(
		f"State comparison results: Add: {len(files_to_add)}, "
		f"Update: {len(files_to_update)}, Delete: {len(files_to_delete)}"
	)
	return files_to_add, files_to_update, files_to_delete
