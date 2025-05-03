"""Module for synchronizing Git state with the KuzuDB graph."""

import logging
from pathlib import Path

from codemap.processor.graph.graph_builder import GraphBuilder
from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.utils.git_utils import get_git_tracked_files
from codemap.processor.utils.sync_utils import compare_states

logger = logging.getLogger(__name__)

# Constant for expected number of columns in _get_db_file_hashes query result
EXPECTED_HASH_QUERY_COLUMNS = 2


class GraphSynchronizer:
	"""Handles synchronization of the Kuzu graph database with Git state."""

	def __init__(
		self,
		repo_path: Path,
		kuzu_manager: KuzuManager,
		graph_builder: GraphBuilder,
	) -> None:
		"""
		Initialize the GraphSynchronizer.

		Args:
		    repo_path: Path to the repository root.
		    kuzu_manager: Instance of KuzuManager.
		    graph_builder: Instance of GraphBuilder.

		"""
		self.repo_path = repo_path
		self.kuzu_manager = kuzu_manager
		self.graph_builder = graph_builder

	def sync_graph(self, current_git_files: dict[str, str] | None = None) -> bool:
		"""
		Synchronizes the Kuzu graph database with the current Git state.

		Args:
		    current_git_files: A dictionary mapping file paths to their Git hashes.
		                       If None, it will be fetched.

		Returns:
		    True if synchronization was successful (or no changes needed), False otherwise.

		"""
		# Handle case where current_git_files might be None if called directly
		if current_git_files is None:
			logger.warning("sync_graph called without current_git_files, fetching...")
			current_git_files = get_git_tracked_files(self.repo_path)
			if current_git_files is None:
				logger.error("Graph sync failed: Could not get Git tracked files.")
				return False

		logger.info("Starting Kuzu graph synchronization...")
		all_success = True

		# 1. Get existing nodes from Kuzu
		existing_db_files = self._get_db_file_hashes()
		if existing_db_files is None:
			logger.error("Graph sync failed: Could not retrieve existing file hashes from Kuzu.")
			return False

		# Log details about states being compared (use f-strings and maybe sample)
		git_files_count = len(current_git_files)
		db_files_count = len(existing_db_files)
		logger.debug(f"Comparing Git state ({git_files_count} entries) with Kuzu state ({db_files_count} entries).")
		if git_files_count > 0:
			logger.debug(f"  Git sample: {dict(list(current_git_files.items())[:5])}")
		if db_files_count > 0:
			logger.debug(f"  Kuzu sample: {dict(list(existing_db_files.items())[:5])}")

		# 2. Compare states
		files_to_add, files_to_update, files_to_delete = compare_states(current_git_files, existing_db_files)

		total_changes = len(files_to_add) + len(files_to_update) + len(files_to_delete)
		if total_changes == 0:
			logger.info("Kuzu graph database is already up-to-date.")
			return True

		log_message = (
			f"Kuzu synchronization required: {len(files_to_add)} to add, "
			f"{len(files_to_update)} to update, {len(files_to_delete)} to delete."
		)
		logger.info(log_message)

		# 3. Process deletions
		if files_to_delete:
			logger.info(f"Deleting {len(files_to_delete)} files/communities from Kuzu...")
			delete_success = self._delete_files_from_graph(files_to_delete)
			if not delete_success:
				logger.warning("Errors occurred during Kuzu deletion phase.")
				all_success = False  # Mark sync as potentially incomplete

		# 4. Process additions and updates
		files_to_process = files_to_add.union(files_to_update)
		if files_to_process:
			logger.info(f"Processing {len(files_to_process)} files for Kuzu addition/update...")
			processed_count = 0
			error_count = 0
			for file_path_str in files_to_process:
				log_prefix = f"File '{file_path_str}'"  # Simplified log prefix

				file_path = self.repo_path / file_path_str
				if not file_path.is_file():
					logger.warning(f"Skipping {log_prefix}: File not found at expected location {file_path}")
					error_count += 1
					continue

				# --- Delete existing nodes for updated files before adding new ones ---
				# This ensures relationships are updated correctly.
				if file_path_str in files_to_update:
					logger.debug(f"Deleting existing nodes for updated file: {file_path_str}")
					# Use the same delete function, passing only the single file
					if not self._delete_files_from_graph({file_path_str}):
						logger.warning(
							f"Failed to delete existing nodes for updated file: {file_path_str}. Skipping update."
						)
						error_count += 1
						continue  # Skip processing this file if deletion failed

				# Add/update file using GraphBuilder
				try:
					self.graph_builder.process_file(file_path)
					processed_count += 1

				except Exception:
					logger.exception(f"Error processing {log_prefix} for Kuzu graph.")
					error_count += 1
					all_success = False

			log_message_processed = (
				f"Kuzu processing finished. Processed: {processed_count}, Errors/Skipped: {error_count}"
			)
			logger.info(log_message_processed)

		logger.info(f"Kuzu graph synchronization finished. Overall success status: {all_success}")
		return all_success

	def _get_db_file_hashes(self) -> dict[str, str] | None:
		"""Retrieves all file paths and their hashes currently stored in Kuzu."""
		try:
			query = "MATCH (f:CodeFile) RETURN f.file_path, f.git_hash"
			results = self.kuzu_manager.execute_query(query)
			if results is None:
				logger.error("Failed to retrieve file hashes from Kuzu.")
				return None
			# Use constant for checking number of columns
			return {row[0]: row[1] for row in results if len(row) == EXPECTED_HASH_QUERY_COLUMNS and row[0] and row[1]}
		except Exception:
			logger.exception("Error retrieving file hashes from Kuzu")
			return None

	def _delete_files_from_graph(self, files_to_delete: set[str]) -> bool:
		"""Deletes CodeFile nodes and associated CodeCommunity nodes (if empty) for given paths."""
		if not files_to_delete:
			return True
		success = True
		# Note: Kuzu doesn't easily support deleting based on a list parameter directly in WHERE IN.
		# Iterate and delete one by one or construct a complex query. Iteration is simpler.
		for file_path_str in files_to_delete:
			log_prefix = f"File '{file_path_str}'"  # Simplified log prefix

			try:
				# Delete the CodeFile node and its relationships
				# Using DETACH DELETE handles relationships automatically
				delete_file_query = "MATCH (f:CodeFile {file_path: $path}) DETACH DELETE f"
				params = {"path": file_path_str}
				result = self.kuzu_manager.execute_query(delete_file_query, params)
				# Check result? KuzuPy doesn't return counts easily on DELETE. Assume success if no exception.
				if result is None:  # Query execution failed
					logger.warning(f"Kuzu query execution failed when trying to delete {log_prefix}.")
					success = False

				# Optional: Add logic here to check if the parent CodeCommunity is now empty
				# and delete it if necessary. This requires more complex queries involving
				# relationship counts or checking for remaining children.
				# Example (conceptual):
				# MATCH (parent:CodeCommunity)<-[:BELONGS_TO_COMMUNITY]-(f:CodeFile {file_path: $path})
				# WITH parent
				# MATCH (parent)-[:CONTAINS_FILE]->(childFile:CodeFile)
				# WITH parent, count(childFile) AS fileCount
				# MATCH (parent)-[:CONTAINS_COMMUNITY]->(childCommunity:CodeCommunity)
				# WITH parent, fileCount, count(childCommunity) AS communityCount
				# WHERE fileCount = 0 AND communityCount = 0
				# DETACH DELETE parent

			except Exception:
				logger.exception(f"Error deleting {log_prefix} from Kuzu graph.")
				success = False
		return success
