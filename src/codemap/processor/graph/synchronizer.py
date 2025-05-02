"""Module for synchronizing Git state with the KuzuDB graph."""

import logging
from pathlib import Path

from codemap.processor.graph.graph_builder import GraphBuilder
from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.utils.git_utils import get_git_tracked_files
from codemap.processor.utils.path_utils import get_workspace_root
from codemap.processor.utils.sync_utils import compare_states

logger = logging.getLogger(__name__)

# Removed constants related to Git commands


class GraphSynchronizer:
	"""Compares Git state with KuzuDB and updates the graph."""

	def __init__(self, repo_path: Path | None, kuzu_manager: KuzuManager, graph_builder: GraphBuilder) -> None:
		"""
		Initialize the synchronizer with repository and database connections.

		Args:
			repo_path (Path | None): Path to the repository root. If None, it's determined.
			kuzu_manager (KuzuManager): Manager for KuzuDB operations.
			graph_builder (GraphBuilder): Builder for creating graph entities.
		"""
		if repo_path is None:
			try:
				self.repo_path = get_workspace_root()
			except FileNotFoundError:
				logger.exception("GraphSynchronizer init failed: Could not determine repository path.")
				raise
		else:
			self.repo_path = repo_path

		self.kuzu_manager = kuzu_manager
		self.graph_builder = graph_builder
		logger.info(f"GraphSynchronizer initialized for repo: {self.repo_path}")

	def sync_graph(self) -> bool:
		"""
		Synchronize the KuzuDB graph with current Git state.

		Returns:
			bool: True if synchronization was successful (or not needed),
				  False otherwise.
		"""
		logger.info(f"Starting graph synchronization for: {self.repo_path}")

		# 1. Get current Git state (files and hashes)
		logger.debug("Fetching current Git state...")
		current_git_files = get_git_tracked_files(self.repo_path)
		if current_git_files is None:
			logger.error("Synchronization failed: Could not get Git tracked files.")
			return False
		logger.debug(f"Found {len(current_git_files)} files in Git.")

		# 2. Get KuzuDB state (files and hashes)
		logger.debug("Fetching current KuzuDB state...")
		try:
			db_file_hashes = self.kuzu_manager.get_all_file_hashes()
			logger.debug(f"Found {len(db_file_hashes)} files represented in KuzuDB.")
		except Exception:
			logger.exception("Synchronization failed: Could not get file hashes from KuzuDB.")
			return False

		# 3. Compare states
		logger.debug("Comparing Git state with KuzuDB state...")
		files_to_add, files_to_update, files_to_delete = compare_states(current_git_files, db_file_hashes)

		total_changes = len(files_to_add) + len(files_to_update) + len(files_to_delete)
		if total_changes == 0:
			logger.info("Graph database is already up-to-date.")
			return True

		log_message = (
			f"Synchronization required: {len(files_to_add)} to add, "
			f"{len(files_to_update)} to update, {len(files_to_delete)} to delete."
		)
		logger.info(log_message)

		# 4. Process deletions
		if files_to_delete:
			logger.info(f"Deleting data for {len(files_to_delete)} files from KuzuDB...")
			deleted_count = 0
			errors_deleting = 0
			for file_path in files_to_delete:
				try:
					self.kuzu_manager.delete_file_data(file_path)
					deleted_count += 1
				except Exception:
					logger.exception(f"Failed to delete data for file: {file_path}")
					errors_deleting += 1
			logger.info(f"Finished deleting files. Deleted: {deleted_count}, Errors: {errors_deleting}")
			if errors_deleting > 0:
				logger.warning("Some errors occurred during file data deletion.")

		# 5. Process additions and updates
		files_to_process = files_to_add.union(files_to_update)
		if files_to_process:
			logger.info(f"Processing {len(files_to_process)} files for graph updates...")
			processed_count = 0
			errors_processing = 0
			for i, file_path_str in enumerate(files_to_process):
				git_hash = current_git_files.get(file_path_str)
				if not git_hash:
					logger.error(f"Consistency error: Cannot find hash for file to process: {file_path_str}. Skipping.")
					errors_processing += 1
					continue

				logger.debug(f"Processing file {i + 1}/{len(files_to_process)}: {file_path_str}")
				try:
					file_path_obj = self.repo_path / file_path_str
					success = self.graph_builder.process_file(file_path_obj, git_hash)
					if success:
						processed_count += 1
					else:
						logger.warning(f"Failed to process file: {file_path_str}")
						errors_processing += 1
				except Exception:
					logger.exception(f"Error processing file: {file_path_str}")
					errors_processing += 1

			logger.info(f"Finished processing files. Processed: {processed_count}, Errors: {errors_processing}")
			if errors_processing > 0:
				logger.error("Synchronization completed with errors during file processing.")
				return False

		logger.info("Graph synchronization finished successfully.")
		return True
