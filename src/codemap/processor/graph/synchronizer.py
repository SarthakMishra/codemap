"""Module for synchronizing Git state with the KuzuDB graph."""

import logging
from pathlib import Path

from rich.progress import Progress, TaskID

from codemap.processor.graph.graph_builder import GraphBuilder
from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.utils.git_utils import get_git_tracked_files
from codemap.processor.utils.sync_utils import compare_states

# Need imports for type hinting progress parameters

logger = logging.getLogger(__name__)

# Constant for expected number of columns in _get_db_file_hashes query result
EXPECTED_HASH_QUERY_COLUMNS = 2

# Constants for progress display
MAX_PATH_DISPLAY_LENGTH = 40
TRUNCATED_PATH_SUFFIX_LENGTH = 37


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

	def sync_graph(
		self,
		current_git_files: dict[str, str] | None = None,
		progress: Progress | None = None,  # Added
		task_id: TaskID | None = None,  # Added
	) -> bool:
		"""
		Synchronizes the Kuzu graph database with the current Git state.

		Args:
		    current_git_files: A dictionary mapping file paths to their Git hashes.
		                       If None, it will be fetched.
		    progress: Optional rich Progress instance passed from the caller.
		    task_id: Optional rich TaskID for the relevant progress task.

		Returns:
		    True if synchronization was successful (or no changes needed), False otherwise.

		"""
		# Handle case where current_git_files might be None if called directly
		if current_git_files is None:
			logger.warning("sync_graph called without current_git_files, fetching...")
			if progress and task_id:
				progress.update(task_id, description="Fetching Git state...")
			current_git_files = get_git_tracked_files(self.repo_path)
			if current_git_files is None:
				logger.error("Graph sync failed: Could not get Git tracked files.")
				if progress and task_id:
					progress.update(task_id, description="[red]Error:[/red] Failed fetching Git state")
				return False

		logger.info("Starting Kuzu graph synchronization...")
		if progress and task_id:
			progress.update(task_id, description="Comparing Git and DB states...", completed=35)
		all_success = True

		# 1. Get existing nodes from Kuzu
		existing_db_files = self.get_db_file_hashes()
		if existing_db_files is None:
			logger.error("Graph sync failed: Could not retrieve existing file hashes from Kuzu.")
			if progress and task_id:
				progress.update(task_id, description="[red]Error:[/red] Failed querying DB state")
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
			if progress and task_id:
				progress.update(task_id, description="[green]✓[/green] Graph up-to-date.", completed=100)
			return True

		log_message = (
			f"Kuzu synchronization required: {len(files_to_add)} to add, "
			f"{len(files_to_update)} to update, {len(files_to_delete)} to delete."
		)
		logger.info(log_message)
		if progress and task_id:
			progress.update(
				task_id,
				description=(
					f"Processing: {len(files_to_add)} add, {len(files_to_update)} update, "
					f"{len(files_to_delete)} delete..."
				),
				completed=40,
			)

		# 3. Process deletions
		if files_to_delete:
			logger.info(f"Deleting {len(files_to_delete)} files/communities from Kuzu...")
			if progress and task_id:
				progress.update(task_id, description=f"Deleting {len(files_to_delete)} files...", completed=45)
			delete_success = self._delete_files_from_graph(files_to_delete)
			if not delete_success:
				logger.warning("Errors occurred during Kuzu deletion phase.")
				all_success = False  # Mark sync as potentially incomplete

		# 4. Process additions and updates
		files_to_process = files_to_add.union(files_to_update)
		if files_to_process:
			logger.info(f"Processing {len(files_to_process)} files for Kuzu addition/update...")
			update_count = 0  # Track updates specifically
			add_count = 0  # Track additions specifically
			error_count = 0

			# Calculate progress increments
			completed_base = 50  # Start at 50% after deletions
			completed_increment = 45 / max(len(files_to_process), 1)  # Distribute remaining 45% across files
			completed_current = completed_base

			# Instead, use a regular loop and manually update the original task_id
			if progress and task_id:
				progress.update(task_id, description="Processing files...", completed=completed_base)

			for i, file_path_str in enumerate(files_to_process):
				# Update description to show current file (truncate if too long)
				display_path = file_path_str
				if len(display_path) > MAX_PATH_DISPLAY_LENGTH:
					display_path = "..." + display_path[-TRUNCATED_PATH_SUFFIX_LENGTH:]

				if progress and task_id:
					file_num = i + 1
					int((file_num / len(files_to_process)) * 100)
					progress.update(
						task_id,
						description=f"Processing file {file_num}/{len(files_to_process)}: {display_path}",
						completed=completed_current,
					)

				log_prefix = f"File '{file_path_str}'"
				file_path = self.repo_path / file_path_str
				if not file_path.is_file():
					logger.warning(f"Skipping {log_prefix}: File not found at expected location {file_path}")
					error_count += 1
					completed_current += completed_increment
					continue

				is_update = file_path_str in files_to_update
				if is_update:
					logger.debug(f"Deleting existing nodes for updated file: {file_path_str}")
					if not self._delete_file_from_graph(file_path_str):
						logger.warning(
							f"Failed to delete existing nodes for updated file: {file_path_str}. Skipping update."
						)
						error_count += 1
						completed_current += completed_increment
						continue

				try:
					git_hash = current_git_files.get(file_path_str)
					self.graph_builder.process_file(file_path, git_hash=git_hash)
					if is_update:
						update_count += 1
					else:
						add_count += 1

					# Update progress
					completed_current += completed_increment
					if progress and task_id:
						progress.update(task_id, completed=min(completed_current, 95))
				except Exception:
					logger.exception(f"Error processing {log_prefix} for Kuzu graph.")
					error_count += 1
					completed_current += completed_increment
					all_success = False

			# --- Rebuild Vector Index --- #
			if files_to_process:  # Only rebuild if there were changes
				if progress and task_id:
					progress.update(task_id, description="Rebuilding vector index...", completed=95)
				try:
					logger.info("Rebuilding vector index after file updates/additions...")
					if not self.kuzu_manager.drop_vector_index():
						logger.warning("Failed to drop vector index, continuing with rebuild attempt...")

					if self.kuzu_manager.create_vector_index():
						logger.info("Vector index rebuilt successfully.")
					else:
						logger.error("Failed to rebuild vector index.")
						# Consider how to handle index rebuild failures
				except Exception:
					logger.exception("Unexpected error during vector index rebuild in sync_graph")
					# all_success = False # Uncomment if rebuild failure should mark the whole sync as failed

			log_message_processed = (
				f"Processed {len(files_to_process)} files: {add_count} added, "
				f"{update_count} updated, {len(files_to_delete)} deleted, "
				f"Errors/Skipped: {error_count}"
			)
			logger.info(log_message_processed)

		logger.info(f"Kuzu graph synchronization finished. Overall success status: {all_success}")
		if progress and task_id:  # Final update for the main task
			final_desc = "[green]✓[/green] Graph sync finished."
			if not all_success:
				final_desc = "[yellow]⚠[/yellow] Graph sync finished with issues."
			progress.update(task_id, description=final_desc, completed=99)
		return all_success

	def get_db_file_hashes(self) -> dict[str, str] | None:
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
		"""Deletes CodeFile nodes and associated data for given paths."""
		if not files_to_delete:
			return True
		success = True
		for file_path_str in files_to_delete:
			if not self._delete_file_from_graph(file_path_str):  # Call helper
				success = False
		return success

	def _delete_file_from_graph(self, file_path_str: str) -> bool:
		"""Helper to delete data for a single file."""
		log_prefix = f"File '{file_path_str}'"
		try:
			self.kuzu_manager.delete_file_data(file_path_str)
			logger.debug(f"Successfully deleted data for {log_prefix}")
			return True
		except Exception:
			logger.exception(f"Error deleting {log_prefix} from Kuzu graph.")
			return False

	def delete_files_from_graph(self, file_paths: set[str]) -> bool:
		"""
		Delete multiple files from the graph database.

		Args:
		        file_paths: Set of file paths to delete

		Returns:
		        True if all files were deleted successfully, False otherwise

		"""
		return self._delete_files_from_graph(file_paths)

	def delete_file_from_graph(self, file_path: str) -> bool:
		"""
		Delete a single file from the graph database.

		Args:
		        file_path: Path of the file to delete

		Returns:
		        True if file was deleted successfully, False otherwise

		"""
		return self._delete_file_from_graph(file_path)
