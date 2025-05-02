"""Module for synchronizing Git state with the KuzuDB graph."""

import logging
import subprocess
from pathlib import Path

from codemap.processor.graph.graph_builder import GraphBuilder
from codemap.processor.graph.kuzu_manager import KuzuManager

logger = logging.getLogger(__name__)

# Constants for git operations
GIT_TRACKED_STAGE = "0"  # Stage 0 means file is tracked and not in conflict
GIT_CMD = "/usr/bin/git"  # Full path to git binary for security
MIN_LS_FILE_PARTS = 4  # Minimum number of parts in ls-files output


class GraphSynchronizer:
	"""Compares Git state with KuzuDB and updates the graph."""

	def __init__(self, repo_path: Path, kuzu_manager: KuzuManager, graph_builder: GraphBuilder) -> None:
		"""
		Initialize the synchronizer with repository and database connections.

		Args:
		        repo_path: Path to the repository root
		        kuzu_manager: Manager for KuzuDB operations
		        graph_builder: Builder for creating graph entities

		"""
		self.repo_path = repo_path
		self.kuzu_manager = kuzu_manager
		self.graph_builder = graph_builder

	def get_current_files(self) -> set[str]:
		"""Get current tracked files in the Git repository."""
		try:
			# Using a full path to git executable for security
			result = subprocess.run(  # noqa: S603
				[GIT_CMD, "ls-files"],
				cwd=self.repo_path,
				capture_output=True,
				text=True,
				check=True,
			)
			files = set(result.stdout.splitlines())
			logger.debug(f"Found {len(files)} tracked files in Git repository")
			return files
		except subprocess.CalledProcessError:
			logger.exception("Failed to get Git tracked files")
			return set()

	def get_file_git_hash(self, file_path: str) -> str:
		"""Get Git hash (blob ID) for a specific file."""
		try:
			# Using ls-files with --stage shows the hash
			result = subprocess.run(  # noqa: S603
				[GIT_CMD, "ls-files", "--stage", file_path],
				cwd=self.repo_path,
				capture_output=True,
				text=True,
				check=True,
			)

			# Parse the output to extract the hash
			for line in result.stdout.splitlines():
				parts = line.split()
				# Expected format: <mode> <hash> <stage>\t<path>
				if len(parts) >= MIN_LS_FILE_PARTS:
					# Stage 0 usually means the file is tracked and not in conflict
					stage = parts[2]
					if stage == GIT_TRACKED_STAGE:
						return parts[1]

			logger.warning(f"Could not find git hash for file: {file_path}")
			return ""
		except subprocess.CalledProcessError:
			logger.exception(f"Failed to get Git hash for file {file_path}")
			return ""

	def sync_graph(self) -> bool:
		"""Synchronize the KuzuDB graph with current Git state."""
		try:
			logger.info("Starting graph synchronization...")
			# Step 1: Get current files in Git
			git_files = self.get_current_files()

			# Step 2: Get current files already in KuzuDB with their hashes
			db_files = self.kuzu_manager.get_all_file_hashes()

			# Step 3: Find files to add/update and remove
			file_paths_to_update = set()
			file_paths_to_remove = set()

			# Add/update files that are in Git but not in DB or have different hash
			for file_path in git_files:
				if file_path not in db_files:
					# New file, need to add
					file_paths_to_update.add(file_path)
				else:
					# Check if hash changed
					git_hash = self.get_file_git_hash(file_path)
					db_hash = db_files[file_path]
					if git_hash != db_hash and git_hash:
						# Hash changed, need to update
						file_paths_to_update.add(file_path)

			# Remove files that are in DB but no longer in Git
			for file_path in db_files:
				if file_path not in git_files:
					file_paths_to_remove.add(file_path)

			# Step 4: Process updates and removals
			if not file_paths_to_update and not file_paths_to_remove:
				logger.info("No changes detected. Graph is up to date.")
				return True

			# Update files in graph
			for file_path in file_paths_to_update:
				git_hash = self.get_file_git_hash(file_path)
				if not git_hash:
					continue  # Skip if we couldn't get a hash

				full_path = self.repo_path / file_path
				if not full_path.exists() or not full_path.is_file():
					logger.warning(f"File not found or not a regular file: {full_path}")
					continue

				# First remove any existing data for this file
				if file_path in db_files:
					self.kuzu_manager.delete_file_data(file_path)

				# Then add the current version
				success = self.graph_builder.process_file(full_path, git_hash)
				if success:
					logger.info(f"Updated file in graph: {file_path}")
				else:
					logger.warning(f"Failed to update file in graph: {file_path}")

			# Remove files from graph
			for file_path in file_paths_to_remove:
				self.kuzu_manager.delete_file_data(file_path)
				logger.info(f"Removed file from graph: {file_path}")

			# Log a summary of the changes
			logger.info(
				f"Graph synchronization completed. "
				f"Updated: {len(file_paths_to_update)}, Removed: {len(file_paths_to_remove)}"
			)
			return True
		except Exception:
			logger.exception("Failed during graph synchronization")
			return False
