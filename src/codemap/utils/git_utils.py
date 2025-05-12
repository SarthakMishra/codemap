"""Utilities for interacting with Git."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pygit2 import Commit, discover_repository
from pygit2.enums import FileStatus
from pygit2.repository import Repository

from codemap.processor.vector.schema import GitBlameSchema, GitMetadataSchema

if TYPE_CHECKING:
	from pygit2.blame import Blame

logger = logging.getLogger(__name__)


class GitError(Exception):
	"""Custom exception for Git-related errors."""


class GitRepoContext:
	"""Context manager for efficient Git operations using pygit2."""

	logger = logging.getLogger(__name__)

	repo_root: Path | None = None
	_instance: "GitRepoContext | None" = None

	@classmethod
	def get_instance(cls) -> "GitRepoContext":
		"""Get a cached instance of GitRepoContext for a given repo_path."""
		if cls._instance is None:
			cls._instance = cls()
		return cls._instance

	@classmethod
	def get_repo_root(cls, path: Path | None = None) -> Path:
		"""Get the root directory of the Git repository."""
		git_dir = discover_repository(str(path or Path.cwd()))
		if git_dir is None:
			msg = "Not a git repository"
			logger.error(msg)
			raise GitError(msg)
		return Path(git_dir)

	def __init__(self) -> None:
		"""Initialize the GitRepoContext with the given repository path."""
		if self.repo_root is None:
			self.repo_root = self.get_repo_root()
		self.repo = Repository(str(self.repo_root))
		self.branch = self.get_current_branch()
		self.exclude_patterns = self._get_exclude_patterns()
		self.tracked_files = self._get_tracked_files()
		self._blame_cache: dict[str, Blame] = {}  # Cache for blame objects

	@staticmethod
	def _get_exclude_patterns() -> list[str]:
		"""
		Get the list of path patterns to exclude from processing.

		Returns:
			List of regex patterns for paths to exclude
		"""
		from codemap.config import ConfigLoader  # Local import to avoid cycles

		config_loader = ConfigLoader.get_instance()
		config_patterns = config_loader.get.sync.exclude_patterns
		default_patterns = [
			r"^node_modules/",
			r"^\.venv/",
			r"^venv/",
			r"^env/",
			r"^__pycache__/",
			r"^\.mypy_cache/",
			r"^\.pytest_cache/",
			r"^\.ruff_cache/",
			r"^dist/",
			r"^build/",
			r"^\.git/",
			r"^typings/",
			r"\.pyc$",
			r"\.pyo$",
			r"\.so$",
			r"\.dll$",
			r"\.lib$",
			r"\.a$",
			r"\.o$",
			r"\.class$",
			r"\.jar$",
		]
		patterns = list(config_patterns)
		for pattern in default_patterns:
			if pattern not in patterns:
				patterns.append(pattern)
		return patterns

	def _should_exclude_path(self, file_path: str) -> bool:
		"""
		Check if a file path should be excluded from processing based on patterns.

		Args:
			file_path: The file path to check

		Returns:
			True if the path should be excluded, False otherwise
		"""
		for pattern in self.exclude_patterns:
			if re.search(pattern, file_path):
				self.logger.debug(f"Excluding file from processing due to pattern '{pattern}': {file_path}")
				return True
		return False

	def _get_tracked_files(self) -> dict[str, str]:
		"""
		Get all tracked files in the Git repository with their blob hashes.

		Returns:
			dict[str, str]: A dictionary of tracked files with their blob hashes.
		"""
		tracked_files: dict[str, str] = {}
		for entry in self.repo.index:
			if not self._should_exclude_path(entry.path):
				tracked_files[entry.path] = str(entry.id)
		self.logger.info(f"Found {len(tracked_files)} tracked files in Git repository: {self.repo.path}")
		return tracked_files

	def get_current_branch(self) -> str:
		"""
		Get the current branch name of the Git repository.

		Returns:
			str: The current branch name, or empty string if detached.
		"""
		if self.repo.head_is_detached:
			return ""
		return self.repo.head.shorthand or ""

	def get_file_git_hash(self, file_path: str) -> str:
		"""
		Get the Git hash (blob ID) for a specific tracked file.

		Args:
			file_path (str): The path to the file relative to the repository root.

		Returns:
			str: The Git blob hash of the file, or empty string if not found.
		"""
		try:
			commit = self.repo.head.peel(Commit)
			if commit is None:
				self.logger.warning(f"HEAD does not point to a commit in repo {self.repo.path}")
				return ""
			tree = commit.tree
			entry = tree[file_path]
			return str(entry.id)
		except KeyError:
			self.logger.warning(f"File {file_path} not found in HEAD tree of repo {self.repo.path}")
			return ""
		except Exception:
			self.logger.exception(f"Failed to get git hash for {file_path}")
			return ""

	def get_git_blame(self, file_path: str, start_line: int, end_line: int) -> list[GitBlameSchema]:
		"""
		Get the Git blame for a specific range of lines in a file.

		Args:
			file_path (str): The path to the file relative to the repository root.
			start_line (int): The starting line number of the range.
			end_line (int): The ending line number of the range.

		Returns:
			list[GitBlameSchema]: A list of Git blame results.
		"""
		try:
			# Check if the file is actually tracked by git
			file_is_tracked = file_path in self.tracked_files
			if not file_is_tracked:
				# Skip blame lookup for untracked files
				logger.debug(f"File '{file_path}' is not tracked in git - skipping blame lookup")
				return []

			# Handle ambiguous file paths without full directory path
			# Only attempt path resolution if we have:
			# 1. A valid repo_root
			# 2. A simple filename without path separators
			if self.repo_root is not None and "/" not in file_path and "\\" not in file_path and self.tracked_files:
				# We have just a filename - try to find it in tracked files
				matching_paths = [
					tracked_path
					for tracked_path in self.tracked_files
					if tracked_path.endswith(f"/{file_path}") or tracked_path == file_path
				]

				if len(matching_paths) == 1:
					# Found exactly one match, use it
					file_path = matching_paths[0]
					self.logger.debug(f"Found unique match for '{file_path}': {matching_paths[0]}")
				elif len(matching_paths) > 1:
					# Multiple matches, use the most specific one or log warning
					self.logger.warning(f"Ambiguous file '{file_path}' has multiple matches: {matching_paths}")
					# For now, pick the first candidate but this could be improved
					file_path = matching_paths[0]
					self.logger.debug(f"Using first match: {file_path}")
				else:
					self.logger.warning(f"No matching file found for '{file_path}' in tracked files")
					return []  # Return empty blame list if no match found

			try:
				blame_obj: Blame
				if file_path not in self._blame_cache:
					# Ensure file_path is relative to repo root for pygit2.blame
					self._blame_cache[file_path] = self.repo.blame(file_path)
				blame_obj = self._blame_cache[file_path]
			except KeyError:
				# File not found in git repository
				logger.debug(f"File '{file_path}' not found in git repository - skipping blame lookup")
				return []

			# Using a dictionary to collect unique commit information for the given line range
			# Key: commit_id_str, Value: GitBlameSchema
			processed_commits: dict[str, GitBlameSchema] = {}

			for line_num_1_indexed in range(start_line, end_line + 1):
				if line_num_1_indexed <= 0:
					continue  # Line numbers are 1-indexed
				try:
					# pygit2.Blame.__getitem__ expects 0-indexed line number
					hunk = blame_obj[line_num_1_indexed - 1]
				except IndexError:
					# This can happen if start_line/end_line are outside the file's actual line count
					self.logger.warning(
						f"Line {line_num_1_indexed} is out of range for "
						f"blame in file {file_path}. Total lines in "
						f"blame: {len(blame_obj)}."
					)
					continue

				commit_id_str = str(hunk.final_commit_id) if hunk.final_commit_id else "Unknown"

				# If this commit_id is already in processed_commits, it means we've started
				# processing this commit's hunk. We update its end_line.
				if commit_id_str in processed_commits:
					current = processed_commits[commit_id_str]
					current.end_line = max(current.end_line, line_num_1_indexed)
					# Continue to the next line, as we only need one entry per commit within the chunk,
					# but covering the full range of lines it affected in this chunk.
					continue

				# New commit_id encountered for this chunk, gather its details
				author_name = "Unknown"
				commit_date_str = "Unknown"

				if hunk.final_commit_id:
					commit_details_obj = None
					try:
						git_object = self.repo.get(hunk.final_commit_id)
						if isinstance(git_object, Commit):
							commit_details_obj = git_object
							if commit_details_obj.author:
								author_name = commit_details_obj.author.name
								commit_date_str = str(commit_details_obj.author.time)
							elif commit_details_obj.committer:
								author_name = commit_details_obj.committer.name
								commit_date_str = str(commit_details_obj.committer.time)
							else:
								self.logger.warning(f"Commit {hunk.final_commit_id} has no author or committer.")
						elif git_object is not None:
							self.logger.warning(f"Object {hunk.final_commit_id} is not a Commit: {type(git_object)}")
					except (KeyError, TypeError, GitError) as e:
						self.logger.warning(f"Error retrieving details for commit {hunk.final_commit_id}: {e}")

				processed_commits[commit_id_str] = GitBlameSchema(
					commit_id=commit_id_str,
					date=commit_date_str,
					author_name=author_name,
					start_line=line_num_1_indexed,  # First line this commit is seen for in the current range
					end_line=line_num_1_indexed,  # Last line this commit is seen for (so far)
				)

			return list(processed_commits.values())

		except Exception:  # Catch broader exceptions like repo.blame() failing or other issues
			self.logger.exception(f"Failed to get git blame for {file_path}")
			return []

	def get_metadata_schema(self, file_path: str, start_line: int, end_line: int) -> GitMetadataSchema:
		"""
		Derive the complete GitMetadataSchema for a given file.

		Args:
			file_path (str): The path to the file relative to the repository root.
			start_line (int): The starting line number of the range.
			end_line (int): The ending line number of the range.

		Returns:
			GitMetadataSchema: The metadata for the file in the git repository.
		"""
		git_hash = self.get_file_git_hash(file_path)
		blame = self.get_git_blame(file_path, start_line, end_line)
		tracked = file_path in self.tracked_files
		return GitMetadataSchema(
			git_hash=git_hash,
			tracked=tracked,
			branch=self.branch,
			blame=blame,
		)

	def get_untracked_files(self) -> list[str]:
		"""Get a list of untracked files in the repository."""
		status = self.repo.status()
		return [path for path, flags in status.items() if flags & FileStatus.WT_NEW]

	def is_git_ignored(self, file_path: str) -> bool:
		"""Check if a file is ignored by Git."""
		return self.repo.path_is_ignored(file_path)

	def is_file_tracked(self, file_path: str) -> bool:
		"""Check if a file is tracked in the Git repository."""
		return file_path in self.tracked_files

	def unstage_files(self, files: list[str]) -> None:
		"""Unstage the specified files."""
		for file in files:
			self.repo.index.remove(file)
		self.repo.index.write()

	def switch_branch(self, branch_name: str) -> None:
		"""Switch the current Git branch to the specified branch name."""
		ref = f"refs/heads/{branch_name}"
		self.repo.checkout(ref)

	def stage_files(self, files: list[str]) -> None:
		"""Stage the specified files."""
		for file in files:
			self.repo.index.add(file)
		self.repo.index.write()

	def commit(self, message: str) -> None:
		"""Create a commit with the given message."""
		author = self.repo.default_signature
		committer = self.repo.default_signature
		tree = self.repo.index.write_tree()
		parents = [self.repo.head.target] if self.repo.head_is_unborn is False else []
		self.repo.create_commit("HEAD", author, committer, message, tree, parents)
