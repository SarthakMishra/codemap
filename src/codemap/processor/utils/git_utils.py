"""Utilities for interacting with Git."""

import logging
import re
from pathlib import Path

from pygit2 import Commit, discover_repository
from pygit2.repository import Repository

from codemap.processor.vector.schema import GitBlameSchema, GitMetadataSchema

logger = logging.getLogger(__name__)

class GitError(Exception):
	"""Custom exception for Git-related errors."""

class GitRepoContext:
	"""Context manager for efficient Git operations using pygit2."""

	logger = logging.getLogger(__name__)

	_repo_root: Path | None = None
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
		if self._repo_root is None:
			self._repo_root = self.get_repo_root()
		self.repo = Repository(str(self._repo_root))
		self.branch = self._get_branch()
		self.tracked_files = self._get_tracked_files()

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

	@classmethod
	def _should_exclude_path(cls, file_path: str) -> bool:
		"""
		Check if a file path should be excluded from processing based on patterns.

		Args:
			file_path: The file path to check

		Returns:
			True if the path should be excluded, False otherwise
		"""
		exclude_patterns = cls._get_exclude_patterns()
		for pattern in exclude_patterns:
			if re.search(pattern, file_path):
				cls.logger.debug(f"Excluding file from processing due to pattern '{pattern}': {file_path}")
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

	def _get_branch(self) -> str:
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
			return entry.hex
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
			blame = self.repo.blame(file_path)
			results = []
			for hunk in blame:
				line_nums = range(hunk.final_start_line_number, hunk.final_start_line_number + hunk.lines_in_hunk)
				results.extend(
					GitBlameSchema(
						commit_id=hunk.final_commit_id.hex,
						date=str(hunk.final_commit_time),
						author_name=hunk.final_signature.name,
						start_line=line_num,
						end_line=line_num,
					)
					for line_num in line_nums
					if start_line <= line_num <= end_line
				)
			return results
		except Exception:
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
