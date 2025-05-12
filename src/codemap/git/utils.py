"""Git utilities for CodeMap."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pygit2 import (
	Commit,
	Diff,
	Patch,
)

from codemap.utils.git_hooks import hook_exists, run_hook
from codemap.utils.git_utils import GitRepoContext

logger = logging.getLogger(__name__)


@dataclass
class GitDiff:
	"""Represents a Git diff chunk."""

	files: list[str]
	content: str
	is_staged: bool = False
	is_untracked: bool = False


class GitError(Exception):
	"""Custom exception for Git-related errors."""


class ExtendedGitRepoContext(GitRepoContext):
	"""Extended context for Git operations using pygit2."""

	_extended_instance: ExtendedGitRepoContext | None = None

	@classmethod
	def get_instance(cls) -> ExtendedGitRepoContext:
		"""Get an instance of the ExtendedGitRepoContext class."""
		if cls._extended_instance is None:
			cls._extended_instance = cls()
		return cls._extended_instance

	def __init__(self) -> None:
		"""Initialize the ExtendedGitRepoContext with the given repository path."""
		super().__init__()

	@classmethod
	def validate_repo_path(cls, path: Path | None = None) -> Path | None:
		"""Validate and return the repository path, or None if not valid."""
		try:
			if path is None:
				path = Path.cwd()
			return cls.get_repo_root(path)
		except GitError:
			return None

	def get_staged_diff(self) -> GitDiff:
		"""Get the diff of staged changes as a GitDiff object."""
		commit = self.repo.head.peel(Commit)
		diff = self.repo.diff(commit.tree, cached=True)

		files = []
		content = ""
		if isinstance(diff, Diff):
			files = [delta.delta.new_file.path for delta in diff]
			content = diff.patch
		elif isinstance(diff, Patch):
			files = [diff.delta.new_file.path]
			content = diff.text
		return GitDiff(files=files, content=content or "", is_staged=True)

	def get_unstaged_diff(self) -> GitDiff:
		"""Get the diff of unstaged changes as a GitDiff object."""
		diff = self.repo.diff()
		files = []

		content = ""

		if isinstance(diff, Diff):
			files = [delta.delta.new_file.path for delta in diff]
			content = diff.patch
		elif isinstance(diff, Patch):
			files = [diff.delta.new_file.path]
			content = diff.text

		return GitDiff(files=files, content=content or "", is_staged=False)

	def get_other_staged_files(self, targeted_files: list[str]) -> list[str]:
		"""Get staged files that are not part of the targeted files."""
		all_staged = self.get_staged_diff().files
		return [f for f in all_staged if f not in targeted_files]

	def stash_staged_changes(self, exclude_files: list[str]) -> bool:
		"""Temporarily stash staged changes except for specified files."""
		try:
			other_files = self.get_other_staged_files(exclude_files)
			if not other_files:
				return False
			self.stage_files(other_files)
		except GitError as e:
			msg = "Failed to stash other staged changes"
			raise GitError(msg) from e
		else:
			return True

	def unstash_changes(self) -> None:
		"""Restore previously stashed changes."""
		try:
			stash_list = self.get_other_staged_files([])
			if "CodeMap: temporary stash for commit" in stash_list:
				self.unstage_files(stash_list)
		except GitError as e:
			msg = "Failed to restore stashed changes; you may need to manually run 'git stash pop'"
			raise GitError(msg) from e

	def commit_only_files(
		self,
		files: list[str],
		message: str,
		ignore_hooks: bool = False,
	) -> list[str]:
		"""
		Commit only the specified files with the given message.

		Runs the pre-commit, commit-msg, and post-commit hooks unless ignore_hooks is True.
		"""
		import tempfile

		# Run pre-commit hook if not ignored
		if not ignore_hooks and hook_exists("pre-commit"):
			exit_code = run_hook("pre-commit")
			if exit_code != 0:
				error_msg = "pre-commit hook failed, aborting commit."
				logger.error(error_msg)
				raise RuntimeError(error_msg)
		try:
			# Prepare commit-msg hook: write message to a temp file if needed
			commit_msg_file = None
			if not ignore_hooks and hook_exists("commit-msg"):
				with tempfile.NamedTemporaryFile("w+", delete=False) as f:
					f.write(message)
					commit_msg_file = f.name
				exit_code = run_hook("commit-msg", repo_root=None)  # Could pass file as env var if needed
				if exit_code != 0:
					error_msg = "commit-msg hook failed, aborting commit."
					logger.error(error_msg)
					if commit_msg_file:
						Path(commit_msg_file).unlink()
					raise RuntimeError(error_msg)
			# self.stage_files(files) # Removed: Index is already prepared by the caller
			other_staged = self.get_other_staged_files(files)
			try:
				self.commit(message)
				logger.info("Created commit with message: %s", message)
			except GitError as e:
				error_msg = "Git commit command failed"
				logger.exception(error_msg)
				raise GitError(error_msg) from e
			# Run post-commit hook if not ignored
			if not ignore_hooks and hook_exists("post-commit"):
				exit_code = run_hook("post-commit")
				if exit_code != 0:
					logger.warning("post-commit hook failed (commit already created)")
			if commit_msg_file:
				Path(commit_msg_file).unlink()
			return other_staged
		except GitError:
			raise
		except Exception as e:
			error_msg = f"Error in commit_only_files: {e!s}"
			logger.exception(error_msg)
			raise GitError(error_msg) from e

	def get_per_file_diff(self, file_path: str, staged: bool = False) -> GitDiff:
		"""
		Get the diff for a single file, either staged or unstaged.

		Args:
			file_path: The path to the file to diff (relative to repo root).
			staged: If True, get the staged diff; otherwise, get the unstaged diff.

		Returns:
			GitDiff: The diff for the specified file.

		Raises:
			GitError: If the diff cannot be generated.
		"""
		logger.debug("get_per_file_diff called with file_path: '%s', staged: %s", file_path, staged)
		try:
			if staged:
				commit = self.repo.head.peel(Commit)
				diff = self.repo.diff(commit.tree, cached=True)
				is_staged = True
			else:
				diff = self.repo.diff()
				is_staged = False

			file_path_set = {file_path}
			if isinstance(diff, Diff):
				for patch in diff:
					new_file_path = patch.delta.new_file.path
					old_file_path = patch.delta.old_file.path
					logger.debug(
						"  Patch details - New: '%s', Old: '%s'",
						new_file_path,
						old_file_path,
					)
					if {new_file_path, old_file_path} & file_path_set:
						content = patch.text or ""
						logger.debug("    Patch text (first 200 chars): %s", repr(content[:200]))
						files = [new_file_path]
						git_diff_obj = GitDiff(files=files, content=content, is_staged=is_staged)
						logger.debug(
							"    Returning GitDiff for '%s', content length: %d",
							file_path,
							len(git_diff_obj.content),
						)
						return git_diff_obj
				logger.debug("  No matching patch found in Diff for '%s'. Returning empty GitDiff.", file_path)
				return GitDiff(files=[file_path], content="", is_staged=is_staged)
			if isinstance(diff, Patch):
				new_file_path = diff.delta.new_file.path
				old_file_path = diff.delta.old_file.path
				logger.debug(
					"  Patch details (standalone) - New: '%s', Old: '%s'",
					new_file_path,
					old_file_path,
				)
				if {new_file_path, old_file_path} & file_path_set:
					content = diff.text or ""
					logger.debug("    Patch text (first 200 chars): %s", repr(content[:200]))
					files = [new_file_path]
					git_diff_obj = GitDiff(files=files, content=content, is_staged=is_staged)
					logger.debug(
						"    Returning GitDiff for '%s' (standalone patch), content length: %d",
						file_path,
						len(git_diff_obj.content),
					)
					return git_diff_obj
				logger.debug("  Standalone Patch does not match '%s'. Returning empty GitDiff.", file_path)
				return GitDiff(files=[file_path], content="", is_staged=is_staged)
			logger.debug("  Diff object is neither Diff nor Patch for '%s'. Returning empty GitDiff.", file_path)
			return GitDiff(files=[file_path], content="", is_staged=is_staged)
		except Exception as e:
			logger.exception("Failed to get %s diff for %s", "staged" if staged else "unstaged", file_path)
			msg = f"Failed to get {'staged' if staged else 'unstaged'} diff for {file_path}: {e}"
			raise GitError(msg) from e
