"""Git utilities for CodeMap."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from pygit2 import (
	Commit,
	Diff,
	Patch,
)
from pygit2.enums import FileStatus
from pygit2.repository import Repository

from codemap.processor.utils.git_utils import GitRepoContext

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


class GitBlameSchema(BaseModel):
	"""Metadata for a git blame."""

	commit_id: str
	date: str
	author_name: str
	start_line: int
	end_line: int


class GitMetadataSchema(BaseModel):
	"""Metadata for a git repository."""

	git_hash: str
	tracked: bool
	branch: str
	blame: list[GitBlameSchema]


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
		if self._repo_root is None:
			self._repo_root = self.get_repo_root()
		self.repo = Repository(str(self._repo_root))
		self.branch = self._get_branch()
		self.tracked_files = self._get_tracked_files()

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
		*,
		commit_options: list[str] | None = None,
		ignore_hooks: bool = False,
	) -> list[str]:
		"""Commit only the specified files with the given message and options."""
		try:
			self.stage_files(files)
			other_staged = self.get_other_staged_files(files)
			commit_cmd = ["git", "commit", "-m", message]
			if commit_options:
				commit_cmd.extend(commit_options)
			if ignore_hooks:
				commit_cmd.append("--no-verify")
			try:
				self.commit(message)
				logger.info("Created commit with message: %s", message)
			except GitError as e:
				error_msg = f"Git commit command failed. Command: '{' '.join(commit_cmd)}'"
				logger.exception("Failed to create commit: %s", error_msg)
				raise GitError(error_msg) from e
			return other_staged
		except GitError:
			raise
		except Exception as e:
			error_msg = f"Error in commit_only_files: {e!s}"
			logger.exception(error_msg)
			raise GitError(error_msg) from e

	def get_untracked_files(self) -> list[str]:
		"""Get a list of untracked files in the repository."""
		status = self.repo.status()
		return [path for path, flags in status.items() if flags & FileStatus.WT_NEW]

	def unstage_files(self, files: list[str]) -> None:
		"""Unstage the specified files."""
		for file in files:
			self.repo.index.remove(file)
		self.repo.index.write()

	def switch_branch(self, branch_name: str) -> None:
		"""Switch the current Git branch to the specified branch name."""
		ref = f"refs/heads/{branch_name}"
		self.repo.checkout(ref)

	def get_current_branch(self) -> str:
		"""Get the name of the current branch."""
		return self.repo.head.shorthand

	def is_git_ignored(self, file_path: str) -> bool:
		"""Check if a file is ignored by Git."""
		return self.repo.path_is_ignored(file_path)

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


def run_git_command(
	command: list[str],
	cwd: Path | str | None = None,
	environment: dict[str, str] | None = None,
) -> str:
	"""
	Run a git command and return its output.

	Args:
	    command: Command to run as a list of string arguments
	    cwd: Working directory to run the command in
	    environment: Environment variables to use

	Returns:
	    The output from the command

	Raises:
	    GitError: If the git command fails

	"""
	try:
		# Using subprocess.run with a list of arguments is safe since we're not using shell=True
		# and the command is not being built from untrusted input
		result = subprocess.run(  # noqa: S603
			command,
			cwd=cwd,
			capture_output=True,
			text=True,
			check=True,
			env=environment,
		)
		return result.stdout.strip()
	except subprocess.CalledProcessError as e:
		# Check if this is a pre-commit hook failure for commit - handled specially by the UI
		if command and len(command) > 1 and command[1] == "commit":
			if "pre-commit" in (e.stderr or ""):
				# This is a pre-commit hook failure - which is handled by the UI, so don't log as exception
				logger.warning("Git hooks failed: %s", e.stderr)
				msg = f"{e.stderr}"
				raise GitError(msg) from e
			# Regular commit error
			logger.exception("Git command failed: %s", " ".join(command))

		cmd_str = " ".join(command)
		error_output = e.stderr or ""
		error_msg = f"Git command failed: {cmd_str}\n{error_output}"
		logger.exception(error_msg)
		raise GitError(error_output or error_msg) from e
	except Exception as e:
		error_msg = f"Error running git command: {e}"
		logger.exception(error_msg)
		raise GitError(error_msg) from e
