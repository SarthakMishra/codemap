"""Git utilities for CodeMap."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GitDiff:
	"""Represents a Git diff chunk."""

	files: list[str]
	content: str
	is_staged: bool = False


class GitError(Exception):
	"""Custom exception for Git-related errors."""


def run_git_command(command: list[str], cwd: Path | str | None = None, check: bool = True) -> str:
	"""
	Run a git command and return its output.

	Args:
	    command: Git command as a list of strings
	    cwd: Working directory
	    check: Whether to check for errors

	Returns:
	    Command output as a string

	Raises:
	    GitError: If the command fails and check is True

	"""
	# Constants to avoid magic numbers
	min_cmd_len_for_merge_base = 3
	merge_base_index = 1
	is_ancestor_index = 2

	# Check if command contains 'merge-base --is-ancestor' which is expected to sometimes fail
	# without it being a true error condition
	is_ancestor_check = (
		len(command) >= min_cmd_len_for_merge_base
		and command[merge_base_index] == "merge-base"
		and command[is_ancestor_index] == "--is-ancestor"
	)

	try:
		# Using subprocess.run with a list of arguments is safe since we're not using shell=True
		# and the command is not being built from untrusted input
		result = subprocess.run(  # noqa: S603
			command,
			cwd=cwd,
			capture_output=True,
			text=True,
			check=check,
		)
		return result.stdout
	except subprocess.CalledProcessError as e:
		stderr = e.stderr.strip() if e.stderr else ""
		stdout = e.stdout.strip() if e.stdout else ""

		# For merge-base --is-ancestor checks, log at debug level as this is expected to fail sometimes
		if is_ancestor_check:
			logger.debug("Git command completed with non-zero status (expected for relationship check): %s", command)
			if check:
				error_message = f"Git command failed with exit code {e.returncode}: {stderr or stdout}"
				raise GitError(error_message) from e
		else:
			# For other commands, log the exception
			logger.exception("Git command failed: %s", " ".join(command))
			if check:
				error_msg = f"Git command failed: {stderr or stdout}"
				raise GitError(error_msg) from e

		# If we're not checking for errors, return an empty string
		return ""


def get_repo_root(path: Path | None = None) -> Path:
	"""
	Get the root directory of the Git repository.

	Args:
	    path: Optional path to start searching from

	Returns:
	    Path to repository root

	Raises:
	    GitError: If not in a Git repository

	"""
	try:
		result = run_git_command(["git", "rev-parse", "--show-toplevel"], path)
		return Path(result.strip())
	except GitError as e:
		msg = "Not in a Git repository"
		raise GitError(msg) from e


def validate_repo_path(path: Path | None = None) -> Path | None:
	"""
	Validate and return the repository path.

	Args:
	    path: Optional path to validate (defaults to current directory)

	Returns:
	    Path to the repository root if valid, None otherwise

	"""
	try:
		# If no path provided, use current directory
		if path is None:
			path = Path.cwd()

		# Get the repository root
		return get_repo_root(path)
	except GitError:
		return None


def get_staged_diff() -> GitDiff:
	"""
	Get the diff of staged changes.

	Returns:
	    GitDiff object containing staged changes

	Raises:
	    GitError: If git command fails

	"""
	try:
		# Get list of staged files
		staged_files = run_git_command(["git", "diff", "--cached", "--name-only"]).splitlines()

		# Get the actual diff
		diff_content = run_git_command(["git", "diff", "--cached"])

		return GitDiff(
			files=staged_files,
			content=diff_content,
			is_staged=True,
		)
	except GitError as e:
		msg = "Failed to get staged changes"
		raise GitError(msg) from e


def get_unstaged_diff() -> GitDiff:
	"""
	Get the diff of unstaged changes.

	Returns:
	    GitDiff object containing unstaged changes

	Raises:
	    GitError: If git command fails

	"""
	try:
		# Get list of modified files
		modified_files = run_git_command(["git", "diff", "--name-only"]).splitlines()

		# Get the actual diff
		diff_content = run_git_command(["git", "diff"])

		return GitDiff(
			files=modified_files,
			content=diff_content,
			is_staged=False,
		)
	except GitError as e:
		msg = "Failed to get unstaged changes"
		raise GitError(msg) from e


def stage_files(files: list[str]) -> None:
	"""
	Stage the specified files.

	This function intelligently handles both existing and deleted files:
	- For existing files, it uses `git add`
	- For files that no longer exist but are tracked by git, it uses `git rm`
	- For files that no longer exist but are still in index, it uses `git rm --cached`

	This prevents errors when trying to stage files that have been deleted
	but not yet tracked in git.

	Args:
	    files: List of files to stage

	Raises:
	    GitError: If staging fails

	"""
	if not files:
		logger.warning("No files provided to stage_files")
		return

	# Keep track of all errors to report at the end
	errors = []

	try:
		# 1. Get information about file status
		# ====================================
		git_status_info = {}
		tracked_files = set()
		index_files = set()

		# 1.1 Get git status information
		try:
			status_output = run_git_command(["git", "status", "--porcelain"])
			for line in status_output.splitlines():
				# Ensure line is a string, not bytes
				line_str = line if isinstance(line, str) else line.decode("utf-8")
				if not line_str:
					continue

				status = line_str[:2]
				file_path = line_str[3:].strip()
				git_status_info[file_path] = status
		except GitError:
			errors.append("Failed to get git status information")

		# 1.2 Get tracked files
		try:
			tracked_files_output = run_git_command(["git", "ls-files"])
			tracked_files = set(tracked_files_output.splitlines())
		except GitError:
			errors.append("Failed to get list of tracked files")

		# 1.3 Get index files
		try:
			index_files_output = run_git_command(["git", "ls-files", "--stage"])
			index_files = {line.split()[-1] for line in index_files_output.splitlines() if line.strip()}
		except GitError:
			errors.append("Failed to get list of files in git index")

		# 2. Filter and categorize files
		# ==============================
		# Filter out invalid filenames
		valid_files = [
			file
			for file in files
			if not (any(char in file for char in ["*", "+", "{", "}", "\\"]) or file.startswith('"'))
		]

		# Skip any invalid filenames that were filtered out
		for file in files:
			if file not in valid_files:
				logger.warning("Skipping invalid filename: %s", file)

		# Categorize files
		existing_files = []
		deleted_tracked_files = []
		deleted_index_files = []
		untracked_nonexistent_files = []

		for file in valid_files:
			path = Path(file)
			if path.exists():
				existing_files.append(file)
			elif file in tracked_files:
				deleted_tracked_files.append(file)
			elif file in index_files:
				deleted_index_files.append(file)
			else:
				untracked_nonexistent_files.append(file)
				logger.warning("Skipping file %s: Does not exist and is not tracked by git", file)

		# Log the categorized files
		logger.debug("Existing files (%d): %s", len(existing_files), existing_files)
		logger.debug("Deleted tracked files (%d): %s", len(deleted_tracked_files), deleted_tracked_files)
		logger.debug("Deleted index files (%d): %s", len(deleted_index_files), deleted_index_files)

		# 3. Process each file category
		# =============================
		# 3.1 Add existing files
		if existing_files:
			try:
				run_git_command(["git", "add", *existing_files])
				logger.debug("Added %d existing files", len(existing_files))
			except GitError as e:
				errors.append(f"Failed to add existing files: {e!s}")

		# 3.2 Remove deleted tracked files
		for file in deleted_tracked_files:
			cmd = ["git", "rm", file]
			try:
				run_git_command(cmd)
				logger.debug("Removed deleted tracked file: %s", file)
			except GitError as e:
				if "did not match any files" in str(e):
					# File exists in tracked_files but can't be found, try with --cached
					deleted_index_files.append(file)
				else:
					errors.append(f"Failed to remove deleted tracked file {file}: {e!s}")

		# 3.3 Remove files from index
		if deleted_index_files:
			try:
				run_git_command(["git", "rm", "--cached", *deleted_index_files])
				logger.debug("Removed %d files from index", len(deleted_index_files))
			except GitError as e:
				errors.append(f"Failed to remove files from index: {e!s}")

		# 4. Report errors if any occurred
		# ================================
		if errors:
			error_msg = "; ".join(errors)
			msg = f"Errors while staging files: {error_msg}"
			logger.error(msg)
			raise GitError(msg)  # noqa: TRY301

	except GitError:
		# Pass through GitError exceptions
		raise
	except Exception as e:
		# Wrap other exceptions in GitError
		msg = f"Unexpected error staging files: {e}"
		logger.exception(msg)
		raise GitError(msg) from e


def commit(message: str) -> None:
	"""
	Create a commit with the given message.

	Args:
	    message: Commit message

	Raises:
	    GitError: If commit fails

	"""
	try:
		# For commit messages, we need to ensure they're properly quoted
		# Use a shell command directly to ensure proper quoting
		import shlex

		quoted_message = shlex.quote(message)
		shell_command = f"git commit -m {quoted_message}"

		# Using shell=True is necessary for proper handling of quoted commit messages
		# Security is maintained by using shlex.quote to escape user input
		subprocess.run(  # noqa: S602
			shell_command,
			cwd=None,  # Use current dir
			capture_output=True,
			text=True,
			check=True,
			shell=True,  # Using shell=True for this operation
		)
	except subprocess.CalledProcessError as e:
		msg = f"Failed to create commit: {e.stderr}"
		raise GitError(msg) from e


def get_other_staged_files(targeted_files: list[str]) -> list[str]:
	"""
	Get staged files that are not part of the targeted files.

	Args:
	    targeted_files: List of files that are meant to be committed

	Returns:
	    List of other staged files that might be committed inadvertently

	Raises:
	    GitError: If git command fails

	"""
	try:
		# Get all staged files
		all_staged = run_git_command(["git", "diff", "--cached", "--name-only"]).splitlines()

		# Filter out the targeted files
		return [f for f in all_staged if f not in targeted_files]
	except GitError as e:
		msg = "Failed to check for other staged files"
		raise GitError(msg) from e


def stash_staged_changes(exclude_files: list[str]) -> bool:
	"""
	Temporarily stash staged changes except for specified files.

	This is used to ensure only specific files are committed when other
	files might be mistakenly staged.

	Args:
	    exclude_files: Files to exclude from stashing (to keep staged)

	Returns:
	    Whether stashing was performed

	Raises:
	    GitError: If git operations fail

	"""
	try:
		# First check if there are any other staged files
		other_files = get_other_staged_files(exclude_files)
		if not other_files:
			return False

		# Create a temporary index to save current state
		run_git_command(["git", "stash", "push", "--keep-index", "--message", "CodeMap: temporary stash for commit"])
	except GitError as e:
		msg = "Failed to stash other staged changes"
		raise GitError(msg) from e
	else:
		return True


def unstash_changes() -> None:
	"""
	Restore previously stashed changes.

	Raises:
	    GitError: If git operations fail

	"""
	try:
		stash_list = run_git_command(["git", "stash", "list"])
		if "CodeMap: temporary stash for commit" in stash_list:
			run_git_command(["git", "stash", "pop"])
	except GitError as e:
		msg = "Failed to restore stashed changes; you may need to manually run 'git stash pop'"
		raise GitError(msg) from e


def commit_only_files(
	files: list[str], message: str, *, commit_options: list[str] | None = None, ignore_hooks: bool = False
) -> list[str]:
	"""
	Commit only the specified files.

	Args:
	    files: List of files to commit
	    message: Commit message
	    commit_options: Additional commit options
	    ignore_hooks: Whether to ignore Git hooks

	Returns:
	    List of other staged files that weren't committed

	"""
	try:
		# Get status to check for deleted files
		status_cmd = ["git", "status", "--porcelain"]
		result = subprocess.run(  # noqa: S603
			status_cmd,
			capture_output=True,
			text=True,
			check=True,
			shell=False,  # Explicitly set shell=False for security
		)
		status_output = result.stdout.strip()

		# Extract files from status output
		status_files = {}
		for line in status_output.splitlines():
			if not line.strip():
				continue
			status = line[:2].strip()
			file_path = line[3:].strip()

			# Handle renamed files
			if isinstance(file_path, bytes):
				file_path = file_path.decode("utf-8")

			if " -> " in file_path:
				file_path = file_path.split(" -> ")[1]

			status_files[file_path] = status

		# Stage all files - our improved stage_files function can handle both existing and deleted files
		stage_files(files)

		# Get other staged files
		other_staged = get_other_staged_files(files)

		# Commit the changes
		commit_cmd = ["git", "commit", "-m", message]

		if commit_options:
			commit_cmd.extend(commit_options)

		if ignore_hooks:
			commit_cmd.append("--no-verify")

		try:
			subprocess.run(  # noqa: S603
				commit_cmd,
				check=True,
				capture_output=True,
				text=True,
				shell=False,  # Explicitly set shell=False for security
			)
			logger.info("Created commit with message: %s", message)
		except subprocess.CalledProcessError as e:
			# Capture stderr and stdout for better error reporting
			error_msg = f"Git commit command failed. Command: '{' '.join(commit_cmd)}'"

			if e.stderr:
				error_msg += f"\n\nGit Error Output:\n{e.stderr.strip()}"
			if e.stdout:
				error_msg += f"\n\nCommand Output:\n{e.stdout.strip()}"

			logger.exception("Failed to create commit: %s", error_msg)
			raise GitError(error_msg) from e

		return other_staged
	except GitError:
		# Re-raise GitErrors directly
		raise
	except Exception as e:
		error_msg = f"Error in commit_only_files: {e!s}"
		logger.exception(error_msg)
		raise GitError(error_msg) from e


def get_untracked_files() -> list[str]:
	"""
	Get a list of untracked files in the repository.

	These are files that are not yet tracked by Git (new files that haven't been staged).

	Returns:
	    List of untracked file paths

	Raises:
	    GitError: If git command fails

	"""
	try:
		# Use ls-files with --others to get untracked files and --exclude-standard to respect gitignore
		return run_git_command(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
	except GitError as e:
		msg = "Failed to get untracked files"
		raise GitError(msg) from e


def unstage_files(files: list[str]) -> None:
	"""
	Unstage the specified files.

	Args:
	    files: List of files to unstage

	Raises:
	    GitError: If unstaging fails

	"""
	try:
		run_git_command(["git", "restore", "--staged", *files])
	except GitError as e:
		msg = f"Failed to unstage files: {', '.join(files)}"
		raise GitError(msg) from e
