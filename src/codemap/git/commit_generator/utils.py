"""Utility functions for commit message generation."""

import logging
import re
from pathlib import Path

from codemap.git.commit_linter.linter import CommitLinter
from codemap.git.utils import GitError, run_git_command
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def clean_message_for_linting(message: str) -> str:
	"""
	Clean a commit message for linting.

	Removes extra newlines, trims whitespace, etc.

	Args:
	        message: The commit message to clean

	Returns:
	        The cleaned commit message

	"""
	# Replace multiple consecutive newlines with a single newline
	cleaned = re.sub(r"\n{3,}", "\n\n", message)
	# Trim leading and trailing whitespace
	return cleaned.strip()


def lint_commit_message(
	message: str, repo_root: Path | None = None, config_loader: ConfigLoader | None = None
) -> tuple[bool, str | None]:
	"""
	Lint a commit message.

	Checks if it adheres to Conventional Commits format using internal CommitLinter.

	Args:
	        message: The commit message to lint
	        repo_root: Repository root path
	        config_loader: Configuration loader instance

	Returns:
	        Tuple of (is_valid, error_message)

	"""
	# Get config loader if not provided
	if config_loader is None:
		config_loader = ConfigLoader(repo_root=repo_root)

	try:
		# Create a CommitLinter instance with the config_loader
		linter = CommitLinter(config_loader=config_loader)

		# Lint the commit message
		is_valid, lint_messages = linter.lint(message)

		# Get error message if not valid
		error_message = None
		if not is_valid and lint_messages:
			error_message = "\n".join(lint_messages)

		return is_valid, error_message

	except Exception as e:
		# Handle any errors during linting
		logger.exception("Error linting commit message")
		return False, f"Linting failed: {e!s}"


def save_working_directory_state(files: list[str], output_file: str) -> bool:
	"""
	Save the current state of specified files to a patch file.

	Args:
	        files: List of file paths
	        output_file: Path to output patch file

	Returns:
	        bool: Whether the operation was successful

	"""
	output_path = Path(output_file)

	try:
		if not files:
			# Nothing to save
			with output_path.open("w") as f:
				f.write("")
			return True

		# Generate diff for the specified files
		diff_cmd = ["git", "diff", "--", *files]
		diff_content = run_git_command(diff_cmd)

		# Write to output file
		with output_path.open("w") as f:
			f.write(diff_content)

		return True

	except (OSError, GitError):
		logger.exception("Error saving working directory state")
		return False


def restore_working_directory_state(patch_file: str) -> bool:
	"""
	Restore the working directory state from a patch file.

	Args:
	        patch_file: Path to patch file

	Returns:
	        bool: Whether the operation was successful

	"""
	patch_path = Path(patch_file)

	try:
		# Check if the patch file exists and is not empty
		if not patch_path.exists() or patch_path.stat().st_size == 0:
			return True  # Nothing to restore

		# Apply the patch
		run_git_command(["git", "apply", patch_file])
		return True

	except GitError:
		logger.exception("Error restoring working directory state")
		return False
