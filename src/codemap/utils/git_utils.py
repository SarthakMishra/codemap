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


def run_git_command(command: list[str], cwd: Path | None = None) -> str:
    """Run a Git command and return its output.

    Args:
        command: Git command to run
        cwd: Working directory (optional)

    Returns:
        Command output as string

    Raises:
        GitError: If the command fails
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
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"Git command failed: {' '.join(command)}\nError: {e.stderr}"
        logger.exception(error_msg)
        raise GitError(error_msg) from e
    else:
        return result.stdout


def get_repo_root(path: Path | None = None) -> Path:
    """Get the root directory of the Git repository.

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


def get_staged_diff() -> GitDiff:
    """Get the diff of staged changes.

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
    """Get the diff of unstaged changes.

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
    """Stage the specified files.

    Args:
        files: List of files to stage

    Raises:
        GitError: If staging fails
    """
    try:
        run_git_command(["git", "add", *files])
    except GitError as e:
        msg = f"Failed to stage files: {', '.join(files)}"
        raise GitError(msg) from e


def commit(message: str) -> None:
    """Create a commit with the given message.

    Args:
        message: Commit message

    Raises:
        GitError: If commit fails
    """
    try:
        run_git_command(["git", "commit", "-m", message])
    except GitError as e:
        msg = "Failed to create commit"
        raise GitError(msg) from e


def commit_only_files(files: list[str], message: str) -> None:
    """Create a commit with only the specified files.

    This ensures that we don't inadvertently commit other staged files.
    Stages the files, then creates a commit using the pathspec to only include
    those specific files in the commit.

    Args:
        files: List of files to commit
        message: Commit message

    Raises:
        GitError: If commit fails
    """
    try:
        # Stage the files
        run_git_command(["git", "add", *files])

        # Commit only the specified files by using pathspec
        run_git_command(["git", "commit", "-m", message, "--", *files])
    except GitError as e:
        msg = f"Failed to commit files: {', '.join(files)}"
        raise GitError(msg) from e
