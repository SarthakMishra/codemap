"""Git utilities for CodeMap."""

from __future__ import annotations

import contextlib
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


def get_other_staged_files(targeted_files: list[str]) -> list[str]:
    """Get staged files that are not part of the targeted files.

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
    """Temporarily stash staged changes except for specified files.

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
    """Restore previously stashed changes.

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


def commit_only_files(files: list[str], message: str, ignore_hooks: bool = False) -> None:
    """Create a commit with only the specified files.

    This ensures that we don't inadvertently commit other staged files.
    Stages the files, then creates a commit using the pathspec to only include
    those specific files in the commit.

    Args:
        files: List of files to commit
        message: Commit message
        ignore_hooks: Whether to ignore git hooks on failure

    Raises:
        GitError: If commit fails
    """
    other_staged = []
    did_stash = False

    try:
        # Check for other staged files
        other_staged = get_other_staged_files(files)

        # Stage the files
        run_git_command(["git", "add", *files])

        # Commit only the specified files by using pathspec
        commit_cmd = ["git", "commit", "-m", message, "--", *files]
        try:
            run_git_command(commit_cmd)
        except GitError as e:
            # Check if failure might be due to git hooks
            if "hook" in str(e).lower() and ignore_hooks:
                # Try again with --no-verify to bypass hooks
                run_git_command([*commit_cmd, "--no-verify"])
            else:
                raise

    except GitError as e:
        if did_stash:
            with contextlib.suppress(GitError):
                unstash_changes()

        msg = f"Failed to commit files: {', '.join(files)}"
        if "hook" in str(e).lower():
            msg += " (git hook failed - check your hook scripts)"
        raise GitError(msg) from e
    finally:
        if did_stash:
            with contextlib.suppress(GitError):
                unstash_changes()

    # Return the list of other staged files that weren't part of this commit
    return other_staged


def get_untracked_files() -> list[str]:
    """Get a list of untracked files in the repository.

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
    """Unstage the specified files.

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
