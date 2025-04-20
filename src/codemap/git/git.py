"""Git utilities wrapper for CodeMap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codemap.git.utils.git_utils import (
    GitDiff,
    GitError,
    commit,
    commit_only_files,
    get_staged_diff,
    get_unstaged_diff,
    get_untracked_files,
    stage_files,
)

if TYPE_CHECKING:
    from pathlib import Path


class GitWrapper:
    """Wrapper for Git operations."""

    def __init__(self, repo_path: Path) -> None:
        """Initialize the Git wrapper.

        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = repo_path

    def has_changes(self) -> bool:
        """Check if the repository has any changes (staged, unstaged, or untracked).

        Returns:
            True if there are any changes, False otherwise
        """
        try:
            staged = get_staged_diff()
            unstaged = get_unstaged_diff()
            untracked = get_untracked_files()

            return bool(staged.files or unstaged.files or untracked)
        except GitError:
            return False

    def get_staged_diff(self) -> GitDiff:
        """Get the diff of staged changes.

        Returns:
            GitDiff object containing staged changes
        """
        return get_staged_diff()

    def get_unstaged_diff(self) -> GitDiff:
        """Get the diff of unstaged changes.

        Returns:
            GitDiff object containing unstaged changes
        """
        return get_unstaged_diff()

    def get_uncommitted_changes(self) -> GitDiff:
        """Get all uncommitted changes (both staged and unstaged).

        Returns:
            GitDiff object with combined diff content
        """
        try:
            staged = get_staged_diff()
            unstaged = get_unstaged_diff()

            # Get untracked files
            untracked_files = get_untracked_files()

            # Combine the diffs into a new GitDiff object
            all_files = list(set(staged.files + unstaged.files + untracked_files))
            combined_content = staged.content + unstaged.content

            return GitDiff(
                files=all_files,
                content=combined_content,
                is_staged=False,  # Mixed staged/unstaged
            )
        except GitError:
            # Return an empty diff in case of error
            return GitDiff(files=[], content="", is_staged=False)

    def stage_files(self, files: list[str]) -> None:
        """Stage the specified files.

        Args:
            files: List of files to stage

        Raises:
            GitError: If staging fails
        """
        stage_files(files)

    def commit_files(self, files: list[str], message: str) -> None:
        """Stage and commit the specified files.

        WARNING: This method will commit all staged changes, not just the files specified.
        Use commit_only_specified_files() instead to commit only specific files.

        Args:
            files: List of files to stage before committing
            message: Commit message

        Raises:
            GitError: If commit fails
        """
        self.stage_files(files)
        commit(message)

    def commit_only_specified_files(self, files: list[str], message: str, ignore_hooks: bool = False) -> list[str]:
        """Stage and commit only the specified files.

        This ensures other staged changes don't get committed.

        Args:
            files: List of files to commit
            message: Commit message
            ignore_hooks: Whether to bypass git hooks on failure

        Returns:
            List of any other files that were staged but not included in this commit

        Raises:
            GitError: If commit fails
        """
        return commit_only_files(files, message, ignore_hooks=ignore_hooks)
