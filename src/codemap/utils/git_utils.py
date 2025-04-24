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


def validate_repo_path(path: Path | None = None) -> Path | None:
    """Validate and return the repository path.

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

    This function intelligently handles both existing and deleted files:
    - For existing files, it uses `git add`
    - For files that no longer exist, it uses `git rm`

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

    try:
        # Get list of deleted but tracked files from git status
        deleted_tracked_files = set()
        already_staged_deletions = set()
        try:
            # Parse git status to find deleted files
            status_output = run_git_command(["git", "status", "--porcelain"])
            for line in status_output.splitlines():
                # Ensure line is a string, not bytes
                line_str = line if isinstance(line, str) else line.decode("utf-8")

                if line_str.startswith(" D"):
                    # Unstaged deletion (space followed by D)
                    deleted_tracked_files.add(line_str[3:])
                elif line_str.startswith("D "):
                    # Staged deletion (D followed by space)
                    already_staged_deletions.add(line_str[3:])
            logger.debug("Found %d deleted tracked files in git status", len(deleted_tracked_files))
            logger.debug("Found %d already staged deletions in git status", len(already_staged_deletions))
        except GitError:
            logger.warning("Failed to get git status for deleted files")

        # Filter out invalid filenames that contain special characters or patterns
        # that would cause git commands to fail
        valid_files = []
        for file in files:
            # Check if the filename looks like a template or pattern rather than a real file
            if any(char in file for char in ["*", "+", "{", "}", "\\"]) or file.startswith('"'):
                logger.warning("Skipping invalid filename: %s", file)
                continue
            valid_files.append(file)

        # Check if files exist in the repository (tracked by git) or filesystem
        original_count = len(valid_files)
        tracked_files_output = run_git_command(["git", "ls-files"])
        tracked_files = set(tracked_files_output.splitlines())

        # Keep files that either:
        # 1. Exist in filesystem
        # 2. Are tracked by git
        # 3. Are known deleted files from git status
        # 4. Are already staged deletions
        filtered_files = []
        for file in valid_files:
            if (
                Path(file).exists()
                or file in tracked_files
                or file in deleted_tracked_files
                or file in already_staged_deletions
            ):
                filtered_files.append(file)
            else:
                logger.warning("Skipping non-existent and untracked file: %s", file)

        valid_files = filtered_files
        if len(valid_files) < original_count:
            logger.warning(
                "Filtered out %d files that don't exist in the repository", original_count - len(valid_files)
            )

        if not valid_files:
            logger.warning("No valid files to stage after filtering")
            return

        # Check if files exist in the filesystem
        for file in valid_files:
            exists = Path(file).exists()
            logger.debug("File %s exists in filesystem: %s", file, exists)

        # Separate files into different categories
        existing_files = []
        deleted_files = []

        for file in valid_files:
            if Path(file).exists():
                existing_files.append(file)
            # If it doesn't exist in the filesystem but is in the tracked files,
            # or is already known to be deleted, use git rm
            elif file in tracked_files or file in deleted_tracked_files or file in already_staged_deletions:
                deleted_files.append(file)
            else:
                logger.warning("Skipping untracked file that doesn't exist: %s", file)

        # Log the categorized files
        logger.debug("Existing files: %s", existing_files)
        logger.debug("Deleted files: %s", deleted_files)

        # Stage the existing files using git add
        if existing_files:
            logger.debug("Adding %d existing files", len(existing_files))
            run_git_command(["git", "add", *existing_files])

        # Remove the deleted files using git rm
        if deleted_files:
            logger.debug("Removing %d deleted files", len(deleted_files))
            run_git_command(["git", "rm", *deleted_files])

    except GitError as e:
        msg = f"Failed to stage files: {e}"
        raise GitError(msg) from e


def commit(message: str) -> None:
    """Create a commit with the given message.

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


def commit_only_files(
    files: list[str], message: str, *, commit_options: list[str] | None = None, ignore_hooks: bool = False
) -> list[str]:
    """Commit only the specified files.

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

        # Classify files as existing or deleted
        existing_files = []
        deleted_files = []

        for file in files:
            if Path(file).exists():
                existing_files.append(file)
            else:
                status = status_files.get(file, "")
                if status and "D" in status:
                    deleted_files.append(file)
                else:
                    logger.warning("File %s does not exist and is not marked as deleted in git status", file)

        # Stage the files
        if existing_files:
            stage_files(existing_files)

        # Stage deleted files separately
        failed_files = []
        if deleted_files:
            # Process all files first, collecting errors
            for file in deleted_files:
                try:
                    git_cmd = ["git", "rm", file]
                    subprocess.run(  # noqa: S603
                        git_cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        shell=False,  # Explicitly set shell=False for security
                    )
                    logger.info("Staged deleted file: %s", file)
                except subprocess.CalledProcessError as e:  # noqa: PERF203
                    failed_files.append((file, e.stderr.strip()))

            # Report failures outside the loop to avoid PERF203
            for file, error in failed_files:
                logger.warning("Failed to stage deleted file %s: %s", file, error)

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
        except subprocess.CalledProcessError:
            logger.exception("Failed to create commit")
            raise

        return other_staged
    except Exception:
        logger.exception("Error in commit_only_files")
        raise


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
