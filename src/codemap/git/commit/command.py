"""Main commit command implementation for CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codemap.git.utils.git_utils import (
    GitDiff,
    GitError,
    commit_only_files,
    get_repo_root,
    get_staged_diff,
    get_unstaged_diff,
    get_untracked_files,
    run_git_command,
    stage_files,
    unstage_files,
)
from codemap.utils.llm_utils import (
    generate_message,
)

from .diff_splitter import DiffChunk, DiffSplitter
from .interactive import ChunkAction, ChunkResult, CommitUI, loading_spinner
from .message_generator import LLMError, MessageGenerator

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CommitCommand:
    """Handles the commit command workflow."""

    def __init__(self, path: Path | None = None, model: str = "gpt-4o-mini") -> None:
        """Initialize the commit command.

        Args:
            path: Optional path to start from
            model: LLM model to use for commit message generation
        """
        try:
            self.repo_root = get_repo_root(path)
            self.ui: CommitUI = CommitUI()
            self.splitter = DiffSplitter(self.repo_root)
            self.message_generator = MessageGenerator(self.repo_root, model=model)
        except GitError as e:
            raise RuntimeError(str(e)) from e

    def _get_changes(self) -> list[GitDiff]:
        """Get all changes in the repository.

        Returns:
            List of GitDiff objects

        Raises:
            RuntimeError: If Git operations fail
        """
        try:
            changes = []

            # First stage all files (including untracked) to ensure we have a complete diff
            # This makes it easier to analyze all changes together
            try:
                # Use git add . to stage everything
                run_git_command(["git", "add", "."])
                logger.info("Staged all changes for analysis")
            except GitError as e:
                logger.warning("Failed to stage all changes: %s", e)
                # Continue with the process even if staging fails

            # Get the staged diff which should now include all changes
            staged = get_staged_diff()
            if staged.files:
                changes.append(staged)

            # We'll still check for any unstaged changes that might have been missed
            unstaged = get_unstaged_diff()
            if unstaged.files:
                changes.append(unstaged)

            # Check for any untracked files that might have been missed by git add .
            # This can happen if there are gitignore rules or other issues
            untracked = get_untracked_files()
            if untracked:
                # Create a GitDiff object for untracked files
                untracked_diff = GitDiff(
                    files=untracked,
                    content="",  # No content for untracked files
                    is_staged=False,
                )
                changes.append(untracked_diff)

        except GitError as e:
            msg = f"Failed to get changes: {e}"
            raise RuntimeError(msg) from e
        else:
            return changes

    def _generate_commit_message(self, chunk: DiffChunk) -> None:
        """Generate a commit message for the chunk.

        Args:
            chunk: DiffChunk to generate message for

        Raises:
            RuntimeError: If message generation fails
        """
        # Constants to avoid magic numbers
        max_log_message_length = 40

        logger.warning("COMMAND: Starting commit message generation for %s", chunk.files)
        try:
            with loading_spinner("Generating commit message using LLM..."):
                # Use the universal message generator
                logger.warning("COMMAND: Calling generate_message")
                message, is_llm = generate_message(chunk, self.message_generator)

                logger.warning(
                    "COMMAND: Got response - is_llm=%s, message=%s",
                    is_llm,
                    message[:max_log_message_length] + "..."
                    if message and len(message) > max_log_message_length
                    else message,
                )
                chunk.description = message

                # Store whether this was LLM-generated for UI
                chunk.is_llm_generated = is_llm

                if is_llm:
                    logger.info("Generated commit message using LLM: %s", message)
                else:
                    logger.warning("Using automatically generated fallback message: %s", message)

        except LLMError as e:
            # If LLM generation fails, try fallback with clear indication
            logger.exception("LLM message generation failed")
            logger.warning("COMMAND: LLM error: %s", str(e))
            with loading_spinner("Falling back to simple message generation..."):
                message = self.message_generator.fallback_generation(chunk)
                chunk.description = message
                # Mark as not LLM-generated
                chunk.is_llm_generated = False
                logger.warning("Using fallback message: %s", message)
        except (ValueError, RuntimeError) as e:
            logger.warning("COMMAND: Other error: %s", str(e))
            msg = f"Failed to generate commit message: {e}"
            raise RuntimeError(msg) from e

    def _process_chunk(self, chunk: DiffChunk, index: int, total_chunks: int) -> bool:
        """Process a single chunk.

        Args:
            chunk: DiffChunk to process
            index: The 0-based index of the current chunk
            total_chunks: The total number of chunks

        Returns:
            True if processing should continue, False to abort

        Raises:
            RuntimeError: If Git operations fail
        """
        # Add logging here
        logger.warning(
            "ENTERING _process_chunk - Chunk ID: %s, Index: %d/%d, Initial Desc: %s",
            id(chunk),
            index + 1,  # Display 1-based index
            total_chunks,
            getattr(chunk, "description", "<None>"),
        )

        # Import here to avoid circular imports
        from .interactive import loading_spinner

        while True:  # Loop to handle regeneration
            # Generate commit message
            self._generate_commit_message(chunk)

            # Get user action
            # Explicitly use the CommitUI.process_chunk method to help type checking
            result: ChunkResult = self.ui.process_chunk(chunk, index, total_chunks)

            if result.action == ChunkAction.ABORT:
                return not self.ui.confirm_abort()

            if result.action == ChunkAction.SKIP:
                self.ui.show_skipped(chunk.files)
                return True

            if result.action == ChunkAction.REGENERATE:
                # Clear the existing description to force regeneration
                chunk.description = None
                chunk.is_llm_generated = False
                self.ui.console.print("\n[yellow]Regenerating commit message...[/yellow]")
                continue  # Go back to the start of the loop

            # For ACCEPT or EDIT actions
            break  # Exit the loop and proceed with committing

        try:
            # Make sure all files are staged first (in case any were missed or unstaged)
            with loading_spinner("Staging files..."):
                run_git_command(["git", "add", "."])

                # Unstage files not in the current chunk to ensure only chunk files are committed
                all_staged_files = get_staged_diff().files
                files_to_unstage = [f for f in all_staged_files if f not in chunk.files]
                if files_to_unstage:
                    unstage_files(files_to_unstage)

                # Make sure the chunk files are staged (should be redundant but ensures consistency)
                stage_files(chunk.files)

            # Create commit using commit_only_files which handles hooks better
            with loading_spinner("Creating commit..."):
                message = result.message or chunk.description or "Update files"
                try:
                    # First try with hooks enabled
                    other_staged = commit_only_files(chunk.files, message, ignore_hooks=False)
                except GitError as hook_error:
                    if "hook failed" in str(hook_error).lower():
                        # If hook failed, ask user if they want to bypass hooks
                        if self.ui.confirm_bypass_hooks():
                            # Retry with hooks disabled
                            other_staged = commit_only_files(chunk.files, message, ignore_hooks=True)
                        else:
                            raise  # Re-raise if user doesn't want to bypass hooks
                    else:
                        raise  # Re-raise if it's not a hook-related error

                # Log if there were other files staged but not included
                if other_staged:
                    logger.warning("Other files were staged but not included in commit: %s", other_staged)

            self.ui.show_success(f"Created commit for {', '.join(chunk.files)}")

            # Re-stage all remaining files for the next commit
            # This ensures we don't lose track of any changes
            with loading_spinner("Re-staging remaining changes..."):
                run_git_command(["git", "add", "."])

        except GitError as e:
            self.ui.show_error(f"Failed to commit changes: {e}")
            return False
        else:
            return True

    def run(self) -> bool:
        """Run the commit command.

        Returns:
            True if successful, False otherwise
        """
        # Import here to avoid circular imports
        from .interactive import loading_spinner

        try:
            # Get all changes
            with loading_spinner("Analyzing repository changes..."):
                changes = self._get_changes()

            if not changes:
                self.ui.show_error("No changes to commit")
                return False

            # Process each change group (staged, unstaged, untracked)
            for diff in changes:
                # Always use semantic strategy for better commit organization
                with loading_spinner("Organizing changes semantically..."):
                    chunks = self.splitter.split_diff(diff, "semantic")
                if not chunks:
                    continue

                # Process each chunk
                for index, chunk in enumerate(chunks):
                    if not self._process_chunk(chunk, index, len(chunks)):
                        return False
        except (RuntimeError, ValueError) as e:
            self.ui.show_error(str(e))
            return False
        else:
            return True
