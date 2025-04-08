"""Main commit command implementation for CodeMap."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codemap.utils.git_utils import (
    GitDiff,
    GitError,
    commit,
    get_repo_root,
    get_staged_diff,
    get_unstaged_diff,
    stage_files,
)

from .diff_splitter import DiffChunk, DiffSplitter
from .interactive import ChunkAction, CommitUI
from .message_generator import LLMError, MessageGenerator

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CommitCommand:
    """Handles the commit command workflow."""

    def __init__(self, path: Path | None = None, model: str = "gpt-3.5-turbo") -> None:
        """Initialize the commit command.

        Args:
            path: Optional path to start from
            model: LLM model to use for commit message generation
        """
        try:
            self.repo_root = get_repo_root(path)
            self.ui = CommitUI()
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

            # First check staged changes
            staged = get_staged_diff()
            if staged.files:
                changes.append(staged)

            # Then check unstaged changes
            unstaged = get_unstaged_diff()
            if unstaged.files:
                changes.append(unstaged)
        except GitError as e:
            msg = f"Failed to get changes: {e}"
            raise RuntimeError(msg) from e
        else:
            return changes

    def _generate_commit_message(self, chunk: DiffChunk) -> None:
        """Generate a commit message for a chunk.

        Args:
            chunk: DiffChunk to generate message for
        """
        try:
            message, used_llm = self.message_generator.generate_message(chunk)
            chunk.description = message

            if used_llm:
                logger.info("Generated commit message using LLM: %s", message)
            else:
                logger.info("Generated commit message using fallback: %s", message)
        except (ValueError, LLMError) as e:
            # If specific errors fail, use a very simple message
            files_str = ", ".join(chunk.files)
            chunk.description = f"chore: update {files_str}"
            logger.warning("Message generation failed: %s", str(e))

    def _process_chunk(self, chunk: DiffChunk) -> bool:
        """Process a single chunk.

        Args:
            chunk: DiffChunk to process

        Returns:
            True if processing should continue, False to abort

        Raises:
            RuntimeError: If Git operations fail
        """
        # Generate commit message
        self._generate_commit_message(chunk)

        # Get user action
        result = self.ui.process_chunk(chunk)

        if result.action == ChunkAction.ABORT:
            return not self.ui.confirm_abort()

        if result.action == ChunkAction.SKIP:
            self.ui.show_skipped(chunk.files)
            return True

        try:
            # Stage files if needed
            if not all(f in get_staged_diff().files for f in chunk.files):
                stage_files(chunk.files)

            # Create commit
            commit(result.message or chunk.description or "Update files")
            self.ui.show_success(f"Created commit for {', '.join(chunk.files)}")
        except GitError as e:
            self.ui.show_error(f"Failed to commit changes: {e}")
            return False
        else:
            return True

    def run(self, strategy: str = "file") -> bool:
        """Run the commit command.

        Args:
            strategy: Diff splitting strategy to use

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all changes
            changes = self._get_changes()
            if not changes:
                self.ui.show_error("No changes to commit")
                return False

            # Process each diff
            for diff in changes:
                # Split into chunks
                chunks = self.splitter.split_diff(diff, strategy)
                if not chunks:
                    continue

                # Process each chunk
                for chunk in chunks:
                    if not self._process_chunk(chunk):
                        return False
        except (RuntimeError, ValueError) as e:
            self.ui.show_error(str(e))
            return False
        else:
            return True
