"""Main commit command implementation for CodeMap."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from codemap.git.utils.git_utils import (
    GitDiff,
    GitError,
    commit,
    get_repo_root,
    get_staged_diff,
    get_unstaged_diff,
    get_untracked_files,
    stage_files,
    unstage_files,
)

from .diff_splitter import DiffChunk, DiffSplitter
from .interactive import ChunkAction, CommitUI
from .message_generator import LLMError, MessageGenerator

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def setup_message_generator(
    repo_path: Path,
    model: str = "gpt-4o-mini",
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    prompt_template: str | None = None,
) -> MessageGenerator:
    """Set up a message generator with the provided options.

    Args:
        repo_path: Repository path
        model: LLM model to use
        provider: LLM provider (e.g., openai, anthropic)
        api_base: Custom API base URL
        api_key: API key for the provider
        prompt_template: Custom prompt template

    Returns:
        Configured message generator
    """
    # Try to load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Extract provider from model if not explicitly provided
    if not provider and "/" in model:
        provider, _ = model.split("/", 1)

    # Set up API key if provided, otherwise try to get from environment
    if api_key and provider:
        _set_provider_api_key(provider, api_key)

    # Create and return the message generator
    return MessageGenerator(
        repo_path,
        prompt_template=prompt_template,
        model=model,
        provider=provider,
        api_base=api_base,
    )


def _set_provider_api_key(provider: str, api_key: str) -> None:
    """Set the API key in the environment for the given provider.

    Args:
        provider: Provider name
        api_key: API key to set
    """
    provider_env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure": "AZURE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "together": "TOGETHER_API_KEY",
        "cohere": "COHERE_API_KEY",
    }

    # Get environment variable name for this provider
    env_var = provider_env_vars.get(provider)
    if env_var:
        os.environ[env_var] = api_key
    else:
        # Default to OpenAI
        os.environ["OPENAI_API_KEY"] = api_key


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

            # First stage all files (including untracked) to ensure we have a complete diff
            # This makes it easier to analyze all changes together
            try:
                # Use git add . to stage everything
                from codemap.git.utils.git_utils import run_git_command

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
            # Make sure all files are staged first (in case any were missed or unstaged)
            from codemap.git.utils.git_utils import run_git_command

            run_git_command(["git", "add", "."])

            # Unstage files not in the current chunk to ensure only chunk files are committed
            all_staged_files = get_staged_diff().files
            files_to_unstage = [f for f in all_staged_files if f not in chunk.files]
            if files_to_unstage:
                unstage_files(files_to_unstage)

            # Make sure the chunk files are staged (should be redundant but ensures consistency)
            stage_files(chunk.files)

            # Create commit
            commit(result.message or chunk.description or "Update files")
            self.ui.show_success(f"Created commit for {', '.join(chunk.files)}")

            # Re-stage all remaining files for the next commit
            # This ensures we don't lose track of any changes
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
        try:
            # Get all changes
            changes = self._get_changes()
            if not changes:
                self.ui.show_error("No changes to commit")
                return False

            # Process each change group (staged, unstaged, untracked)
            for diff in changes:
                # Always use semantic strategy for better commit organization
                chunks = self.splitter.split_diff(diff, "semantic")
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
