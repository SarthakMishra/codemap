"""Main commit command implementation for CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from codemap.utils.cli_utils import loading_spinner
from codemap.utils.git_utils import (
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
from .interactive import ChunkAction, ChunkResult, CommitUI
from .message_generator import DiffChunkData, LLMError, MessageGenerator

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class CommitCommand:
	"""Handles the commit command workflow."""

	def __init__(self, path: Path | None = None, model: str = "gpt-4o-mini", bypass_hooks: bool = False) -> None:
		"""
		Initialize the commit command.

		Args:
		    path: Optional path to start from
		    model: LLM model to use for commit message generation
		    bypass_hooks: Whether to bypass git hooks with --no-verify

		"""
		try:
			self.repo_root = get_repo_root(path)
			self.ui: CommitUI = CommitUI()
			self.splitter = DiffSplitter(self.repo_root)
			self.message_generator = MessageGenerator(self.repo_root, model=model)
			self.error_state = None  # Tracks reason for failure: "failed", "aborted", etc.
			self.bypass_hooks = bypass_hooks  # Whether to bypass git hooks with --no-verify
		except GitError as e:
			raise RuntimeError(str(e)) from e

	def _get_changes(self) -> list[GitDiff]:
		"""
		Get all changes in the repository.

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
				logger.debug("Staged all changes for analysis")
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
		"""
		Generate a commit message for the chunk.

		Args:
		    chunk: DiffChunk to generate message for

		Raises:
		    RuntimeError: If message generation fails

		"""
		# Constants to avoid magic numbers
		max_log_message_length = 40

		logger.debug("Starting commit message generation for %s", chunk.files)
		try:
			with loading_spinner("Generating commit message using LLM..."):
				# Use the universal message generator with linting enabled
				logger.debug("Calling generate_message with linting enabled")
				message, is_llm = generate_message(chunk, self.message_generator, enable_linting=True)

				logger.debug(
					"Got response - is_llm=%s, message=%s",
					is_llm,
					message[:max_log_message_length] + "..."
					if message and len(message) > max_log_message_length
					else message,
				)
				chunk.description = message

				# Store whether this was LLM-generated for UI
				chunk.is_llm_generated = is_llm

				if is_llm:
					logger.debug("Generated commit message using LLM: %s", message)
				else:
					logger.warning("Using automatically generated fallback message: %s", message)

		except LLMError as e:
			# If LLM generation fails, try fallback with clear indication
			logger.exception("LLM message generation failed")
			logger.warning("LLM error: %s", str(e))
			with loading_spinner("Falling back to simple message generation..."):
				# Convert DiffChunk to DiffChunkData before passing to fallback_generation
				description = getattr(chunk, "description", None)
				chunk_dict = DiffChunkData(files=chunk.files, content=chunk.content)
				# Add description only if it exists to match TypedDict total=False
				if description is not None:
					chunk_dict["description"] = description
				message = self.message_generator.fallback_generation(chunk_dict)
				chunk.description = message
				# Mark as not LLM-generated
				chunk.is_llm_generated = False
				logger.warning("Using fallback message: %s", message)
		except (ValueError, RuntimeError) as e:
			logger.warning("Other error: %s", str(e))
			msg = f"Failed to generate commit message: {e}"
			raise RuntimeError(msg) from e

	def _perform_commit(self, chunk: DiffChunk, message: str) -> bool:
		"""
		Perform the actual commit operation.

		Args:
		    chunk: The chunk to commit
		    message: Commit message to use

		Returns:
		    True if successful, False otherwise

		"""
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
				# Use the class's bypass_hooks setting as the default
				initial_ignore_hooks = self.bypass_hooks
				try:
					# First try with hooks enabled (unless bypass_hooks is True)
					other_staged = commit_only_files(chunk.files, message, ignore_hooks=initial_ignore_hooks)
				except GitError as hook_error:
					if "hook failed" in str(hook_error).lower() and not initial_ignore_hooks:
						# If hook failed and we weren't already bypassing hooks, ask user if they want to bypass hooks
						if self.ui.confirm_bypass_hooks():
							# Retry with hooks disabled
							other_staged = commit_only_files(chunk.files, message, ignore_hooks=True)
						else:
							raise  # Re-raise if user doesn't want to bypass hooks
					else:
						raise  # Re-raise if it's not a hook-related error or we were already bypassing hooks

				# Log if there were other files staged but not included
				if other_staged:
					logger.warning("Other files were staged but not included in commit: %s", other_staged)

			self.ui.show_success(f"Created commit for {', '.join(chunk.files)}")

			# Re-stage all remaining changes for the next commit
			# This ensures we don't lose track of any changes
			with loading_spinner("Re-staging remaining changes..."):
				run_git_command(["git", "add", "."])

		except GitError as e:
			# Display the git error in a formatted box
			error_message = str(e)

			# Check if the error contains a detailed git error message with newlines
			if "\n" in error_message:
				# Split the message if it contains a "Git Error Output:" section
				if "Git Error Output:" in error_message:
					parts = error_message.split("Git Error Output:", 1)
					main_error = parts[0].strip()
					git_output = "Git Error Output:" + parts[1].strip()

					# Show the main error first, then the details in a panel
					self.ui.show_error(f"Failed to commit changes: {main_error}")

					from rich.console import Console
					from rich.panel import Panel

					console = Console()
					console.print(Panel(git_output, title="Git Command Output", border_style="red"))
				else:
					# Just show the full error in a formatted panel
					from rich.console import Console
					from rich.panel import Panel

					console = Console()
					console.print(Panel(error_message, title="Git Command Error", border_style="red"))
			else:
				# For simple one-line errors, just display directly
				self.ui.show_error(f"Failed to commit changes: {error_message}")

			return False
		else:
			return True

	def _process_chunk(self, chunk: DiffChunk, index: int, total_chunks: int) -> bool:
		"""
		Process a single chunk.

		Args:
		    chunk: DiffChunk to process
		    index: The 0-based index of the current chunk
		    total_chunks: The total number of chunks

		Returns:
		    True if processing should continue, False to abort

		Raises:
		    RuntimeError: If Git operations fail
		    typer.Exit: If user chooses to exit

		"""
		# Add logging here
		logger.debug(
			"Processing chunk - Chunk ID: %s, Index: %d/%d, Initial Desc: %s",
			id(chunk),
			index + 1,  # Display 1-based index
			total_chunks,
			getattr(chunk, "description", "<None>"),
		)

		# Remove any chunk.index and chunk.total attributes if they exist
		if hasattr(chunk, "index"):
			delattr(chunk, "index")
		if hasattr(chunk, "total"):
			delattr(chunk, "total")

		while True:  # Loop to handle regeneration
			# Generate commit message
			self._generate_commit_message(chunk)

			# Get user action via UI
			result: ChunkResult = self.ui.process_chunk(chunk, index, total_chunks)

			if result.action == ChunkAction.ABORT:
				# Mark as an intended abort (UI.confirm_abort will raise typer.Exit if confirmed)
				self.error_state = "aborted"

				# In production, if confirm_abort returns, it means user declined to abort
				# In tests, mock will return the mocked value and not raise - both cases are handled
				if self.ui.confirm_abort():
					# In tests with a mock that returns True
					return False

				# If we get here, user declined to abort in production, or mock returned False in testing
				continue

			if result.action == ChunkAction.SKIP:
				self.ui.show_skipped(chunk.files)
				return True

			if result.action == ChunkAction.REGENERATE:
				# Clear the existing description to force regeneration
				chunk.description = None
				chunk.is_llm_generated = False
				self.ui.show_regenerating()
				continue  # Go back to the start of the loop

			# For ACCEPT or EDIT actions: perform the commit
			message = result.message or chunk.description or "Update files"
			success = self._perform_commit(chunk, message)
			if not success:
				self.error_state = "failed"
			return success

	def process_all_chunks(self, chunks: list[DiffChunk], interactive: bool = True) -> bool:
		"""
		Process all chunks interactively or automatically.

		Args:
		    chunks: List of diff chunks to process
		    interactive: Whether to process interactively or automatically

		Returns:
		    True if successful, False if failed or aborted

		"""
		i = 0
		while i < len(chunks):
			chunk = chunks[i]

			if interactive:
				# Process chunk interactively
				if not self._process_chunk(chunk, i, len(chunks)):
					self.error_state = "aborted"
					return False
			else:
				# Non-interactive mode: commit all chunks automatically
				self._generate_commit_message(chunk)
				if not self._perform_commit(chunk, chunk.description or "Update files"):
					self.error_state = "failed"
					return False

			i += 1

		self.ui.show_all_committed()
		return True

	def run(self) -> bool:
		"""
		Run the commit command.

		Returns:
		    True if successful, False otherwise

		Note:
		    May raise typer.Exit when users abort

		"""
		try:
			# Get all changes
			with loading_spinner("Analyzing repository changes..."):
				changes = self._get_changes()

			if not changes:
				self.ui.show_error("No changes to commit")
				return False

			# Process each change group (staged, unstaged, untracked)
			for diff in changes:
				# Check sentence-transformers availability first with a separate loading spinner
				with loading_spinner("Checking semantic analysis capabilities..."):
					# Force check of sentence-transformers availability
					self.splitter._check_sentence_transformers_availability()  # noqa: SLF001

				# Show a separate loading spinner for model loading if sentence-transformers is available
				if self.splitter._sentence_transformers_available:  # noqa: SLF001
					with loading_spinner("Loading embedding model..."):
						# Force check of model availability which loads the model
						self.splitter._check_model_availability()  # noqa: SLF001

				# Now proceed with organizing changes semantically
				with loading_spinner("Organizing changes semantically..."):
					chunks = self.splitter.split_diff(diff, "semantic")
				if not chunks:
					continue

				# Process all chunks
				if not self.process_all_chunks(chunks):
					return False
		except typer.Exit:
			# Make sure exit is marked as an intended abort
			self.error_state = "aborted"
			raise
		except (RuntimeError, ValueError) as e:
			self.ui.show_error(str(e))
			self.error_state = "failed"
			return False
		else:
			return True
