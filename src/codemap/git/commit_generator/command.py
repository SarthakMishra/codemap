"""Main commit command implementation for CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from codemap.git.diff_splitter import DiffChunk, DiffSplitter
from codemap.git.interactive import ChunkAction, ChunkResult, CommitUI
from codemap.git.utils import (
	GitDiff,
	GitError,
	commit_only_files,
	get_repo_root,
	get_staged_diff,
	get_unstaged_diff,
	get_untracked_files,
	stage_files,
)
from codemap.llm import LLMError
from codemap.utils.cli_utils import loading_spinner

from . import (
	CommitMessageGenerator,
	DiffChunkData,  # Add this import for the alias
)
from .prompts import DEFAULT_PROMPT_TEMPLATE

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
MAX_FILES_BEFORE_BATCHING = 10


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

			# Store the current branch at initialization to ensure we don't switch branches unexpectedly
			try:
				from codemap.git.pr_generator.utils import get_current_branch

				self.original_branch = get_current_branch()
			except (ImportError, GitError):
				self.original_branch = None

			# Create LLM client and configs
			from codemap.llm import create_client
			from codemap.utils.config_loader import ConfigLoader

			config_loader = ConfigLoader(repo_root=self.repo_root)
			llm_client = create_client(repo_path=self.repo_root, model=model)

			# Create the commit message generator with required parameters
			self.message_generator = CommitMessageGenerator(
				repo_root=self.repo_root,
				llm_client=llm_client,
				prompt_template=DEFAULT_PROMPT_TEMPLATE,
				config_loader=config_loader,
			)

			self.error_state = None  # Tracks reason for failure: "failed", "aborted", etc.
			self.bypass_hooks = bypass_hooks  # Whether to bypass git hooks with --no-verify
		except GitError as e:
			raise RuntimeError(str(e)) from e

	def _get_changes(self) -> list[GitDiff]:
		"""
		Get staged, unstaged, and untracked changes separately.

		Returns:
		    List of GitDiff objects representing changes.

		Raises:
		    RuntimeError: If Git operations fail.

		"""
		changes = []
		try:
			# Get staged changes
			staged = get_staged_diff()
			if staged and staged.files:
				changes.append(staged)
				logger.debug("Found %d staged files.", len(staged.files))

			# Get unstaged changes
			unstaged = get_unstaged_diff()
			if unstaged and unstaged.files:
				changes.append(unstaged)
				logger.debug("Found %d unstaged files.", len(unstaged.files))

			# Get untracked files
			untracked_files = get_untracked_files()
			if untracked_files:
				untracked_diff = GitDiff(files=untracked_files, content="", is_staged=False)
				changes.append(untracked_diff)
				logger.debug("Found %d untracked files.", len(untracked_files))

		except GitError as e:
			msg = f"Failed to get repository changes: {e}"
			logger.exception(msg)
			raise RuntimeError(msg) from e

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
				# Generate the message using the generator
				message, is_llm = self.message_generator.generate_message(chunk)

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
			# Ensure the specific files for this chunk are staged
			# This prevents accidentally committing unrelated staged changes
			with loading_spinner("Staging chunk files..."):
				stage_files(chunk.files)

			# Commit only the files specified in the chunk
			commit_only_files(chunk.files, message, ignore_hooks=self.bypass_hooks)
			self.ui.show_success(f"Committed {len(chunk.files)} files.")
			return True
		except GitError as e:
			error_msg = f"Error during commit: {e}"
			self.ui.show_error(error_msg)
			logger.exception(error_msg)
			self.error_state = "failed"
			return False

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

	def process_all_chunks(self, chunks: list[DiffChunk], grand_total: int, interactive: bool = True) -> bool:
		"""
		Process all chunks interactively or automatically.

		Args:
		    chunks: List of diff chunks to process
		    grand_total: The total number of chunks across all batches
		    interactive: Whether to process interactively or automatically

		Returns:
		    True if successful, False if failed or aborted

		"""
		for i, chunk in enumerate(chunks):
			if interactive:
				if not self._process_chunk(chunk, i, grand_total):
					# _process_chunk sets error_state and returns False on failure/abort
					return False
			else:
				# Non-interactive mode: commit all chunks automatically
				self._generate_commit_message(chunk)
				if not self._perform_commit(chunk, chunk.description or "Update files"):
					self.error_state = "failed"
					return False

		# If loop completes without returning False, it was successful
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
			# 1. Get changes
			with loading_spinner("Analyzing repository changes..."):
				change_diffs = self._get_changes()

			if not change_diffs:
				self.ui.show_error("No changes detected to commit.")
				return True  # Success, nothing to do

			# Combine all changes into one diff for splitting
			# This simplifies logic, assuming splitter can handle combined diff
			combined_files = [f for diff in change_diffs for f in diff.files]
			combined_content = "\n".join(diff.content for diff in change_diffs if diff.content)
			combined_diff = GitDiff(files=combined_files, content=combined_content)

			# 2. Split the combined diff
			with loading_spinner("Organizing changes semantically..."):
				chunks, filtered_files = self.splitter.split_diff(combined_diff)

			# 3. Handle filtered files warning
			if filtered_files:
				self.ui.show_error(f"Skipped {len(filtered_files)} large files from analysis due to size limits.")

			# 4. Check if there are chunks to process
			if not chunks:
				self.ui.show_error("No processable changes found after analysis.")
				return True  # Success, nothing to commit after filtering/splitting

			# 5. Process all chunks
			grand_total = len(chunks)
			if not self.process_all_chunks(chunks, grand_total):
				# Error state is set within process_all_chunks or _process_chunk
				return False

		except typer.Exit:
			self.error_state = "aborted"
			raise
		except (RuntimeError, ValueError, GitError) as e:
			self.ui.show_error(str(e))
			self.error_state = "failed"
			logger.exception("Commit command failed.")  # Log the exception
			return False
		else:
			# Check if we need to restore the original branch
			if self.original_branch:
				try:
					from codemap.git.pr_generator.utils import checkout_branch, get_current_branch

					current_branch = get_current_branch()
					if current_branch != self.original_branch:
						self.ui.show_success(f"Restoring original branch: {self.original_branch}")
						checkout_branch(self.original_branch)
				except (ImportError, GitError) as e:
					logger.warning("Failed to restore original branch: %s", str(e))

			return True
