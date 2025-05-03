"""Main commit command implementation for CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import questionary
import typer

from codemap.git.commit_generator.utils import (  # Import needed for re-linting edits
	clean_message_for_linting,
	lint_commit_message,
)
from codemap.git.diff_splitter import DiffChunk, DiffSplitter
from codemap.git.interactive import ChunkAction, CommitUI
from codemap.git.utils import (
	GitDiff,
	GitError,
	commit_only_files,
	get_current_branch,
	get_repo_root,
	get_staged_diff,
	get_unstaged_diff,
	get_untracked_files,
	stage_files,
	switch_branch,
)
from codemap.llm import LLMError
from codemap.utils.cli_utils import loading_spinner

from . import (
	CommitMessageGenerator,
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
				# Directly use the chunk object with fallback_generation
				message = self.message_generator.fallback_generation(chunk)
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
		Process a single chunk interactively.

		Args:
		    chunk: DiffChunk to process
		    index: The 0-based index of the current chunk
		    total_chunks: The total number of chunks

		Returns:
		    True if processing should continue, False to abort or on failure.

		Raises:
		    typer.Exit: If user chooses to exit.

		"""
		logger.debug(
			"Processing chunk - Chunk ID: %s, Index: %d/%d, Files: %s",
			id(chunk),
			index + 1,
			total_chunks,
			chunk.files,
		)

		# Clear previous generation state if any
		chunk.description = None
		chunk.is_llm_generated = False

		while True:  # Loop to allow regeneration/editing
			message = ""
			used_llm = False
			passed_linting = True  # Assume true unless linting happens and fails
			lint_messages: list[str] = []  # Initialize lint messages list

			# Generate message (potentially with linting retries)
			try:
				# Generate message using the updated method
				message, used_llm, passed_linting, lint_messages = self.message_generator.generate_message_with_linting(
					chunk
				)
				chunk.description = message
				chunk.is_llm_generated = used_llm
			except (LLMError, RuntimeError) as e:
				logger.exception("Failed during message generation for chunk")
				self.ui.show_error(f"Error generating message: {e}")
				# Offer to skip or exit after generation error
				if not questionary.confirm("Skip this chunk and continue?", default=True).ask():
					self.error_state = "aborted"
					return False  # Abort
				# If user chooses to skip after generation error, we continue to the next chunk
				return True

			# -------- Handle Linting Result and User Action ---------
			if not passed_linting:
				# Display the diff chunk info first
				self.ui.display_chunk(chunk, index, total_chunks)
				# Display the failed message and lint errors
				self.ui.display_failed_lint_message(message, lint_messages, used_llm)
				# Ask user what to do on failure
				action = self.ui.get_user_action_on_lint_failure()
			else:
				# Display the valid message and diff chunk
				self.ui.display_chunk(chunk, index, total_chunks)  # Pass correct index and total
				# Ask user what to do with the valid message
				action = self.ui.get_user_action()

			# -------- Process User Action ---------
			if action == ChunkAction.COMMIT:
				# Commit with the current message (which is valid if we got here via the 'else' block)
				if self._perform_commit(chunk, message):
					return True  # Continue to next chunk
				self.error_state = "failed"
				return False  # Abort on commit failure
			if action == ChunkAction.EDIT:
				edited_message = self.ui.edit_message(message)  # Pass current message for editing
				# Clean and re-lint the edited message
				cleaned_edited_message = clean_message_for_linting(edited_message)
				edited_is_valid, edited_lint_messages = lint_commit_message(cleaned_edited_message, self.repo_root)
				if edited_is_valid:
					# Commit with the user-edited, now valid message
					if self._perform_commit(chunk, cleaned_edited_message):
						return True  # Continue to next chunk
					self.error_state = "failed"
					return False  # Abort on commit failure
				# If edited message is still invalid, show errors and loop back
				self.ui.show_warning("Edited message still failed linting.")
				# Update state for the next loop iteration to show the edited (but invalid) message
				message = edited_message
				passed_linting = False
				lint_messages = edited_lint_messages
				# No need to update used_llm as it's now user-edited
				chunk.description = message  # Update chunk description for next display
				chunk.is_llm_generated = False  # Mark as not LLM-generated
				continue  # Go back to the start of the while loop
			if action == ChunkAction.REGENERATE:
				self.ui.show_regenerating()
				chunk.description = None  # Clear description before regenerating
				chunk.is_llm_generated = False
				continue  # Go back to the start of the while loop to regenerate
			if action == ChunkAction.SKIP:
				self.ui.show_skipped(chunk.files)
				return True  # Continue to next chunk
			if action == ChunkAction.EXIT:
				if self.ui.confirm_exit():
					self.error_state = "aborted"
					# Returning False signals to stop processing chunks
					return False
				# If user cancels exit, loop back to show the chunk again
				continue

			# Should not be reached
			logger.error("Unhandled action in _process_chunk: %s", action)
			return False

	def process_all_chunks(self, chunks: list[DiffChunk], grand_total: int, interactive: bool = True) -> bool:
		"""
		Process all generated chunks.

		Args:
		    chunks: List of DiffChunk objects to process
		    grand_total: Total number of chunks initially generated
		    interactive: Whether to run in interactive mode

		Returns:
		    True if all chunks were processed successfully, False otherwise

		"""
		if not chunks:
			self.ui.show_error("No diff chunks found to process.")
			return False

		success = True
		for i, chunk in enumerate(chunks):
			if interactive:
				try:
					if not self._process_chunk(chunk, i, grand_total):
						success = False
						break
				except typer.Exit:
					# User chose to exit via typer.Exit(), which is expected
					success = False  # Indicate not all chunks were processed
					break
				except RuntimeError as e:
					self.ui.show_error(f"Runtime error processing chunk: {e}")
					success = False
					break
			else:
				# Non-interactive mode: generate and attempt commit
				try:
					message, _, passed_linting, _ = self.message_generator.generate_message_with_linting(chunk)
					if not passed_linting:
						logger.warning("Generated message failed linting in non-interactive mode: %s", message)
						# Decide behavior: skip, commit anyway, fail? Let's skip for now.
						self.ui.show_skipped(chunk.files)
						continue
					if not self._perform_commit(chunk, message):
						success = False
						break
				except (LLMError, RuntimeError, GitError) as e:
					self.ui.show_error(f"Error processing chunk non-interactively: {e}")
					success = False
					break

		return success

	def run(self) -> bool:
		"""
		Run the commit command workflow.

		Returns:
		    True if the process completed (even if aborted), False on unexpected error.

		"""
		try:
			with loading_spinner("Analyzing changes..."):
				changes = self._get_changes()

			if not changes:
				self.ui.show_message("No changes detected to commit.")
				return True

			# Combine all diffs for splitting
			all_files = [f for diff in changes for f in diff.files or []]
			# Filter unique files while preserving order
			unique_files = list(dict.fromkeys(all_files))
			all_content = "\n".join([diff.content for diff in changes if diff.content])
			combined_diff = GitDiff(files=unique_files, content=all_content, is_staged=False)

			# Split the combined diff
			chunks, _ = self.splitter.split_diff(combined_diff)
			total_chunks = len(chunks)
			logger.info("Split %d files into %d chunks.", len(unique_files), total_chunks)

			if not chunks:
				self.ui.show_error("Failed to split changes into manageable chunks.")
				return False

			# Process chunks
			success = self.process_all_chunks(chunks, total_chunks)

			if self.error_state == "aborted":
				self.ui.show_message("Commit process aborted by user.")
				return True  # Abort is considered a valid exit
			if self.error_state == "failed":
				self.ui.show_error("Commit process failed due to errors.")
				return False
			if not success:
				# If process_all_chunks returned False without setting error_state
				self.ui.show_error("Commit process failed.")
				return False
			self.ui.show_all_done()
			return True

		except RuntimeError as e:
			self.ui.show_error(str(e))
			return False
		except Exception as e:
			self.ui.show_error(f"An unexpected error occurred: {e}")
			logger.exception("Unexpected error in commit command run loop")
			return False
		finally:
			# Restore original branch if it was changed
			if self.original_branch:
				try:
					# get_current_branch is already imported
					# switch_branch is imported from codemap.git.utils now
					current = get_current_branch()
					if current != self.original_branch:
						logger.info("Restoring original branch: %s", self.original_branch)
						switch_branch(self.original_branch)
				except (GitError, Exception) as e:
					logger.warning("Could not restore original branch %s: %s", self.original_branch, e)
