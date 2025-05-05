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
	get_untracked_files,
	run_git_command,
	switch_branch,
)
from codemap.llm import LLMError
from codemap.utils.cli_utils import loading_spinner
from codemap.utils.file_utils import read_file_content

from . import (
	CommitMessageGenerator,
)
from .prompts import DEFAULT_PROMPT_TEMPLATE

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
MAX_FILES_BEFORE_BATCHING = 10

# Constants for content truncation
MAX_FILE_CONTENT_LINES = 300  # Maximum number of lines to include for a single file
MAX_TOTAL_CONTENT_LINES = 1000  # Maximum total lines across all untracked files


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
		Get staged, unstaged, and untracked changes, generating a GitDiff object per file.

		Returns:
		    List of GitDiff objects, each representing changes for a single file.

		Raises:
		    RuntimeError: If Git operations fail.

		"""
		changes: list[GitDiff] = []
		processed_files: set[str] = set()  # Track files already added

		try:
			# 1. Get Staged Changes (Per File)
			staged_files = run_git_command(["git", "diff", "--cached", "--name-only"]).splitlines()
			if staged_files:
				logger.debug("Found %d staged files. Fetching diffs individually...", len(staged_files))
				for file_path in staged_files:
					if file_path in processed_files:
						continue  # Avoid duplicates if somehow listed again
					try:
						file_diff_content = run_git_command(["git", "diff", "--cached", "--", file_path])
						changes.append(GitDiff(files=[file_path], content=file_diff_content, is_staged=True))
						processed_files.add(file_path)
					except GitError as e:
						logger.warning("Could not get staged diff for %s: %s", file_path, e)

			# 2. Get Unstaged Changes (Per File for files not already staged)
			unstaged_files = run_git_command(["git", "diff", "--name-only"]).splitlines()
			if unstaged_files:
				logger.debug("Found %d unstaged files. Fetching diffs individually...", len(unstaged_files))
				for file_path in unstaged_files:
					# Only process unstaged if not already captured as staged
					if file_path not in processed_files:
						try:
							file_diff_content = run_git_command(["git", "diff", "--", file_path])
							changes.append(GitDiff(files=[file_path], content=file_diff_content, is_staged=False))
							processed_files.add(file_path)
						except GitError as e:
							logger.warning("Could not get unstaged diff for %s: %s", file_path, e)

			# 3. Get Untracked Files (Per File, content formatted as diff)
			untracked_files_paths = get_untracked_files()
			if untracked_files_paths:
				logger.debug("Found %d untracked files. Reading content...", len(untracked_files_paths))
				total_content_lines = 0

				for file_path in untracked_files_paths:
					# Only process untracked if not already captured as staged/unstaged (edge case)
					if file_path not in processed_files:
						abs_path = self.repo_root / file_path
						try:
							content = read_file_content(abs_path)
							if content is not None:
								content_lines = content.splitlines()
								original_line_count = len(content_lines)
								needs_total_truncation_notice = False

								# File-level truncation
								if len(content_lines) > MAX_FILE_CONTENT_LINES:
									logger.info(
										"Untracked file %s is large (%d lines), truncating to %d lines",
										file_path,
										len(content_lines),
										MAX_FILE_CONTENT_LINES,
									)
									truncation_msg = (
										f"[... {len(content_lines) - MAX_FILE_CONTENT_LINES} more lines truncated ...]"
									)
									content_lines = content_lines[:MAX_FILE_CONTENT_LINES]
									content_lines.append(truncation_msg)

								# Total content truncation check
								if total_content_lines + len(content_lines) > MAX_TOTAL_CONTENT_LINES:
									remaining_lines = MAX_TOTAL_CONTENT_LINES - total_content_lines
									if remaining_lines > 0:
										logger.info(
											"Total untracked content size exceeded limit. Truncating %s to %d lines",
											file_path,
											remaining_lines,
										)
										content_lines = content_lines[:remaining_lines]
										needs_total_truncation_notice = True
									else:
										# No space left at all, skip this file and subsequent ones
										logger.warning(
											"Max total untracked lines reached. Skipping remaining untracked files."
										)
										break

								# Format content for the diff
								formatted_content = ["--- /dev/null", f"+++ b/{file_path}"]
								formatted_content.extend(f"+{line}" for line in content_lines)
								if needs_total_truncation_notice:
									formatted_content.append(
										"+[... Further untracked files truncated due to total size limits ...]"
									)

								file_content_str = "\n".join(formatted_content)
								changes.append(
									GitDiff(
										files=[file_path], content=file_content_str, is_staged=False, is_untracked=True
									)
								)
								total_content_lines += len(content_lines)
								processed_files.add(file_path)
								logger.debug(
									"Added content for untracked file %s (%d lines / %d original).",
									file_path,
									len(content_lines),
									original_line_count,
								)
							else:
								# File content is None or empty
								logger.warning(
									"Untracked file %s could not be read or is empty. Creating entry without content.",
									file_path,
								)
								changes.append(
									GitDiff(files=[file_path], content="", is_staged=False, is_untracked=True)
								)
								processed_files.add(file_path)
						except (OSError, UnicodeDecodeError) as file_read_error:
							logger.warning(
								"Could not read untracked file %s: %s. Creating entry without content.",
								file_path,
								file_read_error,
							)
							changes.append(GitDiff(files=[file_path], content="", is_staged=False, is_untracked=True))
							processed_files.add(file_path)

		except GitError as e:
			msg = f"Failed to get repository changes: {e}"
			logger.exception(msg)
			raise RuntimeError(msg) from e

		return changes

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
				edited_is_valid, edited_lint_messages = lint_commit_message(
					cleaned_edited_message, self.repo_root, config_loader=self.message_generator.get_config_loader()
				)
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

			# Process each diff separately to avoid parsing issues
			chunks = []

			for diff in changes:
				# Process each diff individually
				diff_chunks, _ = self.splitter.split_diff(diff)
				chunks.extend(diff_chunks)

			total_chunks = len(chunks)
			logger.info("Split files into %d chunks.", total_chunks)

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
