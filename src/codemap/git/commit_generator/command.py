"""Main commit command implementation for CodeMap."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import questionary
import typer
from pygit2 import Commit, Diff, Patch
from pygit2.enums import FileStatus

from codemap.git.commit_generator.utils import (
	clean_message_for_linting,
	lint_commit_message,
)
from codemap.git.diff_splitter import DiffChunk
from codemap.git.interactive import ChunkAction, CommitUI
from codemap.git.semantic_grouping import SemanticGroup
from codemap.git.utils import (
	ExtendedGitRepoContext,
	GitDiff,
	GitError,
)
from codemap.llm import LLMError
from codemap.utils.cli_utils import progress_indicator
from codemap.utils.file_utils import read_file_content

from .utils import (
	CommitFormattingError,
)

if TYPE_CHECKING:
	from codemap.config import ConfigLoader
	from codemap.git.diff_splitter import DiffSplitter
	from codemap.llm import LLMClient

	from . import CommitMessageGenerator

logger = logging.getLogger(__name__)

# Constants for content truncation
MAX_FILE_CONTENT_LINES = 500  # Maximum number of lines to include for a single file
MAX_TOTAL_CONTENT_LINES = 2000  # Maximum total lines across all untracked files

# Git output constants
MIN_PORCELAIN_LINE_LENGTH = 3  # Minimum length of a valid porcelain status line

# Single file optimization constants
MAX_SMALL_CHANGE_FILES = 3  # Maximum files for small change optimization
MAX_SMALL_CHANGE_SIZE = 5000  # Maximum content size (5KB) for small change optimization


class ExitCommandError(Exception):
	"""Exception to signal an exit command."""


class CommitCommand:
	"""Handles the commit command workflow."""

	def __init__(
		self,
		bypass_hooks: bool = False,
	) -> None:
		"""
		Initialize the commit command.

		Args:
		    bypass_hooks: Whether to bypass git hooks with --no-verify

		"""
		try:
			self.ui: CommitUI = CommitUI()

			self.target_files: list[str] = []
			self.committed_files: set[str] = set()
			self.is_pathspec_mode: bool = False
			self.all_repo_files: set[str] = set()
			self.error_state: str | None = None  # Tracks reason for failure
			self.bypass_hooks: bool = bypass_hooks  # Whether to bypass git hooks with --no-verify
			self._is_single_file_mode: bool = False  # Track if we're processing a single file

			self.git_context = ExtendedGitRepoContext()

			# Store the current branch at initialization to ensure we don't switch branches unexpectedly
			try:
				self.original_branch: str | None = self.git_context.branch
			except (ImportError, GitError):
				self.original_branch = None

			# Remove eager initialization of config_loader, llm_client, splitter, message_generator
			self._config_loader: ConfigLoader | None = None
			self._llm_client: LLMClient | None = None
			self._splitter: DiffSplitter | None = None
			self._message_generator: CommitMessageGenerator | None = None

			current_repo_root = self.config_loader.get.repo_root

			if not current_repo_root:
				current_repo_root = self.git_context.repo_root

			if not current_repo_root:
				current_repo_root = self.git_context.get_repo_root()

			self.repo_root = current_repo_root

		except GitError as e:
			raise RuntimeError(str(e)) from e

	@property
	def config_loader(self) -> ConfigLoader:
		"""Lazily initialize and return the ConfigLoader instance."""
		if self._config_loader is None:
			from codemap.config import ConfigLoader

			self._config_loader = ConfigLoader.get_instance()
		return self._config_loader

	@property
	def llm_client(self) -> LLMClient:
		"""Lazily initialize and return the LLMClient instance."""
		if self._llm_client is None:
			from codemap.llm import LLMClient

			self._llm_client = LLMClient(config_loader=self.config_loader)
		return self._llm_client

	@property
	def splitter(self) -> DiffSplitter:
		"""Lazily initialize and return the DiffSplitter instance."""
		if self._splitter is None:
			from codemap.git.diff_splitter import DiffSplitter

			self._splitter = DiffSplitter()
		return self._splitter

	@property
	def message_generator(self) -> CommitMessageGenerator:
		"""Lazily initialize and return the CommitMessageGenerator instance."""
		if self._message_generator is None:
			from . import CommitMessageGenerator
			from .prompts import DEFAULT_PROMPT_TEMPLATE

			self._message_generator = CommitMessageGenerator(
				repo_root=self.repo_root,
				llm_client=self.llm_client,
				prompt_template=DEFAULT_PROMPT_TEMPLATE,
				config_loader=self.config_loader,
			)
		return self._message_generator

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
			commit = self.git_context.repo.head.peel(Commit)
			staged_files = self.git_context.repo.diff(commit.tree, self.git_context.repo.index)
			staged_file_paths = []
			if isinstance(staged_files, Diff):
				if len(staged_files) > 0:
					staged_file_paths = [delta.delta.new_file.path for delta in staged_files]
			elif isinstance(staged_files, Patch):
				staged_file_paths.append(staged_files.delta.new_file.path)
			if staged_file_paths:
				logger.debug("Found %d staged files. Fetching diffs individually...", len(staged_file_paths))
				for file_path in staged_file_paths:
					if file_path in processed_files:
						continue  # Avoid duplicates if somehow listed again
					try:
						file_diff = self.git_context.get_per_file_diff(file_path, staged=True)
						changes.append(file_diff)
						processed_files.add(file_path)
					except GitError as e:
						logger.warning("Could not get staged diff for %s: %s", file_path, e)

			# 2. Get Unstaged Changes (Per File for files not already staged)
			unstaged_files = self.git_context.repo.diff(self.git_context.repo.index, None)
			unstaged_file_paths = []
			if isinstance(unstaged_files, Diff):
				if len(unstaged_files) > 0:
					unstaged_file_paths = [delta.delta.new_file.path for delta in unstaged_files]
			elif isinstance(unstaged_files, Patch):
				unstaged_file_paths.append(unstaged_files.delta.new_file.path)
			if unstaged_file_paths:
				logger.debug("Found %d unstaged files. Fetching diffs individually...", len(unstaged_file_paths))
				for file_path in unstaged_file_paths:
					if file_path not in processed_files:
						try:
							file_diff = self.git_context.get_per_file_diff(file_path, staged=False)
							changes.append(file_diff)
							processed_files.add(file_path)
						except GitError as e:
							logger.warning("Could not get unstaged diff for %s: %s", file_path, e)

			# 3. Get Untracked Files (Per File, content formatted as diff)
			untracked_files_paths = self.git_context.get_untracked_files()
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
			self.git_context.commit_only_files(chunk.files, message, ignore_hooks=self.bypass_hooks)
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
			passed_validation = True  # Assume valid unless JSON or lint fails
			is_json_error = False  # Flag for JSON formatting errors
			error_messages: list[str] = []  # Stores lint or JSON errors

			# Generate message (potentially with linting retries)
			try:
				# Generate message using the updated method, unpack the 5 values
				(
					message,
					used_llm,
					passed_validation,
					is_json_error,
					error_messages,
				) = self.message_generator.generate_message_with_linting(chunk)

				# Store the potentially failed message in the chunk for display/editing
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

			# -------- Handle Validation Result and User Action ---------
			if not passed_validation:
				# Display the diff chunk info first
				self.ui.display_chunk(chunk, index, total_chunks)
				if is_json_error:
					# Display the raw content and JSON error
					self.ui.display_failed_json_message(message, error_messages, used_llm)
					# Ask user what to do on JSON failure
					action = self.ui.get_user_action_on_lint_failure()
				else:  # It must be a linting error
					# Display the failed message and lint errors
					self.ui.display_failed_lint_message(message, error_messages, used_llm)
					# Ask user what to do on lint failure
					action = self.ui.get_user_action_on_lint_failure()
			else:
				# Display the valid message and diff chunk
				self.ui.display_chunk(chunk, index, total_chunks)  # Pass correct index and total
				# Ask user what to do with the valid message
				action = self.ui.get_user_action()

			# -------- Process User Action ---------
			if action == ChunkAction.COMMIT:
				# Commit with the current message (which passed validation)
				if self._perform_commit(chunk, message):
					return True  # Continue to next chunk
				self.error_state = "failed"
				return False  # Abort on commit failure
			if action == ChunkAction.EDIT:
				# Edit the message (could be formatted message or raw JSON)
				current_message_to_edit = message or ""  # Default to empty string if None

				# Original error was linting, user is editing a formatted message
				edited_message = self.ui.edit_message(current_message_to_edit)
				cleaned_edited_message = clean_message_for_linting(edited_message)
				edited_is_valid, edited_error_msg = lint_commit_message(cleaned_edited_message, self.config_loader)

				if edited_is_valid:
					# Commit with the user-edited, now valid message
					if self._perform_commit(chunk, cleaned_edited_message):
						return True  # Continue to next chunk
					self.error_state = "failed"
					return False  # Abort on commit failure

				# If edited message is still invalid, show errors and loop back
				self.ui.show_warning("Edited message still failed linting.")
				# Show the lint errors for the edited message
				edited_error_messages = [edited_error_msg] if edited_error_msg else []
				self.ui.display_failed_lint_message(
					cleaned_edited_message, edited_error_messages, is_llm_generated=False
				)
				# Loop back to prompt again with the lint failure message
				chunk.description = cleaned_edited_message  # Keep cleaned message in chunk
				chunk.is_llm_generated = False
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
			# Should not be reached, but handle unknown actions
			logger.error("Unhandled action in _process_chunk: %s", action)
			return False

		# This should never be reached due to the while True loop, but add for type checker
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
					# Unpack 5 elements now
					(
						message,
						_,
						passed_validation,
						is_json_error,
						error_messages,
					) = self.message_generator.generate_message_with_linting(chunk)
					if not passed_validation:
						error_type = "JSON validation" if is_json_error else "linting"
						logger.warning(
							f"Generated message failed {error_type} in non-interactive mode: %s\nErrors: %s",
							message,
							"\n".join(error_messages),
						)
						# Decide behavior: skip, commit anyway, fail? Let's skip for now.
						self.ui.show_skipped(chunk.files)
						continue
					if not self._perform_commit(chunk, message):
						success = False
						break
				except (LLMError, RuntimeError, GitError, CommitFormattingError) as e:
					self.ui.show_error(f"Error processing chunk non-interactively: {e}")
					success = False
					break

		return success

	async def run(self, interactive: bool = True) -> bool:
		"""
		Run the commit command workflow.

		Args:
		    interactive: Whether to run in interactive mode. Defaults to True.

		Returns:
		    True if the process completed (even if aborted), False on unexpected error.

		"""
		try:
			with progress_indicator("Analyzing changes..."):
				changes = self._get_changes()

			if not changes:
				self.ui.show_message("No changes detected to commit.")
				return True

			# OPTIMIZATION: For simple changes, skip splitter overhead
			if len(changes) == 1:
				single_diff = changes[0]
				self._is_single_file_mode = True
				logger.info(
					"Simple change detected (files: %s, size: %d chars), skipping splitter for faster processing",
					single_diff.files,
					len(single_diff.content),
				)

				# Create DiffChunk directly without splitter
				from codemap.git.diff_splitter import DiffChunk

				chunk = DiffChunk(
					files=single_diff.files,
					content=single_diff.content,
					description=None,  # Will be generated by message generator
					is_llm_generated=False,
				)
				chunks = [chunk]
				total_chunks = 1
				logger.debug("Created single chunk directly for files: %s", single_diff.files)
			else:
				# Process each diff separately to avoid parsing issues
				chunks = []

				for diff in changes:
					# Process each diff individually
					diff_chunks, _ = await self.splitter.split_diff(diff)
					chunks.extend(diff_chunks)

				total_chunks = len(chunks)
				logger.info("Split %d files into %d chunks.", len(changes), total_chunks)

			if not chunks:
				# Import DiffChunk for clarity

				# If no target files available, try to detect modified files
				if not self.target_files:
					try:
						# Get staged files
						staged_output = self.git_context.repo.diff(
							self.git_context.repo.head.peel(Commit), self.git_context.repo.index
						)
						if isinstance(staged_output, Diff):
							if len(staged_output) > 0:
								self.target_files.extend([delta.delta.new_file.path for delta in staged_output])
						elif isinstance(staged_output, Patch):
							# Patch is a single patch, so always add its file
							self.target_files.append(staged_output.delta.new_file.path)

						# Get unstaged but tracked files
						unstaged_output = self.git_context.repo.diff(self.git_context.repo.index, None)
						if isinstance(unstaged_output, Diff):
							if len(unstaged_output) > 0:
								self.target_files.extend([delta.delta.new_file.path for delta in unstaged_output])
						elif isinstance(unstaged_output, Patch):
							# Patch is a single patch, so always add its file
							self.target_files.append(unstaged_output.delta.new_file.path)

						# Get untracked files
						untracked_files = self.git_context.get_untracked_files()
						if untracked_files:
							self.target_files.extend(untracked_files)

						# Remove duplicates
						self.target_files = list(set(self.target_files))

						if self.target_files:
							logger.info(f"Using detected modified files: {self.target_files}")
					except GitError as e:
						logger.warning(f"Error while getting modified files: {e}")

				# Use helper method to create fallback chunks
				chunks = self._try_create_fallback_chunks(self.target_files)

				# If still no chunks, return error
				if not chunks:
					self.ui.show_error("Failed to split changes into manageable chunks.")
					return False

			# Process chunks, passing the interactive flag
			success = self.process_all_chunks(chunks, total_chunks, interactive=interactive)

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
					current = self.git_context.branch
					if current != self.original_branch:
						logger.info("Restoring original branch: %s", self.original_branch)
						self.git_context.switch_branch(self.original_branch)
				except (GitError, Exception) as e:
					logger.warning("Could not restore original branch %s: %s", self.original_branch, e)

		# This should never be reached due to explicit returns in try/except blocks
		return False

	def _try_create_fallback_chunks(self, files: list[str]) -> list[DiffChunk]:
		"""
		Try to create fallback chunks for files when regular splitting fails.

		Args:
			files: List of file paths to process

		Returns:
			List of created DiffChunk objects
		"""
		from codemap.git.diff_splitter import DiffChunk

		chunks = []

		# Get all tracked files from git
		try:
			all_tracked_files = list(self.git_context.repo.index)
			corrected_files = []
			for file in files:
				if file in all_tracked_files:
					corrected_files.append(file)
					continue
				if file.startswith("rc/") and file.replace("rc/", "src/") in all_tracked_files:
					corrected_file = file.replace("rc/", "src/")
					logger.info(f"Corrected file path from {file} to {corrected_file}")
					corrected_files.append(corrected_file)
					continue
				logger.warning(f"Could not find a matching tracked file for {file}")
			if corrected_files:
				files = corrected_files
				logger.info(f"Using corrected file paths: {files}")
			for file in files:
				try:
					file_diff = self.git_context.get_per_file_diff(file, staged=False)
					if file_diff.content:
						logger.debug(f"Created individual chunk for {file}")
						chunks.append(DiffChunk(files=[file], content=file_diff.content))
						continue
					file_diff = self.git_context.get_per_file_diff(file, staged=True)
					if file_diff.content:
						logger.debug(f"Created individual chunk for staged {file}")
						chunks.append(DiffChunk(files=[file], content=file_diff.content))
				except GitError:
					logger.warning(f"Could not get diff for {file}")
		except GitError as e:
			logger.warning(f"Error while trying to fix file paths: {e}")

		# If still no chunks but we have files, create empty chunks as last resort
		if not chunks and files:
			logger.warning("No diffs found, creating minimal placeholder chunks")
			for file in files:
				placeholder_diff = f"--- a/{file}\n+++ b/{file}\n@@ -1 +1 @@\n No content change detected"
				chunks.append(DiffChunk(files=[file], content=placeholder_diff))

		return chunks

	def _is_simple_single_file_change(self, changes: list[GitDiff]) -> bool:
		"""
		Determine if we have a simple single-file change that can skip splitter overhead.

		Args:
		    changes: List of GitDiff objects

		Returns:
		    True if this is a simple single-file change, False otherwise
		"""
		if len(changes) != 1:
			return False

		single_diff = changes[0]

		# Single file in the diff
		if len(single_diff.files) == 1:
			return True

		# Multiple files but they're all small changes (heuristic)
		return len(single_diff.files) <= MAX_SMALL_CHANGE_FILES and len(single_diff.content) < MAX_SMALL_CHANGE_SIZE


class SemanticCommitCommand(CommitCommand):
	"""Handles the semantic commit command workflow."""

	def __init__(
		self,
		bypass_hooks: bool = False,
	) -> None:
		"""
		Initialize the SemanticCommitCommand.

		Args are similar to CLI options, allowing for programmatic use.
		"""
		# Call parent class initializer first
		super().__init__(bypass_hooks=bypass_hooks)

		config_loader = self.config_loader

		self.clustering_method = config_loader.get.embedding.clustering.method
		self.similarity_threshold = config_loader.get.commit.diff_splitter.similarity_threshold

		# Initialize attributes that may be set during execution
		self.target_files: list[str] = []
		self.is_pathspec_mode: bool = False
		self.all_repo_files: set[str] = set()

		from codemap.git.semantic_grouping.clusterer import DiffClusterer
		from codemap.git.semantic_grouping.embedder import DiffEmbedder
		from codemap.git.semantic_grouping.resolver import FileIntegrityResolver

		# Pass the loaded_model_object to DiffEmbedder
		self.embedder = DiffEmbedder(config_loader=self.config_loader)
		self.clusterer = DiffClusterer(config_loader=self.config_loader)
		self.resolver = FileIntegrityResolver(
			similarity_threshold=self.similarity_threshold, config_loader=self.config_loader
		)

	def _get_target_files(self, pathspecs: list[str] | None = None) -> list[str]:
		"""
		Get the list of target files based on pathspecs.

		Args:
		        pathspecs: Optional list of path specifications

		Returns:
		        List of file paths

		"""
		try:
			cmd = ["git", "status", "--porcelain=v1", "-uall"]
			if pathspecs:
				cmd.extend(["--", *pathspecs])
				self.is_pathspec_mode = True
			output = self.git_context.repo.status()
			target_files = []
			for file_path in output:
				if not file_path or len(file_path) < MIN_PORCELAIN_LINE_LENGTH:
					continue
				# No status parsing here, just add the file
				target_files.append(file_path)
			if self.is_pathspec_mode:
				self.all_repo_files = set(self.git_context.repo.index)
			return target_files
		except GitError as e:
			msg = f"Failed to get target files: {e}"
			logger.exception(msg)
			raise RuntimeError(msg) from e

	def _prepare_untracked_files(self, target_files: list[str]) -> list[str]:
		"""
		Prepare untracked files for diffing by adding them to the index.

		Args:
		        target_files: List of target file paths

		Returns:
		        List of untracked files that were prepared

		"""
		try:
			# Get untracked files
			untracked_files = self.git_context.get_untracked_files()

			# Filter to only those in target_files
			untracked_targets = [f for f in untracked_files if f in target_files]

			if untracked_targets:
				# Add untracked files to the index (but not staging area)
				for path in untracked_targets:
					self.git_context.repo.index.add(path)

			return untracked_targets

		except GitError as e:
			logger.warning("Error preparing untracked files: %s", e)
			return []

	def _get_combined_diff(self, target_files: list[str]) -> GitDiff:
		"""
		Get the combined diff for all target files, including staged, unstaged, and untracked.

		Args:
			target_files: List of target file paths

		Returns:
			GitDiff object with the combined diff content for all specified files.

		Raises:
			RuntimeError: If Git operations fail.
		"""
		logger.debug("SemanticCommitCommand._get_combined_diff called for files: %s", target_files)
		combined_content_parts: list[str] = []
		processed_for_combined_diff: set[str] = set()
		repo_status = self.git_context.repo.status()

		try:
			for file_path in target_files:
				if file_path in processed_for_combined_diff:
					continue

				file_status_flags = repo_status.get(file_path)
				is_untracked = file_status_flags is not None and file_status_flags & FileStatus.WT_NEW

				# Try to get staged diff first
				try:
					staged_diff = self.git_context.get_per_file_diff(file_path, staged=True)
					if staged_diff.content:
						logger.debug("  Adding STAGED diff content for %s", file_path)
						combined_content_parts.append(staged_diff.content)
						processed_for_combined_diff.add(file_path)
						# If a file has staged changes, we typically consider that its primary diff state
						# and might skip unstaged for combined view, or ensure unstaged diff is distinctly handled.
						# For now, if staged, we take it and move to next file to avoid double-adding sections.
						continue
				except GitError as e:
					logger.warning("Could not get staged diff for %s in _get_combined_diff: %s", file_path, e)

				# If no staged content, try unstaged diff
				if file_path not in processed_for_combined_diff:  # Check again in case of error above
					try:
						unstaged_diff = self.git_context.get_per_file_diff(file_path, staged=False)
						if unstaged_diff.content:
							logger.debug("  Adding UNSTAGED diff content for %s", file_path)
							combined_content_parts.append(unstaged_diff.content)
							processed_for_combined_diff.add(file_path)
							continue
					except GitError as e:
						logger.warning("Could not get unstaged diff for %s in _get_combined_diff: %s", file_path, e)

				# Handle untracked files if not already processed
				if file_path not in processed_for_combined_diff and is_untracked:
					repo_root = self.git_context.repo_root

					if repo_root is None:
						repo_root = self.git_context.get_repo_root()

					abs_path = repo_root / file_path

					try:
						content = read_file_content(abs_path)
						if content is not None:
							content_lines = content.splitlines()
							# Apply file-level truncation for combined diff context
							if len(content_lines) > MAX_FILE_CONTENT_LINES:
								logger.info(
									"Untracked file %s (in combined_diff) is large (%d lines), truncating to %d lines",
									file_path,
									len(content_lines),
									MAX_FILE_CONTENT_LINES,
								)
								truncation_msg = (
									f"[... {len(content_lines) - MAX_FILE_CONTENT_LINES} more lines truncated ...]"
								)
								content_lines = content_lines[:MAX_FILE_CONTENT_LINES]
								content_lines.append(truncation_msg)

							formatted_content_for_diff = ["--- /dev/null", f"+++ b/{file_path}"]
							formatted_content_for_diff.extend(f"+{line}" for line in content_lines)
							untracked_content_str = "\n".join(formatted_content_for_diff)
							logger.debug("  Adding UNTRACKED content for %s", file_path)
							combined_content_parts.append(untracked_content_str)
							processed_for_combined_diff.add(file_path)
						else:
							logger.warning(
								"Untracked file %s (in combined_diff) could not be read or is empty.", file_path
							)
					except (OSError, UnicodeDecodeError) as file_read_error:
						logger.warning(
							"Could not read untracked file %s (in combined_diff): %s.",
							file_path,
							file_read_error,
						)

				# If after all attempts, no diff was found for a target file, log it.
				# This could happen if the file exists but has no changes and is not untracked.
				if file_path not in processed_for_combined_diff:
					logger.debug(
						"  No diff content (staged, unstaged, or untracked) found for %s in _get_combined_diff",
						file_path,
					)

			final_combined_content = "\n".join(combined_content_parts)
			logger.debug("_get_combined_diff final content (first 500 chars): %s", repr(final_combined_content[:500]))
			return GitDiff(files=target_files, content=final_combined_content)

		except GitError as e:
			msg = f"Failed to get combined diff: {e}"
			logger.exception(msg)
			raise RuntimeError(msg) from e
		except Exception as e:  # Catch any other unexpected errors during diff aggregation
			msg = f"Unexpected error while generating combined diff: {e}"
			logger.exception(msg)
			raise RuntimeError(msg) from e

	async def _create_semantic_groups(self, chunks: list[DiffChunk]) -> list[SemanticGroup]:
		"""
		Create semantic groups from diff chunks.

		Args:
		        chunks: List of DiffChunk objects

		Returns:
		        List of SemanticGroup objects

		"""
		# Shortcut for small changes - bypass embedding process
		if len(chunks) <= 3:  # Threshold for "small changes" # noqa: PLR2004
			logger.info("Small number of chunks detected (%d), bypassing embedding process", len(chunks))
			# Create a single semantic group with all chunks
			single_group = SemanticGroup(chunks=chunks)
			# Extract all file names from chunks
			files_set = set()
			for chunk in chunks:
				files_set.update(chunk.files)
			single_group.files = list(files_set)
			# Combine all content
			combined_content = "\n".join(chunk.content for chunk in chunks)
			single_group.content = combined_content
			return [single_group]

		# Generate embeddings for chunks
		chunk_embedding_tuples = await self.embedder.embed_chunks(chunks)
		chunk_embeddings = {ce[0]: ce[1] for ce in chunk_embedding_tuples}

		# Cluster chunks
		cluster_lists = self.clusterer.cluster(chunk_embedding_tuples)

		# Create initial semantic groups
		initial_groups = [SemanticGroup(chunks=cluster) for cluster in cluster_lists]

		# Resolve file integrity constraints
		return self.resolver.resolve_violations(initial_groups, chunk_embeddings)

	async def _generate_group_messages(self, groups: list[SemanticGroup]) -> list[SemanticGroup]:
		"""
		Generate commit messages for semantic groups.

		Args:
		        groups: List of SemanticGroup objects

		Returns:
		        List of SemanticGroup objects with messages

		"""
		# Process groups individually
		from codemap.git.diff_splitter import DiffChunk
		from codemap.git.semantic_grouping.context_processor import process_chunks_with_lod

		# Get max token limit and settings from message generator's config
		max_tokens = self.config_loader.get.llm.max_output_tokens
		use_lod_context = self.config_loader.get.commit.use_lod_context

		for group in groups:
			try:
				# Create temporary DiffChunks from the group's chunks
				if use_lod_context and group.chunks and len(group.chunks) > 1:
					logger.debug("Processing semantic group with %d chunks using LOD context", len(group.chunks))
					try:
						# Process all chunks in the group with LOD context processor
						optimized_content = process_chunks_with_lod(group.chunks, max_tokens)

						if optimized_content:
							# Create a temporary chunk with the optimized content
							temp_chunk = DiffChunk(files=group.files, content=optimized_content)
						else:
							# Fallback: create a temp chunk with original content
							temp_chunk = DiffChunk(files=group.files, content=group.content)
					except Exception:
						logger.exception("Error in LOD context processing")
						# Fallback to original content
						temp_chunk = DiffChunk(files=group.files, content=group.content)
				else:
					# Use the original group content
					temp_chunk = DiffChunk(files=group.files, content=group.content)

				# Generate message with linting
				# We ignore linting status - SemanticCommitCommand is less strict
				# Unpack 5 elements, ignore validation/error status for semantic commit
				message, _, _, _, _ = self.message_generator.generate_message_with_linting(temp_chunk)

				# Store the message with the group
				group.message = message

			except Exception:
				logger.exception("Error generating message for group")
				# Use a fallback message
				group.message = f"update: changes to {len(group.files)} files"

		return groups

	async def _stage_and_commit_group(self, group: SemanticGroup) -> bool:
		"""
		Stage and commit a semantic group.

		Args:
		        group: SemanticGroup to commit

		Returns:
		        bool: Whether the commit was successful

		"""
		group_files = group.files
		try:
			# Unstage all files first (if needed, implement as needed)
			# Add the group files to the index individually
			for file_path in group_files:
				# Check file status to handle deletions correctly
				status = self.git_context.repo.status_file(file_path)
				if status == FileStatus.WT_DELETED:
					self.git_context.repo.index.remove(file_path)
				else:
					self.git_context.repo.index.add(file_path)

			self.git_context.repo.index.write()
			# Use commit_only_files utility for commit
			self.git_context.commit_only_files(group_files, group.message or "", ignore_hooks=self.bypass_hooks)
			self.committed_files.update(group_files)
			return True
		except GitError as commit_error:
			self.ui.show_error(f"Failed to commit: {commit_error}")
			return False

	async def run(self, interactive: bool = True, pathspecs: list[str] | None = None) -> bool:
		"""
		Run the semantic commit command workflow.

		Args:
		        interactive: Whether to run in interactive mode
		        pathspecs: Optional list of path specifications

		Returns:
		        bool: Whether the process completed successfully

		"""
		committed_count = 0  # Initialize this at the beginning of the method

		try:
			# Get target files
			with progress_indicator("Analyzing repository..."):
				self.target_files = self._get_target_files(pathspecs)
				logger.debug(f"SemanticCommitCommand.run: Target files after _get_target_files: {self.target_files}")

				if not self.target_files:
					self.ui.show_message("No changes detected to commit.")
					return True

				# OPTIMIZATION: For simple changes, skip splitter overhead
				if len(self.target_files) == 1:
					single_file = self.target_files[0]
					logger.info(
						"Single file change detected (%s), skipping semantic grouping for faster processing",
						single_file,
					)

					# Prepare untracked files if needed
					self._prepare_untracked_files(self.target_files)

					# Get diff for the single file
					combined_diff = self._get_combined_diff(self.target_files)

					# Create a single chunk directly
					from codemap.git.diff_splitter import DiffChunk

					chunk = DiffChunk(
						files=self.target_files, content=combined_diff.content, description=None, is_llm_generated=False
					)

					# Create a single semantic group
					single_group = SemanticGroup(chunks=[chunk])
					single_group.files = self.target_files
					single_group.content = combined_diff.content

					# Generate message for the single group
					groups = await self._generate_group_messages([single_group])

					logger.debug("Created single semantic group directly for file: %s", single_file)
				else:
					# Prepare untracked files
					self._prepare_untracked_files(self.target_files)

					# Get combined diff
					combined_diff = self._get_combined_diff(self.target_files)

					# Log diff details for debugging
					logger.debug(f"Combined diff size: {len(combined_diff.content)} characters")
					logger.debug(f"Target files: {len(self.target_files)} files")

					# Import DiffChunk before using it
					from codemap.git.diff_splitter import DiffChunk

					# Split diff into chunks
					chunks, _ = await self.splitter.split_diff(combined_diff)
					logger.debug(f"Initial chunks created: {len(chunks)}")

					# If no chunks created but we have combined diff content, create a single chunk
					if not chunks and combined_diff.content.strip():
						logger.info("No chunks created from splitter, creating a single chunk")
						chunks = [DiffChunk(files=self.target_files, content=combined_diff.content)]

					# Last resort: try creating individual chunks for each file
					if not chunks:
						logger.info("Attempting to create individual file chunks")
						chunks = self._try_create_fallback_chunks(self.target_files)

					# If still no chunks, return error
					if not chunks:
						self.ui.show_error("Failed to split changes into manageable chunks.")
						return False

					logger.info(f"Final chunk count: {len(chunks)}")

					# Create semantic groups
					with progress_indicator("Creating semantic groups..."):
						# Special case for very few files - create a single group
						if len(chunks) <= 2:  # noqa: PLR2004
							logger.info("Small number of chunks detected, creating a single semantic group")
							# Create a single semantic group with all chunks
							single_group = SemanticGroup(chunks=chunks)
							# Extract all file names from chunks
							files_set = set()
							for chunk in chunks:
								files_set.update(chunk.files)
							single_group.files = list(files_set)
							groups = [single_group]
						else:
							# Normal case - use clustering
							groups = await self._create_semantic_groups(chunks)

						if not groups:
							self.ui.show_error("Failed to create semantic groups.")
							return False

						# Generate messages for groups
						groups = await self._generate_group_messages(groups)

			# Process groups
			self.ui.show_message(f"Found {len(groups)} semantic groups of changes.")

			success = True

			for i, group in enumerate(groups):
				if interactive:
					# Display group info with improved UI
					self.ui.display_group(group, i, len(groups))

					# Get user action
					action = self.ui.get_group_action()

					if action == ChunkAction.COMMIT:
						self.ui.show_message(f"\nCommitting: {group.message}")
						if await self._stage_and_commit_group(group):
							committed_count += 1
						# If commit failed, potentially due to exit request during hook failure
						elif self.error_state == "aborted":
							raise ExitCommandError  # Propagate exit request
						else:
							self.ui.show_error(f"Failed to commit group: {group.message}")
							success = False
					elif action == ChunkAction.EDIT:
						# Allow user to edit the message
						current_message = group.message or ""  # Default to empty string if None
						edited_message = self.ui.edit_message(current_message)
						group.message = edited_message

						# Commit immediately after editing
						self.ui.show_message(f"\nCommitting: {group.message}")
						if await self._stage_and_commit_group(group):
							committed_count += 1
						# If commit failed, potentially due to exit request during hook failure
						elif self.error_state == "aborted":
							raise ExitCommandError  # Propagate exit request
						else:
							self.ui.show_error(f"Failed to commit group: {group.message}")
							success = False
					elif action == ChunkAction.REGENERATE:
						self.ui.show_regenerating()
						# Re-generate the message
						try:
							from codemap.git.diff_splitter import DiffChunk

							temp_chunk = DiffChunk(files=group.files, content=group.content)
							message, _, _, _, _ = self.message_generator.generate_message_with_linting(temp_chunk)
							group.message = message

							# Show the regenerated message
							self.ui.display_group(group, i, len(groups))
							if questionary.confirm("Commit with regenerated message?", default=True).ask():
								self.ui.show_message(f"\nCommitting: {group.message}")
								if await self._stage_and_commit_group(group):
									committed_count += 1
								# If commit failed, potentially due to exit request during hook failure
								elif self.error_state == "aborted":
									raise ExitCommandError  # Propagate exit request
								else:
									self.ui.show_error(f"Failed to commit group: {group.message}")
									success = False
							else:
								self.ui.show_skipped(group.files)
						except (LLMError, GitError, RuntimeError) as e:
							self.ui.show_error(f"Error regenerating message: {e}")
							if questionary.confirm("Skip this group?", default=True).ask():
								self.ui.show_skipped(group.files)
							else:
								success = False
					elif action == ChunkAction.SKIP:
						self.ui.show_skipped(group.files)
					elif action == ChunkAction.EXIT and self.ui.confirm_exit():
						# This is a user-initiated exit, should not be considered a failure
						self.ui.show_message("Commit process exited by user.")
						return True  # Return true to indicate normal exit, not failure
				else:
					# In non-interactive mode, commit each group immediately
					group.message = group.message or f"update: changes to {len(group.files)} files"
					self.ui.show_message(f"\nCommitting: {group.message}")
					if await self._stage_and_commit_group(group):
						committed_count += 1
					# If commit failed, potentially due to exit request during hook failure
					elif self.error_state == "aborted":
						raise ExitCommandError  # Propagate exit request
					else:
						self.ui.show_error(f"Failed to commit group: {group.message}")
						success = False

			if committed_count > 0:
				self.ui.show_message(f"Successfully committed {committed_count} semantic groups.")
				self.ui.show_all_done()
			else:
				self.ui.show_message("No changes were committed.")

			return success
		except ExitCommandError:
			# User requested to exit during lint failure handling
			return committed_count > 0
		except RuntimeError as e:
			self.ui.show_error(str(e))
			return False
		except Exception as e:
			self.ui.show_error(f"An unexpected error occurred: {e}")
			logger.exception("Unexpected error in semantic commit command")
			return False
