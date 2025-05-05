"""Main commit command implementation for CodeMap."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import questionary
import typer

from codemap.git.commit_generator.utils import (
	clean_message_for_linting,
	lint_commit_message,
)
from codemap.git.diff_splitter import DiffChunk, DiffSplitter
from codemap.git.interactive import ChunkAction, CommitUI
from codemap.git.semantic_grouping import SemanticGroup
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

# Git output constants
MIN_PORCELAIN_LINE_LENGTH = 3  # Minimum length of a valid porcelain status line

class ExitCommandError(Exception):
	"""Exception to signal an exit command."""


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
				# Allow user to edit the message
				current_message = chunk.description or ""  # Default to empty string if None
				edited_message = self.ui.edit_message(current_message)
				cleaned_edited_message = clean_message_for_linting(edited_message)
				edited_is_valid, _ = lint_commit_message(cleaned_edited_message)
				# Convert error_message to list for compatibility with the rest of the code
				if edited_is_valid:
					# Commit with the user-edited, now valid message
					if self._perform_commit(chunk, cleaned_edited_message):
						return True  # Continue to next chunk
					self.error_state = "failed"
					return False  # Abort on commit failure
				# If edited message is still invalid, show errors and loop back
				self.ui.show_warning("Edited message still failed linting.")
				# Update state for the next loop iteration to show the edited (but invalid) message
				chunk.description = edited_message
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

	def run(self, interactive: bool = True) -> bool:
		"""
		Run the commit command workflow.

		Args:
		    interactive: Whether to run in interactive mode. Defaults to True.

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
					current = get_current_branch()
					if current != self.original_branch:
						logger.info("Restoring original branch: %s", self.original_branch)
						switch_branch(self.original_branch)
				except (GitError, Exception) as e:
					logger.warning("Could not restore original branch %s: %s", self.original_branch, e)


class SemanticCommitCommand(CommitCommand):
	"""Handles the semantic commit command workflow."""

	def __init__(
		self,
		path: Path | None = None,
		model: str = "gpt-4o-mini",
		bypass_hooks: bool = False,
		embedding_model: str = "all-MiniLM-L6-v2",
		clustering_method: str = "agglomerative",
		similarity_threshold: float = 0.6,
	) -> None:
		"""
		Initialize the semantic commit command.

		Args:
		        path: Optional path to start from
		        model: LLM model to use for commit message generation
		        bypass_hooks: Whether to bypass git hooks with --no-verify
		        embedding_model: Model to use for generating embeddings
		        clustering_method: Method to use for clustering ("agglomerative" or "dbscan")
		        similarity_threshold: Threshold for group similarity to trigger merging

		"""
		super().__init__(path, model, bypass_hooks)

		# Import semantic grouping components
		from codemap.git.semantic_grouping.clusterer import DiffClusterer
		from codemap.git.semantic_grouping.embedder import DiffEmbedder
		from codemap.git.semantic_grouping.resolver import FileIntegrityResolver

		# Initialize semantic grouping components
		self.embedder = DiffEmbedder(model_name=embedding_model)
		self.clusterer = DiffClusterer(method=clustering_method)
		self.resolver = FileIntegrityResolver(similarity_threshold=similarity_threshold)

		# Track state for commits
		self.committed_files: set[str] = set()
		self.is_pathspec_mode = False
		self.all_repo_files: set[str] = set()
		self.target_files: list[str] = []

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

			output = run_git_command(cmd)

			# Parse porcelain output to get file paths
			target_files = []
			for line in output.splitlines():
				if not line or len(line) < MIN_PORCELAIN_LINE_LENGTH:
					continue

				status = line[:2]
				file_path = line[3:].strip()

				# Handle renamed files
				if status.startswith("R"):
					# Extract the new file name after the arrow
					file_path = file_path.split(" -> ")[1]

				target_files.append(file_path)

			# If in pathspec mode, get all repo files for later use
			if self.is_pathspec_mode:
				self.all_repo_files = set(run_git_command(["git", "ls-files"]).splitlines())

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
			untracked_files = get_untracked_files()

			# Filter to only those in target_files
			untracked_targets = [f for f in untracked_files if f in target_files]

			if untracked_targets:
				# Add untracked files to the index (but not staging area)
				run_git_command(["git", "add", "-N", "--", *untracked_targets])

			return untracked_targets

		except GitError as e:
			logger.warning("Error preparing untracked files: %s", e)
			return []

	def _get_combined_diff(self, target_files: list[str]) -> GitDiff:
		"""
		Get the combined diff for all target files.

		Args:
		        target_files: List of target file paths

		Returns:
		        GitDiff object with the combined diff

		"""
		try:
			# Get diff against HEAD for all target files
			diff_content = run_git_command(["git", "diff", "HEAD", "--", *target_files])

			return GitDiff(files=target_files, content=diff_content)

		except GitError as e:
			msg = f"Failed to get combined diff: {e}"
			logger.exception(msg)
			raise RuntimeError(msg) from e

	def _create_semantic_groups(self, chunks: list[DiffChunk]) -> list[SemanticGroup]:
		"""
		Create semantic groups from diff chunks.

		Args:
		        chunks: List of DiffChunk objects

		Returns:
		        List of SemanticGroup objects

		"""
		# Generate embeddings for chunks
		chunk_embedding_tuples = self.embedder.embed_chunks(chunks)
		chunk_embeddings = {ce[0]: ce[1] for ce in chunk_embedding_tuples}

		# Cluster chunks
		cluster_lists = self.clusterer.cluster(chunk_embedding_tuples)

		# Create initial semantic groups
		initial_groups = [SemanticGroup(chunks=cluster) for cluster in cluster_lists]

		# Resolve file integrity constraints
		return self.resolver.resolve_violations(initial_groups, chunk_embeddings)

	def _generate_group_messages(self, groups: list[SemanticGroup]) -> list[SemanticGroup]:
		"""
		Generate commit messages for semantic groups.

		Args:
		        groups: List of SemanticGroup objects

		Returns:
		        List of SemanticGroup objects with messages

		"""
		from codemap.git.diff_splitter import DiffChunk
		from codemap.git.semantic_grouping.context_processor import process_chunks_with_lod

		# Get max token limit and settings from message generator's config
		# since that's where the config_loader is stored
		llm_config = self.message_generator.get_config_loader().get("llm", {})
		max_tokens = llm_config.get("max_context_tokens", 4000)
		use_lod_context = llm_config.get("use_lod_context", True)

		for group in groups:
			try:
				# Create temporary DiffChunks from the group's chunks
				if use_lod_context and len(group.chunks) > 1:
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
				message, _, _, _ = self.message_generator.generate_message_with_linting(temp_chunk)

				# Store the message with the group
				group.message = message

			except Exception:
				logger.exception("Error generating message for group")
				# Use a fallback message
				group.message = f"update: changes to {len(group.files)} files"

		return groups

	def _stage_and_commit_group(self, group: SemanticGroup) -> bool:
		"""
		Stage and commit a semantic group.

		Args:
		        group: SemanticGroup to commit

		Returns:
		        bool: Whether the commit was successful

		"""
		# Get files in this group
		group_files = group.files

		try:
			# First, unstage any previously staged files
			# This ensures we only commit the current group
			run_git_command(["git", "reset"])

			# Add the group files to the index
			run_git_command(["git", "add", "--", *group_files])

			# Create the commit with the group message
			commit_cmd = ["git", "commit", "-m", group.message or ""]

			# Add --no-verify if bypass_hooks is set
			if self.bypass_hooks:
				commit_cmd.append("--no-verify")

			try:
				run_git_command(commit_cmd)

				# Mark files as committed
				self.committed_files.update(group_files)
				return True
			except GitError as commit_error:
				# Check if this is a pre-commit hook failure
				if "pre-commit" in str(commit_error) and not self.bypass_hooks:
					# Show the error message for clarity
					error_msg = str(commit_error)
					if "conventional commit" in error_msg.lower() or "lint" in error_msg.lower():
						# Extract the lint errors if possible
						lint_errors = [
							line.strip()
							for line in error_msg.splitlines()
							if line.strip() and not line.startswith("Command") and "returned non-zero" not in line
						]

						# Show the message with lint warnings
						message = group.message or ""  # Use empty string if None
						self.ui.display_failed_lint_message(message, lint_errors, is_llm_generated=True)

						# Present options specific to lint failures
						lint_action = self.ui.get_user_action_on_lint_failure()

						if lint_action == ChunkAction.REGENERATE:
							self.ui.show_regenerating()
							try:
								# Create temporary DiffChunk for regeneration
								from codemap.git.diff_splitter import DiffChunk

								temp_chunk = DiffChunk(files=group.files, content=group.content)

								# Use the linting-aware prompt this time
								message, _, _, _ = self.message_generator.generate_message_with_linting(temp_chunk)
								group.message = message

								# Try again with the new message
								return self._stage_and_commit_group(group)
							except (LLMError, GitError, RuntimeError) as e:
								self.ui.show_error(f"Error regenerating message: {e}")
								return False
						elif lint_action == ChunkAction.COMMIT:
							# User chose to bypass the linter
							self.ui.show_message("Bypassing linter and committing with --no-verify")
							commit_cmd.append("--no-verify")
							try:
								run_git_command(commit_cmd)
								# Mark files as committed
								self.committed_files.update(group_files)
								return True
							except GitError as e:
								self.ui.show_error(f"Commit failed even with --no-verify: {e}")
								return False
						elif lint_action == ChunkAction.EDIT:
							edited_message = self.ui.edit_message(group.message or "")  # Empty string as fallback
							group.message = edited_message
							return self._stage_and_commit_group(group)
						elif lint_action == ChunkAction.SKIP:
							self.ui.show_skipped(group.files)
							return False
						elif lint_action == ChunkAction.EXIT:
							if self.ui.confirm_exit():
								raise ExitCommandError from None
							return False

					# Generic pre-commit hook failure (not specifically commit message linting)
					if self.ui.confirm_bypass_hooks():
						# Try again with --no-verify
						commit_cmd.append("--no-verify")
						run_git_command(commit_cmd)

						# Mark files as committed
						self.committed_files.update(group_files)
						return True

				# Either not a pre-commit hook error or user declined to bypass
				self.ui.show_error(f"Failed to commit: {commit_error}")
				return False

		except GitError as e:
			self.ui.show_error(f"Git operation failed: {e}")
			return False
		except Exception as e:
			self.ui.show_error(f"Unexpected error during commit: {e}")
			logger.exception("Unexpected error in _stage_and_commit_group")
			return False

	def run(self, interactive: bool = True, pathspecs: list[str] | None = None) -> bool:
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
			with loading_spinner("Analyzing repository..."):
				self.target_files = self._get_target_files(pathspecs)

				if not self.target_files:
					self.ui.show_message("No changes detected to commit.")
					return True

				# Prepare untracked files
				self._prepare_untracked_files(self.target_files)

				# Get combined diff
				combined_diff = self._get_combined_diff(self.target_files)

				# Split diff into chunks
				chunks, _ = self.splitter.split_diff(combined_diff)

				if not chunks:
					self.ui.show_error("Failed to split changes into manageable chunks.")
					return False

			# Create semantic groups
			with loading_spinner("Creating semantic groups..."):
				groups = self._create_semantic_groups(chunks)

				if not groups:
					self.ui.show_error("Failed to create semantic groups.")
					return False

				# Generate messages for groups
				groups = self._generate_group_messages(groups)

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
						if self._stage_and_commit_group(group):
							committed_count += 1
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
						if self._stage_and_commit_group(group):
							committed_count += 1
						else:
							self.ui.show_error(f"Failed to commit group: {group.message}")
							success = False
					elif action == ChunkAction.REGENERATE:
						self.ui.show_regenerating()
						# Re-generate the message
						try:
							from codemap.git.diff_splitter import DiffChunk

							temp_chunk = DiffChunk(files=group.files, content=group.content)
							message, _, _, _ = self.message_generator.generate_message_with_linting(temp_chunk)
							group.message = message

							# Show the regenerated message
							self.ui.display_group(group, i, len(groups))
							if questionary.confirm("Commit with regenerated message?", default=True).ask():
								self.ui.show_message(f"\nCommitting: {group.message}")
								if self._stage_and_commit_group(group):
									committed_count += 1
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
						return committed_count > 0
				else:
					# In non-interactive mode, commit each group immediately
					group.message = group.message or f"update: changes to {len(group.files)} files"
					self.ui.show_message(f"\nCommitting: {group.message}")
					if self._stage_and_commit_group(group):
						committed_count += 1
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
