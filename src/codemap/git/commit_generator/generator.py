"""Generator module for commit messages."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from codemap.config import ConfigLoader
from codemap.git.diff_splitter import DiffChunk
from codemap.git.semantic_grouping.context_processor import process_chunks_with_lod
from codemap.llm import LLMClient, LLMError

from .prompts import (
	COMMIT_SYSTEM_PROMPT,
	MOVE_CONTEXT,
	get_lint_prompt_template,
	prepare_lint_prompt,
	prepare_prompt,
)
from .schemas import CommitMessageSchema
from .utils import (
	CommitFormattingError,
	clean_message_for_linting,
	format_commit,
	lint_commit_message,
)

logger = logging.getLogger(__name__)

MAX_DEBUG_CONTENT_LENGTH = 100
EXPECTED_PARTS_COUNT = 2  # Type+scope and description
MIN_DIRS_FOR_MOVE = 2  # Minimum number of directories for a move operation


class CommitMessageGenerator:
	"""Generates commit messages using LLMs."""

	def __init__(
		self,
		repo_root: Path,
		llm_client: LLMClient,
		prompt_template: str,
		config_loader: ConfigLoader,
	) -> None:
		"""
		Initialize the commit message generator.

		Args:
		    repo_root: Root directory of the Git repository
		    llm_client: LLMClient instance to use
		    prompt_template: Custom prompt template to use
		    config_loader: ConfigLoader instance to use for configuration

		"""
		self.repo_root = repo_root
		self.prompt_template = prompt_template
		self._config_loader = config_loader
		self.client = llm_client

		# Add commit template to client
		self.client.set_template("commit", self.prompt_template)

		# Get max token limit from config
		self.max_tokens = config_loader.get.llm.max_output_tokens

		# Flag to control whether to use the LOD-based context processing
		self.use_lod_context = config_loader.get.commit.use_lod_context

	def extract_file_info(self, chunk: DiffChunk) -> dict[str, Any]:
		"""
		Extract file information from the diff chunk.

		Args:
		    chunk: Diff chunk object to extract information from

		Returns:
		    Dictionary with information about files

		"""
		file_info = {}
		files = chunk.files
		for file in files:
			if not isinstance(file, str):
				continue  # Skip non-string file entries
			file_path = self.repo_root / file
			if not file_path.exists():
				continue
			try:
				extension = file_path.suffix.lstrip(".")
				file_info[file] = {
					"extension": extension,
					"directory": str(file_path.parent.relative_to(self.repo_root)),
				}
				path_parts = file_path.parts
				if len(path_parts) > 1:
					if "src" in path_parts:
						idx = path_parts.index("src")
						if idx + 1 < len(path_parts):
							file_info[file]["module"] = path_parts[idx + 1]
					elif "tests" in path_parts:
						file_info[file]["module"] = "tests"
			except (ValueError, IndexError, TypeError):
				continue
		return file_info

	def _prepare_prompt(self, chunk: DiffChunk) -> str:
		"""
		Prepare the prompt for the LLM.

		Args:
		    chunk: Diff chunk object to prepare prompt for

		Returns:
		    Prepared prompt with diff and file information

		"""
		file_info = self.extract_file_info(chunk)

		# Get the diff content
		diff_content = chunk.content

		# Use the LOD-based context processor if enabled
		if self.use_lod_context:
			logger.debug("Using LOD-based context processing")
			try:
				# Process the chunk with LOD to optimize context length
				enhanced_diff_content = process_chunks_with_lod([chunk], self.max_tokens)

				if enhanced_diff_content:
					diff_content = enhanced_diff_content
					logger.debug("LOD context processing successful")
				else:
					logger.debug("LOD processing returned empty result, using original content")
			except Exception:
				logger.exception("Error during LOD context processing")
				# Continue with the original content if LOD processing fails
		else:
			# Use the original binary file detection logic
			binary_files = []
			for file_path in chunk.files:
				if file_path in file_info:
					extension = file_info[file_path].get("extension", "").lower()
					# Common binary file extensions
					binary_extensions = {
						"png",
						"jpg",
						"jpeg",
						"gif",
						"bmp",
						"tiff",
						"ico",
						"webp",  # Images
						"mp3",
						"wav",
						"ogg",
						"flac",
						"aac",  # Audio
						"mp4",
						"avi",
						"mkv",
						"mov",
						"webm",  # Video
						"pdf",
						"doc",
						"docx",
						"xls",
						"xlsx",
						"ppt",
						"pptx",  # Documents
						"zip",
						"tar",
						"gz",
						"rar",
						"7z",  # Archives
						"exe",
						"dll",
						"so",
						"dylib",  # Binaries
						"ttf",
						"otf",
						"woff",
						"woff2",  # Fonts
						"db",
						"sqlite",
						"mdb",  # Databases
					}

					if extension in binary_extensions:
						binary_files.append(file_path)

				# For absolute paths, try to check if the file is binary
				abs_path = self.repo_root / file_path
				try:
					if abs_path.exists():
						from codemap.utils.file_utils import is_binary_file

						if is_binary_file(abs_path) and file_path not in binary_files:
							binary_files.append(file_path)
				except (OSError, PermissionError) as e:
					# If any error occurs during binary check, log it and continue
					logger.debug("Error checking if %s is binary: %s", file_path, str(e))

			# If we have binary files or no diff content, enhance the prompt
			enhanced_diff_content = diff_content
			if not diff_content or binary_files:
				# Create a specialized header for binary files
				binary_files_header = ""
				if binary_files:
					binary_files_header = "BINARY FILES DETECTED:\n"
					for binary_file in binary_files:
						extension = file_info.get(binary_file, {}).get("extension", "unknown")
						binary_files_header += f"- {binary_file} (binary {extension} file)\n"
					binary_files_header += "\n"

				# If no diff content, create a more informative message about binary files
				if not diff_content:
					file_descriptions = []
					for file_path in chunk.files:
						if file_path in binary_files:
							extension = file_info.get(file_path, {}).get("extension", "unknown")
							file_descriptions.append(f"{file_path} (binary {extension} file)")
						else:
							extension = file_info.get(file_path, {}).get("extension", "")
							file_descriptions.append(f"{file_path} ({extension} file)")

					enhanced_diff_content = (
						f"{binary_files_header}This chunk contains changes to the following files "
						f"with no visible diff content (likely binary changes):\n"
					)
					for desc in file_descriptions:
						enhanced_diff_content += f"- {desc}\n"
				else:
					# If there is diff content but also binary files, add the binary files header
					enhanced_diff_content = binary_files_header + diff_content

			diff_content = enhanced_diff_content

		# Create a context dict with default values for template variables
		context = {
			"diff": diff_content,
			"files": file_info,
			"config_loader": self._config_loader,
			"schema": CommitMessageSchema,
			"original_message": "",  # Default value for original_message
			"lint_errors": "",  # Default value for lint_errors
		}

		# Add move operation context if this is a file move
		if getattr(chunk, "is_move", False):
			# For a move operation, files in chunk.files should include both source and destination paths
			# We need to identify which files are source (deleted) and which are destination (added)

			# First attempt: Try to parse from the diff content to identify actual moved file pairs
			moved_file_pairs = self._extract_moved_file_pairs(chunk)

			if moved_file_pairs:
				# Create context based on actual file pairs extracted from diff
				move_contexts = self._create_move_contexts_from_pairs(moved_file_pairs)
				if move_contexts:
					diff_content += "\n\n" + "\n".join(move_contexts)
					context["diff"] = diff_content
			else:
				# Fallback: Group files by directory and infer move operations
				# Group files by directory
				files_by_dir = {}
				for file_path in chunk.files:
					dir_path = str(Path(file_path).parent)
					if dir_path not in files_by_dir:
						files_by_dir[dir_path] = []
					files_by_dir[dir_path].append(file_path)

				# Find source and target directories
				dirs = list(files_by_dir.keys())
				if len(dirs) >= MIN_DIRS_FOR_MOVE:
					# Simplest case: first directory is source, second is target
					source_dir = dirs[0]
					target_dir = dirs[1]

					# We don't have exact mapping information, so list all files
					files_list = "\n".join([f"- {file}" for file in chunk.files])

					# Format the move context and add it to the diff content
					move_context = MOVE_CONTEXT.format(
						files=files_list,
						source_dir=source_dir if source_dir not in {".", ""} else "root directory",
						target_dir=target_dir if target_dir not in {".", ""} else "root directory",
					)

					diff_content += "\n\n" + move_context
					context["diff"] = diff_content

		# Prepare and return the prompt
		return prepare_prompt(
			template=self.prompt_template,
			diff_content=diff_content,
			file_info=file_info,
			config_loader=self._config_loader,
			extra_context=context,  # Pass the context with default values
		)

	def _extract_moved_file_pairs(self, chunk: DiffChunk) -> list[tuple[str, str]]:
		"""
		Extract moved file pairs from a move operation diff.

		This analyzes diff content to identify pairs of files that were moved
		from one location to another.

		Args:
			chunk: DiffChunk representing a file move operation

		Returns:
			List of (source_path, target_path) tuples
		"""
		if not chunk.content:
			return []

		# Look for patterns in the diff content that indicate moves
		# Git diff for moves typically shows a deletion and an addition
		moved_pairs = []

		try:
			# Parse for deleted/added file patterns
			deleted_files = []
			added_files = []

			# Simple regex-based parsing (could be improved with proper diff parsing)
			deleted_pattern = re.compile(r"diff --git a/(.*?) b/.*?\n.*?deleted file mode")
			added_pattern = re.compile(r"diff --git a/.*? b/(.*?)\n.*?new file mode")

			# Find all deleted files and added files using list comprehensions
			deleted_files = [match.group(1) for match in deleted_pattern.finditer(chunk.content)]
			added_files = [match.group(1) for match in added_pattern.finditer(chunk.content)]

			# Try to match deleted and added files by name
			for deleted in deleted_files:
				deleted_name = Path(deleted).name
				for added in added_files:
					added_name = Path(added).name

					# If filenames match, assume it's a move
					if deleted_name == added_name:
						moved_pairs.append((deleted, added))
						# Remove these files from consideration for other pairs
						added_files.remove(added)
						break

			return moved_pairs
		except Exception:
			logger.exception("Error extracting moved file pairs")
			return []

	def _create_move_contexts_from_pairs(self, moved_file_pairs: list[tuple[str, str]]) -> list[str]:
		"""
		Create move context strings for each group of moved files.

		Args:
			moved_file_pairs: List of (source_path, target_path) tuples

		Returns:
			List of formatted move context strings
		"""
		if not moved_file_pairs:
			return []

		# Group by source/target directories
		move_pairs = {}  # (source_dir, target_dir) -> [(source, target), ...]

		for source, target in moved_file_pairs:
			source_dir = str(Path(source).parent)
			target_dir = str(Path(target).parent)
			dir_pair = (source_dir, target_dir)

			if dir_pair not in move_pairs:
				move_pairs[dir_pair] = []
			move_pairs[dir_pair].append((source, target))

		# Create context for each distinct move operation
		move_contexts = []
		for (src_dir, tgt_dir), file_pairs in move_pairs.items():
			# Create detailed file list with source → target mapping
			files_list = "\n".join([f"- {src} → {tgt}" for src, tgt in file_pairs])

			# Format source/target directory names
			src_dir_display = "root directory" if src_dir in {".", ""} else src_dir
			tgt_dir_display = "root directory" if tgt_dir in {".", ""} else tgt_dir

			# Create context using the template
			move_contexts.append(
				MOVE_CONTEXT.format(files=files_list, source_dir=src_dir_display, target_dir=tgt_dir_display)
			)

		return move_contexts

	def fallback_generation(self, chunk: DiffChunk) -> str:
		"""
		Generate a fallback commit message without LLM.

		This is used when LLM-based generation fails or is disabled.

		Args:
		    chunk: Diff chunk object to generate message for

		Returns:
		    Generated commit message

		"""
		commit_type = "chore"

		# Get files directly from the chunk object
		files = chunk.files

		# Filter only strings (defensive, though DiffChunk.files should be list[str])
		string_files = [f for f in files if isinstance(f, str)]

		for file in string_files:
			if file.startswith("tests/"):
				commit_type = "test"
				break
			if file.startswith("docs/") or file.endswith(".md"):
				commit_type = "docs"
				break

		# Get content directly from the chunk object
		content = chunk.content

		if isinstance(content, str) and ("fix" in content.lower() or "bug" in content.lower()):
			commit_type = "fix"  # Be slightly smarter about 'fix' type

		# Use chunk description if available and seems specific (not just placeholder)
		chunk_desc = chunk.description
		placeholder_descs = ["update files", "changes in", "hunk in", "new file:"]
		# Ensure chunk_desc is not None before calling lower()
		use_chunk_desc = chunk_desc and not any(p in chunk_desc.lower() for p in placeholder_descs)

		if use_chunk_desc and chunk_desc:  # Add explicit check for chunk_desc
			description = chunk_desc
			# Attempt to extract a type from the chunk description if possible
			# Ensure chunk_desc is not None before calling lower() and split()
			if chunk_desc.lower().startswith(
				("feat", "fix", "refactor", "docs", "test", "chore", "style", "perf", "ci", "build")
			):
				parts = chunk_desc.split(":", 1)
				if len(parts) > 1:
					commit_type = parts[0].split("(")[0].strip().lower()  # Extract type before scope
					description = parts[1].strip()
		else:
			# Generate description based on file count/path if no specific chunk desc
			description = "update files"  # Default
			if string_files:
				if len(string_files) == 1:
					description = f"update {string_files[0]}"
				else:
					try:
						common_dir = os.path.commonpath(string_files)
						# Make common_dir relative to repo root if possible
						try:
							common_dir_rel = os.path.relpath(common_dir, self.repo_root)
							if common_dir_rel and common_dir_rel != ".":
								description = f"update files in {common_dir_rel}"
							else:
								description = f"update {len(string_files)} files"
						except ValueError:  # Happens if paths are on different drives (unlikely in repo)
							description = f"update {len(string_files)} files"

					except (ValueError, TypeError):  # commonpath fails on empty list or mixed types
						description = f"update {len(string_files)} files"

		message = f"{commit_type}: {description}"
		logger.debug("Generated fallback message: %s", message)
		return message

	def generate_message(self, chunk: DiffChunk) -> tuple[CommitMessageSchema, bool]:
		"""
		Generate a commit message for a diff chunk.

		Args:
		    chunk: Diff chunk to generate message for

		Returns:
		    Generated message and success flag

		"""
		# Prepare prompt with chunk data
		prompt = self._prepare_prompt(chunk)
		logger.debug("Prompt prepared successfully")

		# Generate message using configured LLM provider
		message = self.client.completion(
			messages=[
				{"role": "system", "content": COMMIT_SYSTEM_PROMPT},
				{"role": "user", "content": prompt},
			],
			pydantic_model=CommitMessageSchema,
		)
		logger.debug("LLM generated message: %s", message)

		if isinstance(message, str):
			msg = "LLM generated message is not a BaseModel"
			logger.error(msg)
			raise TypeError(msg)

		return message, True

	def generate_message_with_linting(
		self, chunk: DiffChunk, retry_count: int = 1, max_retries: int = 3
	) -> tuple[str, bool, bool, bool, list[str]]:
		"""
		Generate a commit message with linting verification.

		Args:
		        chunk: The DiffChunk to generate a message for
		        retry_count: Current retry count (default: 1)
		        max_retries: Maximum number of retries for linting (default: 3)

		Returns:
		        Tuple of (message, used_llm, passed_validation, is_formatting_error, error_messages)
		        - message: Generated message, or original raw content if CommitFormatting failed.
		        - used_llm: Whether LLM was used.
		        - passed_validation: True if both CommitFormatting and linting passed.
		        - is_formatting_error: True if CommitFormatting failed.
		        - error_messages: List of lint or CommitFormatting error messages.

		"""
		# First, generate the initial message
		initial_lint_messages: list[str] = []  # Store initial messages
		message = ""  # Initialize message
		used_llm = False  # Initialize used_llm

		try:
			# --- Initial Generation ---
			commit_obj, used_llm = self.generate_message(chunk)
			logger.debug("Generated initial raw message: %s", commit_obj)

			# --- Format Commit ---
			# This is where CommitFormattingError can occur
			message = format_commit(commit_obj, self._config_loader)
			logger.debug("Formatted initial message: %s", message)

			# --- Clean and Lint ---
			message = clean_message_for_linting(message)
			logger.debug("Cleaned initial message: %s", message)

			is_valid, error_message = lint_commit_message(message, config_loader=self._config_loader)
			initial_lint_messages = [error_message] if error_message is not None else []
			logger.debug("Initial lint result: valid=%s, messages=%s", is_valid, initial_lint_messages)

			if is_valid or retry_count >= max_retries:
				# Return empty list if valid, or initial messages if max retries reached
				# passed_validation is True only if is_valid is True
				# is_json_error is False here
				return message, used_llm, is_valid, False, [] if is_valid else initial_lint_messages

			# --- Regeneration on Lint Failure ---
			logger.info("Regenerating message due to lint failure (attempt %d/%d)", retry_count, max_retries)

			try:
				# Prepare the enhanced prompt for regeneration
				lint_template = get_lint_prompt_template()
				enhanced_prompt = prepare_lint_prompt(
					template=lint_template,
					file_info=self.extract_file_info(chunk),
					config_loader=self._config_loader,
					lint_messages=initial_lint_messages,  # Use initial messages for feedback
					original_message=message,  # Pass the original formatted message that failed linting
				)

				# Generate message with the enhanced prompt
				regenerated_raw_message = self.client.completion(
					messages=[
						{"role": "system", "content": COMMIT_SYSTEM_PROMPT},
						{"role": "user", "content": enhanced_prompt},
					],
					pydantic_model=CommitMessageSchema,
				)
				logger.debug("Regenerated message (RAW LLM output): %s", regenerated_raw_message)
				if isinstance(regenerated_raw_message, str):
					msg = "Regenerated message is not a BaseModel"
					logger.error(msg)
					raise TypeError(msg)

				# --- Format Commit (Regeneration) ---
				# This can also raise JSONFormattingError
				regenerated_message = format_commit(regenerated_raw_message, self._config_loader)
				logger.debug("Formatted regenerated message: %s", regenerated_message)

				# --- Clean and Lint (Regeneration) ---
				cleaned_message = clean_message_for_linting(regenerated_message)
				logger.debug("Cleaned regenerated message: %s", cleaned_message)

				final_is_valid, error_message = lint_commit_message(cleaned_message, config_loader=self._config_loader)
				final_lint_messages = [error_message] if error_message is not None else []
				logger.debug("Regenerated lint result: valid=%s, messages=%s", final_is_valid, final_lint_messages)

				# Return final result and messages (empty if valid)
				# passed_validation is True only if final_is_valid is True
				# is_json_error is False here
				return cleaned_message, True, final_is_valid, False, [] if final_is_valid else final_lint_messages

			except CommitFormattingError:
				# Catch CommitFormattingError during REGENERATION
				logger.exception("Commit formatting failed during regeneration")
				raise
			except (ValueError, TypeError, KeyError, LLMError):
				# If regeneration itself fails (LLM call, prompt prep), log it
				# Return the ORIGINAL message and its lint errors
				logger.exception("Error during message regeneration attempt")
				raise

		except CommitFormattingError:
			# Catch CommitFormattingError during INITIAL formatting
			logger.exception("Initial commit formatting failed")
			raise
		except (ValueError, TypeError, KeyError, LLMError):
			# If initial generation or formatting (non-JSON error) fails completely
			logger.exception("Error during initial message generation/formatting")
			# Use a fallback (fallback doesn't lint, so passed_validation=True, is_json_error=False, empty messages)
			fallback_message = self.fallback_generation(chunk)
			return fallback_message, False, True, False, []
