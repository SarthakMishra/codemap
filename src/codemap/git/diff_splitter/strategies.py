"""Strategies for splitting git diffs into logical chunks."""

import dataclasses
import logging
import re
from io import StringIO
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, cast

from unidiff import Hunk, PatchedFile, PatchSet, UnidiffParseError

from codemap.git.semantic_grouping.embedder import DiffEmbedder
from codemap.git.utils import GitDiff

from .constants import (
	MAX_FILES_PER_GROUP,
)
from .schemas import DiffChunk
from .utils import (
	are_files_related,
	calculate_semantic_similarity,
	create_chunk_description,
	determine_commit_type,
	get_language_specific_patterns,
)

if TYPE_CHECKING:
	from codemap.config import ConfigLoader

logger = logging.getLogger(__name__)

# Constants for numeric comparisons
EXPECTED_TUPLE_SIZE = 2  # Expected size of extract_code_from_diff result


class BaseSplitStrategy:
	"""Base class for diff splitting strategies."""

	def __init__(self) -> None:
		"""Initialize with optional embedding model."""
		# Precompile regex patterns for better performance
		self._file_pattern = re.compile(r"diff --git a/.*? b/(.*?)\n")
		self._hunk_pattern = re.compile(r"@@ -\d+,\d+ \+\d+,\d+ @@")

	async def split(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Split the diff into chunks.

		Args:
		    diff: GitDiff object to split

		Returns:
		    List of DiffChunk objects

		"""
		msg = "Subclasses must implement this method"
		raise NotImplementedError(msg)


class FileSplitStrategy(BaseSplitStrategy):
	"""Strategy to split diffs by file."""

	async def split(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Split a diff into chunks by file.

		Args:
		    diff: GitDiff object to split

		Returns:
		    List of DiffChunk objects, one per file

		"""
		if not diff.content:
			return self._handle_empty_diff_content(diff)

		# Split the diff content by file
		file_chunks = self._file_pattern.split(diff.content)[1:]  # Skip first empty chunk

		# Group files with their content
		chunks = []
		for i in range(0, len(file_chunks), 2):
			if i + 1 >= len(file_chunks):
				break

			file_name = file_chunks[i]
			content = file_chunks[i + 1]

			if self._is_valid_filename(file_name) and content:
				diff_header = f"diff --git a/{file_name} b/{file_name}\n"
				chunks.append(
					DiffChunk(
						files=[file_name],
						content=diff_header + content,
						description=f"Changes in {file_name}",
					)
				)

		return chunks

	def _handle_empty_diff_content(self, diff: GitDiff) -> list[DiffChunk]:
		"""Handle untracked files in empty diff content."""
		if (not diff.is_staged or diff.is_untracked) and diff.files:
			# Filter out invalid file names
			valid_files = [file for file in diff.files if self._is_valid_filename(file)]
			return [DiffChunk(files=[f], content="", description=f"New file: {f}") for f in valid_files]
		return []

	@staticmethod
	def _is_valid_filename(filename: str) -> bool:
		"""Check if the filename is valid (not a pattern or template)."""
		if not filename:
			return False
		invalid_chars = ["*", "+", "{", "}", "\\"]
		return not (any(char in filename for char in invalid_chars) or filename.startswith('"'))


class SemanticSplitStrategy(BaseSplitStrategy):
	"""Strategy to split diffs semantically."""

	def __init__(
		self,
		config_loader: "ConfigLoader",
	) -> None:
		"""
		Initialize the SemanticSplitStrategy.

		Args:
		    config_loader: ConfigLoader instance.
		"""
		# Store thresholds and settings
		self.similarity_threshold = config_loader.get.commit.diff_splitter.similarity_threshold
		self.directory_similarity_threshold = config_loader.get.commit.diff_splitter.directory_similarity_threshold
		self.min_chunks_for_consolidation = config_loader.get.commit.diff_splitter.min_chunks_for_consolidation
		self.max_chunks_before_consolidation = config_loader.get.commit.diff_splitter.max_chunks_before_consolidation
		self.max_file_size_for_llm = config_loader.get.commit.diff_splitter.max_file_size_for_llm
		self.file_move_similarity_threshold = config_loader.get.commit.diff_splitter.file_move_similarity_threshold

		# Set up file extensions, defaulting to config if None is passed
		self.code_extensions = config_loader.get.commit.diff_splitter.default_code_extensions
		# Initialize patterns for related files
		self.related_file_patterns = self._initialize_related_file_patterns()
		self.config_loader = config_loader

		# Create DiffEmbedder instance for embedding operations
		self.embedder = DiffEmbedder(config_loader=config_loader)

	async def split(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Split a diff into chunks based on semantic relationships.

		Args:
		    diff: GitDiff object to split

		Returns:
		    List of DiffChunk objects based on semantic analysis

		"""
		if not diff.files:
			logger.debug("No files to process")
			return []

		# Initialize an empty list to store all chunks
		all_chunks = []

		# Detect moved files
		moved_files = await self._detect_moved_files(diff)
		if moved_files:
			logger.info("Detected %d moved files", len(moved_files))
			move_chunks = self._create_move_chunks(moved_files, diff)
			if move_chunks:
				# Add move chunks to all_chunks rather than returning immediately
				all_chunks.extend(move_chunks)

				# Create a set of files that are part of moves to avoid processing them again
				moved_file_paths = set()
				for chunk in move_chunks:
					moved_file_paths.update(chunk.files)

				# Filter out moved files from diff.files to avoid double processing
				non_moved_files = [f for f in diff.files if f not in moved_file_paths]

				# If all files were moves, return just the move chunks
				if not non_moved_files:
					return all_chunks

				# Update diff.files to only include non-moved files
				diff.files = non_moved_files
				logger.info("Continuing with %d non-moved files", len(non_moved_files))

		# Handle files in manageable groups
		if len(diff.files) > MAX_FILES_PER_GROUP:
			logger.info("Processing large number of files (%d) in smaller groups", len(diff.files))

			# Group files by directory to increase likelihood of related files being processed together
			files_by_dir = {}
			for file in diff.files:
				dir_path = str(Path(file).parent)
				if dir_path not in files_by_dir:
					files_by_dir[dir_path] = []
				files_by_dir[dir_path].append(file)

			# Process each directory group separately, keeping chunks under 5 files
			# Iterate directly over the file lists since the directory path isn't used here
			for files in files_by_dir.values():
				# Process files in this directory in batches of 3-5
				for i in range(0, len(files), 3):
					batch = files[i : i + 3]
					# Create a new GitDiff for the batch, ensuring content is passed
					batch_diff = GitDiff(
						files=batch,
						content=diff.content,  # Pass the original full diff content
						is_staged=diff.is_staged,
					)
					chunks = await self._process_group(batch_diff)
					all_chunks.extend(chunks)

			return all_chunks

		# For smaller groups, process normally and add to all_chunks
		regular_chunks = await self._process_group(diff)
		all_chunks.extend(regular_chunks)

		return all_chunks

	async def _process_group(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Process a GitDiff with one or more files.

		Originally designed for single files, but now supports multiple files.

		"""
		if not diff.files:
			logger.warning("_process_group called with empty files list")
			return []

		# If multiple files, this used to log an error, but now we'll handle it properly
		if len(diff.files) > 1:
			logger.debug("Processing group with multiple files: %s", diff.files)

			# Extract content for each file individually if possible
			chunks = []
			for file_path in diff.files:
				# Try to extract just this file's diff from the full content
				file_diff_content = self._extract_file_diff(diff.content, file_path)

				if file_diff_content:
					# Create a new diff for just this file
					file_diff = GitDiff(files=[file_path], content=file_diff_content, is_staged=diff.is_staged)
					# Process it and add the resulting chunks
					enhanced_chunks = await self._enhance_semantic_split(file_diff)
					chunks.extend(enhanced_chunks)
				else:
					# If we couldn't extract just this file's diff, create a simple chunk
					chunks.append(
						DiffChunk(
							files=[file_path],
							content="",  # Empty content as we couldn't extract it
							description=f"Changes in {file_path}",
						)
					)

			# If we couldn't create any valid chunks, fallback to the original behavior
			if not chunks:
				return [DiffChunk(files=diff.files, content=diff.content, description="Multiple file changes")]

			return chunks

		# Original behavior for single file
		file_path = diff.files[0]

		# Enhance this single file diff
		enhanced_chunks = await self._enhance_semantic_split(diff)  # Pass the original diff directly

		if not enhanced_chunks:
			logger.warning("No chunk generated for file: %s after enhancement.", file_path)
			# Fallback if enhancement yields nothing
			enhanced_chunks = [
				DiffChunk(
					files=[file_path],
					content=diff.content,
					description=f"Changes in {file_path} (enhancement failed)",
				)
			]

		# No further consolidation or grouping needed here as we process file-by-file now
		return enhanced_chunks

	def _extract_file_diff(self, full_diff_content: str, file_path: str) -> str:
		"""
		Extract the diff content for a specific file from a multi-file diff.

		Args:
		        full_diff_content: Complete diff content with multiple files
		        file_path: Path of the file to extract

		Returns:
		        The extracted diff for the specific file, or empty string if not found

		"""
		import re

		# Pattern to match the start of a diff for a file
		diff_start_pattern = re.compile(r"diff --git a/([^\s]+) b/([^\s]+)")

		# Find all diff start positions
		diff_positions = []
		for match in diff_start_pattern.finditer(full_diff_content):
			_, b_file = match.groups()
			# For most changes both files are the same; for renames prefer b_file
			target_file = b_file
			diff_positions.append((match.start(), target_file))

		# Sort by position
		diff_positions.sort()

		# Find the diff for our file
		file_diff = ""
		for i, (start_pos, diff_file) in enumerate(diff_positions):
			if diff_file == file_path:
				# Found our file, now find the end
				if i < len(diff_positions) - 1:
					end_pos = diff_positions[i + 1][0]
					file_diff = full_diff_content[start_pos:end_pos]
				else:
					# Last file in the diff
					file_diff = full_diff_content[start_pos:]
				break

		return file_diff

	def _group_chunks_by_directory(self, chunks: list[DiffChunk]) -> dict[str, list[DiffChunk]]:
		"""Group chunks by their containing directory."""
		dir_groups: dict[str, list[DiffChunk]] = {}

		for chunk in chunks:
			if not chunk.files:
				continue

			file_path = chunk.files[0]
			dir_path = file_path.rsplit("/", 1)[0] if "/" in file_path else "root"

			if dir_path not in dir_groups:
				dir_groups[dir_path] = []

			dir_groups[dir_path].append(chunk)

		return dir_groups

	async def _process_directory_group(
		self, chunks: list[DiffChunk], processed_files: set[str], semantic_chunks: list[DiffChunk]
	) -> None:
		"""Process chunks in a single directory group."""
		if len(chunks) == 1:
			# If only one file in directory, add it directly
			semantic_chunks.append(chunks[0])
			if chunks[0].files:
				processed_files.update(chunks[0].files)
		else:
			# For directories with multiple files, try to group them
			dir_processed: set[str] = set()

			# First try to group by related file patterns
			await self._group_related_files(chunks, dir_processed, semantic_chunks)

			# Then try to group remaining files by content similarity
			remaining_chunks = [c for c in chunks if not c.files or c.files[0] not in dir_processed]

			if remaining_chunks:
				# Use default similarity threshold instead
				await self._group_by_content_similarity(remaining_chunks, semantic_chunks)

			# Add all processed files to the global processed set
			processed_files.update(dir_processed)

	async def _process_remaining_chunks(
		self, all_chunks: list[DiffChunk], processed_files: set[str], semantic_chunks: list[DiffChunk]
	) -> None:
		"""Process any remaining chunks that weren't grouped by directory."""
		remaining_chunks = [c for c in all_chunks if c.files and c.files[0] not in processed_files]

		if remaining_chunks:
			await self._group_by_content_similarity(remaining_chunks, semantic_chunks)

	async def _consolidate_small_chunks(self, initial_chunks: list[DiffChunk]) -> list[DiffChunk]:
		"""
		Merge small or related chunks together.

		First, consolidates chunks originating from the same file.
		Then, consolidates remaining single-file chunks by directory.

		Args:
		    initial_chunks: List of diff chunks to consolidate

		Returns:
		    Consolidated list of chunks

		"""
		# Use instance variable for threshold
		if len(initial_chunks) < self.min_chunks_for_consolidation:
			return initial_chunks

		# Consolidate small chunks for the same file or related files
		consolidated_chunks = []
		processed_indices = set()

		for i, chunk1 in enumerate(initial_chunks):
			if i in processed_indices:
				continue

			merged_chunk = chunk1
			processed_indices.add(i)

			# Check subsequent chunks for merging
			for j in range(i + 1, len(initial_chunks)):
				if j in processed_indices:
					continue

				chunk2 = initial_chunks[j]

				# Check if chunks should be merged (same file or related)
				if self._should_merge_chunks(merged_chunk, chunk2):
					# Combine files if merging related chunks, not just same file chunks
					new_files = merged_chunk.files
					if (
						len(merged_chunk.files) == 1
						and len(chunk2.files) == 1
						and merged_chunk.files[0] != chunk2.files[0]
					):
						new_files = sorted(set(merged_chunk.files + chunk2.files))

					# Merge content and potentially other attributes
					# Ensure a newline between merged content if needed
					separator = "\n" if merged_chunk.content and chunk2.content else ""
					merged_chunk = dataclasses.replace(
						merged_chunk,
						files=new_files,
						content=merged_chunk.content + separator + chunk2.content,
						description=merged_chunk.description,  # Keep first description
					)
					processed_indices.add(j)

			consolidated_chunks.append(merged_chunk)

		return consolidated_chunks

	async def _consolidate_if_needed(self, semantic_chunks: list[DiffChunk]) -> list[DiffChunk]:
		"""Consolidate chunks if we have too many small ones."""
		has_single_file_chunks = any(len(chunk.files) == 1 for chunk in semantic_chunks)

		if len(semantic_chunks) > self.max_chunks_before_consolidation and has_single_file_chunks:
			return await self._consolidate_small_chunks(semantic_chunks)

		return semantic_chunks

	@staticmethod
	def _initialize_related_file_patterns() -> list[tuple[Pattern, Pattern]]:
		"""
		Initialize and compile regex patterns for related files.

		Returns:
		    List of compiled regex pattern pairs

		"""
		# Pre-compile regex for efficiency and validation
		related_file_patterns = []
		# Define patterns using standard strings with escaped backreferences
		default_patterns: list[tuple[str, str]] = [
			# --- File Moves (same name, different directories) ---
			# This helps identify potential file moves regardless of directory structure
			("^(.*/)?(.*?)$", "^(.*/)\\\\2$"),  # Same filename in any directory
			# --- General Code + Test Files ---
			# Python
			("^(.*)\\.py$", "\\\\1_test\\.py$"),
			("^(.*)\\.py$", "test_\\\\1\\.py$"),
			("^(.*)\\.(py)$", "\\\\1_test\\.\\\\2$"),  # For file.py and file_test.py pattern
			("^(.*)\\.(py)$", "\\\\1Test\\.\\\\2$"),  # For file.py and fileTest.py pattern
			("^(.*)\\.py$", "\\\\1_spec\\.py$"),
			("^(.*)\\.py$", "spec_\\\\1\\.py$"),
			# JavaScript / TypeScript (including JSX/TSX)
			("^(.*)\\.(js|jsx|ts|tsx)$", "\\\\1\\.(test|spec)\\.(js|jsx|ts|tsx)$"),
			("^(.*)\\.(js|jsx|ts|tsx)$", "\\\\1\\.stories\\.(js|jsx|ts|tsx)$"),  # Storybook
			("^(.*)\\.(js|ts)$", "\\\\1\\.d\\.ts$"),  # JS/TS + Declaration files
			# Ruby
			("^(.*)\\.rb$", "\\\\1_spec\\.rb$"),
			("^(.*)\\.rb$", "\\\\1_test\\.rb$"),
			("^(.*)\\.rb$", "spec/.*_spec\\.rb$"),  # Common RSpec structure
			# Java
			("^(.*)\\.java$", "\\\\1Test\\.java$"),
			("src/main/java/(.*)\\.java$", "src/test/java/\\\\1Test\\.java$"),  # Maven/Gradle structure
			# Go
			("^(.*)\\.go$", "\\\\1_test\\.go$"),
			# C#
			("^(.*)\\.cs$", "\\\\1Tests?\\.cs$"),
			# PHP
			("^(.*)\\.php$", "\\\\1Test\\.php$"),
			("^(.*)\\.php$", "\\\\1Spec\\.php$"),
			("src/(.*)\\.php$", "tests/\\\\1Test\\.php$"),  # Common structure
			# Rust
			("src/(lib|main)\\.rs$", "tests/.*\\.rs$"),  # Main/Lib and integration tests
			("src/(.*)\\.rs$", "src/\\\\1_test\\.rs$"),  # Inline tests (less common for grouping)
			# Swift
			("^(.*)\\.swift$", "\\\\1Tests?\\.swift$"),
			# Kotlin
			("^(.*)\\.kt$", "\\\\1Test\\.kt$"),
			("src/main/kotlin/(.*)\\.kt$", "src/test/kotlin/\\\\1Test\\.kt$"),  # Common structure
			# --- Frontend Component Bundles ---
			# JS/TS Components + Styles (CSS, SCSS, LESS, CSS Modules)
			("^(.*)\\.(js|jsx|ts|tsx)$", "\\\\1\\.(css|scss|less)$"),
			("^(.*)\\.(js|jsx|ts|tsx)$", "\\\\1\\.module\\.(css|scss|less)$"),
			("^(.*)\\.(js|jsx|ts|tsx)$", "\\\\1\\.styles?\\.(js|ts)$"),  # Styled Components / Emotion convention
			# Vue Components + Styles
			("^(.*)\\.vue$", "\\\\1\\.(css|scss|less)$"),
			("^(.*)\\.vue$", "\\\\1\\.module\\.(css|scss|less)$"),
			# Svelte Components + Styles/Scripts
			("^(.*)\\.svelte$", "\\\\1\\.(css|scss|less)$"),
			("^(.*)\\.svelte$", "\\\\1\\.(js|ts)$"),
			# Angular Components (more specific structure)
			("^(.*)\\.component\\.ts$", "\\\\1\\.component\\.html$"),
			("^(.*)\\.component\\.ts$", "\\\\1\\.component\\.(css|scss|less)$"),
			("^(.*)\\.component\\.ts$", "\\\\1\\.component\\.spec\\.ts$"),  # Component + its test
			("^(.*)\\.service\\.ts$", "\\\\1\\.service\\.spec\\.ts$"),  # Service + its test
			("^(.*)\\.module\\.ts$", "\\\\1\\.routing\\.module\\.ts$"),  # Module + routing
			# --- Implementation / Definition / Generation ---
			# C / C++ / Objective-C
			("^(.*)\\.h$", "\\\\1\\.c$"),
			("^(.*)\\.h$", "\\\\1\\.m$"),
			("^(.*)\\.hpp$", "\\\\1\\.cpp$"),
			("^(.*)\\.h$", "\\\\1\\.cpp$"),  # Allow .h with .cpp
			("^(.*)\\.h$", "\\\\1\\.mm$"),
			# Protocol Buffers / gRPC
			("^(.*)\\.proto$", "\\\\1\\.pb\\.(go|py|js|java|rb|cs|ts)$"),
			("^(.*)\\.proto$", "\\\\1_pb2?\\.py$"),  # Python specific proto generation
			("^(.*)\\.proto$", "\\\\1_grpc\\.pb\\.(go|js|ts)$"),  # gRPC specific
			# Interface Definition Languages (IDL)
			("^(.*)\\.idl$", "\\\\1\\.(h|cpp|cs|java)$"),
			# API Specifications (OpenAPI/Swagger)
			("(openapi|swagger)\\.(yaml|yml|json)$", ".*\\.(go|py|js|java|rb|cs|ts)$"),  # Spec + generated code
			("^(.*)\\.(yaml|yml|json)$", "\\\\1\\.generated\\.(go|py|js|java|rb|cs|ts)$"),  # Another convention
			# --- Web Development (HTML Centric) ---
			("^(.*)\\.html$", "\\\\1\\.(js|ts)$"),
			("^(.*)\\.html$", "\\\\1\\.(css|scss|less)$"),
			# --- Mobile Development ---
			# iOS (Swift)
			("^(.*)\\.swift$", "\\\\1\\.storyboard$"),
			("^(.*)\\.swift$", "\\\\1\\.xib$"),
			# Android (Kotlin/Java)
			("^(.*)\\.(kt|java)$", "res/layout/.*\\.(xml)$"),  # Code + Layout XML (Path sensitive)
			("AndroidManifest\\.xml$", ".*\\.(kt|java)$"),  # Manifest + Code
			("build\\.gradle(\\.kts)?$", ".*\\.(kt|java)$"),  # Gradle build + Code
			# --- Configuration Files ---
			# Package Managers
			("package\\.json$", "(package-lock\\.json|yarn\\.lock|pnpm-lock\\.yaml)$"),
			("requirements\\.txt$", "(setup\\.py|setup\\.cfg|pyproject\\.toml)$"),
			("pyproject\\.toml$", "(setup\\.py|setup\\.cfg|poetry\\.lock|uv\\.lock)$"),
			("Gemfile$", "Gemfile\\.lock$"),
			("Cargo\\.toml$", "Cargo\\.lock$"),
			("composer\\.json$", "composer\\.lock$"),  # PHP Composer
			("go\\.mod$", "go\\.sum$"),  # Go Modules
			("pom\\.xml$", ".*\\.java$"),  # Maven + Java
			("build\\.gradle(\\.kts)?$", ".*\\.(java|kt)$"),  # Gradle + Java/Kotlin
			# Linters / Formatters / Compilers / Type Checkers
			(
				"package\\.json$",
				"(tsconfig\\.json|\\.eslintrc(\\..*)?|\\.prettierrc(\\..*)?|\\.babelrc(\\..*)?|webpack\\.config\\.js|vite\\.config\\.(js|ts))$",
			),
			("pyproject\\.toml$", "(\\.flake8|\\.pylintrc|\\.isort\\.cfg|mypy\\.ini)$"),
			# Docker
			("Dockerfile$", "(\\.dockerignore|docker-compose\\.yml)$"),
			("docker-compose\\.yml$", "\\.env$"),
			# CI/CD
			("\\.github/workflows/.*\\.yml$", ".*\\.(sh|py|js|ts|go)$"),  # Workflow + scripts
			("\\.gitlab-ci\\.yml$", ".*\\.(sh|py|js|ts|go)$"),
			("Jenkinsfile$", ".*\\.(groovy|sh|py)$"),
			# IaC (Terraform)
			("^(.*)\\.tf$", "\\\\1\\.tfvars$"),
			("^(.*)\\.tf$", "\\\\1\\.tf$"),  # Group TF files together
			# --- Documentation ---
			("README\\.md$", ".*$"),  # README often updated with any change
			("^(.*)\\.md$", "\\\\1\\.(py|js|ts|go|java|rb|rs|php|swift|kt)$"),  # Markdown doc + related code
			("docs/.*\\.md$", "src/.*$"),  # Documentation in docs/ related to src/
			# --- Data Science / ML ---
			("^(.*)\\.ipynb$", "\\\\1\\.py$"),  # Notebook + Python script
			("^(.*)\\.py$", "data/.*\\.(csv|json|parquet)$"),  # Script + Data file (path sensitive)
			# --- General Fallbacks (Use with caution) ---
			# Files with same base name but different extensions (already covered by some specifics)
			# ("^(.*)\\..*$", "\\1\\..*$"), # Potentially too broad, rely on specifics above
		]

		for pattern1_str, pattern2_str in default_patterns:
			try:
				# Compile with IGNORECASE for broader matching
				pattern1 = re.compile(pattern1_str, re.IGNORECASE)
				pattern2 = re.compile(pattern2_str, re.IGNORECASE)
				related_file_patterns.append((pattern1, pattern2))
			except re.error as e:
				# Log only if pattern compilation fails
				logger.warning(f"Failed to compile regex pair: ({pattern1_str!r}, {pattern2_str!r}). Error: {e}")

		return related_file_patterns

	# --- New Helper Methods for Refactoring _enhance_semantic_split ---

	def _parse_file_diff(self, diff_content: str, file_path: str) -> PatchedFile | None:
		"""Parse diff content to find the PatchedFile for a specific file path."""
		if not diff_content:
			logger.warning("Cannot parse empty diff content for %s", file_path)
			return None

		filtered_content = ""  # Initialize to handle unbound case
		try:
			# Filter out the truncation marker lines before parsing
			filtered_content_lines = [
				line for line in diff_content.splitlines() if line.strip() != "... [content truncated] ..."
			]
			filtered_content = "\n".join(filtered_content_lines)

			# Use StringIO as PatchSet expects a file-like object or iterable
			try:
				patch_set = PatchSet(StringIO(filtered_content))
			except UnidiffParseError as e:
				logger.warning("UnidiffParseError for %s: %s", file_path, str(e))
				# Try to extract just the diff for this specific file to avoid parsing the entire diff
				file_diff_content_raw = re.search(
					rf"diff --git a/.*? b/{re.escape(file_path)}\n(.*?)(?=diff --git a/|\Z)",
					diff_content,
					re.DOTALL | re.MULTILINE,
				)
				content_for_chunk = file_diff_content_raw.group(0) if file_diff_content_raw else ""
				if content_for_chunk:
					logger.debug("Extracted raw content for %s after parse error", file_path)
					# Create a manual PatchedFile since we can't parse it properly
					return None
				return None

			matched_file: PatchedFile | None = None
			for patched_file in patch_set:
				# unidiff paths usually start with a/ or b/
				if patched_file.target_file == f"b/{file_path}" or patched_file.path == file_path:
					matched_file = patched_file
					break
			if not matched_file:
				logger.warning("Could not find matching PatchedFile for: %s in unidiff output", file_path)
				return None
			return matched_file
		except UnidiffParseError:
			# Log the specific parse error and the content that caused it (first few lines)
			preview_lines = "\n".join(filtered_content.splitlines()[:10])  # Log first 10 lines
			logger.exception(
				"UnidiffParseError for %s\nContent Preview:\n%s",  # Corrected format string
				file_path,
				preview_lines,
			)
			return None  # Return None on parse error
		except Exception:
			logger.exception("Failed to parse diff content using unidiff for %s", file_path)
			return None

	def _reconstruct_file_diff(self, patched_file: PatchedFile) -> tuple[str, str]:
		"""Reconstruct the diff header and full diff content for a PatchedFile."""
		file_diff_hunks_content = "\n".join(str(hunk) for hunk in patched_file)
		file_header_obj = getattr(patched_file, "patch_info", None)
		file_header = str(file_header_obj) if file_header_obj else ""

		if not file_header.startswith("diff --git") and patched_file.source_file and patched_file.target_file:
			logger.debug("Reconstructing missing diff header for %s", patched_file.path)
			file_header = f"diff --git {patched_file.source_file} {patched_file.target_file}\n"
			if hasattr(patched_file, "index") and patched_file.index:
				file_header += f"index {patched_file.index}\n"
			# Use timestamps if available for more accurate header reconstruction
			source_ts = f"\t{patched_file.source_timestamp}" if patched_file.source_timestamp else ""
			target_ts = f"\t{patched_file.target_timestamp}" if patched_file.target_timestamp else ""
			file_header += f"--- {patched_file.source_file}{source_ts}\n"
			file_header += f"+++ {patched_file.target_file}{target_ts}\n"

		full_file_diff_content = file_header + file_diff_hunks_content
		return file_header, full_file_diff_content

	def _split_large_file_diff(self, patched_file: PatchedFile, file_header: str) -> list[DiffChunk]:
		"""Split a large file's diff by grouping hunks under the size limit."""
		file_path = patched_file.path
		max_chunk_size = self.max_file_size_for_llm  # Use instance config
		logger.info(
			"Splitting large file diff for %s by hunks (limit: %d bytes)",
			file_path,
			max_chunk_size,
		)
		large_file_chunks = []
		current_hunk_group: list[Hunk] = []
		current_group_size = len(file_header)  # Start with header size

		for hunk in patched_file:
			hunk_content_str = str(hunk)
			hunk_size = len(hunk_content_str) + 1  # +1 for newline separator

			# If adding this hunk exceeds the limit (and group isn't empty), finalize the current chunk
			if current_hunk_group and current_group_size + hunk_size > max_chunk_size:
				group_content = file_header + "\n".join(str(h) for h in current_hunk_group)
				description = f"Chunk {len(large_file_chunks) + 1} of large file {file_path}"
				large_file_chunks.append(DiffChunk(files=[file_path], content=group_content, description=description))
				# Start a new chunk with the current hunk
				current_hunk_group = [hunk]
				current_group_size = len(file_header) + hunk_size
			# Edge case: If a single hunk itself is too large, create a chunk just for it
			elif not current_hunk_group and len(file_header) + hunk_size > max_chunk_size:
				logger.warning(
					"Single hunk in %s exceeds size limit (%d bytes). Creating oversized chunk.",
					file_path,
					len(file_header) + hunk_size,
				)
				group_content = file_header + hunk_content_str
				description = f"Chunk {len(large_file_chunks) + 1} (oversized hunk) of large file {file_path}"
				large_file_chunks.append(DiffChunk(files=[file_path], content=group_content, description=description))
				# Reset for next potential chunk (don't carry this huge hunk forward)
				current_hunk_group = []
				current_group_size = len(file_header)
			else:
				# Add hunk to the current group
				current_hunk_group.append(hunk)
				current_group_size += hunk_size

		# Add the last remaining chunk group if any
		if current_hunk_group:
			group_content = file_header + "\n".join(str(h) for h in current_hunk_group)
			description = f"Chunk {len(large_file_chunks) + 1} of large file {file_path}"
			large_file_chunks.append(DiffChunk(files=[file_path], content=group_content, description=description))

		return large_file_chunks

	# --- Refactored Orchestrator Method ---

	async def _enhance_semantic_split(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Enhance the semantic split by using NLP and chunk detection.

		Args:
		    diff: The GitDiff object to split

		Returns:
		    List of enhanced DiffChunk objects

		"""
		if not diff.files:
			return []

		# Special handling for untracked files - avoid unidiff parsing errors
		if diff.is_untracked:
			# Create a basic chunk with only file info for untracked files
			# Use a list comprehension for performance (PERF401)
			return [
				DiffChunk(
					files=[file_path],
					content=diff.content if len(diff.files) == 1 else f"New untracked file: {file_path}",
					description=f"New file: {file_path}",
				)
				for file_path in diff.files
				if self._is_valid_filename(file_path)
			]

		if not diff.files or len(diff.files) != 1:
			logger.error("_enhance_semantic_split called with invalid diff object (files=%s)", diff.files)
			return []

		file_path = diff.files[0]
		extension = Path(file_path).suffix[1:].lower()

		if not diff.content:
			logger.warning("No diff content provided for %s, creating basic chunk.", file_path)
			return [DiffChunk(files=[file_path], content="", description=f"New file: {file_path}")]

		# 1. Parse the diff to get the PatchedFile object
		matched_file = self._parse_file_diff(diff.content, file_path)
		if not matched_file:
			# If parsing failed, return a basic chunk with raw content attempt
			file_diff_content_raw = re.search(
				rf"diff --git a/.*? b/{re.escape(file_path)}\n(.*?)(?=diff --git a/|\Z)",
				diff.content,
				re.DOTALL | re.MULTILINE,
			)
			content_for_chunk = file_diff_content_raw.group(0) if file_diff_content_raw else ""
			return [
				DiffChunk(
					files=[file_path],
					content=content_for_chunk,
					description=f"Changes in {file_path} (parsing failed)",
				)
			]

		# 2. Reconstruct the full diff content for this file
		_, full_file_diff_content = self._reconstruct_file_diff(matched_file)

		# 3. Check if the reconstructed diff is too large
		if len(full_file_diff_content) > self.max_file_size_for_llm:
			return self._split_large_file_diff(matched_file, "")

		# 4. Try splitting by semantic patterns (if applicable)
		patterns = get_language_specific_patterns(extension)
		if patterns:
			logger.debug("Attempting semantic pattern splitting for %s", file_path)
			pattern_chunks = self._split_by_semantic_patterns(matched_file, patterns)
			if pattern_chunks:
				return pattern_chunks
			logger.debug("Pattern splitting yielded no chunks for %s, falling back.", file_path)

		# 5. Fallback: Split by individual hunks
		logger.debug("Falling back to hunk splitting for %s", file_path)
		hunk_chunks = []
		for hunk in matched_file:
			hunk_content = str(hunk)
			hunk_chunks.append(
				DiffChunk(
					files=[file_path],
					content=hunk_content,  # Combine header + hunk
					description=f"Hunk in {file_path} starting near line {hunk.target_start}",
				)
			)

		# If no hunks were found at all, return the single reconstructed chunk
		if not hunk_chunks:
			logger.warning("No hunks detected for %s after parsing, returning full diff.", file_path)
			return [
				DiffChunk(
					files=[file_path],
					content=full_file_diff_content,
					description=f"Changes in {file_path} (no hunks detected)",
				)
			]

		return hunk_chunks

	# --- Existing Helper Methods (Potentially need review/updates) ---

	async def _group_by_content_similarity(
		self,
		chunks: list[DiffChunk],
		result_chunks: list[DiffChunk],
		similarity_threshold: float | None = None,
	) -> None:
		"""
		Group chunks by content similarity.

		Args:
		    chunks: List of chunks to process
		    result_chunks: List to append grouped chunks to (modified in place)
		    similarity_threshold: Optional custom threshold to override default

		"""
		if not chunks:
			return

		processed_indices = set()
		threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

		# Extract content from all chunks and get embeddings in batch
		chunk_contents = [chunk.content for chunk in chunks]
		chunk_embeddings = await self.embedder.embed_contents(chunk_contents)

		# For each chunk, find similar chunks and group them
		for i, chunk in enumerate(chunks):
			if i in processed_indices or not chunk_embeddings[i]:
				continue

			related_chunks = [chunk]
			processed_indices.add(i)

			# Find similar chunks
			for j, other_chunk in enumerate(chunks):
				if i == j or j in processed_indices or not chunk_embeddings[j]:
					continue

				# Calculate similarity between chunks using embeddings
				# Cast to remove None type since we've checked above
				emb1 = cast("list[float]", chunk_embeddings[i])
				emb2 = cast("list[float]", chunk_embeddings[j])
				similarity = calculate_semantic_similarity(emb1, emb2)

				if similarity >= threshold:
					related_chunks.append(other_chunk)
					processed_indices.add(j)

			# Create a semantic chunk from related chunks
			if related_chunks:
				self._create_semantic_chunk(related_chunks, result_chunks)

	async def _group_related_files(
		self,
		file_chunks: list[DiffChunk],
		processed_files: set[str],
		semantic_chunks: list[DiffChunk],
	) -> None:
		"""
		Group related files into semantic chunks.

		Args:
		    file_chunks: List of file-based chunks
		    processed_files: Set of already processed files (modified in place)
		    semantic_chunks: List of semantic chunks (modified in place)

		"""
		if not file_chunks:
			return

		# Group clearly related files
		for i, chunk in enumerate(file_chunks):
			if not chunk.files or chunk.files[0] in processed_files:
				continue

			related_chunks = [chunk]
			processed_files.add(chunk.files[0])

			# Find related files
			for j, other_chunk in enumerate(file_chunks):
				if i == j or not other_chunk.files or other_chunk.files[0] in processed_files:
					continue

				if are_files_related(chunk.files[0], other_chunk.files[0], self.related_file_patterns):
					related_chunks.append(other_chunk)
					processed_files.add(other_chunk.files[0])

			# Create a semantic chunk from related files
			if related_chunks:
				self._create_semantic_chunk(related_chunks, semantic_chunks)

	def _create_semantic_chunk(
		self,
		related_chunks: list[DiffChunk],
		semantic_chunks: list[DiffChunk],
	) -> None:
		"""
		Create a semantic chunk from related file chunks.

		Args:
		    related_chunks: List of related file chunks
		    semantic_chunks: List of semantic chunks to append to (modified in place)

		"""
		if not related_chunks:
			return

		all_files = []
		combined_content: list[str] = []
		is_move = any(getattr(chunk, "is_move", False) for chunk in related_chunks)

		for rc in related_chunks:
			all_files.extend(rc.files)
			combined_content.append(rc.content)

		# Determine if this is a move or a normal change
		if is_move:
			commit_type = "chore"  # For moves, we always use chore
			# Description will be handled separately for moves
			description = related_chunks[0].description if related_chunks else "Move files"
		else:
			# For normal changes, use the regular commit type detection
			commit_type = determine_commit_type(all_files)
			# Create description based on file count
			description = create_chunk_description(commit_type, all_files)

		# Join the content from all related chunks
		content = "\n\n".join(combined_content)

		semantic_chunks.append(
			DiffChunk(
				files=all_files,
				content=content,
				description=description,
				is_move=is_move,
			)
		)

	def _should_merge_chunks(self, chunk1: DiffChunk, chunk2: DiffChunk) -> bool:
		"""Determine if two chunks should be merged."""
		# Condition 1: Same single file
		same_file = len(chunk1.files) == 1 and chunk1.files == chunk2.files

		# Condition 2: Related single files
		related_files = (
			len(chunk1.files) == 1
			and len(chunk2.files) == 1
			and are_files_related(chunk1.files[0], chunk2.files[0], self.related_file_patterns)
		)

		# Return True if either condition is met
		return same_file or related_files

	def _split_by_semantic_patterns(self, patched_file: PatchedFile, patterns: list[str]) -> list[DiffChunk]:
		"""
		Split a PatchedFile's content by grouping hunks based on semantic patterns.

		This method groups consecutive hunks together until a hunk is encountered
		that contains an added line matching one of the semantic boundary patterns.
		It does *not* split within a single hunk, only between hunks where a boundary
		is detected in the *first* line of the subsequent hunk group.

		Args:
		    patched_file: The PatchedFile object from unidiff.
		    patterns: List of regex pattern strings to match as boundaries.

		Returns:
		    List of DiffChunk objects, potentially splitting the file into multiple chunks.

		"""
		compiled_patterns = [re.compile(p) for p in patterns]
		file_path = patched_file.path  # Or target_file? Need consistency

		final_chunks_data: list[list[Hunk]] = []
		current_semantic_chunk_hunks: list[Hunk] = []

		# Get header info once using the reconstruction helper
		_, _ = self._reconstruct_file_diff(patched_file)

		for hunk in patched_file:
			hunk_has_boundary = False
			for line in hunk:
				if line.is_added and any(pattern.match(line.value) for pattern in compiled_patterns):
					hunk_has_boundary = True
					break  # Found a boundary in this hunk

			# Start a new semantic chunk if the current hunk has a boundary
			# and we already have hunks accumulated.
			if hunk_has_boundary and current_semantic_chunk_hunks:
				final_chunks_data.append(current_semantic_chunk_hunks)
				current_semantic_chunk_hunks = [hunk]  # Start new chunk with this hunk
			else:
				# Append the current hunk to the ongoing semantic chunk
				current_semantic_chunk_hunks.append(hunk)

		# Add the last accumulated semantic chunk
		if current_semantic_chunk_hunks:
			final_chunks_data.append(current_semantic_chunk_hunks)

		# Convert grouped hunks into DiffChunk objects
		result_chunks: list[DiffChunk] = []
		for i, hunk_group in enumerate(final_chunks_data):
			if not hunk_group:
				continue
			# Combine content of all hunks in the group
			group_content = "\n".join(str(h) for h in hunk_group)
			# Generate description (could be more sophisticated)
			description = f"Semantic section {i + 1} in {file_path}"
			result_chunks.append(
				DiffChunk(
					files=[file_path],
					content=group_content,  # Combine header + hunks
					description=description,
				)
			)

		logger.debug("Split %s into %d chunks based on semantic patterns", file_path, len(result_chunks))
		return result_chunks

	@staticmethod
	def _is_valid_filename(filename: str) -> bool:
		"""Check if the filename is valid (not a pattern or template)."""
		if not filename:
			return False
		invalid_chars = ["*", "+", "{", "}", "\\"]
		return not (any(char in filename for char in invalid_chars) or filename.startswith('"'))

	async def _detect_moved_files(self, diff: GitDiff) -> dict[str, str]:
		"""
		Detect files that have been moved (deleted and added elsewhere).

		This analyzes the diff to find files that appear to have been deleted from
		one location and added to another location with similar content.

		Args:
		    diff: The git diff to analyze

		Returns:
		    Dictionary mapping from deleted file paths to their new locations
		"""
		# Parse the diff to identify deleted and added files with their content
		deleted_files: dict[str, str] = {}  # path -> content
		added_files: dict[str, str] = {}  # path -> content

		try:
			# Use PatchSet to parse the diff
			patch_set = PatchSet(StringIO(diff.content))

			# Identify deleted and added files
			for patched_file in patch_set:
				if patched_file.is_removed_file:
					# Extract content from the source (deleted) file
					file_path = patched_file.source_file.replace("a/", "", 1)
					file_content = ""
					for hunk in patched_file:
						for line in hunk:
							# Line type ' ' is context, '-' is removed
							if line.line_type in (" ", "-"):
								file_content += line.value
					deleted_files[file_path] = file_content
				elif patched_file.is_added_file:
					# Extract content from the target (added) file
					file_path = patched_file.target_file.replace("b/", "", 1)
					file_content = ""
					for hunk in patched_file:
						for line in hunk:
							# Line type ' ' is context, '+' is added
							if line.line_type in (" ", "+"):
								file_content += line.value
					added_files[file_path] = file_content

		except (ValueError, UnidiffParseError, Exception) as e:
			logger.warning(f"Failed to parse diff for move detection: {e}")
			return {}

		# Match deleted files with added files based on content similarity
		moved_files = {}

		# Group files with same name to avoid unnecessary embedding comparisons
		potential_moves = {}
		for deleted_path, deleted_content in deleted_files.items():
			if not deleted_content.strip():
				continue

			deleted_name = Path(deleted_path).name
			potential_moves.setdefault(deleted_name, {"deleted": [], "added": []})
			potential_moves[deleted_name]["deleted"].append((deleted_path, deleted_content))

		for added_path, added_content in added_files.items():
			if not added_content.strip():
				continue

			added_name = Path(added_path).name
			if added_name in potential_moves:  # Only add if there's a matching deleted file
				potential_moves[added_name]["added"].append((added_path, added_content))

		# Process each group of potential moves
		for group in potential_moves.values():
			deleted_items = group["deleted"]
			added_items = group["added"]

			if not deleted_items or not added_items:
				continue

			# Get embeddings for all deleted and added contents in batch
			all_contents = []
			for _, content in deleted_items + added_items:
				all_contents.append(content)
			all_embeddings = await self.embedder.embed_contents(all_contents)

			# Split embeddings back to deleted and added
			deleted_count = len(deleted_items)
			deleted_embeddings = all_embeddings[:deleted_count]
			added_embeddings = all_embeddings[deleted_count:]

			# Match deleted and added files based on embedding similarity
			for i, (deleted_path, _) in enumerate(deleted_items):
				if deleted_embeddings[i] is None:
					continue

				best_match = None
				best_similarity = 0.0

				for j, (added_path, _) in enumerate(added_items):
					if added_embeddings[j] is None:
						continue

					# Cast to remove None type since we've checked above
					emb1 = cast("list[float]", deleted_embeddings[i])
					emb2 = cast("list[float]", added_embeddings[j])
					similarity = calculate_semantic_similarity(emb1, emb2)

					if similarity > best_similarity:
						best_similarity = similarity
						best_match = added_path

				# If we found a good match above the threshold, consider it a move
				if best_match and best_similarity >= self.file_move_similarity_threshold:
					moved_files[deleted_path] = best_match
					logger.debug(f"Detected move: {deleted_path} -> {best_match} (similarity: {best_similarity:.2f})")

		return moved_files

	def _create_move_chunks(self, moved_files: dict[str, str], diff: GitDiff) -> list[DiffChunk]:
		"""
		Create diff chunks for moved files.

		Args:
		    moved_files: Dictionary mapping from source (deleted) paths to target (added) paths
		    diff: Original diff containing the move changes

		Returns:
		    List of DiffChunk objects representing file moves
		"""
		if not moved_files:
			return []

		# Group moves by common source/target directories
		# This helps create logical commit groups for moves within the same directories
		dir_moves: dict[tuple[str, str], list[tuple[str, str]]] = {}

		for source, target in moved_files.items():
			source_dir = str(Path(source).parent)
			target_dir = str(Path(target).parent)
			dir_key = (source_dir, target_dir)

			if dir_key not in dir_moves:
				dir_moves[dir_key] = []

			dir_moves[dir_key].append((source, target))

		# Create chunks for each move group
		move_chunks = []

		for (source_dir, target_dir), moves in dir_moves.items():
			# Combine source and target files
			all_files = []
			for source, target in moves:
				all_files.extend([source, target])

			# Create descriptive move information
			if source_dir == target_dir:
				# Rename within same directory
				if len(moves) == 1:
					source, target = moves[0]
					description = f"chore: rename {Path(source).name} to {Path(target).name}"
				else:
					description = f"chore: rename {len(moves)} files in {source_dir}"
			else:
				# Move between directories
				source_dir_desc = "root directory" if source_dir in {".", ""} else source_dir

				target_dir_desc = "root directory" if target_dir in {".", ""} else target_dir

				if len(moves) == 1:
					description = f"chore: move {Path(moves[0][0]).name} from {source_dir_desc} to {target_dir_desc}"
				else:
					description = f"chore: move {len(moves)} files from {source_dir_desc} to {target_dir_desc}"

			# Extract all content related to these moves from the original diff
			chunk_content = ""
			try:
				patch_set = PatchSet(StringIO(diff.content))
				for patched_file in patch_set:
					source_file = patched_file.source_file.replace("a/", "", 1)
					target_file = patched_file.target_file.replace("b/", "", 1)

					# Check if this patched file is part of our move group
					is_move_source = source_file in [source for source, _ in moves]
					is_move_target = target_file in [target for _, target in moves]

					if is_move_source or is_move_target:
						# Reconstruct this file's diff content
						_, file_content = self._reconstruct_file_diff(patched_file)
						chunk_content += file_content + "\n"

			except Exception:
				logger.exception("Error extracting content for move chunk")
				# Use a placeholder if extraction failed
				chunk_content = f"# File moves between {source_dir} and {target_dir}:\n"
				for source, target in moves:
					chunk_content += f"# - {source} -> {target}\n"

			# Create a single chunk for this move group
			move_chunks.append(
				DiffChunk(
					files=all_files,
					content=chunk_content,
					description=description,
					is_move=True,  # Indicate this is a move operation
				)
			)

		return move_chunks
