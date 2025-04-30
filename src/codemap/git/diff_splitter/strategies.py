"""Strategies for splitting git diffs into logical chunks."""

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from re import Pattern
from typing import Any, Protocol

import numpy as np

from codemap.git.utils import GitDiff

from .constants import (
	DEFAULT_CODE_EXTENSIONS,
	DEFAULT_SIMILARITY_THRESHOLD,
	MAX_CHUNKS_BEFORE_CONSOLIDATION,
	MAX_FILE_SIZE_FOR_LLM,
	MAX_FILES_PER_GROUP,
	MIN_CHUNKS_FOR_CONSOLIDATION,
)
from .schemas import DiffChunk
from .utils import (
	are_files_related,
	calculate_semantic_similarity,
	create_chunk_description,
	determine_commit_type,
	extract_code_from_diff,
	get_language_specific_patterns,
	is_test_environment,
)

logger = logging.getLogger(__name__)

# Constants for numeric comparisons
EXPECTED_TUPLE_SIZE = 2  # Expected size of extract_code_from_diff result


class EmbeddingModel(Protocol):
	"""Protocol for embedding models."""

	def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:  # noqa: ANN401
		"""Encode texts into embeddings."""
		...


class BaseSplitStrategy:
	"""Base class for diff splitting strategies."""

	def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
		"""Initialize with optional embedding model."""
		self._embedding_model = embedding_model
		# Precompile regex patterns for better performance
		self._file_pattern = re.compile(r"diff --git a/.*? b/(.*?)\n")
		self._hunk_pattern = re.compile(r"@@ -\d+,\d+ \+\d+,\d+ @@")

	def split(self, diff: GitDiff) -> list[DiffChunk]:
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

	def split(self, diff: GitDiff) -> list[DiffChunk]:
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
		if not diff.is_staged and diff.files:
			# Filter out invalid file names
			valid_files = [file for file in diff.files if self._is_valid_filename(file)]
			return [DiffChunk(files=[f], content="") for f in valid_files]
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
		embedding_model: EmbeddingModel | None = None,
		code_extensions: set[str] | None = None,
		related_file_patterns: list[tuple[Pattern, Pattern]] | None = None,
	) -> None:
		"""
		Initialize the SemanticSplitStrategy.

		Args:
		    embedding_model: Optional embedding model instance
		    code_extensions: Optional set of code file extensions
		    related_file_patterns: Optional list of related file patterns

		"""
		super().__init__(embedding_model)
		# Set up file extensions that are likely to contain code
		self.code_extensions = code_extensions or DEFAULT_CODE_EXTENSIONS
		# Initialize patterns for related files
		self.related_file_patterns = related_file_patterns or self._initialize_related_file_patterns()

	def split(self, diff: GitDiff) -> list[DiffChunk]:
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

		# Validate embedding model is available
		self._validate_embedding_model()

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
			all_chunks = []
			# Iterate directly over the file lists since the directory path isn't used here
			for files in files_by_dir.values():
				# Process files in this directory in batches of 3-5
				for i in range(0, len(files), 3):
					batch = files[i : i + 3]
					batch_diff = GitDiff(
						files=batch,
						content=diff.content,  # Keep same content
						is_staged=diff.is_staged,
					)
					all_chunks.extend(self._process_group(batch_diff))

			return all_chunks

		# For smaller groups, process normally
		return self._process_group(diff)

	def _process_group(self, diff: GitDiff) -> list[DiffChunk]:
		"""Process a manageable group of files."""
		if not diff.files:
			return []

		# 1. Create one chunk per file initially
		initial_file_chunks: list[DiffChunk] = []
		for file_path in diff.files:
			file_diff = GitDiff(
				files=[file_path],
				content=diff.content,
				is_staged=diff.is_staged,
			)
			enhanced_chunks = self._enhance_semantic_split(file_diff)
			if enhanced_chunks:
				initial_file_chunks.extend(enhanced_chunks)
			else:
				logger.warning("No chunk generated for file: %s", file_path)

		if not initial_file_chunks:
			return []

		# 2. Consolidate chunks from the same file (though step 1 should make this rare)
		#    and potentially by directory if that logic is re-enabled later.
		consolidated_chunks = self._consolidate_small_chunks(initial_file_chunks)

		# 3. Group remaining chunks by relatedness and similarity
		processed_files: set[str] = set()
		final_semantic_chunks: list[DiffChunk] = []
		self._group_related_files(consolidated_chunks, processed_files, final_semantic_chunks)
		self._process_remaining_chunks(consolidated_chunks, processed_files, final_semantic_chunks)

		# 4. Final consolidation check
		return self._consolidate_if_needed(final_semantic_chunks)

	def _validate_embedding_model(self) -> None:
		"""Validate that the embedding model is available."""
		if self._embedding_model is None and not is_test_environment():
			msg = (
				"Semantic analysis unavailable: embedding model not available. "
				"Make sure the model is properly loaded before calling this method."
			)
			raise ValueError(msg)

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

	def _process_directory_group(
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
			self._group_related_files(chunks, dir_processed, semantic_chunks)

			# Then try to group remaining files by content similarity
			remaining_chunks = [c for c in chunks if not c.files or c.files[0] not in dir_processed]

			if remaining_chunks:
				# Use default similarity threshold instead
				self._group_by_content_similarity(remaining_chunks, semantic_chunks)

			# Add all processed files to the global processed set
			processed_files.update(dir_processed)

	def _process_remaining_chunks(
		self, all_chunks: list[DiffChunk], processed_files: set[str], semantic_chunks: list[DiffChunk]
	) -> None:
		"""Process any remaining chunks that weren't grouped by directory."""
		remaining_chunks = [c for c in all_chunks if c.files and c.files[0] not in processed_files]

		if remaining_chunks:
			self._group_by_content_similarity(remaining_chunks, semantic_chunks)

	def _consolidate_if_needed(self, semantic_chunks: list[DiffChunk]) -> list[DiffChunk]:
		"""Consolidate chunks if we have too many small ones."""
		has_single_file_chunks = any(len(chunk.files) == 1 for chunk in semantic_chunks)

		if len(semantic_chunks) > MAX_CHUNKS_BEFORE_CONSOLIDATION and has_single_file_chunks:
			return self._consolidate_small_chunks(semantic_chunks)

		return semantic_chunks

	@staticmethod
	def _initialize_related_file_patterns() -> list[tuple[Pattern, Pattern]]:
		"""
		Initialize and compile regex patterns for related files.

		Returns:
		    List of compiled regex pattern pairs

		"""
		patterns = [
			# Frontend component pairs
			(r".*\.jsx?$", r".*\.css$"),
			(r".*\.tsx?$", r".*\.css$"),
			(r".*\.vue$", r".*\.css$"),
			(r".*\.jsx?$", r".*\.scss$"),
			(r".*\.tsx?$", r".*\.scss$"),
			(r".*\.vue$", r".*\.scss$"),
			(r".*\.jsx?$", r".*\.less$"),
			(r".*\.tsx?$", r".*\.less$"),
			# React component pairs
			(r".*\.jsx$", r".*\.jsx$"),
			(r".*\.tsx$", r".*\.tsx$"),
			(r".*Component\.jsx?$", r".*Container\.jsx?$"),
			(r".*Component\.tsx?$", r".*Container\.tsx?$"),
			# Implementation and definition pairs
			(r".*\.h$", r".*\.c$"),
			(r".*\.hpp$", r".*\.cpp$"),
			(r".*\.h$", r".*\.m$"),
			(r".*\.h$", r".*\.mm$"),
			(r".*\.proto$", r".*\.pb\.(go|py|js|java|rb|cs)$"),
			(r".*\.idl$", r".*\.(h|cpp|cs|java)$"),
			# Web development pairs
			(r".*\.html$", r".*\.js$"),
			(r".*\.html$", r".*\.css$"),
			(r".*\.html$", r".*\.scss$"),
			(r".*\.html$", r".*\.ts$"),
			# Python related files
			(r".*\.py$", r".*_test\.py$"),
			(r".*\.py$", r"test_.*\.py$"),
			(r".*\.py$", r".*_spec\.py$"),
			# JavaScript/TypeScript related files
			(r".*\.js$", r".*\.test\.js$"),
			(r".*\.js$", r".*\.spec\.js$"),
			(r".*\.ts$", r".*\.test\.ts$"),
			(r".*\.ts$", r".*\.spec\.ts$"),
			# Ruby related files
			(r".*\.rb$", r".*_spec\.rb$"),
			(r".*\.rb$", r".*_test\.rb$"),
			# Java related files
			(r".*\.java$", r".*Test\.java$"),
			# Go related files
			(r".*\.go$", r".*_test\.go$"),
			# Configuration files
			(r"package\.json$", r"package-lock\.json$"),
			(r"package\.json$", r"yarn\.lock$"),
			(r"package\.json$", r"tsconfig\.json$"),
			(r"package\.json$", r"\.eslintrc(\.js|\.json|\.yml)?$"),
			(r"package\.json$", r"\.prettierrc(\.js|\.json|\.yml)?$"),
			(r"requirements\.txt$", r"setup\.py$"),
			(r"pyproject\.toml$", r"setup\.py$"),
			(r"pyproject\.toml$", r"setup\.cfg$"),
			(r"Gemfile$", r"Gemfile\.lock$"),
			(r"Cargo\.toml$", r"Cargo\.lock$"),
			# Documentation
			(r".*\.md$", r".*\.(js|ts|py|rb|java|go|c|cpp|h|hpp)$"),
			(r"README\.md$", r".*$"),
		]

		# Compile all patterns for better performance
		return [(re.compile(p1), re.compile(p2)) for p1, p2 in patterns]

	def _get_code_embedding(self, content: str) -> list[float] | None:
		"""
		Get embedding vector for code content.

		Args:
		    content: Code content to embed

		Returns:
		    List of floats representing code embedding or None if unavailable

		"""
		# Skip empty content
		if not content or not content.strip():
			return None

		# Check if embedding model exists
		if self._embedding_model is None:
			logger.warning("Embedding model is None, cannot generate embedding")
			return None

		# Generate embedding
		try:
			embeddings = self._embedding_model.encode([content], show_progress_bar=False)
			return embeddings[0].tolist()
		except (ValueError, TypeError, RuntimeError, IndexError):
			logger.exception("Failed to generate embedding")
			return None

	def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
		"""
		Calculate semantic similarity between two code chunks.

		Args:
		    content1: First code content
		    content2: Second code content

		Returns:
		    Similarity score between 0 and 1

		"""
		# Get embeddings
		emb1 = self._get_code_embedding(content1)
		emb2 = self._get_code_embedding(content2)

		if not emb1 or not emb2:
			return 0.0

		# Calculate cosine similarity using utility function
		return calculate_semantic_similarity(emb1, emb2)

	def _semantic_hunk_splitting(self, file_path: str, diff_content: str) -> list[str]:
		"""
		Split a diff hunk by semantic boundaries in the code.

		Args:
		    file_path: Path to the file
		    diff_content: Git diff content

		Returns:
		    List of semantically separated diff chunks

		"""
		# Extract file extension
		extension = Path(file_path).suffix.lstrip(".")

		# Get language-specific patterns
		patterns = get_language_specific_patterns(extension)

		if not patterns:
			# If no language patterns available, return the whole diff as one chunk
			return [diff_content]

		# Extract new code from diff
		extraction_result = extract_code_from_diff(diff_content)
		if not extraction_result or len(extraction_result) != EXPECTED_TUPLE_SIZE:
			return [diff_content]

		_, new_code = extraction_result

		if not new_code:
			return [diff_content]

		# Find all pattern matches
		boundaries = []
		for pattern in patterns:
			matches = list(re.finditer(pattern, new_code, re.MULTILINE))
			boundaries.extend(match.start() for match in matches)

		if not boundaries:
			return [diff_content]

		# Sort and deduplicate boundaries
		boundaries = sorted(set(boundaries))

		# Split the diff using line tracking
		return self._split_diff_at_boundaries(diff_content, boundaries)

	def _split_diff_at_boundaries(self, diff_content: str, boundaries: list[int]) -> list[str]:
		"""Split diff content at the given semantic boundaries."""
		lines = diff_content.splitlines()
		chunks = []
		current_chunk = []
		current_line_idx = 0
		line_positions = {}  # Maps line indices to positions in the unified code

		# First pass: build the line position mapping
		pos = 0
		temp_line_idx = 0
		for line in diff_content.splitlines(keepends=True):  # Keep line endings
			if line.startswith("@@"):
				hunk_match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
				if hunk_match:
					temp_line_idx = int(hunk_match.group(1)) - 1
			elif not line.startswith("-"):  # Additions or context lines
				line_positions[temp_line_idx] = pos
				temp_line_idx += 1
			pos += len(line)

		# Second pass: split at boundaries
		current_line_idx = 0
		for line in lines:
			current_chunk.append(line)

			# Reset line counter at hunk headers
			if line.startswith("@@"):
				hunk_match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
				if hunk_match:
					current_line_idx = int(hunk_match.group(1)) - 1
				continue

			# Track line number for content lines (skip removals)
			if not line.startswith("-"):
				if current_line_idx in line_positions and line_positions[current_line_idx] in boundaries:
					# Finish current chunk
					chunks.append("\n".join(current_chunk))
					current_chunk = []

				current_line_idx += 1

		# Add the final chunk if there's anything left
		if current_chunk:
			chunks.append("\n".join(current_chunk))

		return chunks if chunks else [diff_content]

	def _enhance_semantic_split(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Enhance semantic analysis with language-specific patterns.

		Args:
		    diff: GitDiff object to analyze

		Returns:
		    List of semantically grouped chunks

		"""
		if not diff.content or not diff.files:
			return []

		file_path = diff.files[0]
		file_suffix = Path(file_path).suffix

		# Check for large diff content and handle appropriately
		if len(diff.content) > MAX_FILE_SIZE_FOR_LLM:
			logger.warning(
				(
					"Diff content for %s is too large (%d bytes) for detailed "
					"semantic analysis. Creating a simplified chunk."
				),
				file_path,
				len(diff.content),
			)
			commit_type = determine_commit_type([file_path])
			# Create a simplified chunk instead of detailed semantic analysis
			return [
				DiffChunk(
					files=[file_path],
					content="// Large file content - truncated for API limits",
					description=create_chunk_description(commit_type, [file_path]),
				)
			]

		# If no file extension, create a single chunk
		if not file_suffix:
			return [
				DiffChunk(
					files=[file_path],
					content=diff.content,
					description=f"Changes in {file_path}",
				)
			]

		# Try to apply language-specific patterns
		patterns = get_language_specific_patterns(file_suffix.lstrip("."))

		if not patterns:
			# Create a single chunk for unsupported file types
			return [
				DiffChunk(
					files=[file_path],
					content=diff.content,
					description=f"Changes in {file_path}",
				)
			]

		# Try semantic splitting
		chunks = self._split_by_semantic_patterns(diff, file_path, patterns)

		return chunks or [
			DiffChunk(
				files=[file_path],
				content=diff.content,
				description=f"Changes in {file_path}",
			)
		]

	def _split_by_semantic_patterns(self, diff: GitDiff, file_path: str, patterns: list[str]) -> list[DiffChunk]:
		"""Split a diff by semantic patterns in the code."""
		# Extract code from diff
		extraction_result = extract_code_from_diff(diff.content)
		if not extraction_result or len(extraction_result) != EXPECTED_TUPLE_SIZE:
			return []

		_, new_code = extraction_result

		if not new_code:
			return []

		# Find semantic boundaries
		boundaries = []
		for pattern in patterns:
			pattern_boundaries = [m.start() for m in re.finditer(pattern, new_code, re.MULTILINE)]
			boundaries.extend(pattern_boundaries)

		if not boundaries:
			return []

		# Sort boundaries for splitting
		boundaries.sort()

		# Create chunks based on boundaries
		chunks = []
		prev_boundary = 0

		for i, boundary in enumerate(boundaries):
			if boundary <= prev_boundary:
				continue

			chunk_content = new_code[prev_boundary:boundary]
			if chunk_content.strip():
				chunks.append(
					DiffChunk(
						files=[file_path],
						content=chunk_content,
						description=f"Semantic section {i + 1} in {file_path}",
					)
				)
			prev_boundary = boundary

		# Add the last chunk
		if prev_boundary < len(new_code):
			final_content = new_code[prev_boundary:]
			if final_content.strip():
				chunks.append(
					DiffChunk(
						files=[file_path],
						content=final_content,
						description=f"Semantic section {len(boundaries) + 1} in {file_path}",
					)
				)

		return chunks

	def _group_by_content_similarity(
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

		# Check if model is available
		if self._embedding_model is None:
			logger.debug("Embedding model not available, using fallback grouping strategy")
			# If model is unavailable, try to group by file path patterns
			grouped_paths: dict[str, list[DiffChunk]] = {}

			# Group by common path prefixes
			for chunk in chunks:
				if not chunk.files:
					result_chunks.append(chunk)
					continue

				file_path = chunk.files[0]
				# Get directory or file prefix as the grouping key
				if "/" in file_path:
					# Use directory as key
					key = file_path.rsplit("/", 1)[0]
				else:
					# Use file prefix (before extension) as key
					key = file_path.split(".", 1)[0] if "." in file_path else file_path

				if key not in grouped_paths:
					grouped_paths[key] = []
				grouped_paths[key].append(chunk)

			# Create chunks from each group
			for related_chunks in grouped_paths.values():
				self._create_semantic_chunk(related_chunks, result_chunks)
			return

		processed_indices = set()
		threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_SIMILARITY_THRESHOLD

		# For each chunk, find similar chunks and group them
		for i, chunk in enumerate(chunks):
			if i in processed_indices:
				continue

			related_chunks = [chunk]
			processed_indices.add(i)

			# Find similar chunks
			for j, other_chunk in enumerate(chunks):
				if i == j or j in processed_indices:
					continue

				# Calculate similarity between chunks
				similarity = self._calculate_semantic_similarity(chunk.content, other_chunk.content)

				if similarity >= threshold:
					related_chunks.append(other_chunk)
					processed_indices.add(j)

			# Create a semantic chunk from related chunks
			if related_chunks:
				self._create_semantic_chunk(related_chunks, result_chunks)

	def _group_related_files(
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
		combined_content = []

		for rc in related_chunks:
			all_files.extend(rc.files)
			combined_content.append(rc.content)

		# Determine the appropriate commit type based on the files
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
			)
		)

	def _consolidate_small_chunks(self, chunks: list[DiffChunk]) -> list[DiffChunk]:
		"""
		Consolidate small chunks into larger, more meaningful groups.

		First, consolidates chunks originating from the same file.
		Then, consolidates remaining single-file chunks by directory.

		Args:
		    chunks: List of diff chunks to consolidate

		Returns:
		    Consolidated list of chunks

		"""
		# If we have fewer than MIN_CHUNKS_FOR_CONSOLIDATION chunks, no need to consolidate
		if len(chunks) < MIN_CHUNKS_FOR_CONSOLIDATION:
			return chunks

		# --- Step 1: Consolidate chunks from the same file ----
		file_groups: dict[str, list[DiffChunk]] = {}
		other_chunks: list[DiffChunk] = []  # Chunks with multiple files or no files

		for chunk in chunks:
			if len(chunk.files) == 1:
				file_path = chunk.files[0]
				if file_path not in file_groups:
					file_groups[file_path] = []
				file_groups[file_path].append(chunk)
			else:
				other_chunks.append(chunk)  # Keep multi-file chunks separate for now

		consolidated_same_file_chunks: list[DiffChunk] = []
		for file_path, file_chunk_list in file_groups.items():
			if len(file_chunk_list) > 1:
				# Merge chunks for this file
				combined_content = "\n".join([c.content for c in file_chunk_list])
				# Use the description from the first chunk or generate a default one
				description = file_chunk_list[0].description or f"Changes in {file_path}"
				consolidated_same_file_chunks.append(
					DiffChunk(files=[file_path], content=combined_content, description=description)
				)
			else:
				# Keep single chunks as they are
				consolidated_same_file_chunks.extend(file_chunk_list)

		# Combine same-file consolidated chunks and the multi-file chunks
		final_chunks = consolidated_same_file_chunks + other_chunks

		logger.debug("Consolidated (file-level only) from %d to %d chunks", len(chunks), len(final_chunks))
		return final_chunks
