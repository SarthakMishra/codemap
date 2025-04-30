"""Diff splitting implementation for CodeMap."""

import logging
import re
from pathlib import Path
from typing import cast

import numpy as np

from codemap.git.utils import GitDiff
from codemap.utils.cli_utils import console, loading_spinner

from .constants import MAX_FILE_SIZE_FOR_LLM, MAX_LOG_DIFF_SIZE, MODEL_NAME
from .schemas import DiffChunk
from .strategies import EmbeddingModel, FileSplitStrategy, SemanticSplitStrategy
from .utils import (
	calculate_semantic_similarity,
	filter_valid_files,
	get_language_specific_patterns,
	is_test_environment,
)
from .utils import extract_code_from_diff as _extract_code_from_diff

logger = logging.getLogger(__name__)


class DiffSplitter:
	"""Splits Git diffs into logical chunks."""

	# Class-level cache for the embedding model
	_embedding_model = None
	# Track availability of sentence-transformers and the model
	_sentence_transformers_available = None
	_model_available = None

	def __init__(self, repo_root: Path) -> None:
		"""
		Initialize the diff splitter.

		Args:
		    repo_root: Root directory of the Git repository

		"""
		self.repo_root = repo_root

		# Do NOT automatically check availability - let the command class do this explicitly
		# This avoids checks happening during initialization without visible loading states

	@classmethod
	def _check_sentence_transformers_availability(cls) -> bool:
		"""
		Check if sentence-transformers package is available.

		Returns:
		    True if sentence-transformers is available, False otherwise

		"""
		try:
			# This is needed for the import check, but don't flag as unused
			import sentence_transformers  # type: ignore  # noqa: F401, PGH003

			# Set the class flag for future reference
			cls._sentence_transformers_available = True
			logger.debug("sentence-transformers is available")
			return True
		except ImportError as e:
			# Log the specific import error for better debugging
			cls._sentence_transformers_available = False
			logger.warning(
				"sentence-transformers import failed: %s. Semantic similarity features will be limited. "
				"Install with: pip install sentence-transformers numpy",
				e,
			)
			return False
		except (RuntimeError, ValueError, AttributeError) as e:
			# Catch specific errors during import
			cls._sentence_transformers_available = False
			logger.warning(
				"Unexpected error importing sentence-transformers: %s. Semantic similarity features will be limited.", e
			)
			return False

	@classmethod
	def _check_model_availability(cls, model_name: str = MODEL_NAME) -> bool:
		"""
		Check if the embedding model is available.

		Args:
		    model_name: Name of the model to check

		Returns:
		    True if model is available, False otherwise

		"""
		if not DiffSplitter._sentence_transformers_available:
			return False

		try:
			from sentence_transformers import SentenceTransformer

			# Create model instance if not already created
			if DiffSplitter._embedding_model is None:
				logger.debug("Loading embedding model: %s", model_name)

				try:
					# Use a simpler loading approach without Progress bar
					# to avoid "Only one live display may be active at once" error
					console.print("Loading embedding model...")

					# Load the model without progress tracking
					DiffSplitter._embedding_model = SentenceTransformer(model_name)

					console.print("[green]✓[/green] Model loaded successfully")

					logger.debug("Initialized embedding model: %s", model_name)
					# Explicitly set the class variable to True when model loads successfully
					cls._model_available = True
					return True
				except ImportError as e:
					logger.exception("Missing dependencies for embedding model")
					console.print(f"[red]Error: Missing dependencies: {e}[/red]")
					cls._model_available = False
					return False
				except MemoryError:
					logger.exception("Not enough memory to load embedding model")
					console.print("[red]Error: Not enough memory to load embedding model[/red]")
					cls._model_available = False
					return False
				except ValueError as e:
					logger.exception("Invalid model configuration")
					console.print(f"[red]Error: Invalid model configuration: {e}[/red]")
					cls._model_available = False
					return False
				except RuntimeError as e:
					error_msg = str(e)
					# Check for CUDA/GPU related errors
					if "CUDA" in error_msg or "GPU" in error_msg:
						logger.exception("GPU error when loading model")
						console.print("[red]Error: GPU/CUDA error. Try using CPU only mode.[/red]")
					else:
						logger.exception("Runtime error when loading model")
						console.print(f"[red]Error loading model: {error_msg}[/red]")
					cls._model_available = False
					return False
				except Exception as e:
					logger.exception("Unexpected error loading embedding model")
					console.print(f"[red]Unexpected error loading model: {e}[/red]")
					cls._model_available = False
					return False
			# If we already have a model loaded, make sure to set the flag to True
			cls._model_available = True
			return True
		except Exception as e:
			# This is the outer exception handler for any unexpected errors
			logger.exception("Failed to load embedding model %s", model_name)
			console.print(f"[red]Failed to load embedding model: {e}[/red]")
			cls._model_available = False
			return False

	def split_diff(self, diff: GitDiff) -> tuple[list[DiffChunk], list[str]]:
		"""
		Split a diff into logical chunks using semantic splitting.

		Args:
		    diff: GitDiff object to split

		Returns:
		    Tuple of (List of DiffChunk objects based on semantic analysis, List of filtered large files)

		Raises:
		    ValueError: If semantic splitting is not available or fails

		"""
		filtered_large_files = []

		if not diff.files:
			return [], filtered_large_files

		# In test environments, log the diff content for debugging
		if is_test_environment():
			logger.debug("Processing diff in test environment with %d files", len(diff.files) if diff.files else 0)
			if diff.content and len(diff.content) < MAX_LOG_DIFF_SIZE:  # Only log short diffs to avoid spamming logs
				logger.debug("Diff content: %s", diff.content)

		# Check for excessively large diff content and handle appropriately
		if diff.content and len(diff.content) > MAX_FILE_SIZE_FOR_LLM:
			logger.warning("Diff content is very large (%d bytes). Processing might be limited.", len(diff.content))

			# Try to extract file names directly from the diff content for large diffs
			file_list = re.findall(r"diff --git a/(.*?) b/(.*?)$", diff.content, re.MULTILINE)
			if file_list:
				logger.info("Extracted %d files from large diff content", len(file_list))
				files_to_process = [f[1] for f in file_list]  # Use the "b" side of each diff

				# Override diff.files with extracted file list to bypass content processing
				diff.files = files_to_process

				# Optional: Clear the content to avoid processing it
				original_content_size = len(diff.content)
				diff.content = ""
				logger.info("Cleared %d bytes of diff content to avoid payload limits", original_content_size)

		# Process files in the diff
		if diff.files:
			diff.files, large_files = filter_valid_files(diff.files, is_test_environment())
			filtered_large_files.extend(large_files)

		if not diff.files:
			logger.warning("No valid files to process after filtering")
			return [], filtered_large_files

		# Set up availability flags if not already set
		cls = type(self)
		cls._sentence_transformers_available = (
			cls._sentence_transformers_available or cls._check_sentence_transformers_availability()
		)

		if not cls._sentence_transformers_available:
			msg = (
				"Semantic splitting is not available. sentence-transformers package is required. "
				"Install with: pip install sentence-transformers numpy"
			)
			raise ValueError(msg)

		# Try to load the model
		with loading_spinner("Loading embedding model..."):
			cls._model_available = cls._model_available or cls._check_model_availability()

		if not cls._model_available:
			msg = (
				"Semantic splitting failed: embedding model could not be loaded. "
				"Check logs for details or try a different model."
			)
			raise ValueError(msg)

		# Use semantic splitting
		chunks = self._split_semantic(diff)
		return chunks, filtered_large_files

	def _extract_code_from_diff(self, diff_content: str) -> tuple[str, str]:
		"""
		Extract old and new code from diff content.

		Args:
		    diff_content: Git diff content

		Returns:
		    Tuple of (old_code, new_code)

		"""
		return _extract_code_from_diff(diff_content)

	def _semantic_hunk_splitting(self, file_path: str, diff_content: str) -> list[str]:
		"""
		Split a diff into semantic hunks based on code structure, preserving hunk integrity.

		Args:
		    file_path: Path to the file
		    diff_content: Git diff content for a single file

		Returns:
		    List of diff chunk strings, where each chunk contains one or more full hunks.

		"""
		if not diff_content.strip():
			return []  # Return empty list if diff content is empty

		# Extract language-specific patterns
		extension = Path(file_path).suffix.lower()
		patterns = [re.compile(p) for p in get_language_specific_patterns(extension.lstrip("."))]

		if not patterns:
			logger.debug("No language patterns found for %s, returning whole diff as one chunk", extension)
			return [diff_content]

		diff_lines = diff_content.splitlines()
		raw_hunks: list[list[str]] = []
		current_hunk_lines: list[str] = []
		is_first_hunk = True

		# Split into raw hunks (including headers and file context lines)
		for line in diff_lines:
			if line.startswith("@@ "):
				if not is_first_hunk and current_hunk_lines:  # Don't add if it's the first @@ and list is empty
					raw_hunks.append(current_hunk_lines)
					current_hunk_lines = [line]  # Start new hunk with the @@ line
				else:
					current_hunk_lines.append(line)  # Add first @@ line or lines before first @@
				is_first_hunk = False
			else:
				current_hunk_lines.append(line)  # Add non-@@ lines

		if current_hunk_lines:
			raw_hunks.append(current_hunk_lines)

		if not raw_hunks:
			logger.debug("No hunks found in diff content for %s", file_path)
			return [diff_content]  # Return original content if parsing fails

		# Process hunks to create semantic chunks
		final_chunks: list[list[str]] = []
		current_semantic_chunk_lines: list[str] = []

		for hunk_lines in raw_hunks:
			hunk_has_boundary = False
			# Check added lines within this hunk for semantic boundaries
			for line in hunk_lines:
				if line.startswith("+"):
					added_line_content = line[1:]  # Check content without the '+'
					if any(pattern.match(added_line_content) for pattern in patterns):
						hunk_has_boundary = True
						break  # Found a boundary in this hunk

			# Decision: Start a new semantic chunk IF the current hunk has a boundary
			# AND we already have lines accumulated in the current semantic chunk.
			if hunk_has_boundary and current_semantic_chunk_lines:
				final_chunks.append(current_semantic_chunk_lines)  # Finalize previous semantic chunk
				current_semantic_chunk_lines = hunk_lines  # Start new semantic chunk with current hunk
			else:
				# Otherwise, append the current hunk to the ongoing semantic chunk
				current_semantic_chunk_lines.extend(hunk_lines)

		# Add the last accumulated semantic chunk
		if current_semantic_chunk_lines:
			final_chunks.append(current_semantic_chunk_lines)

		# Join the lines back into strings for each chunk
		result_chunks = ["\n".join(chunk_lines) for chunk_lines in final_chunks]

		# Handle the case where splitting results in no usable chunks (e.g., only headers split)
		if not result_chunks or all(not c.strip() for c in result_chunks):
			logger.debug("Semantic splitting resulted in empty chunks for %s, returning whole diff.", file_path)
			return [diff_content]

		logger.debug("Split %s into %d semantic chunks", file_path, len(result_chunks))
		return result_chunks

	def _enhance_semantic_split(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Enhance semantic splitting by analyzing code structure.

		This method now aims to return a SINGLE chunk per file, using semantic
		splitting internally perhaps for description generation, but not for splitting.

		Args:
		    diff: GitDiff object to split (should contain only one file)

		Returns:
		    A list containing at most one DiffChunk.

		"""
		if not diff.content or not diff.files:
			return []

		# This function should now only handle single-file diffs
		if len(diff.files) != 1:
			logger.warning("_enhance_semantic_split called with %d files, expected 1. Skipping.", len(diff.files))
			# Optionally, handle multi-file diffs by falling back to file splitting
			# return FileSplitStrategy().split(diff)
			return []

		file_path = diff.files[0]

		# --- Removed semantic hunk splitting logic for chunk creation ---
		# semantic_chunks_content = self._semantic_hunk_splitting(file_path, diff.content)
		# if not semantic_chunks_content:
		# 	return [DiffChunk(files=[file_path], content=diff.content, description=f"Changes in {file_path}")]

		# Simply return one chunk containing the full diff content for the file.
		# Semantic analysis might still happen elsewhere (e.g., for description generation)
		# but we don't split the chunk itself here.
		return [
			DiffChunk(
				files=[file_path],
				content=diff.content,  # Use the original full diff content
				description=f"Changes in {file_path}",  # Basic description, can be enhanced later
			)
		]

	def _split_semantic(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Split a diff semantically considering code structure.

		Args:
		    diff: GitDiff object to split

		Returns:
		    List of DiffChunk objects based on semantic analysis

		"""
		# Try semantic strategy first
		try:
			# Apply semantic splitting
			semantic_strategy = SemanticSplitStrategy(embedding_model=cast("EmbeddingModel", self._embedding_model))
			chunks = semantic_strategy.split(diff)

			if chunks:
				return chunks

			# If semantic splitting produced no chunks, log a warning and continue to fallbacks
			logger.warning(
				"Semantic splitting failed to produce any chunks. Falling back to simpler strategies. "
				"This commonly happens with large refactoring operations."
			)
		except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as e:
			# Log specific exceptions from semantic splitting and fall back
			logger.warning("Semantic splitting encountered an error: %s. Falling back to simpler strategies.", e)

		# First fallback: directory-based grouping
		logger.info("Using directory-based grouping as fallback strategy")
		dir_chunks = []

		try:
			# Group files by directory
			files_by_dir = {}
			for file in diff.files:
				dir_path = str(Path(file).parent)
				if dir_path not in files_by_dir:
					files_by_dir[dir_path] = []
				files_by_dir[dir_path].append(file)

			# Create one chunk per directory
			for dir_path, files in files_by_dir.items():
				if files:
					display_name = dir_path if dir_path != "." else "root directory"
					dir_chunks.append(
						DiffChunk(
							files=files,
							content=diff.content,
							description=f"Changes in {display_name}",
						)
					)

			if dir_chunks:
				return dir_chunks
		except (ValueError, KeyError, TypeError, AttributeError, OSError) as dir_error:
			logger.warning("Directory-based fallback failed: %s", dir_error)

		# Second fallback: file-based splitting
		logger.info("Using file-based splitting as final fallback strategy")
		file_strategy = FileSplitStrategy()
		chunks = file_strategy.split(diff)

		if chunks:
			return chunks

		# Last resort: create one chunk per file if all else fails
		logger.info("File splitting produced no chunks. Creating basic chunks (one per file).")
		return [
			DiffChunk(
				files=[file],
				content=f"File: {file}\n{diff.content}",
				description=f"Changes in {file}",
			)
			for file in diff.files
		]

	def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
		"""
		Calculate semantic similarity between two text segments.

		Args:
		    text1: First text
		    text2: Second text

		Returns:
		    Similarity score between 0 and 1

		"""
		if not text1 or not text2:
			return 0.0

		# Check if embedding model is available
		cls = type(self)
		cls._sentence_transformers_available = (
			cls._sentence_transformers_available or cls._check_sentence_transformers_availability()
		)

		if not cls._sentence_transformers_available:
			logger.debug("Sentence transformers not available, returning zero similarity")
			return 0.0

		cls._model_available = cls._model_available or cls._check_model_availability()

		if not cls._model_available or cls._embedding_model is None:
			logger.debug("Embedding model not available, returning zero similarity")
			return 0.0

		try:
			# Encode both texts
			embeddings = cls._embedding_model.encode([text1, text2])

			# Calculate cosine similarity using utility function
			return calculate_semantic_similarity(embeddings[0].tolist(), embeddings[1].tolist())

		except (ValueError, TypeError, IndexError, RuntimeError) as e:
			logger.warning("Error calculating semantic similarity: %s", e)
			return 0.0

	@classmethod
	def encode_chunks(cls, chunks: list[str]) -> dict[str, np.ndarray]:
		"""
		Encode text chunks into embeddings.

		Args:
		    chunks: List of text chunks

		Returns:
		    Dict with keys 'embeddings' containing numpy array of embeddings

		"""
		# Ensure the model is initialized
		cls._sentence_transformers_available = (
			cls._sentence_transformers_available or cls._check_sentence_transformers_availability()
		)
		if cls._sentence_transformers_available:
			cls._model_available = cls._model_available or cls._check_model_availability(model_name=MODEL_NAME)

		if not cls._model_available:
			logger.debug("Embedding model not available, returning empty embeddings")
			return {"embeddings": np.array([])}

		if not chunks:
			return {"embeddings": np.array([])}

		# At this point we know model is initialized and available
		if cls._embedding_model is None:
			logger.debug("Embedding model is None but was marked as available, reinitializing")
			cls._model_available = cls._check_model_availability(model_name=MODEL_NAME)
			if not cls._model_available:
				return {"embeddings": np.array([])}

		# Use runtime check instead of assert
		if cls._embedding_model is None:
			logger.error("Embedding model is None but should be initialized at this point")
			return {"embeddings": np.array([])}

		embeddings = cls._embedding_model.encode(chunks)
		return {"embeddings": embeddings}
