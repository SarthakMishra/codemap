"""Diff splitting implementation for CodeMap."""

import logging
import os
import re
from pathlib import Path
from typing import cast

import numpy as np

from codemap.git.utils import GitDiff
from codemap.utils.cli_utils import console, loading_spinner

from .constants import MODEL_NAME
from .schemas import DiffChunk, SplitStrategy
from .strategies import EmbeddingModel, FileSplitStrategy, HunkSplitStrategy, SemanticSplitStrategy
from .utils import extract_code_from_diff as _extract_code_from_diff
from .utils import filter_valid_files

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

	def split_diff(self, diff: GitDiff, strategy: str | SplitStrategy | None = None) -> list[DiffChunk]:
		"""
		Split a diff into logical chunks.

		Args:
		    diff: GitDiff object to split
		    strategy: Strategy to use for splitting (FILE, HUNK, or SEMANTIC)

		Returns:
		    List of DiffChunk objects

		"""
		if not diff.content and not diff.files:
			return []

		# Check if we're in a test environment
		is_test_environment = "PYTEST_CURRENT_TEST" in os.environ

		# Process files in the diff
		if diff.files:
			diff.files = filter_valid_files(diff.files, is_test_environment)

		if not diff.files:
			logger.warning("No valid files to process after filtering")
			return []

		# Convert string strategy to enum if needed
		if isinstance(strategy, str):
			try:
				strategy = SplitStrategy(strategy)
			except ValueError:
				logger.warning("Invalid strategy: %s. Using SEMANTIC instead.", strategy)
				strategy = SplitStrategy.SEMANTIC

		# Use semantic strategy by default
		if strategy is None:
			strategy = SplitStrategy.SEMANTIC

		# Apply the selected strategy
		if strategy == SplitStrategy.FILE:
			file_strategy = FileSplitStrategy()
			return file_strategy.split(diff)

		if strategy == SplitStrategy.HUNK:
			hunk_strategy = HunkSplitStrategy()
			return hunk_strategy.split(diff)

		# SEMANTIC
		# Check if we need to load the model for semantic strategy
		if strategy == SplitStrategy.SEMANTIC:
			# Set up availability flags if not already set
			cls = type(self)
			cls._sentence_transformers_available = (
				cls._sentence_transformers_available or cls._check_sentence_transformers_availability()
			)

			if cls._sentence_transformers_available:
				# Try to load the model if sentence-transformers is available
				with loading_spinner("Loading embedding model..."):
					cls._model_available = cls._model_available or cls._check_model_availability()

			# Create the semantic strategy with the model if available
			return self._split_semantic(diff)

		# Fallback to file strategy if we somehow reach here
		file_strategy = FileSplitStrategy()
		return file_strategy.split(diff)

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
		Split a diff into semantic hunks based on code structure.

		Args:
		    file_path: Path to the file
		    diff_content: Git diff content

		Returns:
		    List of diff chunks

		"""
		# Extract language-specific patterns
		extension = Path(file_path).suffix.lower()
		patterns = self._get_language_specific_patterns(extension)

		if not patterns:
			logger.debug("No language patterns found for %s, using basic splitting", extension)
			return [diff_content]  # Return the whole diff as one chunk if no patterns found

		# Extract old and new code
		_, new_code = self._extract_code_from_diff(diff_content)

		if not new_code:
			logger.debug("No code extracted from diff, using basic splitting")
			return [diff_content]

		# Find boundaries of structural elements in the code
		chunk_boundaries = []

		# Search for all pattern matches in the code
		for pattern in patterns:
			for match in re.finditer(pattern, new_code, re.MULTILINE):
				start_pos = match.start()
				chunk_boundaries.append(start_pos)

		# Sort boundaries
		chunk_boundaries.sort()

		if not chunk_boundaries:
			logger.debug("No semantic boundaries found, using basic splitting")
			return [diff_content]

		# Create chunks around semantic boundaries
		diff_lines = diff_content.splitlines()
		chunks = []
		current_chunk_lines = []
		current_line_number = 0

		for line in diff_lines:
			current_chunk_lines.append(line)

			# Check if this is a hunk header
			if line.startswith("@@"):
				hunk_match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
				if hunk_match:
					current_line_number = int(hunk_match.group(1)) - 1

			# Increment line number for added/context lines, skip removed lines
			if line.startswith("+") or not line.startswith("-"):
				current_line_number += 1

			# Check if we've reached a semantic boundary
			if current_line_number in chunk_boundaries:
				# Finish current chunk
				chunks.append("\n".join(current_chunk_lines))
				current_chunk_lines = []

		# Add the last chunk if any
		if current_chunk_lines:
			chunks.append("\n".join(current_chunk_lines))

		return chunks if chunks else [diff_content]

	def _enhance_semantic_split(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Enhance semantic splitting by analyzing code structure.

		Args:
		    diff: GitDiff object to split

		Returns:
		    List of DiffChunk objects

		"""
		if not diff.content or not diff.files:
			return []

		file_path = diff.files[0] if len(diff.files) == 1 else None

		if not file_path:
			# If multiple files, use basic file-based splitting
			logger.debug("Multiple files in diff, using file-based splitting")
			file_strategy = FileSplitStrategy()
			return file_strategy.split(diff)

		# Try semantic splitting based on language structures
		semantic_chunks = self._semantic_hunk_splitting(file_path, diff.content)

		if len(semantic_chunks) <= 1:
			# Fallback to hunk-based splitting if semantic splitting produced only one chunk
			logger.debug("Semantic splitting produced only one chunk, using hunk-based splitting")
			hunk_strategy = HunkSplitStrategy()
			return hunk_strategy.split(diff)

		# Create DiffChunk objects for each semantic chunk
		diff_chunks = []
		for i, chunk_content in enumerate(semantic_chunks):
			chunk = DiffChunk(
				files=[file_path],
				content=chunk_content,
				description=f"Semantic chunk {i + 1} of {len(semantic_chunks)} in {file_path}",
			)
			diff_chunks.append(chunk)

		return diff_chunks

	def _split_semantic(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Split a diff semantically considering code structure.

		Args:
		    diff: GitDiff object to split

		Returns:
		    List of DiffChunk objects

		"""
		# First try basic semantic splitting
		semantic_strategy = SemanticSplitStrategy(embedding_model=cast("EmbeddingModel", self._embedding_model))
		chunks = semantic_strategy.split(diff)

		# If semantic strategy failed or produced only one chunk, try enhanced semantic splitting
		if len(chunks) <= 1 and len(diff.files) == 1:
			logger.debug("Basic semantic splitting produced only one chunk, trying enhanced semantic splitting")
			return self._enhance_semantic_split(diff)

		return chunks

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

			# Calculate cosine similarity
			embedding1 = embeddings[0]
			embedding2 = embeddings[1]

			# Compute dot product
			dot_product = np.dot(embedding1, embedding2)

			# Compute magnitudes
			magnitude1 = np.linalg.norm(embedding1)
			magnitude2 = np.linalg.norm(embedding2)

			# Compute cosine similarity
			if magnitude1 > 0 and magnitude2 > 0:
				similarity = dot_product / (magnitude1 * magnitude2)
				return float(similarity)

			return 0.0

		except (ValueError, TypeError, IndexError, RuntimeError) as e:
			logger.warning("Error calculating semantic similarity: %s", e)
			return 0.0

	def _get_language_specific_patterns(self, extension: str) -> list[str]:
		"""
		Get regex patterns for semantic boundaries based on language.

		Args:
		    extension: File extension

		Returns:
		    List of regex patterns

		"""
		# Define language-specific patterns for semantic boundaries
		patterns = {
			".py": [
				r"^import\s+.*",  # Import statements
				r"^from\s+.*",  # From imports
				r"^class\s+\w+",  # Class definitions
				r"^def\s+\w+",  # Function definitions
				r"^if\s+__name__\s*==\s*['\"]__main__['\"]",  # Main block
			],
			".js": [
				r"^import\s+.*",  # ES6 imports
				r"^const\s+\w+\s*=\s*require",  # CommonJS imports
				r"^function\s+\w+",  # Function declarations
				r"^const\s+\w+\s*=\s*function",  # Function expressions
				r"^class\s+\w+",  # Class declarations
				r"^export\s+",  # Exports
			],
			".java": [
				r"^import\s+.*",  # Import statements
				r"^public\s+class",  # Public class
				r"^private\s+class",  # Private class
				r"^(public|private|protected)(\s+static)?\s+\w+\s+\w+\(",  # Methods
			],
			".go": [
				r"^import\s+",  # Import statements
				r"^func\s+",  # Function definitions
				r"^type\s+\w+\s+struct",  # Struct definitions
			],
			".rb": [
				r"^require\s+",  # Requires
				r"^class\s+",  # Class definitions
				r"^def\s+",  # Method definitions
				r"^module\s+",  # Module definitions
			],
			".php": [
				r"^namespace\s+",  # Namespace declarations
				r"^use\s+",  # Use statements
				r"^class\s+",  # Class definitions
				r"^(public|private|protected)\s+function",  # Methods
			],
			".ts": [
				r"^import\s+.*",  # Imports
				r"^export\s+",  # Exports
				r"^interface\s+",  # Interfaces
				r"^type\s+",  # Type definitions
				r"^class\s+",  # Classes
				r"^function\s+",  # Functions
				r"^const\s+\w+\s*=\s*",  # Constants
			],
			".cs": [
				r"^using\s+",  # Using directives
				r"^namespace\s+",  # Namespace declarations
				r"^(public|private|protected|internal)\s+class",  # Classes
				r"^(public|private|protected|internal)(\s+static)?\s+\w+\s+\w+\(",  # Methods
			],
		}

		return patterns.get(extension, [])

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
