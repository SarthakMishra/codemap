"""Diff splitting implementation for CodeMap."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from codemap.config import DEFAULT_CONFIG
from codemap.git.diff_splitter.strategies import FileSplitStrategy, SemanticSplitStrategy
from codemap.git.diff_splitter.utils import calculate_semantic_similarity, filter_valid_files, is_test_environment
from codemap.git.utils import GitDiff
from codemap.utils.cli_utils import console, loading_spinner
from codemap.utils.config_loader import ConfigLoader

from .schemas import DiffChunk

if TYPE_CHECKING:
	# If SentenceTransformer is importable under TYPE_CHECKING, use it for better type safety
	# from sentence_transformers import SentenceTransformer as EmbeddingModelType
	# else:
	# EmbeddingModelType = Any
	EmbeddingModelType = Any  # Simpler for now

logger = logging.getLogger(__name__)

# Define EmbeddingModel type alias at module level
EmbeddingModel = Any  # Placeholder if SentenceTransformer is truly optional for class methods

# Constants for truncation and sampling
MAX_DIFF_CONTENT_LENGTH = 100000  # ~100KB maximum size for diff content
MAX_DIFF_LINES = 1000  # Maximum number of lines to process
SMALL_SECTION_SIZE = 50  # Maximum size for a "small" diff section
COMPLEX_SECTION_SIZE = 100  # Minimum size for a "complex" diff section (with middle sample)


class DiffSplitter:
	"""Splits Git diffs into logical chunks."""

	_embedding_model: EmbeddingModel | None = None  # Type hint for class attribute
	_sentence_transformers_available: bool | None = None  # Explicitly None initially
	_model_available: bool | None = None  # Explicitly None initially

	def __init__(
		self,
		repo_root: Path,
		config_loader: ConfigLoader | None = None,  # Added config_loader
		# Parameters will now try to fetch from config_loader first, then DEFAULT_CONFIG
		embedding_model: str | None = None,  # Renamed from model_name, allow None
		similarity_threshold: float | None = None,
		directory_similarity_threshold: float | None = None,
		min_chunks_for_consolidation: int | None = None,
		max_chunks_before_consolidation: int | None = None,
		max_file_size_for_llm: int | None = None,
		max_log_diff_size: int | None = None,
	) -> None:
		"""
		Initialize the diff splitter.

		Args:
		    repo_root: Root directory of the Git repository
		    config_loader: ConfigLoader object for loading configuration
		    embedding_model: Name of the sentence-transformer model to use
		    similarity_threshold: Threshold for grouping by content similarity.
		    directory_similarity_threshold: Threshold for directory similarity.
		    min_chunks_for_consolidation: Min chunks to trigger consolidation.
		    max_chunks_before_consolidation: Max chunks allowed before forced consolidation.
		    max_file_size_for_llm: Max file size (bytes) to process for LLM context.
		    max_log_diff_size: Max diff size (bytes) to log in debug mode.

		"""
		self.repo_root = repo_root
		self.config_loader = config_loader or ConfigLoader(repo_root=self.repo_root)

		# Get config for diff_splitter, fallback to empty dict if not found
		ds_config = self.config_loader.get("commit", {}).get("diff_splitter", {})

		# Determine parameters: CLI/direct arg > Config file > DEFAULT_CONFIG
		self.embedding_model = (
			embedding_model
			or ds_config.get("model_name")  # model_name is the key in config
			or DEFAULT_CONFIG["commit"]["diff_splitter"]["model_name"]
		)
		self.similarity_threshold = (
			similarity_threshold
			or ds_config.get("similarity_threshold")
			or DEFAULT_CONFIG["commit"]["diff_splitter"]["similarity_threshold"]
		)
		self.directory_similarity_threshold = (
			directory_similarity_threshold
			or ds_config.get("directory_similarity_threshold")
			or DEFAULT_CONFIG["commit"]["diff_splitter"]["directory_similarity_threshold"]
		)
		self.min_chunks_for_consolidation = (
			min_chunks_for_consolidation
			or ds_config.get("min_chunks_for_consolidation")
			or DEFAULT_CONFIG["commit"]["diff_splitter"]["min_chunks_for_consolidation"]
		)
		self.max_chunks_before_consolidation = (
			max_chunks_before_consolidation
			or ds_config.get("max_chunks_before_consolidation")
			or DEFAULT_CONFIG["commit"]["diff_splitter"]["max_chunks_before_consolidation"]
		)
		self.max_file_size_for_llm = (
			max_file_size_for_llm
			or ds_config.get("max_file_size_for_llm")
			or DEFAULT_CONFIG["commit"]["diff_splitter"]["max_file_size_for_llm"]
		)
		self.max_log_diff_size = (
			max_log_diff_size
			or ds_config.get("max_log_diff_size")
			or DEFAULT_CONFIG["commit"]["diff_splitter"]["max_log_diff_size"]
		)

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
	def are_sentence_transformers_available(cls) -> bool:
		"""
		Check if sentence transformers are available.

		Returns:
		    True if sentence transformers are available, False otherwise

		"""
		return cls._sentence_transformers_available or cls._check_sentence_transformers_availability()

	@classmethod
	def is_model_available(cls) -> bool:
		"""
		Check if embedding model is available.

		Returns:
		    True if embedding model is available, False otherwise

		"""
		return bool(cls._model_available)

	@classmethod
	def set_model_available(cls, value: bool) -> None:
		"""
		Set model availability flag.

		Args:
		    value: Boolean indicating if model is available

		"""
		cls._model_available = value

	@classmethod
	def get_embedding_model(cls) -> EmbeddingModel | None:
		"""
		Get the embedding model.

		Returns:
		    The embedding model or None if not available

		"""
		return cls._embedding_model

	@classmethod
	def set_embedding_model(cls, model: EmbeddingModel) -> None:
		"""
		Set the embedding model.

		Args:
		    model: The embedding model to set

		"""
		cls._embedding_model = model

	def check_model_availability(self) -> bool:
		"""
		Check if the embedding model is available using the instance's configured model name.

		Returns:
		    True if model is available, False otherwise

		"""
		# Use class method to access class-level cache check
		if not self.__class__.are_sentence_transformers_available():
			return False

		try:
			from sentence_transformers import SentenceTransformer

			# Use class method to access class-level cache
			if self.__class__.get_embedding_model() is None:
				# Use self.embedding_model from instance configuration
				logger.debug("Loading embedding model: %s", self.embedding_model)

				try:
					console.print("Loading embedding model...")
					# Load the model using self.embedding_model
					model = SentenceTransformer(self.embedding_model)
					self.__class__.set_embedding_model(cast("EmbeddingModel", model))
					console.print("[green]✓[/green] Model loaded successfully")
					logger.debug("Initialized embedding model: %s", self.embedding_model)
					# Set class-level flag via class method
					self.__class__.set_model_available(True)
					return True
				except ImportError as e:
					logger.exception("Missing dependencies for embedding model")
					console.print(f"[red]Error: Missing dependencies: {e}[/red]")
					self.__class__.set_model_available(False)
					return False
				except MemoryError:
					logger.exception("Not enough memory to load embedding model")
					console.print("[red]Error: Not enough memory to load embedding model[/red]")
					self.__class__.set_model_available(False)
					return False
				except ValueError as e:
					logger.exception("Invalid model configuration")
					console.print(f"[red]Error: Invalid model configuration: {e}[/red]")
					self.__class__.set_model_available(False)
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
					self.__class__.set_model_available(False)
					return False
				except Exception as e:
					logger.exception("Unexpected error loading embedding model")
					console.print(f"[red]Unexpected error loading model: {e}[/red]")
					self.__class__.set_model_available(False)
					return False
			# If we already have a model loaded, make sure to set the flag to True
			self.__class__.set_model_available(True)
			return True
		except Exception as e:
			# This is the outer exception handler for any unexpected errors
			logger.exception("Failed to load embedding model %s", self.embedding_model)
			console.print(f"[red]Failed to load embedding model: {e}[/red]")
			self.__class__.set_model_available(False)
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
		if not diff.files:
			return [], []

		# Special handling for untracked files - bypass semantic split since the content isn't a proper diff format
		if diff.is_untracked:
			logger.debug("Processing untracked files with special handling: %d files", len(diff.files))
			# Create a simple chunk per file to avoid errors with unidiff parsing
			chunks = []
			for file_path in diff.files:
				# Create a basic chunk with file info but without trying to parse the content as a diff
				chunks = [
					DiffChunk(
						files=[file_path],
						content=f"New untracked file: {file_path}",
						description=f"New file: {file_path}",
					)
					for file_path in diff.files
				]
			return chunks, []

		# In test environments, log the diff content for debugging
		if is_test_environment():
			logger.debug("Processing diff in test environment with %d files", len(diff.files) if diff.files else 0)
			if diff.content and len(diff.content) < self.max_log_diff_size:  # Use configured max log size
				logger.debug("Diff content: %s", diff.content)

		# Process files in the diff
		if diff.files:
			# Filter for valid files (existence, tracked status), max_size check removed here
			logger.debug(f"DiffSplitter.split_diff: Files before filter_valid_files: {diff.files}")
			diff.files, _ = filter_valid_files(diff.files, self.repo_root, is_test_environment())
			logger.debug(f"DiffSplitter.split_diff: Files after filter_valid_files: {diff.files}")
			# filtered_large_files list is no longer populated or used here

		if not diff.files:
			logger.warning("No valid files to process after filtering")
			return [], []  # Return empty lists

		# Set up availability flags if not already set
		# Use class method to check sentence transformers availability
		if not self.__class__.are_sentence_transformers_available():
			msg = (
				"Semantic splitting is not available. sentence-transformers package is required. "
				"Install with: pip install sentence-transformers numpy"
			)
			raise ValueError(msg)

		# Try to load the model using the instance method
		with loading_spinner("Loading embedding model..."):
			# Use self.check_model_availability() - it uses self.embedding_model internally
			if not self.__class__.is_model_available():
				self.check_model_availability()

		if not self.__class__.is_model_available():
			msg = "Semantic splitting failed: embedding model could not be loaded. Check logs for details."
			raise ValueError(msg)

		try:
			chunks = self._split_semantic(diff)

			# If we truncated the content, restore the original content for the actual chunks
			if diff.content and chunks:
				# Create a mapping of file paths to chunks for quick lookup
				chunks_by_file = {}
				for chunk in chunks:
					for file_path in chunk.files:
						if file_path not in chunks_by_file:
							chunks_by_file[file_path] = []
						chunks_by_file[file_path].append(chunk)

				# For chunks that represent files we can find in the original content,
				# update their content to include the full original diff for that file
				for chunk in chunks:
					# Use a heuristic to match file sections in the original content
					for file_path in chunk.files:
						file_marker = f"diff --git a/{file_path} b/{file_path}"
						if file_marker in diff.content:
							# Found a match for this file in the original content
							# Extract that file's complete diff section
							start_idx = diff.content.find(file_marker)
							end_idx = diff.content.find("diff --git", start_idx + len(file_marker))
							if end_idx == -1:  # Last file in the diff
								end_idx = len(diff.content)

							file_diff = diff.content[start_idx:end_idx].strip()

							# Now replace just this file's content in the chunk
							# This is a heuristic that may need adjustment based on your diff format
							if chunk.content and file_marker in chunk.content:
								chunk_start = chunk.content.find(file_marker)
								chunk_end = chunk.content.find("diff --git", chunk_start + len(file_marker))
								if chunk_end == -1:  # Last file in the chunk
									chunk_end = len(chunk.content)

								# Replace this file's truncated diff with the full diff
								chunk.content = chunk.content[:chunk_start] + file_diff + chunk.content[chunk_end:]

			return chunks, []
		except Exception as e:
			logger.exception("Semantic splitting failed")
			console.print(f"[red]Semantic splitting failed: {e}[/red]")

			# Try basic splitting as a fallback
			logger.warning("Falling back to basic file splitting")
			console.print("[yellow]Falling back to basic file splitting[/yellow]")
			# Return empty list for filtered_large_files as it's no longer tracked here
			return self._create_basic_file_chunk(diff), []

	def _create_basic_file_chunk(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Create a basic chunk per file without semantic analysis.

		Args:
		    diff: GitDiff object to split

		Returns:
		    List of DiffChunk objects, one per file

		"""
		chunks = []

		if diff.files:
			# Create a basic chunk, one per file in this strategy, no semantic grouping
			strategy = FileSplitStrategy()
			chunks = strategy.split(diff)

		return chunks

	def _split_semantic(self, diff: GitDiff) -> list[DiffChunk]:
		"""
		Perform semantic splitting, falling back if needed.

		Args:
		    diff: GitDiff object to split

		Returns:
		    List of DiffChunk objects

		Raises:
		    ValueError: If semantic splitting fails and fallback is not possible.

		"""
		if not self.are_sentence_transformers_available():
			logger.warning("Sentence transformers unavailable. Falling back to file-based splitting.")
			# Directly use FileSplitStrategy when ST is unavailable
			file_splitter = FileSplitStrategy()
			return file_splitter.split(diff)

		# Existing logic for semantic splitting when ST is available
		try:
			semantic_strategy = SemanticSplitStrategy(embedding_model=self._embedding_model)
			return semantic_strategy.split(diff)
		except Exception:
			logger.exception("Semantic splitting failed: %s. Falling back to file splitting.")
			# Fallback to FileSplitStrategy on any semantic splitting error
			file_splitter = FileSplitStrategy()
			return file_splitter.split(diff)

	def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
		"""
		Calculate semantic similarity between two texts using the embedding model.

		Args:
		    text1: First text
		    text2: Second text

		Returns:
		    Similarity score between 0 and 1

		"""
		# Check if embedding model is available
		if not self.__class__.are_sentence_transformers_available():
			logger.debug("Sentence transformers not available, returning zero similarity")
			return 0.0

		# Call instance method self.check_model_availability()
		if not self.__class__.is_model_available():
			self.check_model_availability()

		if not self.__class__.is_model_available() or self.__class__.get_embedding_model() is None:
			logger.debug("Embedding model not available, returning zero similarity")
			return 0.0

		# Assign to local variable after check guarantees it's not None
		embedding_model_maybe_none = self.__class__.get_embedding_model()
		if embedding_model_maybe_none is None:
			# This case should have been caught earlier, but log just in case
			logger.error("Embedding model unexpectedly None after availability check")
			return 0.0

		embedding_model = embedding_model_maybe_none  # Now we know it's not None

		try:
			# Get embeddings for both texts
			emb1 = embedding_model.encode([text1])[0]
			emb2 = embedding_model.encode([text2])[0]

			# Calculate similarity using numpy
			return calculate_semantic_similarity(emb1.tolist(), emb2.tolist())
		except (ValueError, TypeError, IndexError, RuntimeError) as e:
			logger.warning("Failed to calculate semantic similarity: %s", e)
			return 0.0

	def encode_chunks(self, chunks: list[str]) -> dict[str, np.ndarray]:
		"""
		Encode a list of text chunks using the embedding model.

		Args:
		    chunks: List of text chunks to encode

		Returns:
		    Dictionary with embeddings array

		"""
		# Ensure the model is initialized
		if self.__class__.are_sentence_transformers_available() and not self.__class__.is_model_available():
			self.check_model_availability()

		if not self.__class__.is_model_available():
			logger.debug("Embedding model not available, returning empty embeddings")
			return {"embeddings": np.array([])}

		# Skip empty chunks
		if not chunks:
			logger.debug("No chunks to encode")
			return {"embeddings": np.array([])}

		# Use class method for class cache access
		if self.__class__.get_embedding_model() is None:
			logger.debug("Embedding model is None but was marked as available, reinitializing")
			# Re-check availability using instance method
			self.check_model_availability()

		# Check again after potential re-initialization and assign to local variable
		if self.__class__.get_embedding_model() is None:
			logger.error("Embedding model is still None after re-check")
			return {"embeddings": np.array([])}

		# Explicitly cast after the check
		embedding_model_maybe_none = self.__class__.get_embedding_model()
		if embedding_model_maybe_none is None:
			logger.error("Embedding model unexpectedly None in encode_chunks")
			return {"embeddings": np.array([])}

		embedding_model = embedding_model_maybe_none  # Now we know it's not None

		try:
			logger.debug("Encoding %d chunks", len(chunks))
			embeddings = embedding_model.encode(chunks)
			logger.debug("Successfully encoded %d chunks to shape %s", len(chunks), embeddings.shape)
			return {"embeddings": embeddings}
		except Exception:
			logger.exception("Error encoding chunks")
			return {"embeddings": np.array([])}  # Return empty on error
