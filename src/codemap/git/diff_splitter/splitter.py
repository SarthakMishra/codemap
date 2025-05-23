"""Diff splitting implementation for CodeMap."""

import logging
from typing import TYPE_CHECKING

from codemap.git.diff_splitter.strategies import FileSplitStrategy, SemanticSplitStrategy
from codemap.git.diff_splitter.utils import filter_valid_files, is_test_environment
from codemap.git.utils import ExtendedGitRepoContext, GitDiff

from .schemas import DiffChunk

if TYPE_CHECKING:
	from codemap.config import ConfigLoader

logger = logging.getLogger(__name__)

# Constants for truncation and sampling
MAX_DIFF_CONTENT_LENGTH = 100000  # ~100KB maximum size for diff content
MAX_DIFF_LINES = 1000  # Maximum number of lines to process
SMALL_SECTION_SIZE = 50  # Maximum size for a "small" diff section
COMPLEX_SECTION_SIZE = 100  # Minimum size for a "complex" diff section (with middle sample)


class DiffSplitter:
	"""Splits Git diffs into logical chunks."""

	def __init__(
		self,
		config_loader: "ConfigLoader | None" = None,
	) -> None:
		"""
		Initialize the diff splitter.

		Args:
		    config_loader: ConfigLoader object for loading configuration
		"""
		if config_loader:
			self.config_loader = config_loader
		else:
			from codemap.config import ConfigLoader  # Import locally

			self.config_loader = ConfigLoader.get_instance()

		if self.config_loader.get.repo_root is None:
			self.repo_root = ExtendedGitRepoContext.get_repo_root()
		else:
			self.repo_root = self.config_loader.get.repo_root

		# Get config for diff_splitter, fallback to empty dict if not found
		ds_config = self.config_loader.get.commit.diff_splitter

		# Determine parameters: CLI/direct arg > Config file > DEFAULT_CONFIG
		self.similarity_threshold = ds_config.similarity_threshold
		self.directory_similarity_threshold = ds_config.directory_similarity_threshold
		self.min_chunks_for_consolidation = ds_config.min_chunks_for_consolidation
		self.max_chunks_before_consolidation = ds_config.max_chunks_before_consolidation
		self.max_file_size_for_llm = ds_config.max_file_size_for_llm
		self.max_log_diff_size = ds_config.max_log_diff_size

	async def split_diff(self, diff: GitDiff) -> tuple[list[DiffChunk], list[str]]:
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

		try:
			semantic_strategy = SemanticSplitStrategy(config_loader=self.config_loader)
			chunks = await semantic_strategy.split(diff)

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
						if isinstance(diff.content, str) and file_marker in diff.content:
							# Found a match for this file in the original content
							# Extract that file's complete diff section
							start_idx = diff.content.find(file_marker)
							end_idx = diff.content.find("diff --git", start_idx + len(file_marker))
							if end_idx == -1:  # Last file in the diff
								end_idx = len(diff.content)

							file_diff = diff.content[start_idx:end_idx].strip()

							# Now replace just this file's content in the chunk
							# This is a heuristic that may need adjustment based on your diff format
							if chunk.content and isinstance(chunk.content, str) and file_marker in chunk.content:
								chunk_start = chunk.content.find(file_marker)
								chunk_end = chunk.content.find("diff --git", chunk_start + len(file_marker))
								if chunk_end == -1:  # Last file in the chunk
									chunk_end = len(chunk.content)

								# Replace this file's truncated diff with the full diff
								chunk.content = chunk.content[:chunk_start] + file_diff + chunk.content[chunk_end:]

			return chunks, []
		except Exception:
			logger.exception("Semantic splitting failed")

			# Try basic splitting as a fallback
			logger.warning("Falling back to basic file splitting")
			# Return empty list for filtered_large_files as it's no longer tracked here
			return await self._create_basic_file_chunk(diff), []

	async def _create_basic_file_chunk(self, diff: GitDiff) -> list[DiffChunk]:
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
			chunks = await strategy.split(diff)

		return chunks
