"""
Diff splitting package for CodeMap.

This package provides utilities for splitting Git diffs into logical
chunks.

"""

from codemap.git.diff_splitter.constants import (
	MIN_NAME_LENGTH_FOR_SIMILARITY,
)
from codemap.git.diff_splitter.schemas import DiffChunk, DiffChunkData
from codemap.git.diff_splitter.splitter import DiffSplitter
from codemap.git.diff_splitter.strategies import (
	BaseSplitStrategy,
	FileSplitStrategy,
	SemanticSplitStrategy,
)
from codemap.git.diff_splitter.utils import (
	calculate_semantic_similarity,
	create_chunk_description,
	determine_commit_type,
	filter_valid_files,
	get_language_specific_patterns,
	is_test_environment,
)

__all__ = [
	"MIN_NAME_LENGTH_FOR_SIMILARITY",
	# Strategy Classes
	"BaseSplitStrategy",
	# Classes
	"DiffChunk",
	"DiffChunkData",
	"DiffSplitter",
	"FileSplitStrategy",
	"SemanticSplitStrategy",
	"calculate_semantic_similarity",
	"create_chunk_description",
	"determine_commit_type",
	# Utility Functions
	"filter_valid_files",
	"get_language_specific_patterns",
	"is_test_environment",
]
