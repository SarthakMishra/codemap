"""
Diff splitting package for CodeMap.

This package provides utilities for splitting Git diffs into logical
chunks.

"""

from codemap.git.diff_splitter.constants import (
	DEFAULT_CODE_EXTENSIONS,
	DEFAULT_SIMILARITY_THRESHOLD,
	DIRECTORY_SIMILARITY_THRESHOLD,
	MAX_CHUNKS_BEFORE_CONSOLIDATION,
	MIN_CHUNKS_FOR_CONSOLIDATION,
	MIN_NAME_LENGTH_FOR_SIMILARITY,
	MODEL_NAME,
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
	extract_code_from_diff,
	filter_valid_files,
	get_language_specific_patterns,
	is_test_environment,
)

__all__ = [
	"DEFAULT_CODE_EXTENSIONS",
	# Constants
	"DEFAULT_SIMILARITY_THRESHOLD",
	"DIRECTORY_SIMILARITY_THRESHOLD",
	"MAX_CHUNKS_BEFORE_CONSOLIDATION",
	"MIN_CHUNKS_FOR_CONSOLIDATION",
	"MIN_NAME_LENGTH_FOR_SIMILARITY",
	"MODEL_NAME",
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
	"extract_code_from_diff",
	"filter_valid_files",
	"get_language_specific_patterns",
	"is_test_environment",
]
