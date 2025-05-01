"""
CodeMap processor module.

This module provides functionality for code processing and analysis,
with a focus on generating code structure at different levels of detail
using tree-sitter.

"""

from __future__ import annotations

from pathlib import Path

from codemap.processor.lod import LODEntity, LODGenerator, LODLevel
from codemap.processor.pipeline import ProcessingPipeline

__all__ = [
	"LODEntity",
	"LODGenerator",
	"LODLevel",
	"ProcessingPipeline",
	"create_processor",
]


def create_processor(
	repo_path: str | Path,
	max_workers: int = 4,
	default_lod_level: LODLevel = LODLevel.SIGNATURES,
) -> ProcessingPipeline:
	"""
	Create a processing pipeline for a repository.

	Args:
	    repo_path: Path to the repository root
	    max_workers: Maximum number of worker threads for processing
	    default_lod_level: Default Level of Detail to use for processing

	Returns:
	    A configured ProcessingPipeline instance

	"""
	return ProcessingPipeline(
		repo_path=Path(repo_path),
		max_workers=max_workers,
		default_lod_level=default_lod_level,
	)
