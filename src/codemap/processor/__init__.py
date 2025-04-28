"""
Code processing modules for CodeMap.

This package contains modules for processing and analyzing code:
- chunking: Strategies for breaking code into semantic chunks
- analysis: Tools for analyzing code structure and metadata
- embedding: Tools for generating vector embeddings of code

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codemap.processor.pipeline import ProcessingPipeline

if TYPE_CHECKING:
	from codemap.processor.embedding.models import EmbeddingConfig
	from codemap.processor.storage.base import StorageConfig

logger = logging.getLogger(__name__)

__all__ = ["ProcessingPipeline", "initialize_processor"]


def initialize_processor(
	repo_path: str | Path,
	storage_config: StorageConfig | None = None,
	embedding_config: EmbeddingConfig | None = None,
	enable_lsp: bool = True,
	max_workers: int = 4,
) -> ProcessingPipeline:
	"""
	Initialize the processor module with proper directory structure.

	This is a convenience function that sets up the processing pipeline
	with the appropriate directory structure for storing embeddings and
	vector databases.

	Args:
	    repo_path: Path to the repository to process
	    storage_config: Optional custom storage configuration
	    embedding_config: Optional custom embedding configuration
	    enable_lsp: Whether to enable LSP analysis
	    max_workers: Maximum number of worker threads

	Returns:
	    Initialized processing pipeline

	"""
	# Convert string path to Path object
	if isinstance(repo_path, str):
		repo_path = Path(repo_path)

	# Create and return the processing pipeline
	pipeline = ProcessingPipeline(
		repo_path=repo_path,
		storage_config=storage_config,
		embedding_config=embedding_config,
		enable_lsp=enable_lsp,
		max_workers=max_workers,
	)

	logger.info("Processor module initialized with repository: %s", repo_path)
	return pipeline
