"""Embedding module for generating and storing vector embeddings of code
chunks."""

from codemap.processor.embedding.generator import EmbeddingGenerator
from codemap.processor.embedding.models import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingGenerator",
    "EmbeddingProvider",
    "EmbeddingResult",
]
