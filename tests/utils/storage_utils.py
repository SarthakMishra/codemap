"""Utility functions for storage-related tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codemap.processor.analysis.tree_sitter.base import EntityType
from codemap.processor.chunking.base import Chunk, ChunkMetadata, Location
from codemap.processor.embedding.models import EmbeddingResult


def create_test_chunk(
    content: str = "def test_function():\n    return True",
    file_path: str = "test_file.py",
    start_line: int = 1,
    end_line: int = 2,
    language: str = "python",
    entity_type: EntityType = EntityType.FUNCTION,
    entity_name: str = "test_function",
) -> Chunk:
    """Create a test chunk for use in tests.

    Args:
        content: The content of the chunk
        file_path: The file path
        start_line: The start line
        end_line: The end line
        language: The language
        entity_type: The entity type (an EntityType enum value)
        entity_name: The entity name

    Returns:
        A chunk instance for testing

    """
    # Create location object
    location = Location(file_path=Path(file_path), start_line=start_line, end_line=end_line)

    # Create metadata object
    metadata = ChunkMetadata(entity_type=entity_type, name=entity_name, location=location, language=language)

    # Create and return chunk
    return Chunk(content=content, metadata=metadata)


def create_test_embedding(
    content: str = "def test_function():\n    return True",
    embedding_values: list[float] | None = None,
    chunk_id: str | None = "test-chunk-id",
    model: str = "test-model",
    tokens: int = 10,
    file_path: str | None = "test_file.py",
) -> EmbeddingResult:
    """Create a test embedding result for use in tests.

    Args:
        content: The content that was embedded
        embedding_values: The embedding vector values
        chunk_id: The chunk ID
        model: The model used to generate the embedding
        tokens: Number of tokens in the content
        file_path: The file path

    Returns:
        An EmbeddingResult instance for testing

    """
    if embedding_values is None:
        embedding_values = [0.1, 0.2, 0.3, 0.4]
    embedding_vector = np.array(embedding_values, dtype=np.float32)
    file_path_obj = Path(file_path) if file_path else None

    return EmbeddingResult(
        content=content,
        embedding=embedding_vector,
        tokens=tokens,
        model=model,
        chunk_id=chunk_id,
        file_path=file_path_obj,
    )
