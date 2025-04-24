"""Utility functions for storage backends."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from codemap.processor.chunking.base import Chunk, ChunkMetadata, Location
from codemap.processor.embedding.models import EmbeddingResult

logger = logging.getLogger(__name__)


def chunk_to_dict(chunk: Chunk, commit_id: str | None = None) -> dict[str, Any]:
    """Convert a chunk to a dictionary representation.

    Args:
        chunk: The chunk to convert
        commit_id: Optional Git commit ID to associate with the chunk

    Returns:
        Dictionary representation of the chunk
    """
    # Build unique ID that includes file path and entity name to ensure uniqueness
    # If a commit_id is provided, include it as well
    commit_part = f":{commit_id}" if commit_id else ""
    chunk_id = f"{chunk.full_name}{commit_part}"

    # Convert location to dictionary
    location = chunk.metadata.location
    location_dict = {
        "file_path": str(location.file_path),
        "start_line": location.start_line,
        "end_line": location.end_line,
        "start_col": location.start_col,
        "end_col": location.end_col,
    }

    # Prepare additional metadata
    metadata_dict = {
        "entity_type": chunk.metadata.entity_type,
        "name": chunk.metadata.name,
        "language": chunk.metadata.language,
        "description": chunk.metadata.description,
        "dependencies": chunk.metadata.dependencies,
        "attributes": chunk.metadata.attributes,
    }

    # Prepare parent reference if available
    parent_id = None
    if chunk.parent:
        parent_id = chunk.parent.full_name

    # Create the chunk dictionary
    return {
        "id": chunk_id,
        "content": chunk.content,
        "file_path": str(location.file_path),  # Duplicate for easier querying
        "language": chunk.metadata.language,  # Duplicate for easier querying
        "entity_type": chunk.metadata.entity_type,  # Duplicate for easier querying
        "full_name": chunk.full_name,
        "parent_id": parent_id,
        "location": json.dumps(location_dict),
        "metadata": json.dumps(metadata_dict),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_id": commit_id or "",
    }



def dict_to_chunk(data: dict[str, Any]) -> Chunk:
    """Convert a dictionary representation back to a Chunk.

    Args:
        data: Dictionary representation of a chunk

    Returns:
        Reconstructed Chunk object
    """
    # Parse location
    location_dict = json.loads(data["location"]) if isinstance(data["location"], str) else data["location"]

    location = Location(
        file_path=location_dict["file_path"],
        start_line=location_dict["start_line"],
        end_line=location_dict["end_line"],
        start_col=location_dict["start_col"],
        end_col=location_dict["end_col"],
    )

    # Parse metadata
    metadata_dict = json.loads(data["metadata"]) if isinstance(data["metadata"], str) else data["metadata"]

    metadata = ChunkMetadata(
        location=location,
        entity_type=metadata_dict["entity_type"],
        name=metadata_dict["name"],
        language=metadata_dict["language"],
        description=metadata_dict.get("description"),
        dependencies=metadata_dict.get("dependencies", []),
        attributes=metadata_dict.get("attributes", {}),
    )

    # Store parent_id for later reconstruction
    parent_id = data.get("parent_id")
    if parent_id:
        # We can't set the parent directly because Chunk is frozen,
        # so we'll save the ID temporarily
        object.__setattr__(metadata, "_parent_id", parent_id)

    # Recreate the chunk
    chunk = Chunk(
        content=data["content"],
        metadata=metadata,
        children=[],  # Will be reconstructed later
    )

    # Also store the parent ID on the chunk for later use
    if parent_id:
        object.__setattr__(chunk, "_parent_id", parent_id)

    return chunk


def embedding_to_dict(embedding: EmbeddingResult, chunk_id: str) -> dict[str, Any]:
    """Convert an embedding result to a dictionary representation.

    Args:
        embedding: The embedding result to convert
        chunk_id: ID of the associated chunk

    Returns:
        Dictionary representation of the embedding
    """
    # Generate a unique ID for the embedding
    embedding_id = str(uuid.uuid4())

    # Convert embedding to list if it's numpy array
    embedding_vector = embedding.embedding
    if hasattr(embedding_vector, "tolist"):
        embedding_vector = embedding_vector.tolist()

    # Create the embedding dictionary
    return {
        "id": embedding_id,
        "chunk_id": chunk_id,
        "embedding": embedding_vector,
        "model": embedding.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }



def create_pyarrow_schema_for_chunks() -> dict[str, list]:
    """Create a schema for the chunks table using pandas DataFrame.

    Returns:
        Dictionary with empty lists for each column
    """
    # Create an empty schema compatible with LanceDB/pandas
    return {
        "id": [],
        "content": [],
        "file_path": [],
        "language": [],
        "entity_type": [],
        "full_name": [],
        "parent_id": [],
        "location": [],
        "metadata": [],
        "created_at": [],
        "commit_id": [],
        "vector": [],  # For storing vector representation
    }


def create_pyarrow_schema_for_embeddings() -> dict[str, list]:
    """Create a schema for the embeddings table using pandas DataFrame.

    Returns:
        Dictionary with empty lists for each column
    """
    # Create an empty schema compatible with LanceDB/pandas
    return {
        "id": [],
        "chunk_id": [],
        "embedding": [],
        "model": [],
        "created_at": [],
    }
