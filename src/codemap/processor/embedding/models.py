"""Data models for embeddings module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class EmbeddingProvider(Enum):
    """Supported embedding providers."""

    OPENAI = auto()
    AZURE = auto()
    COHERE = auto()
    HUGGINGFACE = auto()
    BEDROCK = auto()  # AWS Bedrock
    VERTEX = auto()  # Google Vertex AI
    MISTRAL = auto()
    GEMINI = auto()
    VOYAGE = auto()
    LITELLM_AUTO = auto()  # Let LiteLLM choose based on the model name


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    """The embedding provider to use."""

    model: str = "text-embedding-3-small"
    """The model to use for embeddings."""

    api_key: str | None = None
    """API key for the provider, if not using environment variables."""

    api_base: str | None = None
    """Base URL for the API, if not using the default."""

    api_version: str | None = None
    """API version, required for some providers like Azure."""

    dimensions: int | None = None
    """Number of dimensions for the embedding, if supported by the model."""

    batch_size: int = 32
    """Maximum number of chunks to process in a single API call."""

    provider_params: dict = field(default_factory=dict)
    """Additional provider-specific parameters to pass to the model."""


@dataclass
class EmbeddingResult:
    """Result of embedding generation for a chunk."""

    content: str
    """The content that was embedded."""

    embedding: np.ndarray
    """The embedding vector."""

    tokens: int
    """Number of tokens in the content."""

    model: str
    """The model used to generate the embedding."""

    file_path: Path | None = None
    """The path to the file containing this chunk."""

    chunk_id: str | None = None
    """Identifier for the chunk."""

    @classmethod
    def from_litellm_response(cls, response: dict, content: str, file_path: Path | None = None) -> EmbeddingResult:
        """Create an EmbeddingResult from a LiteLLM embedding response.

        Args:
            response: The response from LiteLLM's embedding function
            content: The content that was embedded
            file_path: Optional file path associated with the content

        Returns:
            An EmbeddingResult instance

        """
        # Convert LiteLLM embedding response to our internal format
        embedding_data = response["data"][0]["embedding"]
        embedding_vector = np.array(embedding_data, dtype=np.float32)

        return cls(
            content=content,
            embedding=embedding_vector,
            tokens=response["usage"]["prompt_tokens"],
            model=response["model"],
            file_path=file_path,
        )
