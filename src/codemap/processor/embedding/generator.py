"""Embedding generator for converting code chunks into vector embeddings."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Sequence

import litellm
import numpy as np

from codemap.processor.chunking import Chunk
from codemap.processor.embedding.models import EmbeddingConfig, EmbeddingProvider, EmbeddingResult
from codemap.utils.config_loader import ConfigError, ConfigLoader

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generator for code embeddings using LiteLLM."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the embedding generator.

        Args:
            config: Configuration for the embedding generator. If None, a default
                   configuration is created using environment variables.
        """
        self.config = config or self._create_default_config()
        self._setup_litellm()

    @staticmethod
    def _create_default_config() -> EmbeddingConfig:
        """Create a default configuration based on environment variables and config file.

        Returns:
            A default embedding configuration
        """
        # Get configuration from ConfigLoader if available
        try:
            config_loader = ConfigLoader()
            embedding_config = config_loader.config.get("embedding", {})
        except (ConfigError, FileNotFoundError, OSError) as e:
            logger.warning("Failed to load config, using environment variables: %s", str(e))
            embedding_config = {}

        # Extract provider from config or use default
        provider_str = embedding_config.get("provider", "openai").upper()
        try:
            provider = EmbeddingProvider[provider_str]
        except KeyError:
            logger.warning("Unknown embedding provider '%s', falling back to OPENAI", provider_str)
            provider = EmbeddingProvider.OPENAI

        # Create config with values from environment or config file
        return EmbeddingConfig(
            provider=provider,
            model=embedding_config.get("model", os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")),
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_API_BASE"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
            dimensions=embedding_config.get("dimensions"),
            batch_size=embedding_config.get("batch_size", 32),
        )

    def _setup_litellm(self) -> None:
        """Configure LiteLLM with the current settings."""
        # Configure LiteLLM based on provider
        if self.config.api_key:
            if self.config.provider == EmbeddingProvider.OPENAI:
                os.environ["OPENAI_API_KEY"] = self.config.api_key
            elif self.config.provider == EmbeddingProvider.AZURE:
                os.environ["AZURE_API_KEY"] = self.config.api_key
            # Add other providers as needed

        # Set custom API base if provided
        if self.config.api_base and self.config.provider == EmbeddingProvider.OPENAI:
            os.environ["OPENAI_API_BASE"] = self.config.api_base
            # Add other providers as needed

    def _get_model_string(self) -> str:
        """Get the model string to use with LiteLLM.

        Returns:
            The model string for the configured provider
        """
        model = self.config.model

        # For Azure, prefix with 'azure/'
        if self.config.provider == EmbeddingProvider.AZURE and not model.startswith("azure/"):
            model = f"azure/{model}"

        # For some providers, we might need special prefixing
        provider_prefixes = {
            EmbeddingProvider.COHERE: "cohere/",
            EmbeddingProvider.HUGGINGFACE: "huggingface/",
            EmbeddingProvider.BEDROCK: "bedrock/",
            EmbeddingProvider.VERTEX: "vertex/",
            EmbeddingProvider.MISTRAL: "mistral/",
            EmbeddingProvider.GEMINI: "gemini/",
            EmbeddingProvider.VOYAGE: "voyage/",
        }

        prefix = provider_prefixes.get(self.config.provider, "")
        if prefix and not model.startswith(prefix):
            model = f"{prefix}{model}"

        return model

    async def generate_embeddings_async(
        self,
        chunks: Sequence[Chunk],
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple chunks asynchronously.

        Args:
            chunks: Sequence of chunks to generate embeddings for

        Returns:
            A list of embedding results
        """
        # Process chunks in batches to avoid rate limits
        batch_size = self.config.batch_size
        results = []

        # Group chunks into batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_results = await self._process_batch_async(batch)
            results.extend(batch_results)

        return results

    async def _process_batch_async(self, chunks: Sequence[Chunk]) -> list[EmbeddingResult]:
        """Process a batch of chunks asynchronously.

        Args:
            chunks: Batch of chunks to process

        Returns:
            List of embedding results for the batch
        """
        model = self._get_model_string()
        texts = [chunk.content for chunk in chunks]

        try:
            # Build embedding kwargs
            kwargs: dict[str, Any] = {
                "model": model,
                "input": texts,
            }

            # Add dimensions if specified and supported
            if self.config.dimensions is not None:
                kwargs["dimensions"] = self.config.dimensions

            # Add any additional provider-specific parameters
            kwargs.update(self.config.provider_params)

            # Call LiteLLM embedding API
            response = await litellm.aembedding(**kwargs)

            # Convert response to EmbeddingResult objects
            results = []
            for i, chunk in enumerate(chunks):
                if i < len(response.data):
                    embedding_data = response.data[i].embedding
                    file_path = chunk.metadata.location.file_path if hasattr(chunk, "metadata") else None
                    chunk_id = f"{chunk.metadata.name}_{i}" if hasattr(chunk, "metadata") else f"chunk_{i}"

                    # Safely get token count
                    token_count = 0
                    if (
                        hasattr(response, "usage")
                        and response.usage is not None
                        and hasattr(response.usage, "prompt_tokens")
                    ):
                        token_count = response.usage.prompt_tokens // len(texts)  # Approximate per-text token count

                    # Ensure model name is not None
                    model_name = response.model if response.model is not None else self.config.model

                    result = EmbeddingResult(
                        content=chunk.content,
                        embedding=np.array(embedding_data, dtype=np.float32),
                        tokens=token_count,
                        model_used=model_name,
                        file_path=file_path,
                        chunk_id=chunk_id,
                    )
                    results.append(result)

            return results

        except Exception:
            logger.exception("Error generating embeddings")
            # Return empty embeddings for the chunks
            return []

    def generate_embeddings(self, chunks: Sequence[Chunk]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple chunks.

        Args:
            chunks: Sequence of chunks to generate embeddings for

        Returns:
            A list of embedding results
        """
        return asyncio.run(self.generate_embeddings_async(chunks))

    def generate_embedding(self, chunk: Chunk) -> EmbeddingResult | None:
        """Generate embedding for a single chunk.

        Args:
            chunk: The chunk to generate an embedding for

        Returns:
            An embedding result or None if generation failed
        """
        results = self.generate_embeddings([chunk])
        return results[0] if results else None
