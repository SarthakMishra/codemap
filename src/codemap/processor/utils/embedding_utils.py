"""Utilities for generating text embeddings."""

import logging
import os
from typing import cast

import voyageai

from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Define retryable exceptions for voyageai (based on potential errors)
# Note: voyageai.Client handles retries internally based on max_retries parameter
# We will rely on the client's retry mechanism.


def get_retry_settings(config_loader: ConfigLoader) -> tuple[int, int]:
	"""Get retry settings from config."""
	embedding_config = config_loader.get("embedding", {})
	# Use max_retries directly for voyageai.Client
	max_retries = embedding_config.get("max_retries", 3)
	# retry_delay is handled internally by voyageai client's exponential backoff
	# We can still keep the config value if needed elsewhere, but timeout is more relevant here.
	# Increased default timeout
	timeout = embedding_config.get("timeout", 180)  # Default timeout for requests (increased from 60)
	return max_retries, timeout


async def generate_embeddings_batch(
	texts: list[str], model: str | None = None, config_loader: ConfigLoader | None = None
) -> list[list[float]] | None:
	"""
	Generates embeddings for a batch of texts using the Voyage AI async client.

	Args:
	    texts (List[str]): A list of text strings to embed.
	    model (str): The embedding model to use (defaults to config value).
	    config_loader: Configuration loader instance.

	Returns:
	    Optional[List[List[float]]]: A list of embedding vectors,
	                                 or None if embedding fails after retries.

	"""
	if not texts:
		logger.warning("generate_embeddings_batch called with empty list.")
		return []

	# Create ConfigLoader if not provided
	if config_loader is None:
		config_loader = ConfigLoader()

	embedding_config = config_loader.get("embedding", {})

	# Use model from parameter or fallback to config
	embedding_model = model or embedding_config.get("model_name", "voyage-code-3")

	# Get retry and timeout settings from config
	max_retries, timeout = get_retry_settings(config_loader)

	# Ensure VOYAGE_API_KEY is available (voyageai client checks this, but explicit check is good)
	if "voyage" in embedding_model and "VOYAGE_API_KEY" not in os.environ:
		logger.error("VOYAGE_API_KEY environment variable not set, but required for model '%s'", embedding_model)
		return None

	try:
		logger.info(f"Generating embeddings for {len(texts)} texts using model: {embedding_model} via voyageai client")

		# Instantiate the async client with retry and timeout settings
		# API key is automatically picked up from VOYAGE_API_KEY env var by default
		# Explicitly reference voyageai.AsyncClient
		# type: ignore because linter incorrectly flags AsyncClient as not exported
		vo = voyageai.AsyncClient(max_retries=max_retries, timeout=timeout)  # type: ignore[arg-type]

		# Call the voyageai embed method
		# Use keyword argument 'texts=' for clarity and future compatibility
		result = await vo.embed(texts=texts, model=embedding_model)

		# Check response structure (based on typical patterns, might need adjustment)
		# Assuming result has an 'embeddings' attribute which is a list of lists of floats
		if result and hasattr(result, "embeddings") and isinstance(result.embeddings, list):
			embeddings = result.embeddings
			if len(embeddings) == len(texts):
				# Check if embeddings are valid lists of floats
				if all(isinstance(emb, list) and all(isinstance(x, float) for x in emb) for emb in embeddings):
					logger.debug(f"Successfully generated {len(embeddings)} embeddings.")
					# Use cast to assure the type checker
					return cast("list[list[float]]", embeddings)
				logger.error("Generated embeddings list contains non-float or non-list items.")
				return None
			logger.error(
				"Mismatch between input texts (%d) and generated embeddings (%d).", len(texts), len(embeddings)
			)
			return None  # Indicate partial failure
		logger.error("Unexpected response structure from voyageai.embed: %s", result)
		return None  # Indicate unexpected response

	except Exception:
		# Catch specific Voyage AI errors (includes API errors, rate limits, etc.)
		# Catch any unexpected errors during the API call
		# Use logger.exception without redundant variable (per TRY401)
		logger.exception("Error during voyageai embedding generation")
		return None


async def generate_embedding(
	text: str, model: str | None = None, config_loader: ConfigLoader | None = None
) -> list[float] | None:
	"""
	Generates an embedding for a single text using the Voyage AI client.

	Args:
	    text (str): The text to embed.
	    model (str): The embedding model to use.
	    config_loader: Configuration loader instance.

	Returns:
	    Optional[List[float]]: The embedding vector, or None if embedding fails.

	"""
	if not text:
		logger.warning("generate_embedding called with empty string.")
		return None

	# Await the async batch function (now using voyageai client)
	embeddings_batch = await generate_embeddings_batch(texts=[text], model=model, config_loader=config_loader)

	if embeddings_batch and len(embeddings_batch) == 1:
		return embeddings_batch[0]

	# Error logging is now handled within generate_embeddings_batch
	logger.error("Failed to generate embedding for single text using voyageai client.")
	return None


# Example Usage (remains the same, but now uses voyageai backend)
# import asyncio
#
# async def main():
#     # Ensure VOYAGE_API_KEY is set in your environment for this example
#     texts_to_embed = [
#         "This is the first document.",
#         "This document is the second document.",
#         "And this is the third one.",
#         "Is this the first document?",
#     ]
#     embeddings = await generate_embeddings_batch(texts_to_embed)
#
#     if embeddings:
#         print(f"Generated {len(embeddings)} embeddings.")
#         for i, emb in enumerate(embeddings):
#             print(f"Embedding {i+1} (first 5 dims): {emb[:5]}...")
#             print(f"Embedding dimension: {len(emb)}")
#     else:
#         print("Failed to generate embeddings.")
#
#     single_embedding = await generate_embedding("A single piece of text.")
#     if single_embedding:
#         print(f"Single embedding (first 5 dims): {single_embedding[:5]}...")
#         print(f"Single embedding dimension: {len(single_embedding)}")
#     else:
#         print("Failed to generate single embedding.")
#
# if __name__ == "__main__":
#    asyncio.run(main())
