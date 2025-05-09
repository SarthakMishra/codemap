"""Utilities for generating text embeddings."""

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Literal, cast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
	from voyageai.client import Client
	from voyageai.client_async import AsyncClient

	from codemap.config import ConfigLoader

# Create a synchronous client for token counting
_sync_voyage_client = None


def get_retry_settings(config_loader: "ConfigLoader") -> tuple[int, int]:
	"""Get retry settings from config."""
	embedding_config = config_loader.get.embedding
	# Use max_retries directly for voyageai.Client
	max_retries = embedding_config.max_retries
	# retry_delay is handled internally by voyageai client's exponential backoff
	# We can still keep the config value if needed elsewhere, but timeout is more relevant here.
	# Increased default timeout
	timeout = embedding_config.timeout
	return max_retries, timeout


def get_voyage_client() -> "Client":
	"""
	Get or initialize the synchronous VoyageAI client for token counting.

	Returns:
		Client instance for token counting
	"""
	global _sync_voyage_client  # noqa: PLW0603
	if _sync_voyage_client is None:
		try:
			# Import lazily - only when needed
			from voyageai.client import Client

			# API key is picked up from environment automatically
			_sync_voyage_client = Client()
			logger.debug("Initialized synchronous VoyageAI client for token counting")
		except Exception as e:
			message = f"Failed to initialize VoyageAI client: {e}"
			logger.exception(message)
			raise RuntimeError(message) from e
	return _sync_voyage_client


def count_tokens(texts: list[str], model: str) -> int:
	"""
	Count tokens for a list of texts using the VoyageAI API.

	Args:
		texts: List of text strings to count tokens for
		model: The model name to use for token counting

	Returns:
		int: Total token count
	"""
	if not texts:
		return 0

	client = get_voyage_client()
	try:
		return client.count_tokens(texts, model=model)
	except (ValueError, TypeError, OSError, KeyError, AttributeError):
		logger.warning("Token counting failed")
		# Fallback - estimate 4 tokens per word
		return sum(len(text.split()) * 4 for text in texts)


def split_batch(texts: list[str], token_limit: int, model: str) -> list[list[str]]:
	"""
	Split a batch of texts into smaller batches based on token limits.

	Args:
		texts: List of text strings to split
		token_limit: Maximum token count per batch
		model: Model name for token counting

	Returns:
		list[list[str]]: List of batches, each below the token limit
	"""
	if not texts:
		return []

	batches = []
	current_batch = []
	current_token_count = 0

	for text in texts:
		# Count tokens for this text
		try:
			text_tokens = count_tokens([text], model)
		except (ValueError, OSError, TimeoutError, ConnectionError):
			# If token counting fails, estimate based on text length
			text_tokens = len(text.split()) * 4  # Rough estimate

		# Check if this single text exceeds the token limit
		if text_tokens > token_limit:
			logger.warning(f"Text exceeds token limit ({text_tokens} > {token_limit}). Truncating.")
			# Handle the oversized text somehow - truncate or skip
			continue

		# If adding this text would exceed the limit, finalize the current batch
		if current_batch and current_token_count + text_tokens > token_limit:
			batches.append(current_batch)
			current_batch = []
			current_token_count = 0

		# Add the text to the current batch
		current_batch.append(text)
		current_token_count += text_tokens

	# Add the last batch if it's not empty
	if current_batch:
		batches.append(current_batch)

	return batches


async def process_batch_with_backoff(
	client: "AsyncClient",
	texts: list[str],
	model: str,
	output_dimension: Literal[256, 512, 1024, 2048],
	truncation: bool,
	max_retries: int,
	base_delay: float = 1.0,
) -> list[list[float]]:
	"""
	Process a batch with exponential backoff for rate limits.

	Args:
		client: VoyageAI async client
		texts: Texts to embed
		model: Embedding model name
		output_dimension: Embedding dimension
		truncation: Whether to truncate texts
		max_retries: Maximum number of retries
		base_delay: Base delay for exponential backoff

	Returns:
		list[list[float]]: List of embeddings
	"""
	retries = 0
	while True:
		try:
			result = await client.embed(
				texts=texts,
				model=model,
				output_dimension=output_dimension,
				truncation=truncation,
			)
			# Convert any integer lists to float lists
			return [[float(val) for val in emb] for emb in result.embeddings]
		except (ValueError, OSError, TimeoutError, ConnectionError) as e:
			# Check if it's a rate limit error by examining the error message
			is_rate_limit = any(term in str(e).lower() for term in ["rate limit", "too many requests", "429"])

			if is_rate_limit and retries < max_retries:
				delay = base_delay * (2**retries)  # Exponential backoff
				logger.warning(f"Rate limit hit. Retrying in {delay:.2f}s (attempt {retries + 1}/{max_retries})")
				await asyncio.sleep(delay)
				retries += 1
			elif retries < max_retries:
				# For other errors, try a few times with backoff as well
				delay = base_delay * (2**retries)
				logger.warning(f"API error: {e}. Retrying in {delay:.2f}s (attempt {retries + 1}/{max_retries})")
				await asyncio.sleep(delay)
				retries += 1
			else:
				logger.exception(f"Error after {retries} retries")
				raise


async def generate_embeddings_batch(
	texts: list[str],
	truncation: bool = True,
	output_dimension: Literal[256, 512, 1024, 2048] = 1024,
	model: str | None = None,
	config_loader: "ConfigLoader | None" = None,
) -> list[list[float]] | None:
	"""
	Generates embeddings for a batch of texts using the Voyage AI async client.

	Automatically handles token limits, batch splitting, and rate limiting.

	Args:
	    texts (List[str]): A list of text strings to embed.
	    truncation (bool): Whether to truncate the texts.
	    output_dimension (Literal[256, 512, 1024, 2048]): The dimension of the output embeddings.
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
		from codemap.config import ConfigLoader

		config_loader = ConfigLoader.get_instance()

	embedding_config = config_loader.get.embedding

	# Use model from parameter or fallback to config
	embedding_model = model or embedding_config.model_name

	# Get token limit and batch settings
	token_limit = embedding_config.token_limit
	max_retries, timeout = get_retry_settings(config_loader)

	# Ensure VOYAGE_API_KEY is available
	if "voyage" in embedding_model and "VOYAGE_API_KEY" not in os.environ:
		logger.error("VOYAGE_API_KEY environment variable not set, but required for model '%s'", embedding_model)
		return None

	# Import AsyncClient lazily
	from voyageai.client_async import AsyncClient

	# Initialize the async client with retry settings
	client = AsyncClient(max_retries=max_retries, timeout=timeout)
	logger.info("Initialized Voyage AI async client for batch embeddings")

	try:
		# Split into batches based on token limit
		all_batches = split_batch(texts, token_limit, embedding_model)
		logger.info(f"Split {len(texts)} texts into {len(all_batches)} batches based on token limit")

		# Process each batch with backoff and combine results
		all_embeddings = []
		for i, batch in enumerate(all_batches):
			logger.info(f"Processing batch {i + 1}/{len(all_batches)} with {len(batch)} texts")
			batch_embeddings = await process_batch_with_backoff(
				client=client,
				texts=batch,
				model=embedding_model,
				output_dimension=output_dimension,
				truncation=truncation,
				max_retries=max_retries,
			)
			all_embeddings.extend(batch_embeddings)

		# Verify we got the right number of embeddings
		if len(all_embeddings) != len(texts):
			logger.error(f"Embedding count mismatch: got {len(all_embeddings)}, expected {len(texts)}")
			return None

		return cast("list[list[float]]", all_embeddings)

	except (ValueError, OSError, TimeoutError, ConnectionError, KeyError):
		logger.exception("Error during embedding generation")
		return None


async def generate_embedding(
	text: str, model: str | None = None, config_loader: "ConfigLoader | None" = None
) -> list[float] | None:
	"""
	Generates an embedding for a single text string.

	Args:
	    text (str): The text string to embed.
	    model (str): The embedding model to use (defaults to config value).
	    config_loader (ConfigLoader, optional): Configuration loader instance. Defaults to None.

	Returns:
	    Optional[List[float]]: The embedding vector, or None if embedding fails.

	"""
	if not text.strip():
		logger.warning("generate_embedding called with empty or whitespace-only text.")
		return None  # Return None for empty or whitespace-only strings

	# Create ConfigLoader if not provided
	if config_loader is None:
		from codemap.config import ConfigLoader

		config_loader = ConfigLoader.get_instance()

	embedding_config = config_loader.get.embedding
	model_name = model or embedding_config.model_name

	# Call generate_embeddings_batch with a single text
	embeddings_list = await generate_embeddings_batch(
		texts=[text],
		model=model_name,
		config_loader=config_loader,  # Pass the potentially newly created config_loader
		# output_dimension and truncation will use defaults from generate_embeddings_batch
	)

	if embeddings_list and embeddings_list[0]:
		return embeddings_list[0]

	return None
