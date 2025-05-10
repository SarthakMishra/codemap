"""Utilities for generating text embeddings."""

import logging
import os
from typing import TYPE_CHECKING, Literal, cast

from voyageai.client import Client
from voyageai.client_async import AsyncClient
from voyageai.object.embeddings import EmbeddingsObject

from codemap.utils.cli_utils import progress_indicator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:

	from codemap.config import ConfigLoader

# Create a synchronous client for token counting
_sync_voyage_client = None

# Create an asynchronous client for embedding generation
_async_voyage_client = None

TOKEN_WINDOW_SECONDS = 60


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


def get_voyage_client(config_loader: "ConfigLoader") -> Client:
	"""
	Get or initialize the synchronous VoyageAI client for token counting.

	Returns:
		Client instance for token counting
	"""
	global _sync_voyage_client  # noqa: PLW0603
	if _sync_voyage_client is None:
		try:
			# API key is picked up from environment automatically
			_sync_voyage_client = Client(
				api_key=os.getenv("VOYAGE_API_KEY"),
				max_retries=config_loader.get.embedding.max_retries,
				timeout=config_loader.get.embedding.timeout,
			)
			logger.debug("Initialized synchronous VoyageAI client for token counting")
		except Exception as e:
			message = f"Failed to initialize VoyageAI client: {e}"
			logger.exception(message)
			raise RuntimeError(message) from e
	return _sync_voyage_client


async def get_voyage_async_client(config_loader: "ConfigLoader") -> AsyncClient:
	"""
	Get or initialize the asynchronous VoyageAI client for embedding generation.

	Returns:
		AsyncClient instance for embedding generation
	"""
	global _async_voyage_client  # noqa: PLW0603
	if _async_voyage_client is None:
		try:
			_async_voyage_client = AsyncClient(
				api_key=os.getenv("VOYAGE_API_KEY"),
				max_retries=config_loader.get.embedding.max_retries,
				timeout=config_loader.get.embedding.timeout,
			)
			logger.debug("Initialized asynchronous VoyageAI client for embedding generation")
		except Exception as e:
			message = f"Failed to initialize VoyageAI client: {e}"
			logger.exception(message)
			raise RuntimeError(message) from e
	return _async_voyage_client


def count_tokens(texts: list[str], model: str, config_loader: "ConfigLoader") -> int:
	"""
	Count tokens for a list of texts using the VoyageAI API.

	Args:
		texts: List of text strings to count tokens for
		model: The model name to use for token counting
		config_loader: The config loader to use for token counting
	Returns:
		int: Total token count
	"""
	if not texts:
		return 0

	client = get_voyage_client(config_loader)
	try:
		return client.count_tokens(texts, model=model)
	except (ValueError, TypeError, OSError, KeyError, AttributeError):
		logger.warning("Token counting failed")
		# Fallback - estimate 4 tokens per word
		return sum(len(text.split()) * 4 for text in texts)


async def split_batch(
	texts: list[str], token_limit: int, model: str, max_batch_size: int, config_loader: "ConfigLoader"
) -> list[list[str]]:
	"""
	Split a batch of texts into smaller batches based on token and batch size limits.

	Args:
		texts: List of text strings to split
		token_limit: Maximum token count per batch
		model: Model name for token counting
		max_batch_size: Maximum number of items per batch
		config_loader: The config loader to use for token counting
	Returns:
		list[list[str]]: List of batches, each below the token and batch size limits
	"""
	if not texts:
		return []

	batches = []
	current_batch = []
	current_token_count = 0
	skipped_texts = []  # Track skipped texts and reasons

	for text in texts:
		# Count tokens for this text
		try:
			text_tokens = count_tokens([text], model, config_loader)
		except (ValueError, OSError, TimeoutError, ConnectionError):
			# If token counting fails, estimate based on text length
			text_tokens = len(text.split()) * 4  # Rough estimate

		# Check if this single text exceeds the token limit
		if text_tokens > token_limit:
			logger.warning(f"Text exceeds token limit ({text_tokens} > {token_limit}). Skipping.")
			skipped_texts.append((text, text_tokens))
			continue

		# If adding this text would exceed the limit, or batch size is reached, finalize the current batch
		if current_batch and (current_token_count + text_tokens > token_limit or len(current_batch) >= max_batch_size):
			batches.append(current_batch)
			current_batch = []
			current_token_count = 0

		# Add the text to the current batch
		current_batch.append(text)
		current_token_count += text_tokens

	# Add the last batch if it's not empty
	if current_batch:
		batches.append(current_batch)

	if skipped_texts:
		logger.warning(
			f"Skipped {len(skipped_texts)} texts that exceeded the token limit. "
			f"Example: {skipped_texts[0] if skipped_texts else ''}"
		)

	return batches


async def generate_embeddings_batch(
	texts: list[str],
	truncation: bool = True,
	output_dimension: Literal[256, 512, 1024, 2048] = 1024,
	model: str | None = None,
	config_loader: "ConfigLoader | None" = None,
) -> list[list[float]] | None:
	"""
	Generates embeddings for a batch of texts using the Voyage AI client.

	Handles per-call and per-minute token limits using a rolling window.
	Optimizes throughput by batching requests asynchronously up to the per-minute token limit.

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
	embedding_model = model or embedding_config.model_name
	per_call_token_limit = embedding_config.per_call_token_limit
	per_call_batch_size = getattr(embedding_config, "per_call_batch_size", 32)

	# Ensure VOYAGE_API_KEY is available
	if "voyage" in embedding_model and "VOYAGE_API_KEY" not in os.environ:
		logger.error("VOYAGE_API_KEY environment variable not set, but required for model '%s'", embedding_model)
		return None

	client = await get_voyage_async_client(config_loader)
	logger.info("Initialized Voyage AI async client for batch embeddings")

	with progress_indicator("Splitting texts into batches..."):
		# Split into batches based on per-call token and batch size limits
		all_batches = await split_batch(
			texts, per_call_token_limit, embedding_model, per_call_batch_size, config_loader
		)

	logger.info(
		f"Split {len(texts)} texts into {len(all_batches)} batches based on"
		f"per_call_token_limit ({per_call_token_limit}) and per_call_batch_size ({per_call_batch_size})"
	)
	with progress_indicator("Counting tokens for each batch..."):
		batch_token_counts = [count_tokens(batch, embedding_model, config_loader) for batch in all_batches]

	batch_indices = []  # Track the indices of texts in each batch
	idx = 0
	for batch in all_batches:
		batch_indices.append(list(range(idx, idx + len(batch))))
		idx += len(batch)

	# Log the lengths before zipping
	logger.debug(
		f"All batches: "
		f"all_batches={len(all_batches)}, "
		f"batch_token_counts={len(batch_token_counts)}, "
		f"batch_indices={len(batch_indices)}"
	)
	if not (len(all_batches) == len(batch_token_counts) == len(batch_indices)):
		msg = (
			f"Length mismatch: "
			f"all_batches={len(all_batches)}, "
			f"batch_token_counts={len(batch_token_counts)}, "
			f"batch_indices={len(batch_indices)}"
		)
		logger.error(msg)
		raise ValueError(msg)

	set_dimension_none: int | None = output_dimension

	# Patch for voyage SDK - dimension must be set to None for models with fixed dimensions
	if embedding_model not in ("voyage-code-3", "voyage-3"):
		set_dimension_none = None

	async def send_batch(
		batch: list[str], batch_token_count: int, batch_idx: list[int]
	) -> tuple[list[int], EmbeddingsObject, int]:
		embeddings = await client.embed(
			texts=batch,
			model=embedding_model,
			output_dimension=set_dimension_none,
			truncation=truncation,
		)
		return batch_idx, embeddings, batch_token_count

	pending_batches = list(zip(all_batches, batch_token_counts, batch_indices, strict=True))
	results = [[] for _ in range(len(all_batches))]
	i = 0

	with progress_indicator(
		"Processing embedding batches...", style="progress", total=len(all_batches)
	) as update_progress:
		while i < len(pending_batches):
			batch, batch_token_count, batch_idx = pending_batches[i]
			batch_idx, embeddings, _ = await send_batch(batch, batch_token_count, batch_idx)
			results[batch_indices.index(batch_idx)] = embeddings.embeddings
			i += 1
			update_progress("Processing embedding batches...", i, len(all_batches))

	# Flatten results and assign to all_embeddings
	flat_embeddings = []
	for batch_embeds in results:
		if batch_embeds:
			flat_embeddings.extend(batch_embeds)
	if len(flat_embeddings) != len(texts):
		logger.error(f"Embedding count mismatch: got {len(flat_embeddings)}, expected {len(texts)}")
		return None
	return cast("list[list[float]]", flat_embeddings)


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
