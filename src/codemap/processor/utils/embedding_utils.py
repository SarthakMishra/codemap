"""Utilities for generating text embeddings."""

import logging
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_fixed

from codemap.config import DEFAULT_CONFIG

if TYPE_CHECKING:
	import numpy as np

logger = logging.getLogger(__name__)

# Configuration from DEFAULT_CONFIG
MODEL_NAME = DEFAULT_CONFIG["embedding"]["model_name"]
EMBEDDING_DIMENSION = DEFAULT_CONFIG["embedding"]["dimension"]
MAX_EMBEDDING_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 2

# Global variable to hold the loaded model (lazy loading)
_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
	"""Loads and returns the SentenceTransformer model (singleton)."""
	global _embedding_model  # noqa: PLW0603
	if _embedding_model is None:
		logger.info("Loading sentence transformer model: %s...", MODEL_NAME)
		try:
			_embedding_model = SentenceTransformer(MODEL_NAME)
			# Break the long line into multiple lines
			model_dimension = _embedding_model.get_sentence_embedding_dimension()
			logger.info("Sentence transformer model loaded successfully with dimension %d", model_dimension)
		except Exception:
			logger.exception("Failed to load sentence transformer model: %s", MODEL_NAME)
			# Re-raise to prevent returning None, as the model is essential
			raise
	return _embedding_model


@retry(stop=stop_after_attempt(MAX_EMBEDDING_ATTEMPTS), wait=wait_fixed(RETRY_WAIT_SECONDS))
def generate_embedding(text: str) -> list[float] | None:
	"""
	Generates an embedding for the given text using the configured model.

	Args:
	    text (str): The text to embed.

	Returns:
	    Optional[List[float]]: The embedding vector as a list of floats,
	                           or None if embedding fails after retries.

	"""
	if not text or not text.strip():
		logger.warning("Attempted to generate embedding for empty or whitespace-only text.")
		return None

	try:
		model = _get_embedding_model()
		logger.debug(f"Generating embedding for text (length: {len(text)}, preview: '{text[:100]}...')")
		# Encode returns a numpy array
		embedding_tensor = model.encode(text, convert_to_tensor=True)
		embedding_array: np.ndarray = embedding_tensor.cpu().numpy()

		# Check dimension
		embedding_dimension = embedding_array.shape[0]
		# Declare global before using it
		global EMBEDDING_DIMENSION  # noqa: PLW0603
		if embedding_dimension != EMBEDDING_DIMENSION:
			logger.warning(
				f"Embedding dimension mismatch. Expected {EMBEDDING_DIMENSION}, got {embedding_dimension}. "
				f"Updating EMBEDDING_DIMENSION to match the model."
			)
			EMBEDDING_DIMENSION = embedding_dimension

		# Convert numpy array to list of floats
		embedding_list = embedding_array.tolist()
		logger.debug(f"Successfully generated embedding with dimension {len(embedding_list)}")
		return embedding_list

	except Exception:  # No need to capture the exception as 'e' if not used
		# Logger exception handles the traceback
		logger.exception("Failed to generate embedding for text")
		# Return None here; the @retry decorator will handle retries
		return None


# Example Usage (can be removed later)
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	test_text = "def example_function(param1: int) -> str:"
	embedding = generate_embedding(test_text)
	if embedding:
		pass
	else:
		pass

	empty_embedding = generate_embedding("")
