"""Handles the generation of embeddings for text chunks."""

import logging

import numpy as np

# Import torch for device selection
import torch
from sentence_transformers import SentenceTransformer

from . import config

logger = logging.getLogger(__name__)

# Module-level variable for the singleton model
_embedding_model: SentenceTransformer | None = None


def _initialize_embedding_model() -> SentenceTransformer | None:
	"""Loads and returns the SentenceTransformer model."""
	model_name = config.EMBEDDING_MODEL_NAME
	try:
		# Determine device (CPU or CUDA)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		logger.info(f"Loading embedding model '{model_name}' on device: {device}")
		# trust_remote_code=True might be needed for some models
		model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
		logger.info(f"Embedding model '{model_name}' loaded successfully.")
		return model
	except Exception:
		logger.exception(f"Failed to load embedding model '{model_name}'")
		return None


def _get_embedding_model() -> SentenceTransformer | None:
	"""Loads and returns the SentenceTransformer model (singleton)."""
	global _embedding_model  # noqa: PLW0603
	if _embedding_model is None:
		logger.debug("Embedding model is None, attempting initialization.")
		_embedding_model = _initialize_embedding_model()
		# Check if initialization succeeded
		if _embedding_model:
			logger.debug("Embedding model initialized successfully.")
		else:
			logger.error("Failed to initialize embedding model.")
	return _embedding_model


def generate_embeddings(texts: str | list[str]) -> np.ndarray | list[np.ndarray] | None:
	"""
	Generates embeddings for a single text or a list of texts.

	Args:
	    texts (Union[str, List[str]]): A single text string or a list of text strings.

	Returns:
	    Union[np.ndarray, List[np.ndarray], None]:
	        - A single numpy array if input is a string.
	        - A list of numpy arrays if input is a list of strings.
	        - None if embedding generation fails.

	"""
	model = _get_embedding_model()
	if model is None:
		logger.error("Embedding model not available. Cannot generate embeddings.")
		return None

	try:
		logger.debug(f"Generating embeddings for {len(texts) if isinstance(texts, list) else 1} text(s)...")
		# SentenceTransformer handles both single string and list of strings
		embeddings = model.encode(texts, show_progress_bar=False)  # Adjust progress bar based on verbosity?
		logger.debug("Embeddings generated successfully.")
		return embeddings
	except Exception:
		logger.exception("Error during embedding generation")
		return None


# Example Usage
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	test_texts = [
		"This is the first test sentence.",
		"Here is another sentence for testing embedding.",
		'def hello_world():\n    print("Hello, world!")',
	]

	single_text = "Just one sentence."

	logger.info("Testing single embedding generation...")
	single_embedding = generate_embeddings(single_text)
	if single_embedding is not None:
		# Check if it's a single embedding (numpy array)
		if isinstance(single_embedding, np.ndarray):
			logger.info(f"Single embedding shape: {single_embedding.shape}")
		else:
			logger.warning(f"Expected single embedding to be ndarray, got {type(single_embedding)}")
	else:
		logger.error("Single embedding generation failed.")

	logger.info("\nTesting batch embedding generation...")
	batch_embeddings = generate_embeddings(test_texts)
	if batch_embeddings is not None:
		# Check if it's a batch of embeddings (numpy array for SentenceTransformer)
		if isinstance(batch_embeddings, np.ndarray):
			logger.info(f"Batch embeddings shape: {batch_embeddings.shape}")
		# Note: Some SentenceTransformer versions might return List[np.ndarray]
		# elif isinstance(batch_embeddings, list) and all(isinstance(e, np.ndarray) for e in batch_embeddings):
		#     logger.info(f"Batch embeddings count: {len(batch_embeddings)}, "
		#                 f"first shape: {batch_embeddings[0].shape if batch_embeddings else 'N/A'}")
		else:
			logger.warning(f"Expected batch embeddings to be ndarray, got {type(batch_embeddings)}")
	else:
		logger.error("Batch embedding generation failed.")
