"""Utilities for generating text embeddings."""

import logging
import signal
import time
from functools import lru_cache
from types import FrameType
from typing import TYPE_CHECKING

from codemap.utils.cli_utils import progress_indicator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
	from codemap.config import ConfigLoader

# Global cache for loaded models
_model_cache: dict[str, object] = {}


class ModelLoadTimeoutError(Exception):
	"""Raised when model loading times out."""


def _timeout_handler(_signum: int, _frame: FrameType | None) -> None:
	"""Handle timeout signal."""
	msg = "Model loading timed out"
	raise ModelLoadTimeoutError(msg)


@lru_cache(maxsize=1)
def _get_cached_model(model_name: str) -> object:
	"""
	Load and cache a model with timeout protection.

	Args:
		model_name: Name of the model to load

	Returns:
		Loaded model instance

	Raises:
		ModelLoadTimeoutError: If model loading times out
		ImportError: If model2vec is not available
	"""
	if model_name in _model_cache:
		logger.info(f"Using cached model: {model_name}")
		return _model_cache[model_name]

	try:
		from model2vec import StaticModel
	except ImportError as e:
		logger.exception("model2vec package not found. Please install it with: uv add model2vec")
		msg = "model2vec is required for embedding generation"
		raise ImportError(msg) from e

	# Set up timeout (5 minutes for model download)
	timeout_seconds = 300

	# Set up signal handler for timeout (Unix-like systems only)
	if hasattr(signal, "SIGALRM"):
		old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
		signal.alarm(timeout_seconds)

	try:
		logger.info(f"Loading model: {model_name} (timeout: {timeout_seconds}s)")
		start_time = time.time()

		# Try to load the model
		model = StaticModel.from_pretrained(model_name)

		load_time = time.time() - start_time
		logger.info(f"Model loaded successfully in {load_time:.2f}s")

		# Cache the model
		_model_cache[model_name] = model
		return model

	except ModelLoadTimeoutError:
		logger.exception(f"Model loading timed out after {timeout_seconds}s")
		raise
	except Exception:
		logger.exception(f"Failed to load model {model_name}")
		raise
	finally:
		# Reset alarm
		if hasattr(signal, "SIGALRM"):
			signal.alarm(0)
			signal.signal(signal.SIGALRM, old_handler)


def generate_embedding(texts: list[str], config_loader: "ConfigLoader") -> list[list[float]]:
	"""
	Generate embeddings for a list of texts using model2vec.

	Args:
		texts: List of text strings to embed.
		config_loader: ConfigLoader instance used to load embedding model configuration.

	Returns:
		List of embeddings (each embedding is a list of floats)

	"""
	model_name = config_loader.get.embedding.model_name

	with progress_indicator("Loading model..."):
		try:
			model = _get_cached_model(model_name)
		except ModelLoadTimeoutError:
			logger.exception("Model loading timed out.")
			raise
		except Exception:
			logger.exception("Failed to load embedding model")
			raise

	with progress_indicator("Generating embeddings..."):
		try:
			embeddings = model.encode(texts)  # type: ignore[attr-defined]
			return embeddings.tolist()  # Convert np.ndarray to list of lists
		except Exception:
			logger.exception("Error generating embeddings")
			raise
