"""Utilities for generating text embeddings."""

import logging
from typing import TYPE_CHECKING

from codemap.utils.cli_utils import progress_indicator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
	from codemap.config import ConfigLoader


def generate_embedding(texts: list[str], config_loader: "ConfigLoader") -> list[list[float]]:
	"""
	Generate embeddings for a list of texts using model2vec.

	Args:
		texts: List of text strings to embed.
		config_loader: ConfigLoader instance used to load embedding model configuration.

	Returns:
		List of embeddings (each embedding is a list of floats)

	"""
	with progress_indicator("Loading model..."):
		from model2vec import StaticModel

		model_name = config_loader.get.embedding.model_name
		model = StaticModel.from_pretrained(model_name)

	with progress_indicator("Generating embeddings..."):
		try:
			embeddings = model.encode(texts)
			return embeddings.tolist()  # Convert np.ndarray to list of lists
		except Exception:
			logger.exception("Error generating embeddings")
			raise
