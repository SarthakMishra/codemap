"""Module for generating embeddings from diff chunks."""

import logging
from typing import TYPE_CHECKING

import numpy as np

# Define EmbeddingModelType at module level for type hinting
if TYPE_CHECKING:
	from sentence_transformers import SentenceTransformer

from codemap.config import ConfigLoader
from codemap.git.diff_splitter import DiffChunk

logger = logging.getLogger(__name__)


class DiffEmbedder:
	"""Generates embeddings for diff chunks."""

	def __init__(
		self,
		model: "SentenceTransformer",
		config_loader: ConfigLoader,
	) -> None:
		"""
		Initialize the embedder with a pre-loaded SentenceTransformer model.

		Args:
		    model: The pre-loaded sentence_transformers.SentenceTransformer model instance.
		    config_loader: Optional ConfigLoader instance.
		"""
		self.config_loader = config_loader or ConfigLoader()
		self.model = model

	def preprocess_diff(self, diff_text: str) -> str:
		"""
		Preprocess diff text to make it more suitable for embedding.

		Args:
		    diff_text: Raw diff text

		Returns:
		    Preprocessed text

		"""
		# Remove diff headers, line numbers, etc.
		# Focus on actual content changes
		lines = []
		for line in diff_text.splitlines():
			# Skip diff metadata lines
			if line.startswith(("diff --git", "index ", "+++", "---")):
				continue

			# Keep actual content changes, removing the +/- prefix
			if line.startswith(("+", "-", " ")):
				lines.append(line[1:])

		return "\n".join(lines)

	def embed_chunk(self, chunk: DiffChunk) -> np.ndarray:
		"""
		Generate an embedding for a diff chunk.

		Args:
		    chunk: DiffChunk object

		Returns:
		    numpy.ndarray: Embedding vector

		"""
		if self.model is None:
			# This should not happen if __init__ guarantees a model or raises an error.
			message = "Attempted to use DiffEmbedder.embed_chunk with no model loaded."
			logger.error(message)
			raise RuntimeError(message)

		# Get the diff content from the chunk
		diff_text = chunk.content

		# Preprocess the diff text
		processed_text = self.preprocess_diff(diff_text)

		# If the processed text is empty, use the file paths as context
		if not processed_text.strip():
			processed_text = " ".join(chunk.files)

		# Generate the embedding and convert to numpy array
		embedding = self.model.encode(processed_text)
		return np.array(embedding)

	def embed_chunks(self, chunks: list[DiffChunk]) -> list[tuple[DiffChunk, np.ndarray]]:
		"""
		Generate embeddings for multiple chunks.

		Args:
		    chunks: List of DiffChunk objects

		Returns:
		    List of (chunk, embedding) tuples

		"""
		return [(chunk, self.embed_chunk(chunk)) for chunk in chunks]
