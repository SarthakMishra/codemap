"""Module for generating embeddings from diff chunks."""

import logging

import numpy as np

from codemap.git.diff_splitter import DiffChunk

logger = logging.getLogger(__name__)


class DiffEmbedder:
	"""Generates embeddings for diff chunks."""

	def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
		"""
		Initialize the embedder with a specific model.

		Args:
		    model_name: Name of the sentence-transformers model to use

		"""
		# Import here to avoid making sentence-transformers a hard dependency
		try:
			from sentence_transformers import SentenceTransformer

			self.model = SentenceTransformer(model_name)
		except ImportError as e:
			logger.exception(
				"Failed to import sentence-transformers. Please install it with: uv add sentence-transformers"
			)
			msg = "sentence-transformers is required for semantic grouping"
			raise ImportError(msg) from e

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
