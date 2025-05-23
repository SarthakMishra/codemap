"""Module for generating embeddings from diff chunks."""

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from codemap.git.diff_splitter import DiffChunk
from codemap.processor.utils.embedding_utils import generate_embedding

if TYPE_CHECKING:
	from codemap.config import ConfigLoader

logger = logging.getLogger(__name__)


class DiffEmbedder:
	"""Generates embeddings for diff chunks."""

	def __init__(
		self,
		config_loader: "ConfigLoader",
	) -> None:
		"""
		Initialize the embedder with configuration.

		Args:
		    config_loader: ConfigLoader instance for embedding configuration.
		"""
		self.config_loader = config_loader

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

	async def embed_chunk(self, chunk: DiffChunk) -> np.ndarray:
		"""
		Generate an embedding for a diff chunk using Voyage AI.

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

		# Generate embeddings in batch (of 1)
		embeddings = generate_embedding([processed_text], self.config_loader)

		if not embeddings:
			message = f"Failed to generate embedding for chunk with files: {', '.join(chunk.files)}"
			logger.error(message)
			# Return a zero vector as a fallback
			return np.zeros(1024)  # Using default dimension of 1024

		return np.array(embeddings[0])

	async def embed_contents(self, contents: list[str]) -> list[list[float] | None]:
		"""
		Generate embeddings for multiple content strings.

		Args:
		    contents: List of text content strings to embed

		Returns:
		    List of embedding vectors or None for each content
		"""
		# Filter out empty contents
		contents_to_embed = []
		valid_indices = []

		for i, content in enumerate(contents):
			if content and content.strip():
				# Preprocess if it looks like diff content
				if content.startswith(("diff --git", "+", "-", " ")):
					processed = self.preprocess_diff(content)
					if processed.strip():
						contents_to_embed.append(processed)
						valid_indices.append(i)
				else:
					# Use as-is if it doesn't look like a diff
					contents_to_embed.append(content)
					valid_indices.append(i)

		# Return early if no valid contents
		if not contents_to_embed:
			return cast("list[list[float] | None]", [None] * len(contents))

		# Generate embeddings in batch
		try:
			embeddings_batch = generate_embedding(contents_to_embed, self.config_loader)

			# Rebuild result list with None for invalid contents
			result: list[list[float] | None] = cast("list[list[float] | None]", [None] * len(contents))
			if embeddings_batch:
				for idx, valid_idx in enumerate(valid_indices):
					if idx < len(embeddings_batch):
						result[valid_idx] = embeddings_batch[idx]
			return result

		except Exception:
			logger.exception("Unexpected error during embedding generation")
			return cast("list[list[float] | None]", [None] * len(contents))

	async def embed_chunks(self, chunks: list[DiffChunk]) -> list[tuple[DiffChunk, np.ndarray]]:
		"""
		Generate embeddings for multiple chunks using efficient batch processing.

		Args:
		    chunks: List of DiffChunk objects

		Returns:
		    List of (chunk, embedding) tuples

		"""
		if not chunks:
			return []

		# Preprocess all chunk texts
		preprocessed_texts = []
		for chunk in chunks:
			diff_text = chunk.content
			processed_text = self.preprocess_diff(diff_text)

			# If the processed text is empty, use the file paths as context
			if not processed_text.strip():
				processed_text = " ".join(chunk.files)

			preprocessed_texts.append(processed_text)

		# Generate embeddings in batch
		embeddings = generate_embedding(preprocessed_texts, self.config_loader)

		# Create result tuples
		result = []
		if embeddings:
			for i, chunk in enumerate(chunks):
				if i < len(embeddings):
					embedding = np.array(embeddings[i])
				else:
					logger.error(f"Missing embedding for chunk with files: {', '.join(chunk.files)}")
					embedding = np.zeros(1024)  # Fallback
				result.append((chunk, embedding))
		else:
			# Fallback if batch embedding failed
			logger.error("Batch embedding generation failed, using fallback zeros")
			result.extend((chunk, np.zeros(1024)) for chunk in chunks)

		return result
