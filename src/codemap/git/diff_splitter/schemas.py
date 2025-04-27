"""Schema definitions for diff splitting."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SplitStrategy(str, Enum):
	"""Strategy for splitting diffs into logical chunks."""

	FILE = "file"  # Split by file
	HUNK = "hunk"  # Split by change hunk
	SEMANTIC = "semantic"  # Split by semantic meaning


@dataclass
class DiffChunk:
	"""Represents a logical chunk of changes."""

	files: list[str]
	content: str
	description: str | None = None
	is_llm_generated: bool = False


@dataclass
class DiffChunkData:
	"""Dictionary-based representation of a DiffChunk for serialization."""

	files: list[str]
	content: str
	description: str | None = None
	is_llm_generated: bool = False

	@classmethod
	def from_chunk(cls, chunk: DiffChunk) -> "DiffChunkData":
		"""Create a DiffChunkData from a DiffChunk."""
		return cls(
			files=chunk.files,
			content=chunk.content,
			description=chunk.description,
			is_llm_generated=chunk.is_llm_generated,
		)

	def to_chunk(self) -> DiffChunk:
		"""Convert DiffChunkData to a DiffChunk."""
		return DiffChunk(
			files=self.files, content=self.content, description=self.description, is_llm_generated=self.is_llm_generated
		)

	def to_dict(self) -> dict[str, Any]:
		"""Convert to a dictionary."""
		return {
			"files": self.files,
			"content": self.content,
			"description": self.description,
			"is_llm_generated": self.is_llm_generated,
		}
