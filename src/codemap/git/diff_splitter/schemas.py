"""Schema definitions for diff splitting."""

from dataclasses import dataclass
from typing import Any


@dataclass
class DiffChunk:
	"""Represents a logical chunk of changes."""

	files: list[str]
	content: str
	description: str | None = None
	is_llm_generated: bool = False
	filtered_files: list[str] | None = None

	def __post_init__(self) -> None:
		"""Initialize default values."""
		if self.filtered_files is None:
			self.filtered_files = []


@dataclass
class DiffChunkData:
	"""Dictionary-based representation of a DiffChunk for serialization."""

	files: list[str]
	content: str
	description: str | None = None
	is_llm_generated: bool = False
	filtered_files: list[str] | None = None

	@classmethod
	def from_chunk(cls, chunk: DiffChunk) -> "DiffChunkData":
		"""Create a DiffChunkData from a DiffChunk."""
		return cls(
			files=chunk.files,
			content=chunk.content,
			description=chunk.description,
			is_llm_generated=chunk.is_llm_generated,
			filtered_files=chunk.filtered_files,
		)

	def to_chunk(self) -> DiffChunk:
		"""Convert DiffChunkData to a DiffChunk."""
		return DiffChunk(
			files=self.files,
			content=self.content,
			description=self.description,
			is_llm_generated=self.is_llm_generated,
			filtered_files=self.filtered_files,
		)

	def to_dict(self) -> dict[str, Any]:
		"""Convert to a dictionary."""
		return {
			"files": self.files,
			"content": self.content,
			"description": self.description,
			"is_llm_generated": self.is_llm_generated,
			"filtered_files": self.filtered_files,
		}
