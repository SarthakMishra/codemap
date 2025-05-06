"""Schema definitions for diff splitting."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiffChunk:
	"""Represents a logical chunk of changes."""

	files: list[str] = field(default_factory=list)
	content: str = ""
	description: str | None = None
	embedding: list[float] | None = None
	is_move: bool = False  # Indicates if this chunk represents a file move operation
	is_llm_generated: bool = False
	filtered_files: list[str] | None = None

	def __post_init__(self) -> None:
		"""Initialize default values."""
		if self.filtered_files is None:
			self.filtered_files = []

	def __hash__(self) -> int:
		"""
		Make DiffChunk hashable by using the object's id.

		Returns:
		        Hash value based on the object's id

		"""
		return hash(id(self))

	def __eq__(self, other: object) -> bool:
		"""
		Compare DiffChunk objects for equality.

		Args:
		        other: Another object to compare with

		Returns:
		        True if the objects are the same instance, False otherwise

		"""
		if not isinstance(other, DiffChunk):
			return False
		return id(self) == id(other)


@dataclass
class DiffChunkData:
	"""Dictionary-based representation of a DiffChunk for serialization."""

	files: list[str]
	content: str
	description: str | None = None
	is_llm_generated: bool = False
	filtered_files: list[str] | None = None
	is_move: bool = False  # Indicates if this chunk represents a file move operation

	@classmethod
	def from_chunk(cls, chunk: DiffChunk) -> "DiffChunkData":
		"""Create a DiffChunkData from a DiffChunk."""
		return cls(
			files=chunk.files,
			content=chunk.content,
			description=chunk.description,
			is_llm_generated=chunk.is_llm_generated,
			filtered_files=chunk.filtered_files,
			is_move=getattr(chunk, "is_move", False),
		)

	def to_chunk(self) -> DiffChunk:
		"""Convert DiffChunkData to a DiffChunk."""
		return DiffChunk(
			files=self.files,
			content=self.content,
			description=self.description,
			is_llm_generated=self.is_llm_generated,
			filtered_files=self.filtered_files,
			is_move=self.is_move,
		)

	def to_dict(self) -> dict[str, Any]:
		"""Convert to a dictionary."""
		return {
			"files": self.files,
			"content": self.content,
			"description": self.description,
			"is_llm_generated": self.is_llm_generated,
			"filtered_files": self.filtered_files,
			"is_move": self.is_move,
		}
