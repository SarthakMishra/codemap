"""Module for semantic grouping of diff chunks."""

from codemap.git.diff_splitter import DiffChunk


class SemanticGroup:
	"""Represents a group of semantically related diff chunks."""

	def __init__(self, chunks: list[DiffChunk] | None = None, name: str | None = None) -> None:
		"""
		Initialize a semantic group.

		Args:
		    chunks: List of DiffChunk objects
		    name: Optional name for the group

		"""
		self.chunks = chunks or []
		self.name = name
		self.message: str | None = None
		self.approved = False

	@property
	def files(self) -> list[str]:
		"""Get the set of files affected by this group."""
		files: set[str] = set()
		for chunk in self.chunks:
			files.update(chunk.files)
		return sorted(files)

	@property
	def content(self) -> str:
		"""Get the combined diff content of all chunks."""
		return "\n".join(chunk.content for chunk in self.chunks)

	def merge_with(self, other_group: "SemanticGroup") -> "SemanticGroup":
		"""
		Merge this group with another group.

		Args:
		    other_group: Another SemanticGroup to merge with

		Returns:
		    A new SemanticGroup containing chunks from both groups

		"""
		return SemanticGroup(
			chunks=self.chunks + other_group.chunks, name=f"Merged: {self.name or ''} + {other_group.name or ''}"
		)

	def __repr__(self) -> str:
		"""Return a string representation of the group with file and chunk counts."""
		return f"SemanticGroup(files={len(self.files)}, chunks={len(self.chunks)})"
