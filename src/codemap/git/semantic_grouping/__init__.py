"""
Semantic grouping implementation for the CodeMap project.

This module provides functionality to group related diff chunks into
semantic groups for more meaningful commit messages.

"""

from dataclasses import dataclass, field

from codemap.git.diff_splitter import DiffChunk

# --- These imports are after the class definition to avoid circular imports ---
# isort: skip
from codemap.git.semantic_grouping.clusterer import DiffClusterer

# isort: skip
from codemap.git.semantic_grouping.context_processor import (
	format_chunk,
	prioritize_chunks,
	process_chunks_with_lod,
)

# isort: skip
from codemap.git.semantic_grouping.embedder import DiffEmbedder

# isort: skip
from codemap.git.semantic_grouping.resolver import FileIntegrityResolver


@dataclass
class SemanticGroup:
	"""
	A semantic group of related diff chunks.

	This class represents a group of related diff chunks that should be
	committed together because they are semantically related.

	"""

	chunks: list[DiffChunk] = field(default_factory=list)
	files: list[str] = field(default_factory=list)
	content: str = ""
	message: str | None = None
	approved: bool = False
	embedding: list[float] | None = None

	def __post_init__(self) -> None:
		"""Initialize files and content from chunks if not provided."""
		if not self.files and self.chunks:
			# Extract all unique files from chunks
			all_files = []
			for chunk in self.chunks:
				all_files.extend(chunk.files)
			self.files = sorted(set(all_files))

		if not self.content and self.chunks:
			# Combine content from all chunks
			self.content = "\n\n".join(chunk.content for chunk in self.chunks if chunk.content)

	def merge_with(self, other: "SemanticGroup") -> "SemanticGroup":
		"""
		Merge this group with another group.

		Args:
		        other: Another SemanticGroup to merge with

		Returns:
		        A new SemanticGroup containing chunks from both groups

		"""
		# Combine chunks from both groups
		combined_chunks = list(self.chunks)
		combined_chunks.extend(other.chunks)

		# Create a new group with the combined chunks
		result = SemanticGroup(chunks=combined_chunks)

		# If both groups have a message, prefer the one from self
		if self.message:
			result.message = self.message
		elif other.message:
			result.message = other.message

		return result


__all__ = [
	"DiffClusterer",
	"DiffEmbedder",
	"FileIntegrityResolver",
	"SemanticGroup",
	"format_chunk",
	"prioritize_chunks",
	"process_chunks_with_lod",
]
