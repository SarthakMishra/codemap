"""Syntax-based code chunking using tree-sitter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from codemap.processor.analysis.tree_sitter import TreeSitterAnalyzer, get_language_by_extension
from codemap.processor.analysis.tree_sitter.base import EntityType
from codemap.processor.chunking.base import Chunk, ChunkingStrategy, ChunkMetadata, GitMetadata, Location
from codemap.processor.chunking.reg_exp import RegExpChunker

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class TreeSitterChunker(ChunkingStrategy):
    """Chunking strategy based on syntax tree analysis using language-specific handlers."""

    def __init__(self) -> None:
        """Initialize the syntax chunker."""
        self.analyzer = TreeSitterAnalyzer()
        self.fallback_chunker = RegExpChunker()

    def chunk(
        self,
        content: str,
        file_path: Path,
        git_metadata: GitMetadata | None = None,
        language: str | None = None,
    ) -> Sequence[Chunk]:
        """Chunk the content into semantic chunks using tree-sitter analysis.

        Args:
            content: The content to chunk
            file_path: Path to the file being chunked
            git_metadata: Optional Git metadata for the file
            language: Optional language override (if not inferred from file extension)

        Returns:
            A sequence of chunks
        """
        # Determine the language if not provided
        if not language:
            language = get_language_by_extension(file_path)
            if not language:
                logger.warning("Could not determine language for file %s, using fallback chunking", file_path)
                return self.fallback_chunker.chunk(content, file_path, git_metadata, language)

        # Analyze the file using tree-sitter
        analysis_result = self.analyzer.analyze_file(file_path, content, language)

        # Check if analysis was successful
        if not analysis_result.get("success", False):
            logger.warning(
                "Failed to analyze file %s with tree-sitter, using regex-based fallback chunking: %s",
                file_path,
                analysis_result.get("error", "Unknown error"),
            )
            return self.fallback_chunker.chunk(content, file_path, git_metadata, language)

        # Create chunks from the analysis result
        return self._create_chunks_from_analysis(analysis_result, content, file_path, git_metadata)

    def _create_chunks_from_analysis(
        self,
        analysis: dict,
        content: str,
        file_path: Path,
        git_metadata: GitMetadata | None = None,
        parent_chunk: Chunk | None = None,
    ) -> list[Chunk]:
        """Create chunks from the analysis result.

        Args:
            analysis: Analysis result from TreeSitterAnalyzer
            content: Original file content
            file_path: Path to the file
            git_metadata: Optional Git metadata
            parent_chunk: Parent chunk if any

        Returns:
            List of chunks
        """
        # Create metadata for this chunk
        try:
            entity_type = EntityType[analysis["type"]] if "type" in analysis else EntityType.UNKNOWN
        except (KeyError, ValueError):
            entity_type = EntityType.UNKNOWN

        location_data = analysis.get("location", {})
        metadata = ChunkMetadata(
            entity_type=entity_type,
            name=analysis.get("name", ""),
            location=Location(
                file_path=file_path,
                start_line=location_data.get("start_line", 1),
                end_line=location_data.get("end_line", 1),
                start_col=location_data.get("start_col", 0),
                end_col=location_data.get("end_col", 0),
            ),
            language=analysis.get("language", ""),
            git=git_metadata,
            description=analysis.get("docstring"),
        )

        # Get chunk content - use the specific content from analysis or extract from original
        if analysis.get("content"):
            chunk_content = analysis["content"]
        else:
            # Fall back to extracting from the original content if needed
            lines = content.splitlines()
            start_line = max(0, location_data.get("start_line", 1) - 1)
            end_line = min(len(lines), location_data.get("end_line", len(lines)))
            chunk_content = "\n".join(lines[start_line:end_line])

        # Create the chunk
        current_chunk = Chunk(content=chunk_content, metadata=metadata, parent=parent_chunk)

        # Process children
        child_chunks = []
        for child in analysis.get("children", []):
            processed_children = self._create_chunks_from_analysis(
                child, content, file_path, git_metadata, current_chunk
            )
            child_chunks.extend(processed_children)

        # Attach children to current chunk
        if child_chunks:
            object.__setattr__(current_chunk, "children", tuple(child_chunks))

        return [current_chunk]

    def merge(self, chunks: Sequence[Chunk], merge_fn: Callable[[Sequence[Chunk]], Chunk]) -> Sequence[Chunk]:
        """Merge chunks based on a merging function.

        Args:
            chunks: The chunks to merge
            merge_fn: Function that takes a sequence of chunks and returns a merged chunk

        Returns:
            A sequence of merged chunks
        """
        # This is a placeholder - actual implementation would be more sophisticated
        return [merge_fn(chunks)]

    def split(self, chunk: Chunk, max_size: int) -> Sequence[Chunk]:
        """Split a chunk if it exceeds max_size.

        Args:
            chunk: The chunk to split
            max_size: Maximum size of a chunk

        Returns:
            A sequence of split chunks
        """
        # If the chunk is small enough, return it as is
        if len(chunk.content) <= max_size:
            return [chunk]

        # Use the more sophisticated fallback chunker's split implementation
        return self.fallback_chunker.split(chunk, max_size)
