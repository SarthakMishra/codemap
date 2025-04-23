"""Regular expression-based code chunking for fallback when tree-sitter is unavailable."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Callable, Sequence

from codemap.processor.chunking.base import Chunk, ChunkingStrategy, ChunkMetadata, EntityType, GitMetadata, Location

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
MAX_DOCSTRING_OFFSET = 100  # Maximum line offset for considering a docstring at the top of file


class RegExpChunker(ChunkingStrategy):
    """Chunking strategy based on regular expressions for general-purpose code parsing.

    This chunker serves as a fallback when more sophisticated parsing (like tree-sitter)
    is unavailable or fails. It performs basic chunking based on common patterns
    without attempting language-specific entity recognition.
    """

    def __init__(self) -> None:
        """Initialize the regex chunker."""
        super().__init__()

    def chunk(
        self,
        content: str,
        file_path: Path,
        git_metadata: GitMetadata | None = None,
        language: str | None = None,
    ) -> Sequence[Chunk]:
        """Chunk the content using basic regular expressions.

        Args:
            content: The content to chunk
            file_path: Path to the file being chunked
            git_metadata: Optional Git metadata for the file
            language: Optional language (ignored in this implementation)

        Returns:
            A sequence of chunks
        """
        # Create a module-level chunk as the parent
        module_chunk = self._create_module_chunk(content, file_path, language, git_metadata)

        # Extract chunks using basic patterns
        child_chunks = self._basic_line_chunking(content, file_path, language, git_metadata, module_chunk)

        # Set the children on the module chunk
        if child_chunks:
            object.__setattr__(module_chunk, "children", tuple(child_chunks))

        return [module_chunk]

    def _create_module_chunk(
        self, content: str, file_path: Path, language: str | None, git_metadata: GitMetadata | None
    ) -> Chunk:
        """Create a module-level chunk for the entire file.

        Args:
            content: The file content
            file_path: Path to the file
            language: Programming language (optional)
            git_metadata: Optional Git metadata

        Returns:
            A Chunk object representing the module
        """
        # Try to extract a potential docstring using a very generic pattern
        docstring = None
        # Look for either Python or JSDoc style docstrings at the top of the file
        docstring_patterns = [
            # Python docstring
            re.compile(r'"""(.*?)"""', re.DOTALL),
            # JSDoc style
            re.compile(r"/\*\*(.*?)\*/", re.DOTALL),
        ]

        for pattern in docstring_patterns:
            match = pattern.search(content)
            if match and match.start() < MAX_DOCSTRING_OFFSET:  # Only if it's near the top
                docstring = match.group(1).strip()
                break

        # Use empty string as default when language is None
        lang = language or ""

        metadata = ChunkMetadata(
            entity_type=EntityType.MODULE,
            name=file_path.stem,
            location=Location(
                file_path=file_path,
                start_line=1,
                end_line=content.count("\n") + 1,
            ),
            language=lang,
            git=git_metadata,
            description=docstring,
        )
        return Chunk(content=content, metadata=metadata)

    def _basic_line_chunking(
        self, content: str, file_path: Path, language: str | None, git_metadata: GitMetadata | None, parent_chunk: Chunk
    ) -> list[Chunk]:
        """Perform basic line-based chunking for any language.

        Args:
            content: The file content
            file_path: Path to the file
            language: Programming language (optional)
            git_metadata: Optional Git metadata
            parent_chunk: The parent chunk (module)

        Returns:
            A list of child chunks
        """
        chunks = []
        lines = content.splitlines()

        # Use empty string as default when language is None
        lang = language or ""

        # Group lines into logical blocks based on empty lines and indentation
        current_block = []
        current_block_start = 1

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip empty lines when not in a block
            if not stripped and not current_block:
                continue

            # Start of a new block or continuing a block
            if stripped and not current_block:
                current_block = [line]
                current_block_start = i
            elif not stripped and i < len(lines) - 1 and not lines[i].strip():
                # End of block (empty line followed by another empty line)
                if current_block:
                    chunk_content = "\n".join(current_block)
                    name = self._extract_name_from_block(current_block)

                    metadata = ChunkMetadata(
                        entity_type=EntityType.UNKNOWN,
                        name=name or f"block_{current_block_start}",
                        location=Location(
                            file_path=file_path,
                            start_line=current_block_start,
                            end_line=i - 1,
                        ),
                        language=lang,
                        git=git_metadata,
                    )

                    chunk = Chunk(content=chunk_content, metadata=metadata, parent=parent_chunk)
                    chunks.append(chunk)
                    current_block = []
            else:
                # Continue current block
                current_block.append(line)

        # Handle the last block if any
        if current_block:
            chunk_content = "\n".join(current_block)
            name = self._extract_name_from_block(current_block)

            metadata = ChunkMetadata(
                entity_type=EntityType.UNKNOWN,
                name=name or f"block_{current_block_start}",
                location=Location(
                    file_path=file_path,
                    start_line=current_block_start,
                    end_line=len(lines),
                ),
                language=lang,
                git=git_metadata,
            )

            chunk = Chunk(content=chunk_content, metadata=metadata, parent=parent_chunk)
            chunks.append(chunk)

        return chunks

    def _extract_name_from_block(self, block: list[str]) -> str | None:
        """Try to extract a meaningful name from a block of code.

        Args:
            block: List of code lines

        Returns:
            A name for the block if one can be inferred, None otherwise
        """
        if not block:
            return None

        # Look for common patterns in the first line
        first_line = block[0].strip()

        # Check for comments (language-agnostic approach)
        if first_line.startswith(("//", "#", "/*", "*")):
            comment_text = first_line.lstrip("/#* ")
            words = comment_text.split()
            if words:
                return words[0]

        # Very basic attempt to find a name without being language-specific
        # This will just look for word patterns that might be definitions
        basic_patterns = [
            r"\b(\w+)\s*\(",  # Function call pattern
            r"\b(\w+)\s*=",  # Assignment pattern
            r"\b(\w+)\s*{",  # Block definition pattern
        ]

        for pattern in basic_patterns:
            match = re.search(pattern, first_line)
            if match:
                return match.group(1)

        return None

    def merge(self, chunks: Sequence[Chunk], merge_fn: Callable[[Sequence[Chunk]], Chunk]) -> Sequence[Chunk]:
        """Merge chunks based on a merging function.

        Args:
            chunks: The chunks to merge
            merge_fn: Function that takes a sequence of chunks and returns a merged chunk

        Returns:
            A sequence of merged chunks
        """
        return [merge_fn(chunks)]

    def split(self, chunk: Chunk, max_size: int) -> Sequence[Chunk]:
        """Split a chunk if it exceeds max_size.

        Args:
            chunk: The chunk to split
            max_size: Maximum size of a chunk

        Returns:
            A sequence of split chunks
        """
        if len(chunk.content) <= max_size:
            return [chunk]

        # Try to split on meaningful boundaries
        content = chunk.content
        lines = content.splitlines(keepends=True)

        # Find logical splitting points
        split_points = []
        current_size = 0
        last_good_split = 0

        for i, line in enumerate(lines):
            current_size += len(line)

            # Look for good splitting points (blank lines, end of blocks)
            if line.strip() == "" or line.strip().endswith(("{", "}", ";", ":")):
                last_good_split = i

            # If we've exceeded max_size, split at the last good point
            if current_size > max_size:
                if last_good_split > 0:
                    split_points.append(last_good_split + 1)  # Split after the line
                    current_size = sum(len(line_item) for line_item in lines[last_good_split + 1 : i + 1])
                    last_good_split = 0
                else:
                    # If no good split point, just split at the current line
                    split_points.append(i)
                    current_size = len(line)

        # Create chunks based on split points
        result_chunks = []
        start_idx = 0

        for split_idx in split_points:
            # Create chunk from start_idx to split_idx
            split_content = "".join(lines[start_idx:split_idx])

            # Calculate line numbers for this chunk
            start_line = chunk.metadata.location.start_line + start_idx
            end_line = chunk.metadata.location.start_line + split_idx - 1

            # Create metadata
            metadata = ChunkMetadata(
                entity_type=chunk.metadata.entity_type,
                name=f"{chunk.metadata.name}_part{len(result_chunks) + 1}",
                location=Location(
                    file_path=chunk.metadata.location.file_path,
                    start_line=start_line,
                    end_line=end_line,
                ),
                language=chunk.metadata.language,
                git=chunk.metadata.git,
                description=None,  # Don't copy description to split chunks
            )

            result_chunks.append(
                Chunk(
                    content=split_content,
                    metadata=metadata,
                    parent=chunk.parent,
                )
            )

            start_idx = split_idx

        # Add the final chunk
        if start_idx < len(lines):
            final_content = "".join(lines[start_idx:])

            # Calculate line numbers
            start_line = chunk.metadata.location.start_line + start_idx
            end_line = chunk.metadata.location.end_line

            metadata = ChunkMetadata(
                entity_type=chunk.metadata.entity_type,
                name=f"{chunk.metadata.name}_part{len(result_chunks) + 1}",
                location=Location(
                    file_path=chunk.metadata.location.file_path,
                    start_line=start_line,
                    end_line=end_line,
                ),
                language=chunk.metadata.language,
                git=chunk.metadata.git,
                description=None,
            )

            result_chunks.append(
                Chunk(
                    content=final_content,
                    metadata=metadata,
                    parent=chunk.parent,
                )
            )

        return result_chunks
