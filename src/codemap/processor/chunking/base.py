"""Base classes and interfaces for code chunking.

This module defines the core data structures and interfaces for code chunking strategies.
It provides:
- Entity type definitions for code elements
- Metadata structures for chunks and git information
- Base chunking strategy interface
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of code entities that can be extracted."""

    # File-level entities
    MODULE = auto()
    NAMESPACE = auto()
    PACKAGE = auto()

    # Type definitions
    CLASS = auto()
    INTERFACE = auto()
    PROTOCOL = auto()  # Similar to interface but for structural typing
    STRUCT = auto()
    ENUM = auto()
    TYPE_ALIAS = auto()

    # Functions and methods
    FUNCTION = auto()
    METHOD = auto()
    PROPERTY = auto()  # For getter/setter methods
    TEST_CASE = auto()
    TEST_SUITE = auto()

    # Variables and constants
    VARIABLE = auto()
    CONSTANT = auto()
    CLASS_FIELD = auto()  # For class-level variables/fields

    # Code organization
    IMPORT = auto()
    DECORATOR = auto()

    # Documentation
    COMMENT = auto()
    DOCSTRING = auto()

    # Special cases
    UNKNOWN = auto()


@dataclass(frozen=True)
class Location:
    """Location of a code chunk in a file.

    This class represents the precise location of a code chunk within its source file,
    including optional column information for exact positioning.
    """

    file_path: Path
    """Path to the source file containing the chunk."""

    start_line: int
    """1-based line number where the chunk starts."""

    end_line: int
    """1-based line number where the chunk ends (inclusive)."""

    start_col: int | None = None
    """Optional 0-based column number where the chunk starts."""

    end_col: int | None = None
    """Optional 0-based column number where the chunk ends (exclusive)."""

    def __str__(self) -> str:
        """Get string representation of location."""
        if self.start_col is not None and self.end_col is not None:
            return f"{self.file_path}:{self.start_line}:{self.start_col}-{self.end_line}:{self.end_col}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class GitMetadata:
    """Git-related metadata for a code chunk.

    Tracks version control information for the chunk, including both original
    authorship and last modification details.
    """

    commit_id: str
    """Git commit hash where this chunk was last modified."""

    author: str
    """Original author of the code chunk."""

    timestamp: datetime
    """Timestamp of the original commit."""

    branch: str
    """Branch name where this chunk exists."""

    last_modified_by: str | None = None
    """User who last modified this chunk, if different from author."""

    last_modified_at: datetime | None = None
    """Timestamp of the last modification, if different from original commit."""


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata for a code chunk.

    Contains all contextual information about a code chunk, including its type,
    location, and relationships to other code elements.
    """

    entity_type: EntityType
    """Type of code entity this chunk represents."""

    name: str
    """Local name of the entity (e.g., class name, function name).
    For qualified names including parent scope, use Chunk.full_name."""

    location: Location
    """Physical location of the chunk in the source code."""

    language: str
    """Programming language of the chunk (e.g., 'python', 'javascript')."""

    git: GitMetadata | None = None
    """Optional version control metadata."""

    description: str | None = None
    """Optional human-readable description of the chunk's purpose."""

    dependencies: list[str] = field(default_factory=list)
    """List of fully-qualified names of chunks this chunk depends on.
    These could be imported modules, parent classes, called functions, etc."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Additional language or tool-specific attributes.
    Common keys might include:
    - 'visibility': 'public'/'private'/'protected'
    - 'complexity': Cyclomatic complexity score
    - 'deprecated': bool
    - 'async': bool
    - 'static': bool
    """


@dataclass(frozen=True)
class Chunk:
    """A chunk of code with its metadata and hierarchical relationships.

    A chunk represents a meaningful unit of code (e.g., function, class) along with:
    - Its content and metadata
    - Its position in the code hierarchy (parent/children relationships)
    - Helper methods for common operations
    """

    content: str
    """The actual source code text of the chunk."""

    metadata: ChunkMetadata
    """Metadata describing the chunk's type, location, and attributes."""

    parent: Chunk | None = None
    """Parent chunk in the code hierarchy (e.g., class for a method)."""

    children: list[Chunk] = field(default_factory=list)
    """Child chunks (e.g., methods and fields for a class)."""

    def __post_init__(self) -> None:
        """Initialize computed fields."""
        # Ensure children list is immutable
        object.__setattr__(self, "children", tuple(self.children))

    @property
    def full_name(self) -> str:
        """Get the fully qualified name of the chunk.

        Returns:
            Dot-separated path from root to this chunk
            (e.g., 'module.MyClass.my_method').
        """
        if self.parent:
            return f"{self.parent.full_name}.{self.metadata.name}"
        return self.metadata.name

    @property
    def size(self) -> int:
        """Get the size of the chunk in characters.

        Returns:
            Number of characters in the chunk's content.
        """
        return len(self.content)

    @property
    def line_count(self) -> int:
        """Get the number of lines in the chunk.

        Returns:
            Number of lines in the chunk's content.
        """
        return self.content.count("\n") + 1

    def find_child(self, name: str) -> Chunk | None:
        """Find a direct child chunk by name.

        Args:
            name: The local name of the child chunk to find.

        Returns:
            The child chunk if found, None otherwise.
        """
        return next((child for child in self.children if child.metadata.name == name), None)


class ChunkingStrategy(abc.ABC):
    """Base class for chunking strategies.

    A chunking strategy defines how to break down source code into meaningful chunks.
    Different strategies might use different approaches:
    - Syntax-based (using AST)
    - Rule-based (using regular expressions)
    - ML-based (using embeddings)

    The only required method is `chunk`. The `merge` and `split` operations
    are optional and have default implementations.
    """

    @abc.abstractmethod
    def chunk(
        self,
        content: str,
        file_path: Path,
        git_metadata: GitMetadata | None = None,
        language: str | None = None,
    ) -> Sequence[Chunk]:
        """Chunk the content into semantic chunks based on the strategy.

        Args:
            content: The content to chunk.
            file_path: Path to the file being chunked.
            git_metadata: Optional Git metadata for the file.
            language: Optional language override (if not inferred from file extension).

        Returns:
            A sequence of top-level chunks representing the file structure.
            The chunks may have children forming a tree structure.

        Raises:
            ValueError: If the language is not supported or cannot be determined.
        """

    def merge(self, chunks: Sequence[Chunk], merge_fn: Callable[[Sequence[Chunk]], Chunk]) -> Sequence[Chunk]:
        """Merge chunks based on a merging function.

        This is an optional operation. The default implementation returns chunks unchanged.
        Override this method if your strategy needs to support post-processing merging
        of chunks (e.g., combining small related chunks).

        Args:
            chunks: The chunks to potentially merge.
            merge_fn: Function that takes a sequence of chunks and returns a merged chunk.
                     The function should handle metadata merging appropriately.

        Returns:
            A sequence of chunks, potentially after merging.
        """
        # Default implementation: no merging, but use merge_fn to satisfy linter
        _ = merge_fn
        return chunks

    def split(self, chunk: Chunk, max_size: int) -> Sequence[Chunk]:
        """Split a chunk if it exceeds max_size.

        This is an optional operation. The default implementation returns the chunk as-is,
        with a warning if it exceeds max_size. Override this method if your strategy
        needs to support splitting large chunks into smaller ones.

        Args:
            chunk: The chunk to potentially split.
            max_size: Maximum size in characters. The interpretation might vary
                     by implementation (e.g., characters, tokens, lines).

        Returns:
            A sequence of chunks, where each is ideally <= max_size.
            The default implementation returns [chunk].
        """
        if chunk.size > max_size:
            logger.warning(
                "Chunk %s exceeds max size (%d > %d chars) but splitting is not implemented",
                chunk.metadata.name,
                chunk.size,
                max_size,
            )
        return [chunk]
