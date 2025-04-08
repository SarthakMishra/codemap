"""Diff splitting utilities for CodeMap commit feature."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from codemap.utils.git_utils import GitDiff

logger = logging.getLogger(__name__)


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


class DiffSplitter:
    """Splits Git diffs into logical chunks."""

    def __init__(self, repo_root: Path) -> None:
        """Initialize the diff splitter.

        Args:
            repo_root: Root directory of the Git repository
        """
        self.repo_root = repo_root

    def _split_by_file(self, diff: GitDiff) -> list[DiffChunk]:
        """Split a diff into chunks by file.

        Args:
            diff: GitDiff object to split

        Returns:
            List of DiffChunk objects, one per file
        """
        if not diff.content:
            return []

        # Split the diff content by file
        file_pattern = r"diff --git a/.*? b/(.*?)\n"
        file_chunks = re.split(file_pattern, diff.content)[1:]  # Skip first empty chunk

        # Group files with their content
        chunks = []
        for i in range(0, len(file_chunks), 2):
            file_name = file_chunks[i]
            content = file_chunks[i + 1] if i + 1 < len(file_chunks) else ""
            if file_name and content:
                chunks.append(
                    DiffChunk(
                        files=[file_name],
                        content=f"diff --git a/{file_name} b/{file_name}\n{content}",
                    ),
                )

        return chunks

    def _split_by_hunk(self, diff: GitDiff) -> list[DiffChunk]:
        """Split a diff into chunks by hunk.

        Args:
            diff: GitDiff object to split

        Returns:
            List of DiffChunk objects, one per hunk
        """
        if not diff.content:
            return []

        # Regex to match the start of a file diff
        file_pattern = r"diff --git a/(.*?) b/(.*?)\n"

        # Regex to match the start of a hunk within a file
        hunk_pattern = r"@@ -\d+,\d+ \+\d+,\d+ @@"

        # First split by file
        file_chunks = re.split(file_pattern, diff.content)

        # Skip the first empty chunk if present
        if file_chunks and not file_chunks[0].strip():
            file_chunks = file_chunks[1:]

        chunks = []

        # Process each file
        i = 0
        while i < len(file_chunks):
            if i + 2 >= len(file_chunks):
                break

            file_name = file_chunks[i]
            file_content = file_chunks[i + 2]

            # Skip to next file
            i += 3

            if not file_name or not file_content:
                continue

            # Split the file content by hunks
            hunk_starts = [m.start() for m in re.finditer(hunk_pattern, file_content)]

            if not hunk_starts:
                # If no hunks found, treat the entire file as one chunk
                chunks.append(
                    DiffChunk(
                        files=[file_name],
                        content=f"diff --git a/{file_name} b/{file_name}\n{file_content}",
                    ),
                )
                continue

            # Process each hunk
            for j in range(len(hunk_starts)):
                hunk_start = hunk_starts[j]
                hunk_end = hunk_starts[j + 1] if j + 1 < len(hunk_starts) else len(file_content)

                # Extract hunk content
                hunk_content = file_content[hunk_start:hunk_end]

                # Get the file header (everything before the first hunk)
                file_header = file_content[:hunk_start] if j == 0 else ""

                # Create chunk
                chunks.append(
                    DiffChunk(
                        files=[file_name],
                        content=f"diff --git a/{file_name} b/{file_name}\n{file_header}{hunk_content}",
                    ),
                )

        return chunks

    def _split_semantic(self, diff: GitDiff) -> list[DiffChunk]:
        """Split a diff into semantic chunks using code analysis.

        Args:
            diff: GitDiff object to split

        Returns:
            List of DiffChunk objects based on semantic grouping
        """
        # Start with file-based splitting as a base
        file_chunks = self._split_by_file(diff)

        if not file_chunks:
            return []

        # Group related files based on semantic analysis
        processed_files = set()
        semantic_chunks = []

        # Process file chunks to create semantic groups
        self._group_related_files(file_chunks, processed_files, semantic_chunks)

        # Process any remaining files
        remaining_chunks = [c for c in file_chunks if c.files[0] not in processed_files]
        semantic_chunks.extend(remaining_chunks)

        return semantic_chunks

    def _are_files_related(self, file1: str, file2: str) -> bool:
        """Determine if two files are semantically related.

        Args:
            file1: First file path
            file2: Second file path

        Returns:
            True if the files are related, False otherwise
        """
        # 1. Files in the same directory
        dir1 = file1.rsplit("/", 1)[0] if "/" in file1 else ""
        dir2 = file2.rsplit("/", 1)[0] if "/" in file2 else ""
        if dir1 and dir1 == dir2:
            return True

        # 2. Test files and implementation files
        if (file1.startswith("tests/") and file2 in file1) or (file2.startswith("tests/") and file1 in file2):
            return True

        # 3. Files with similar names (e.g., user.py and user_test.py)
        base1 = file1.rsplit(".", 1)[0] if "." in file1 else file1
        base2 = file2.rsplit(".", 1)[0] if "." in file2 else file2
        if base1 in base2 or base2 in base1:
            return True

        # 4. Check for related file patterns
        return self._has_related_file_pattern(file1, file2)

    def _has_related_file_pattern(self, file1: str, file2: str) -> bool:
        """Check if files match known related patterns.

        Args:
            file1: First file path
            file2: Second file path

        Returns:
            True if the files match a known pattern, False otherwise
        """
        # Common file patterns that are likely to be related
        related_patterns = [
            # Frontend component pairs
            (r".*\.jsx?$", r".*\.css$"),
            (r".*\.tsx?$", r".*\.css$"),
            (r".*\.vue$", r".*\.css$"),
            # Implementation and definition pairs
            (r".*\.h$", r".*\.c$"),
            (r".*\.hpp$", r".*\.cpp$"),
            (r".*\.proto$", r".*\.pb\.(go|py|js|java)$"),
            # Web development pairs
            (r".*\.html$", r".*\.js$"),
            (r".*\.html$", r".*\.css$"),
        ]

        for pattern1, pattern2 in related_patterns:
            if (re.match(pattern1, file1) and re.match(pattern2, file2)) or (
                re.match(pattern2, file1) and re.match(pattern1, file2)
            ):
                return True

        return False

    def _group_related_files(
        self,
        file_chunks: list[DiffChunk],
        processed_files: set[str],
        semantic_chunks: list[DiffChunk],
    ) -> None:
        """Group related files into semantic chunks.

        Args:
            file_chunks: List of file-based chunks
            processed_files: Set of already processed files (modified in place)
            semantic_chunks: List of semantic chunks (modified in place)
        """
        # First pass: group clearly related files
        for i, chunk in enumerate(file_chunks):
            if chunk.files[0] in processed_files:
                continue

            related_chunks = [chunk]
            processed_files.add(chunk.files[0])

            # Find related files
            for j, other_chunk in enumerate(file_chunks):
                if i == j or other_chunk.files[0] in processed_files:
                    continue

                if self._are_files_related(chunk.files[0], other_chunk.files[0]):
                    related_chunks.append(other_chunk)
                    processed_files.add(other_chunk.files[0])

            # Create a semantic chunk from related files
            if related_chunks:
                self._create_semantic_chunk(related_chunks, semantic_chunks)

    def _create_semantic_chunk(
        self,
        related_chunks: list[DiffChunk],
        semantic_chunks: list[DiffChunk],
    ) -> None:
        """Create a semantic chunk from related file chunks.

        Args:
            related_chunks: List of related file chunks
            semantic_chunks: List of semantic chunks to append to (modified in place)
        """
        all_files = []
        combined_content = []

        for rc in related_chunks:
            all_files.extend(rc.files)
            combined_content.append(rc.content)

        semantic_chunks.append(
            DiffChunk(
                files=all_files,
                content="\n".join(combined_content),
            ),
        )

    def split_diff(self, diff: GitDiff, strategy: str | SplitStrategy = "file") -> list[DiffChunk]:
        """Split a diff into logical chunks using the specified strategy.

        Args:
            diff: GitDiff object to split
            strategy: Splitting strategy ("file", "hunk", "semantic") or SplitStrategy enum

        Returns:
            List of DiffChunk objects

        Raises:
            ValueError: If an invalid strategy is specified
        """
        if not diff.content:
            return []

        # Convert strategy to string if it's an enum
        strategy_str = strategy.value if isinstance(strategy, SplitStrategy) else strategy

        # Use the string value to determine which method to call
        if strategy_str == SplitStrategy.FILE.value:
            return self._split_by_file(diff)
        if strategy_str == SplitStrategy.HUNK.value:
            return self._split_by_hunk(diff)
        if strategy_str == SplitStrategy.SEMANTIC.value:
            return self._split_semantic(diff)

        msg = f"Invalid diff splitting strategy: {strategy}"
        raise ValueError(msg)
