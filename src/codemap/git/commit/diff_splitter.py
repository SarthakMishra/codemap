"""Diff splitting utilities for CodeMap commit feature."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

# Constants for configuration
MIN_CHUNKS_FOR_CONSOLIDATION = 1
MAX_CHUNKS_BEFORE_CONSOLIDATION = 10
MIN_NAME_LENGTH_FOR_SIMILARITY = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DIRECTORY_SIMILARITY_THRESHOLD = 0.5

if TYPE_CHECKING:
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
    is_llm_generated: bool = False


class DiffSplitter:
    """Splits Git diffs into logical chunks."""

    # Class-level cache for the embedding model
    _embedding_model = None

    def __init__(self, repo_root: Path) -> None:
        """Initialize the diff splitter.

        Args:
            repo_root: Root directory of the Git repository
        """
        self.repo_root = repo_root
        self.code_extensions = {
            "py",
            "js",
            "ts",
            "java",
            "kt",
            "go",
            "c",
            "cpp",
            "cs",
            "rb",
            "php",
            "swift",
            "jsx",
            "tsx",
        }

        # Precompile regex patterns for better performance
        self._file_pattern = re.compile(r"diff --git a/.*? b/(.*?)\n")
        self._hunk_pattern = re.compile(r"@@ -\d+,\d+ \+\d+,\d+ @@")

        # Initialize related file patterns
        self._related_file_patterns = self._initialize_related_file_patterns()

    def _initialize_related_file_patterns(self) -> list[tuple[re.Pattern, re.Pattern]]:
        """Initialize and compile regex patterns for related files.

        Returns:
            List of compiled regex pattern pairs
        """
        patterns = [
            # Frontend component pairs
            (r".*\.jsx?$", r".*\.css$"),
            (r".*\.tsx?$", r".*\.css$"),
            (r".*\.vue$", r".*\.css$"),
            (r".*\.jsx?$", r".*\.scss$"),
            (r".*\.tsx?$", r".*\.scss$"),
            (r".*\.vue$", r".*\.scss$"),
            (r".*\.jsx?$", r".*\.less$"),
            (r".*\.tsx?$", r".*\.less$"),
            # React component pairs
            (r".*\.jsx$", r".*\.jsx$"),
            (r".*\.tsx$", r".*\.tsx$"),
            (r".*Component\.jsx?$", r".*Container\.jsx?$"),
            (r".*Component\.tsx?$", r".*Container\.tsx?$"),
            # Implementation and definition pairs
            (r".*\.h$", r".*\.c$"),
            (r".*\.hpp$", r".*\.cpp$"),
            (r".*\.h$", r".*\.m$"),
            (r".*\.h$", r".*\.mm$"),
            (r".*\.proto$", r".*\.pb\.(go|py|js|java|rb|cs)$"),
            (r".*\.idl$", r".*\.(h|cpp|cs|java)$"),
            # Web development pairs
            (r".*\.html$", r".*\.js$"),
            (r".*\.html$", r".*\.css$"),
            (r".*\.html$", r".*\.scss$"),
            (r".*\.html$", r".*\.ts$"),
            # Python related files
            (r".*\.py$", r".*_test\.py$"),
            (r".*\.py$", r"test_.*\.py$"),
            (r".*\.py$", r".*_spec\.py$"),
            # JavaScript/TypeScript related files
            (r".*\.js$", r".*\.test\.js$"),
            (r".*\.js$", r".*\.spec\.js$"),
            (r".*\.ts$", r".*\.test\.ts$"),
            (r".*\.ts$", r".*\.spec\.ts$"),
            # Ruby related files
            (r".*\.rb$", r".*_spec\.rb$"),
            (r".*\.rb$", r".*_test\.rb$"),
            # Java related files
            (r".*\.java$", r".*Test\.java$"),
            # Go related files
            (r".*\.go$", r".*_test\.go$"),
            # Configuration files
            (r"package\.json$", r"package-lock\.json$"),
            (r"package\.json$", r"yarn\.lock$"),
            (r"package\.json$", r"tsconfig\.json$"),
            (r"package\.json$", r"\.eslintrc(\.js|\.json|\.yml)?$"),
            (r"package\.json$", r"\.prettierrc(\.js|\.json|\.yml)?$"),
            (r"requirements\.txt$", r"setup\.py$"),
            (r"pyproject\.toml$", r"setup\.py$"),
            (r"pyproject\.toml$", r"setup\.cfg$"),
            (r"Gemfile$", r"Gemfile\.lock$"),
            (r"Cargo\.toml$", r"Cargo\.lock$"),
            # Documentation
            (r".*\.md$", r".*\.(js|ts|py|rb|java|go|c|cpp|h|hpp)$"),
            (r"README\.md$", r".*$"),
        ]

        # Compile all patterns for better performance
        return [(re.compile(p1), re.compile(p2)) for p1, p2 in patterns]

    def _get_code_embedding(self, content: str) -> list[float] | None:
        """Get embedding vector for code content.

        Args:
            content: Code content to embed

        Returns:
            List of floats representing code embedding or None if unavailable

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        # Skip empty content
        if not content or not content.strip():
            return None

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers or numpy not installed. Install with: pip install sentence-transformers numpy"
            )
            return None

        # Generate embedding using sentence-transformers
        try:
            # Initialize model with fixed model name (singleton pattern)
            model_name = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"

            # Create model instance if not already created
            if DiffSplitter._embedding_model is None:
                DiffSplitter._embedding_model = SentenceTransformer(model_name)
                logger.info("Initialized embedding model: %s", model_name)

            # Generate embedding (returns numpy array)
            return DiffSplitter._embedding_model.encode(content, show_progress_bar=False).tolist()

        except Exception:
            logger.exception("Failed to generate embedding")
            return None

    def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two code chunks.

        Args:
            content1: First code content
            content2: Second code content

        Returns:
            Similarity score between 0 and 1
        """
        # Get embeddings
        emb1 = self._get_code_embedding(content1)
        emb2 = self._get_code_embedding(content2)

        if not emb1 or not emb2:
            return 0.0

        # Calculate cosine similarity
        try:
            import numpy as np

            # Convert to numpy arrays
            vec1 = np.array(emb1)
            vec2 = np.array(emb2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except ImportError:
            logger.warning("numpy not installed. Install with: pip install numpy")
            return 0.0
        except (ArithmeticError, ValueError, TypeError) as e:
            logger.warning("Failed to calculate similarity: %s", str(e))
            return 0.0

    def _split_by_file(self, diff: GitDiff) -> list[DiffChunk]:
        """Split a diff into chunks by file.

        Args:
            diff: GitDiff object to split

        Returns:
            List of DiffChunk objects, one per file
        """
        if not diff.content:
            # Handle untracked files specifically
            if not diff.is_staged and diff.files:
                # Filter out invalid file names
                valid_files = []
                for file in diff.files:
                    # Skip files that look like patterns or templates
                    if any(char in file for char in ["*", "+", "{", "}", "\\"]) or file.startswith('"'):
                        logger.warning("Skipping invalid filename in diff: %s", file)
                        continue
                    valid_files.append(file)

                return [DiffChunk(files=[f], content="") for f in valid_files]
            return []

        # Split the diff content by file
        file_chunks = self._file_pattern.split(diff.content)[1:]  # Skip first empty chunk

        # Group files with their content
        chunks = []
        for i in range(0, len(file_chunks), 2):
            if i + 1 >= len(file_chunks):
                break

            file_name = file_chunks[i]
            content = file_chunks[i + 1]

            # Skip files that look like patterns or templates
            if any(char in file_name for char in ["*", "+", "{", "}", "\\"]) or file_name.startswith('"'):
                logger.warning("Skipping invalid filename in diff: %s", file_name)
                continue

            if file_name and content:
                # Build the diff header manually without using an f-string
                diff_header = "diff --git a/" + file_name + " b/" + file_name + "\n"
                chunks.append(
                    DiffChunk(
                        files=[file_name],
                        content=diff_header + content,
                        description=f"Changes in {file_name}",
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

        # First split by file
        file_chunks = self._split_by_file(diff)

        if not file_chunks:
            return []

        chunks = []

        # Process each file
        for file_chunk in file_chunks:
            if not file_chunk.files or not file_chunk.content:
                continue

            file_name = file_chunk.files[0]

            # Skip files that look like patterns or templates
            # (redundant check since _split_by_file should already filter)
            if any(char in file_name for char in ["*", "+", "{", "}", "\\"]) or file_name.startswith('"'):
                logger.warning("Skipping invalid filename in hunk processing: %s", file_name)
                continue

            file_content = file_chunk.content

            # Split the file content by hunks
            hunk_starts = [m.start() for m in self._hunk_pattern.finditer(file_content)]

            if not hunk_starts:
                # If no hunks found, treat the entire file as one chunk
                chunks.append(file_chunk)
                continue

            # Find the start of the actual diff content (after the file header)
            header_end = file_content.find("\n", file_content.find("diff --git"))
            if header_end == -1:
                header_end = 0

            file_header = file_content[: header_end + 1]

            # Process each hunk
            for j in range(len(hunk_starts)):
                hunk_start = hunk_starts[j]
                hunk_end = hunk_starts[j + 1] if j + 1 < len(hunk_starts) else len(file_content)

                # Extract hunk content
                hunk_content = file_content[hunk_start:hunk_end]

                # Create chunk
                chunks.append(
                    DiffChunk(
                        files=[file_name], content=file_header + hunk_content, description=f"Update in {file_name}"
                    ),
                )

        return chunks

    def _extract_code_from_diff(self, diff_content: str) -> tuple[str, str]:
        """Extract actual code content from a diff.

        Args:
            diff_content: The raw diff content

        Returns:
            Tuple of (old_code, new_code) extracted from the diff
        """
        old_lines = []
        new_lines = []

        # Skip diff header lines
        lines = diff_content.split("\n")
        in_hunk = False
        context_function = None

        for line in lines:
            # Check for hunk header
            if line.startswith("@@"):
                in_hunk = True
                # Try to extract function context if available
                context_match = re.search(r"@@ .+ @@ (.*)", line)
                if context_match and context_match.group(1):
                    context_function = context_match.group(1).strip()
                    # Add function context to both old and new lines
                    if context_function:
                        old_lines.append(f"// {context_function}")
                        new_lines.append(f"// {context_function}")
                continue

            if not in_hunk:
                continue

            # Extract code content
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            else:
                # Context lines appear in both old and new
                old_lines.append(line)
                new_lines.append(line)

        return "\n".join(old_lines), "\n".join(new_lines)

    def _get_language_specific_patterns(self, language: str) -> re.Pattern | None:
        """Get language-specific regex patterns for code structure.

        Args:
            language: Programming language identifier

        Returns:
            Compiled regex pattern or None if language not supported
        """
        patterns = {
            "py": r'(^class\s+\w+|^def\s+\w+|^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:|^import\s+|^from\s+\w+\s+import)',  # noqa: E501
            "js": r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)",  # noqa: E501
            "ts": r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)",  # noqa: E501
            "jsx": r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)",  # noqa: E501
            "tsx": r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)",  # noqa: E501
            "java": r"(^public\s+|^private\s+|^protected\s+|^class\s+\w+|^interface\s+\w+|^enum\s+\w+|^import\s+|^package\s+)",  # noqa: E501
            "kt": r"(^public\s+|^private\s+|^protected\s+|^class\s+\w+|^interface\s+\w+|^enum\s+\w+|^import\s+|^package\s+)",  # noqa: E501
            "scala": r"(^public\s+|^private\s+|^protected\s+|^class\s+\w+|^interface\s+\w+|^enum\s+\w+|^import\s+|^package\s+)",  # noqa: E501
            "go": r"(^func\s+|^type\s+\w+|^import\s+|^package\s+\w+)",
        }

        if language in patterns:
            return re.compile(patterns[language], re.MULTILINE)
        return None

    def _semantic_hunk_splitting(self, file_path: str, diff_content: str) -> list[str]:
        """Split a diff into more semantically meaningful chunks based on code structure.

        Args:
            file_path: Path to the file being diffed (used to determine language)
            diff_content: The diff content to split

        Returns:
            List of split diff contents
        """
        # Get language based on file extension
        ext = Path(file_path).suffix
        language = ext.lstrip(".").lower() if ext else ""

        # Extract old and new code from diff
        _, new_code = self._extract_code_from_diff(diff_content)

        # If no meaningful code to analyze, return the original diff
        if not new_code.strip():
            return [diff_content]

        # Get language-specific pattern
        pattern = self._get_language_specific_patterns(language)

        if not pattern or language not in self.code_extensions:
            return [diff_content]

        # Find boundaries in the code
        boundaries = [m.start() for m in pattern.finditer(new_code)]

        # If no meaningful boundaries found, return original
        if not boundaries:
            return [diff_content]

        # Add start and end positions
        boundaries = [0, *boundaries, len(new_code)]

        # Create chunks based on boundaries
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk = new_code[start:end]
            if chunk.strip():
                chunks.append(chunk)

        # If we couldn't split meaningfully, return the original
        if not chunks:
            return [diff_content]

        # Convert code chunks back to diff format
        # This is a simplified approach - in a real implementation,
        # you would need to map these code chunks back to the original diff format
        # Returning the conceptual chunks for now to allow _enhance_semantic_split to create multiple DiffChunks
        return chunks  # Return the split code chunks

    def _enhance_semantic_split(self, diff: GitDiff) -> list[DiffChunk]:
        """Enhanced semantic splitting that considers code structure.

        Args:
            diff: GitDiff object to split

        Returns:
            List of DiffChunk objects based on semantic grouping
        """
        # First split by file
        file_chunks = self._split_by_file(diff)

        if not file_chunks:
            return []

        enhanced_chunks = []

        # Process each file chunk
        for chunk in file_chunks:
            if not chunk.files or not chunk.content:
                continue

            file_path = chunk.files[0]
            ext = Path(file_path).suffix
            language = ext.lstrip(".").lower() if ext else ""

            # Only apply semantic chunking to recognized code files
            if language in self.code_extensions:
                # Apply semantic code splitting
                semantic_chunks = self._semantic_hunk_splitting(file_path, chunk.content)

                # Create diff chunks from semantic chunks based on the number of semantic parts identified
                if semantic_chunks and len(semantic_chunks) > 0:  # Check if any semantic chunks were identified
                    for i, _semantic_part_content in enumerate(semantic_chunks):
                        # Create a meaningful description, using the original diff content for the chunk
                        part_number = i + 1
                        description = f"update: {file_path} (part {part_number})"
                        enhanced_chunks.append(
                            DiffChunk(
                                files=[file_path],
                                # Use the original diff content for now, as re-splitting the diff is complex
                                content=chunk.content,
                                description=description,
                            ),
                        )
                else:
                    # If no meaningful semantic chunks found, keep the original file chunk
                    enhanced_chunks.append(chunk)
            else:
                # For non-code files, keep the original chunk
                enhanced_chunks.append(chunk)

        return enhanced_chunks

    def _group_by_content_similarity(
        self,
        chunks: list[DiffChunk],
        result_chunks: list[DiffChunk],
        similarity_threshold: float | None = None,
    ) -> None:
        """Group chunks by content similarity.

        Args:
            chunks: List of chunks to process
            result_chunks: List to append grouped chunks to (modified in place)
            similarity_threshold: Optional custom threshold to override default
        """
        if not chunks:
            return

        processed_indices = set()
        threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_SIMILARITY_THRESHOLD

        # For each chunk, find similar chunks and group them
        for i, chunk in enumerate(chunks):
            if i in processed_indices:
                continue

            related_chunks = [chunk]
            processed_indices.add(i)

            # Find similar chunks
            for j, other_chunk in enumerate(chunks):
                if i == j or j in processed_indices:
                    continue

                # Calculate similarity between chunks
                similarity = self._calculate_semantic_similarity(chunk.content, other_chunk.content)

                if similarity >= threshold:
                    related_chunks.append(other_chunk)
                    processed_indices.add(j)

            # Create a semantic chunk from related chunks
            if related_chunks:
                self._create_semantic_chunk(related_chunks, result_chunks)

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

        # 2. Files in closely related directories (parent/child)
        if dir1 and dir2 and (dir1.startswith(dir2 + "/") or dir2.startswith(dir1 + "/")):
            return True

        # 3. Test files and implementation files
        if (file1.startswith("tests/") and file2 in file1) or (file2.startswith("tests/") and file1 in file2):
            return True

        # 4. More test file patterns
        file1_name = file1.rsplit("/", 1)[-1] if "/" in file1 else file1
        file2_name = file2.rsplit("/", 1)[-1] if "/" in file2 else file2

        # Check for test_X.py and X.py patterns
        if file1_name.startswith("test_") and file1_name[5:] == file2_name:
            return True
        if file2_name.startswith("test_") and file2_name[5:] == file1_name:
            return True

        # Check for X_test.py and X.py patterns
        if file1_name.endswith("_test.py") and file1_name[:-8] + ".py" == file2_name:
            return True
        if file2_name.endswith("_test.py") and file2_name[:-8] + ".py" == file1_name:
            return True

        # 5. Files with similar names
        base1 = file1_name.rsplit(".", 1)[0] if "." in file1_name else file1_name
        base2 = file2_name.rsplit(".", 1)[0] if "." in file2_name else file2_name

        if (base1 in base2 or base2 in base1) and min(len(base1), len(base2)) >= MIN_NAME_LENGTH_FOR_SIMILARITY:
            return True

        # 6. Check for related file patterns
        return self._has_related_file_pattern(file1, file2)

    def _has_related_file_pattern(self, file1: str, file2: str) -> bool:
        """Check if files match known related patterns.

        Args:
            file1: First file path
            file2: Second file path

        Returns:
            True if the files match a known pattern, False otherwise
        """
        for pattern1, pattern2 in self._related_file_patterns:
            if (pattern1.match(file1) and pattern2.match(file2)) or (pattern2.match(file1) and pattern1.match(file2)):
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
        if not file_chunks:
            return

        # Group clearly related files
        for i, chunk in enumerate(file_chunks):
            if not chunk.files or chunk.files[0] in processed_files:
                continue

            related_chunks = [chunk]
            processed_files.add(chunk.files[0])

            # Find related files
            for j, other_chunk in enumerate(file_chunks):
                if i == j or not other_chunk.files or other_chunk.files[0] in processed_files:
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
        if not related_chunks:
            return

        all_files = []
        combined_content = []

        for rc in related_chunks:
            all_files.extend(rc.files)
            combined_content.append(rc.content)

        # Determine the appropriate commit type based on the files
        commit_type = self._determine_commit_type(all_files)

        # Create description based on file count
        description = self._create_chunk_description(commit_type, all_files)

        # Join the content from all related chunks
        content = "\n\n".join(combined_content)

        semantic_chunks.append(
            DiffChunk(
                files=all_files,
                content=content,
                description=description,
            ),
        )

    def _determine_commit_type(self, files: list[str]) -> str:
        """Determine the appropriate commit type based on the files.

        Args:
            files: List of file paths

        Returns:
            Commit type string (e.g., "feat", "fix", "test", "docs", "chore")
        """
        # Check for test files
        if any(f.startswith("tests/") or "_test." in f or "test_" in f for f in files):
            return "test"

        # Check for documentation files
        if any(f.startswith("docs/") or f.endswith(".md") for f in files):
            return "docs"

        # Check for configuration files
        if any(f.endswith((".json", ".yml", ".yaml", ".toml", ".ini", ".cfg")) for f in files):
            return "chore"

        # Default to "chore" for general updates
        return "chore"

    def _create_chunk_description(self, commit_type: str, files: list[str]) -> str:
        """Create a meaningful description for a chunk.

        Args:
            commit_type: Type of commit (e.g., "feat", "fix")
            files: List of file paths

        Returns:
            Description string
        """
        if len(files) == 1:
            return f"{commit_type}: update {files[0]}"

        # Try to find a common directory
        common_dir = os.path.commonpath(files)
        if common_dir and common_dir != ".":
            return f"{commit_type}: update files in {common_dir}"

        return f"{commit_type}: update {len(files)} related files"

    def _consolidate_small_chunks(self, chunks: list[DiffChunk]) -> list[DiffChunk]:
        """Consolidate small chunks into larger, more meaningful groups.

        Args:
            chunks: List of diff chunks to consolidate

        Returns:
            Consolidated list of chunks
        """
        # If we have fewer than MIN_CHUNKS_FOR_CONSOLIDATION chunks, no need to consolidate
        if len(chunks) < MIN_CHUNKS_FOR_CONSOLIDATION:
            return chunks

        # Separate single-file chunks from multi-file chunks
        single_file_chunks = [c for c in chunks if len(c.files) == 1]
        multi_file_chunks = [c for c in chunks if len(c.files) > 1]

        # If we don't have many single-file chunks, no need to consolidate
        if len(single_file_chunks) < MIN_CHUNKS_FOR_CONSOLIDATION:
            return chunks

        # Group single-file chunks by directory
        dir_groups: dict[str, list[DiffChunk]] = {}
        for chunk in single_file_chunks:
            file_path = chunk.files[0]
            dir_path = file_path.rsplit("/", 1)[0] if "/" in file_path else "root"

            if dir_path not in dir_groups:
                dir_groups[dir_path] = []

            dir_groups[dir_path].append(chunk)

        # Create consolidated chunks for each directory with multiple files
        consolidated_chunks = []
        for dir_path, dir_chunks in dir_groups.items():
            if len(dir_chunks) > 1:
                # Combine all chunks in this directory
                all_files = []
                combined_content = []

                for c in dir_chunks:
                    all_files.extend(c.files)
                    combined_content.append(c.content)

                commit_type = self._determine_commit_type(all_files)
                consolidated_chunks.append(
                    DiffChunk(
                        files=all_files,
                        content="\n".join(combined_content),
                        description=f"{commit_type}: update files in {dir_path}",
                    ),
                )
            else:
                # Keep single chunks in directories with only one file
                consolidated_chunks.extend(dir_chunks)

        # Add back the multi-file chunks
        consolidated_chunks.extend(multi_file_chunks)

        return consolidated_chunks

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

        # Try enhanced splitting for code files
        enhanced_chunks = self._enhance_semantic_split(diff)

        # Use enhanced chunks if available, otherwise fall back to file chunks
        all_chunks = enhanced_chunks if enhanced_chunks else file_chunks

        # Group related files based on semantic analysis
        processed_files: set[str] = set()
        semantic_chunks: list[DiffChunk] = []

        # First, group files by directory structure
        dir_groups: dict[str, list[DiffChunk]] = {}
        for chunk in all_chunks:
            if not chunk.files:
                continue

            file_path = chunk.files[0]
            # Get directory path (or use root if file is in root)
            dir_path = file_path.rsplit("/", 1)[0] if "/" in file_path else "root"

            if dir_path not in dir_groups:
                dir_groups[dir_path] = []

            dir_groups[dir_path].append(chunk)

        # Process each directory group
        for chunks in dir_groups.values():
            if len(chunks) == 1:
                # If only one file in directory, add it directly
                semantic_chunks.append(chunks[0])
                for file in chunks[0].files:
                    processed_files.add(file)
            else:
                # For directories with multiple files, try to group them
                dir_processed: set[str] = set()

                # First try to group by related file patterns
                self._group_related_files(chunks, dir_processed, semantic_chunks)

                # Then try to group remaining files by content similarity
                remaining_chunks = [c for c in chunks if not c.files or c.files[0] not in dir_processed]
                if remaining_chunks:
                    # Use a lower similarity threshold for files in the same directory
                    self._group_by_content_similarity(
                        remaining_chunks,
                        semantic_chunks,
                        similarity_threshold=DIRECTORY_SIMILARITY_THRESHOLD,
                    )

                # Add all processed files to the global processed set
                for file in dir_processed:
                    processed_files.add(file)

        # Process any remaining files that weren't grouped by directory
        remaining_chunks = [c for c in all_chunks if c.files and c.files[0] not in processed_files]

        # Try to group remaining chunks by content similarity
        if remaining_chunks:
            self._group_by_content_similarity(remaining_chunks, semantic_chunks)

        # If we ended up with too many small chunks, try to consolidate them
        has_single_file_chunks = any(len(chunk.files) == 1 for chunk in semantic_chunks)
        if len(semantic_chunks) > MAX_CHUNKS_BEFORE_CONSOLIDATION and has_single_file_chunks:
            return self._consolidate_small_chunks(semantic_chunks)

        return semantic_chunks

    def split_diff(self, diff: GitDiff, strategy: str | SplitStrategy | None = None) -> list[DiffChunk]:
        """Split a diff into logical chunks.

        Args:
            diff: GitDiff object to split
            strategy: Strategy to use for splitting (FILE, HUNK, or SEMANTIC)

        Returns:
            List of DiffChunk objects
        """
        if not diff.content and not diff.files:
            return []

        # Filter invalid filenames from the diff
        if diff.files:
            valid_files = []
            for file in diff.files:
                # Skip files that look like patterns or templates
                if any(char in file for char in ["*", "+", "{", "}", "\\"]) or file.startswith('"'):
                    logger.warning("Skipping invalid filename in diff processing: %s", file)
                    continue
                valid_files.append(file)

            # Replace files list with valid files only
            diff.files = valid_files

        # Convert string strategy to enum if needed
        if isinstance(strategy, str):
            try:
                strategy = SplitStrategy(strategy)
            except ValueError:
                logger.warning("Invalid strategy: %s. Using SEMANTIC instead.", strategy)
                strategy = SplitStrategy.SEMANTIC

        # Use semantic strategy by default
        if strategy is None:
            strategy = SplitStrategy.SEMANTIC

        # Apply the selected strategy
        if strategy == SplitStrategy.FILE:
            return self._split_by_file(diff)
        if strategy == SplitStrategy.HUNK:
            return self._split_by_hunk(diff)
        # SEMANTIC
        return self._split_semantic(diff)
