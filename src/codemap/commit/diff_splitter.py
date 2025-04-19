"""Diff splitting utilities for CodeMap commit feature."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


class DiffSplitter:
    """Splits Git diffs into logical chunks."""

    def __init__(self, repo_root: Path) -> None:
        """Initialize the diff splitter.

        Args:
            repo_root: Root directory of the Git repository
        """
        self.repo_root = repo_root
        self._embeddings_cache: dict[str, list[float]] = {}

        # Load semantic chunking configuration
        self._config = self._load_semantic_config()

    def _load_semantic_config(self) -> dict[str, Any]:  # noqa: C901
        """Load semantic chunking configuration from .codemap.yml.

        Returns:
            Dictionary with semantic chunking settings
        """
        # Default configuration
        config = {
            "similarity_threshold": 0.7,
            "embedding_model": "flax-sentence-embeddings/st-codesearch-distroberta-base",
            "fallback_model": "all-MiniLM-L6-v2",
            "languages": {
                "extensions": ["py", "js", "ts", "java", "kt", "go", "c", "cpp", "cs", "rb", "php", "swift"],
                "cache_embeddings": True,
                "max_cache_size": 1000,
            },
        }

        # Try to load from config file
        config_file = self.repo_root / ".codemap.yml"
        if config_file.exists():
            try:
                import yaml

                with config_file.open("r") as f:
                    yaml_config = yaml.safe_load(f)

                if yaml_config and "commit" in yaml_config and "semantic" in yaml_config["commit"]:
                    semantic_config = yaml_config["commit"]["semantic"]

                    # Update configuration with values from file
                    if "similarity_threshold" in semantic_config:
                        config["similarity_threshold"] = semantic_config["similarity_threshold"]

                    if "embedding_model" in semantic_config:
                        config["embedding_model"] = semantic_config["embedding_model"]

                    if "fallback_model" in semantic_config:
                        config["fallback_model"] = semantic_config["fallback_model"]

                    if "languages" in semantic_config:
                        languages_config = semantic_config["languages"]

                        if "extensions" in languages_config:
                            config["languages"]["extensions"] = languages_config["extensions"]

                        if "cache_embeddings" in languages_config:
                            config["languages"]["cache_embeddings"] = languages_config["cache_embeddings"]

                        if "max_cache_size" in languages_config:
                            config["languages"]["max_cache_size"] = languages_config["max_cache_size"]

            except (ImportError, yaml.YAMLError) as e:
                logger.warning("Failed to load semantic config: %s", e)

        logger.debug("Loaded semantic chunking config: %s", config)
        return config

    def _get_code_embedding(self, content: str) -> list[float] | None:  # noqa: C901
        """Get embedding vector for code content.

        This method could be implemented with various embedding models.
        For lightweight local processing, models like:
        - code-embedding models from SentenceTransformers
        - MiniLM or CodeBERT based models

        Args:
            content: Code content to embed

        Returns:
            List of floats representing code embedding or None if unavailable
        """
        # Skip empty content
        if not content or not content.strip():
            return None

        # If embedding is already cached, return it
        if content in self._embeddings_cache:
            return self._embeddings_cache[content]

        # Generate embedding using sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            # Initialize model (will be cached after first load)
            # Using a code-optimized model or a general purpose one
            model_name = self._config["embedding_model"]

            # Create model instance (singleton pattern to avoid loading multiple times)
            if not hasattr(self, "_embedding_model"):
                try:
                    self._embedding_model = SentenceTransformer(model_name)
                    logger.info("Initialized embedding model: %s", model_name)
                except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
                    # Fallback to a more common model if the code-specific one fails
                    fallback_model = self._config["fallback_model"]
                    logger.warning("Failed to load code model, falling back to %s: %s", fallback_model, e)
                    self._embedding_model = SentenceTransformer(fallback_model)

            # Generate embedding (returns numpy array)
            embedding = self._embedding_model.encode(content, show_progress_bar=False).tolist()

            # Cache the result if enabled
            if self._config["languages"]["cache_embeddings"]:
                # Manage cache size if needed
                max_cache_size = self._config["languages"]["max_cache_size"]
                if len(self._embeddings_cache) >= max_cache_size:
                    # Simple approach: remove random item when cache is full
                    # A more sophisticated approach could use LRU cache
                    try:
                        key_to_remove = next(iter(self._embeddings_cache))
                        del self._embeddings_cache[key_to_remove]
                    except (StopIteration, KeyError):
                        pass

                self._embeddings_cache[content] = embedding
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            return None
        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.warning("Failed to generate embedding: %s", e)
            return None
        else:
            return embedding

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
        # This is a simple implementation, could be replaced with a proper vector library
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

            return dot_product / (norm1 * norm2)
        except (ImportError, Exception) as e:  # pylint: disable=broad-except
            logger.warning("Failed to calculate similarity: %s", e)
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
                return [DiffChunk(files=[f], content="") for f in diff.files]
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
                        old_lines.append(context_function)
                        new_lines.append(context_function)
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

    def _semantic_hunk_splitting(  # noqa: C901
        self,
        file_path: str,
        diff_content: str,
    ) -> list[str]:
        """Split a diff into more semantically meaningful chunks based on code structure.

        This method attempts to identify logical code blocks within the diff
        and split it at semantically meaningful boundaries.

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

        # Language-specific splitting based on syntactic boundaries
        chunks = []

        # Handle specific languages with more sophisticated splitting
        if language in ("py", "python"):
            # Split Python code at class/function definitions and logical blocks
            pattern = (
                r'(^class\s+\w+|^def\s+\w+|^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:|'
                r"^import\s+|^from\s+\w+\s+import)"
            )
            boundaries = [m.start() for m in re.finditer(pattern, new_code, re.MULTILINE)]

            # Add start and end positions
            boundaries = [0, *boundaries, len(new_code)]

            # Create chunks based on boundaries
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                chunks.append(new_code[start:end])

        elif language in ("js", "javascript", "ts", "typescript"):
            # Split JS/TS at function declarations, class methods, etc.
            pattern = (
                r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|"
                r"^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)"
            )
            boundaries = [m.start() for m in re.finditer(pattern, new_code, re.MULTILINE)]

            # Add start and end positions
            boundaries = [0, *boundaries, len(new_code)]

            # Create chunks based on boundaries
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                chunks.append(new_code[start:end])

        elif language in ("java", "kt", "kotlin", "scala"):
            # Split Java/Kotlin/Scala at class/method declarations
            pattern = (
                r"(^public\s+|^private\s+|^protected\s+|^class\s+\w+|"
                r"^interface\s+\w+|^enum\s+\w+|^import\s+|^package\s+)"
            )
            boundaries = [m.start() for m in re.finditer(pattern, new_code, re.MULTILINE)]

            # Add start and end positions
            boundaries = [0, *boundaries, len(new_code)]

            # Create chunks based on boundaries
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                chunks.append(new_code[start:end])

        elif language in ("go"):
            # Split Go code at function/type declarations
            pattern = r"(^func\s+|^type\s+\w+|^import\s+|^package\s+\w+)"
            boundaries = [m.start() for m in re.finditer(pattern, new_code, re.MULTILINE)]

            # Add start and end positions
            boundaries = [0, *boundaries, len(new_code)]

            # Create chunks based on boundaries
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                chunks.append(new_code[start:end])

        # If we couldn't split meaningfully, return the original
        if not chunks:
            return [diff_content]

        return chunks

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

            # Apply semantic code chunking for suitable file types
            ext = Path(file_path).suffix
            language = ext.lstrip(".").lower() if ext else ""

            # Only apply semantic chunking to recognized code files
            code_extensions = set(self._config["languages"]["extensions"])

            if language in code_extensions:
                # Apply semantic code splitting
                semantic_chunks = self._semantic_hunk_splitting(file_path, chunk.content)

                # Create diff chunks from semantic chunks
                if len(semantic_chunks) > 1:
                    for i, content in enumerate(semantic_chunks):
                        if content.strip():
                            # For semantic chunks, we need to transform them back to diff format
                            # This is important because the semantic chunking extracts actual code
                            diff_content = chunk.content  # Use the original diff content
                            enhanced_chunks.append(
                                DiffChunk(
                                    files=[file_path],
                                    content=diff_content,
                                    description=f"Semantic chunk {i + 1} of {file_path}",
                                ),
                            )
                else:
                    # If we couldn't split meaningfully, keep the original chunk
                    enhanced_chunks.append(chunk)
            else:
                # For non-code files, keep the original chunk
                enhanced_chunks.append(chunk)

        return enhanced_chunks

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
        
        # Combine both approaches for better results
        all_chunks = enhanced_chunks if enhanced_chunks else file_chunks
        
        # Group related files based on semantic analysis with a more aggressive approach
        processed_files = set()
        semantic_chunks = []

        # First, group files by directory structure
        dir_groups = {}
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
        for dir_path, chunks in dir_groups.items():
            if len(chunks) == 1:
                # If only one file in directory, add it directly
                semantic_chunks.append(chunks[0])
                for file in chunks[0].files:
                    processed_files.add(file)
            else:
                # For directories with multiple files, try to group them
                dir_processed = set()
                
                # First try to group by related file patterns
                self._group_related_files(chunks, dir_processed, semantic_chunks)
                
                # Then try to group remaining files by content similarity
                remaining_chunks = [c for c in chunks if c.files[0] not in dir_processed]
                if remaining_chunks:
                    # Use a lower similarity threshold for files in the same directory
                    self._group_by_content_similarity(
                        remaining_chunks, 
                        semantic_chunks, 
                        similarity_threshold=0.5  # Lower threshold for same-directory files
                    )
                
                # Add all processed files to the global processed set
                for file in dir_processed:
                    processed_files.add(file)
        
        # Process any remaining files that weren't grouped by directory
        remaining_chunks = [c for c in all_chunks if c.files[0] not in processed_files]
        
        # Try to group remaining chunks by content similarity
        if remaining_chunks:
            self._group_by_content_similarity(remaining_chunks, semantic_chunks)
            
        # If we ended up with too many small chunks, try to consolidate them
        if len(semantic_chunks) > 5 and any(len(chunk.files) == 1 for chunk in semantic_chunks):
            return self._consolidate_small_chunks(semantic_chunks)
            
        return semantic_chunks

    def _group_by_content_similarity(
        self,
        remaining_chunks: list[DiffChunk],
        semantic_chunks: list[DiffChunk],
        similarity_threshold: float = None,
    ) -> None:
        """Group remaining chunks by content similarity.

        Args:
            remaining_chunks: List of chunks not yet processed
            semantic_chunks: List of semantic chunks to append to (modified in place)
            similarity_threshold: Optional custom threshold to override config
        """
        processed_indices = set()
        # Use provided threshold or fall back to configured threshold
        threshold = similarity_threshold if similarity_threshold is not None else self._config["similarity_threshold"]

        # For each chunk, find similar chunks and group them
        for i, chunk in enumerate(remaining_chunks):
            if i in processed_indices:
                continue

            related_chunks = [chunk]
            processed_indices.add(i)

            # Find similar chunks
            for j, other_chunk in enumerate(remaining_chunks):
                if i == j or j in processed_indices:
                    continue

                # Calculate similarity between chunks
                similarity = self._calculate_semantic_similarity(chunk.content, other_chunk.content)

                if similarity >= threshold:
                    related_chunks.append(other_chunk)
                    processed_indices.add(j)

            # Create a semantic chunk from related chunks
            if len(related_chunks) > 1:  # Only create a group if there are multiple related chunks
                self._create_semantic_chunk(related_chunks, semantic_chunks)
            else:
                # Add single chunks directly
                semantic_chunks.append(related_chunks[0])
                
    def _consolidate_small_chunks(self, chunks: list[DiffChunk]) -> list[DiffChunk]:
        """Consolidate small chunks into larger, more meaningful groups.
        
        This helps prevent having too many small commits with single files.
        
        Args:
            chunks: List of diff chunks to consolidate
            
        Returns:
            Consolidated list of chunks
        """
        # If we have fewer than 3 chunks, no need to consolidate
        if len(chunks) < 3:
            return chunks
            
        # Separate single-file chunks from multi-file chunks
        single_file_chunks = [c for c in chunks if len(c.files) == 1]
        multi_file_chunks = [c for c in chunks if len(c.files) > 1]
        
        # If we don't have many single-file chunks, no need to consolidate
        if len(single_file_chunks) < 3:
            return chunks
            
        # Group single-file chunks by directory
        dir_groups = {}
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
                    
                consolidated_chunks.append(
                    DiffChunk(
                        files=all_files,
                        content="\n".join(combined_content),
                        description=f"Changes in {dir_path} directory"
                    )
                )
            else:
                # Keep single chunks in directories with only one file
                consolidated_chunks.extend(dir_chunks)
                
        # Add back the multi-file chunks
        consolidated_chunks.extend(multi_file_chunks)
        
        return consolidated_chunks

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
            
        # 1.5. Files in closely related directories (parent/child)
        if dir1 and dir2:
            if dir1.startswith(dir2 + "/") or dir2.startswith(dir1 + "/"):
                return True

        # 2. Test files and implementation files
        if (file1.startswith("tests/") and file2 in file1) or (file2.startswith("tests/") and file1 in file2):
            return True
            
        # 2.5. More test file patterns
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

        # 3. Files with similar names (e.g., user.py and user_test.py)
        base1 = file1_name.rsplit(".", 1)[0] if "." in file1_name else file1_name
        base2 = file2_name.rsplit(".", 1)[0] if "." in file2_name else file2_name
        
        # More aggressive name matching
        if (base1 in base2 or base2 in base1) and min(len(base1), len(base2)) >= 3:
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
            (r".*\.jsx?$", r".*\.scss$"),
            (r".*\.tsx?$", r".*\.scss$"),
            (r".*\.vue$", r".*\.scss$"),
            (r".*\.jsx?$", r".*\.less$"),
            (r".*\.tsx?$", r".*\.less$"),
            # React component pairs
            (r".*\.jsx$", r".*\.jsx$"),  # Related React components
            (r".*\.tsx$", r".*\.tsx$"),  # Related TypeScript React components
            (r".*Component\.jsx?$", r".*Container\.jsx?$"),  # Component/Container pattern
            (r".*Component\.tsx?$", r".*Container\.tsx?$"),  # TypeScript Component/Container pattern
            # Implementation and definition pairs
            (r".*\.h$", r".*\.c$"),
            (r".*\.hpp$", r".*\.cpp$"),
            (r".*\.h$", r".*\.m$"),  # Objective-C
            (r".*\.h$", r".*\.mm$"),  # Objective-C++
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
            (r"README\.md$", r".*$"),  # README changes often relate to other files
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
