"""Syntax-based code chunking using tree-sitter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from tree_sitter import Language, Node, Parser
from tree_sitter_language_pack import SupportedLanguage, get_language

from codemap.processor.chunking.base import Chunk, ChunkingStrategy, ChunkMetadata, EntityType, GitMetadata, Location
from codemap.processor.chunking.languages import LANGUAGE_CONFIGS, LANGUAGE_HANDLERS, LanguageSyntaxHandler

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Language name mapping for tree-sitter-language-pack
LANGUAGE_NAMES: dict[str, SupportedLanguage] = {
    "python": "python",
    # Add more languages as needed
}


def _get_language_by_extension(file_path: Path) -> str | None:
    """Get language name from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Language name if supported, None otherwise
    """
    ext = file_path.suffix
    for lang, config in LANGUAGE_CONFIGS.items():
        if ext in config.file_extensions:
            return lang
    return None


class SyntaxChunker(ChunkingStrategy):
    """Chunking strategy based on syntax tree analysis using language-specific handlers."""

    def __init__(self) -> None:
        """Initialize the syntax chunker."""
        self.parsers: dict[str, Parser] = {}
        self._load_parsers()

    def _load_parsers(self) -> None:
        """Load tree-sitter parsers for supported languages.

        This method attempts to load parsers for all configured languages using tree-sitter-language-pack.
        If a language fails to load, it will be logged but won't prevent other languages from loading.
        """
        self.parsers: dict[str, Parser] = {}
        failed_languages: list[tuple[str, str]] = []

        for lang in LANGUAGE_CONFIGS:
            try:
                # Get the language name for tree-sitter-language-pack
                ts_lang_name = LANGUAGE_NAMES.get(lang)
                if not ts_lang_name:
                    continue

                # Get the language from tree-sitter-language-pack
                language: Language = get_language(ts_lang_name)

                # Create a new parser and set its language
                parser: Parser = Parser()
                parser.language = language

                self.parsers[lang] = parser
            except (ValueError, RuntimeError, ImportError) as e:
                failed_languages.append((lang, str(e)))
                logger.debug("Failed to load language %s: %s", lang, str(e))

        if failed_languages:
            failed_names = ", ".join(f"{lang} ({err})" for lang, err in failed_languages)
            logger.warning("Failed to load parsers for languages: %s", failed_names)

    def _get_parser(self, language: str) -> Parser | None:
        """Get the parser for a language.

        Args:
            language: The language to get a parser for

        Returns:
            A tree-sitter parser or None if not supported
        """
        return self.parsers.get(language)

    def _process_node(
        self,
        node: Node,
        content_bytes: bytes,
        file_path: Path,
        language: str,
        handler: LanguageSyntaxHandler,
        git_metadata: GitMetadata | None = None,
        parent_chunk: Chunk | None = None,
    ) -> list[Chunk]:
        """Process a tree-sitter node and its children using a language handler.

        Args:
            node: The tree-sitter node
            content_bytes: Source code content as bytes
            file_path: Path to the source file
            language: Programming language
            handler: Language-specific syntax handler
            git_metadata: Optional Git metadata
            parent_chunk: Parent chunk if any

        Returns:
            List of chunks created from this node and its children
        """
        # Check if we should skip this node
        if handler.should_skip_node(node):
            return []

        logger.debug("Processing node: type=%s", node.type)

        # Get entity type for this node from the handler
        entity_type = handler.get_entity_type(node, node.parent, content_bytes)
        logger.debug("Entity type for node %s: %s", node.type, entity_type)

        # Skip unknown/uninteresting nodes unless they might contain interesting children
        if entity_type == EntityType.UNKNOWN and not node.named_child_count > 0:
            # No need to process further if it's unknown and has no children
            return []

        # Initialize variables for this node's processing
        current_chunk = None
        docstring_text = None
        docstring_node_to_skip = None

        # Only create a chunk if it's a recognized entity type
        if entity_type != EntityType.UNKNOWN:
            # Get docstring and the node to skip from the handler
            docstring_text, docstring_node_to_skip = handler.find_docstring(node, content_bytes)

            # Extract name from the handler
            name = handler.extract_name(node, content_bytes)

            # Create chunk metadata
            metadata = ChunkMetadata(
                entity_type=entity_type,
                name=name,
                location=Location(
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,  # Convert to 1-based
                    end_line=node.end_point[0] + 1,
                    start_col=node.start_point[1],
                    end_col=node.end_point[1],
                ),
                language=language,
                git=git_metadata,
                description=docstring_text,
            )

            # Get chunk content
            chunk_content = content_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

            # Create the chunk with metadata
            current_chunk = Chunk(content=chunk_content, metadata=metadata, parent=parent_chunk)

        # Process child nodes
        child_chunks = []

        # Get the body node from the handler (if applicable)
        body_node = handler.get_body_node(node)

        # Get the children to process from the handler
        children_to_process = handler.get_children_to_process(node, body_node)

        for child in children_to_process:
            # Skip the docstring node if we found it
            if docstring_node_to_skip and child == docstring_node_to_skip:
                continue

            # Recursively process child nodes
            processed_child_chunks = self._process_node(
                child,
                content_bytes,
                file_path,
                language,
                handler,
                git_metadata,
                current_chunk or parent_chunk,  # Use current chunk as parent if we created one
            )

            child_chunks.extend(processed_child_chunks)

        # Attach children to the current chunk if it exists
        if current_chunk and child_chunks:
            object.__setattr__(current_chunk, "children", tuple(child_chunks))
            return [current_chunk]
        if current_chunk:
            # Return the chunk even if it has no children
            return [current_chunk]
        # If no chunk was created at this level, return all child chunks
        return child_chunks

    def chunk(
        self,
        content: str,
        file_path: Path,
        git_metadata: GitMetadata | None = None,
        language: str | None = None,
    ) -> Sequence[Chunk]:
        """Chunk the content into semantic chunks using language-specific handlers.

        Args:
            content: The content to chunk
            file_path: Path to the file being chunked
            git_metadata: Optional Git metadata for the file
            language: Optional language override (if not inferred from file extension)

        Returns:
            A sequence of chunks

        Raises:
            ValueError: If language is not supported or cannot be determined
        """
        # Determine the language if not provided
        if not language:
            language = _get_language_by_extension(file_path)
            if not language:
                logger.warning("Could not determine language for file %s, using fallback chunking", file_path)
                return self._fallback_chunk(content, file_path, file_path.suffix.lstrip(".").lower(), git_metadata)

        # Get the parser and handler for this language
        parser = self._get_parser(language)
        handler_class = LANGUAGE_HANDLERS.get(language)

        # Fall back if no parser or handler is available
        if not parser or not handler_class:
            logger.warning("No parser or handler for language %s, using fallback chunking", language)
            return self._fallback_chunk(content, file_path, language, git_metadata)

        # Instantiate the language-specific handler
        handler = handler_class()

        try:
            # Parse the content using tree-sitter
            content_bytes = content.encode("utf-8")
            tree = parser.parse(content_bytes)
            root_node = tree.root_node
        except Exception:
            logger.exception("Failed to parse file %s", file_path)
            return self._fallback_chunk(content, file_path, language, git_metadata)

        logger.debug("Root node type: %s", root_node.type)

        # Create a module chunk for the entire file
        module_entity_type = handler.get_entity_type(root_node, None, content_bytes)
        if module_entity_type == EntityType.UNKNOWN:
            module_entity_type = EntityType.MODULE  # Default to MODULE type

        # Extract module-level docstring if available
        module_description, module_docstring_node = handler.find_docstring(root_node, content_bytes)

        # Create module chunk metadata
        metadata = ChunkMetadata(
            entity_type=module_entity_type,
            name=file_path.stem,
            location=Location(
                file_path=file_path,
                start_line=root_node.start_point[0] + 1,
                end_line=root_node.end_point[0] + 1,
                start_col=root_node.start_point[1],
                end_col=root_node.end_point[1],
            ),
            language=language,
            git=git_metadata,
            description=module_description,
        )

        # Create the module chunk
        module_chunk = Chunk(content=content, metadata=metadata)

        # Process children of the root node
        children_to_process = handler.get_children_to_process(root_node, None)

        root_children_chunks = []
        for child in children_to_process:
            # Skip the module docstring node if we found it
            if module_docstring_node and child == module_docstring_node:
                continue

            # Process the child nodes
            processed_chunks = self._process_node(
                child, content_bytes, file_path, language, handler, git_metadata, module_chunk
            )
            root_children_chunks.extend(processed_chunks)

        # Attach the child chunks to the module chunk
        if root_children_chunks:
            object.__setattr__(module_chunk, "children", tuple(root_children_chunks))

        return [module_chunk]

    def _fallback_chunk(
        self,
        content: str,
        file_path: Path,
        language: str,
        git_metadata: GitMetadata | None = None,
    ) -> Sequence[Chunk]:
        """Fallback chunking when no parser is available.

        Args:
            content: The content to chunk
            file_path: Path to the file being chunked
            language: The programming language
            git_metadata: Optional Git metadata

        Returns:
            A sequence of chunks
        """
        # Create a single chunk for the entire file
        metadata = ChunkMetadata(
            entity_type=EntityType.MODULE,
            name=file_path.stem,
            location=Location(
                file_path=file_path,
                start_line=1,
                end_line=content.count("\n") + 1,
            ),
            language=language,
            git=git_metadata,
        )
        return [Chunk(content=content, metadata=metadata)]

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
        # This is a placeholder - actual implementation would need to be smarter
        # about splitting while preserving syntax
        if len(chunk.content) <= max_size:
            return [chunk]

        # For now, just split on newlines near the middle
        content = chunk.content
        split_point = content.rfind("\n", 0, max_size)
        if split_point == -1:
            split_point = max_size

        first_content = content[:split_point]
        second_content = content[split_point:]

        first_chunk = Chunk(
            content=first_content,
            metadata=ChunkMetadata(
                entity_type=chunk.metadata.entity_type,
                name=f"{chunk.metadata.name}_part1",
                location=Location(
                    file_path=chunk.metadata.location.file_path,
                    start_line=chunk.metadata.location.start_line,
                    end_line=chunk.metadata.location.start_line + first_content.count("\n"),
                ),
                language=chunk.metadata.language,
                git=chunk.metadata.git,
            ),
            parent=chunk.parent,
        )

        second_chunk = Chunk(
            content=second_content,
            metadata=ChunkMetadata(
                entity_type=chunk.metadata.entity_type,
                name=f"{chunk.metadata.name}_part2",
                location=Location(
                    file_path=chunk.metadata.location.file_path,
                    start_line=chunk.metadata.location.start_line + first_content.count("\n") + 1,
                    end_line=chunk.metadata.location.end_line,
                ),
                language=chunk.metadata.language,
                git=chunk.metadata.git,
            ),
            parent=chunk.parent,
        )

        return [first_chunk, second_chunk]
