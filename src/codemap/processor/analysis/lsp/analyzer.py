"""LSP analyzer implementation using MultiLSPy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, cast

from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
from typing_extensions import Self

from codemap.processor.analysis.lsp.models import LSPMetadata, LSPReference, LSPTypeInfo
from codemap.processor.chunking.base import Chunk, Location

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


class LSPAnalyzer:
    """Analyzer that uses Language Server Protocol to enrich code chunks with semantic information."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the LSP analyzer.

        Args:
            project_root: The root directory of the project being analyzed.
        """
        self.project_root = project_root
        self.language_servers: dict[str, SyncLanguageServer] = {}
        self.logger = MultilspyLogger()

    def _get_language_server(self, language: str) -> SyncLanguageServer | None:
        """Get or create a language server for the given language.

        Args:
            language: The programming language to get a server for.

        Returns:
            A language server instance or None if the language is not supported.
        """
        if language in self.language_servers:
            return self.language_servers[language]

        # Map CodeMap language names to MultiLSPy language names
        language_map = {
            "python": "python",
            "javascript": "javascript",
            "typescript": "typescript",
            "java": "java",
            "csharp": "csharp",
            "go": "go",
            "rust": "rust",
            "ruby": "ruby",
            "php": "php",
            "cpp": "cpp",
        }

        multilspy_language = language_map.get(language.lower())
        if not multilspy_language:
            logger.warning("LSP analysis not supported for language: %s", language)
            return None

        try:
            config = MultilspyConfig.from_dict({"code_language": multilspy_language})
            server = SyncLanguageServer.create(config, self.logger, str(self.project_root))
            self.language_servers[language] = server
            return server
        except Exception:
            logger.exception("Failed to initialize language server for %s", language)
            return None

    def _normalize_path(self, file_path: Path) -> str:
        """Convert an absolute path to a project-relative path for LSP.

        Args:
            file_path: The absolute file path.

        Returns:
            A project-relative path string.
        """
        try:
            return str(file_path.relative_to(self.project_root))
        except ValueError:
            # If it's not relative to project_root, return as is
            return str(file_path)

    def _position_from_location(self, location: Location) -> tuple[int, int]:
        """Convert a Location to LSP position (line, column).

        Args:
            location: The source location.

        Returns:
            A tuple of (line, column) with 0-based indexing.
        """
        # LSP uses 0-based line numbers, while Location uses 1-based
        line = location.start_line - 1
        column = location.start_col if location.start_col is not None else 0
        return (line, column)

    def analyze_chunk(self, chunk: Chunk) -> LSPMetadata:
        """Analyze a code chunk using LSP and return enhanced metadata.

        Args:
            chunk: The code chunk to analyze.

        Returns:
            LSP metadata containing semantic information.
        """
        language = chunk.metadata.language
        file_path = chunk.metadata.location.file_path

        lsp = self._get_language_server(language)
        if not lsp:
            return LSPMetadata()

        relative_path = self._normalize_path(file_path)

        # Get position for LSP queries
        line, column = self._position_from_location(chunk.metadata.location)

        lsp_metadata = LSPMetadata()

        # Start the language server
        with lsp.start_server():
            try:
                # Get hover information (can include docstrings, type hints)
                hover_result = lsp.request_hover(relative_path, line, column)
                hover_text = None
                if hover_result and isinstance(hover_result, dict) and "contents" in hover_result:
                    contents = hover_result["contents"]
                    if isinstance(contents, dict) and "value" in contents:
                        hover_text = contents["value"]
                    elif isinstance(contents, str):
                        hover_text = contents

                # Get symbol references
                references = []
                try:
                    refs_result = lsp.request_references(relative_path, line, column)
                    if refs_result and isinstance(refs_result, list):
                        for ref in refs_result:
                            if isinstance(ref, dict):
                                # Convert URI to filesystem path
                                uri = ref.get("uri", "")
                                # Use Path.name instead of os.path.basename
                                target_name = Path(uri).name.split(".")[0]  # Simplified extraction

                                references.append(
                                    LSPReference(
                                        target_name=target_name,
                                        target_uri=uri,
                                        # Cast the range to Dict[str, Any] to satisfy type checker
                                        target_range=cast("Dict[str, Any]", ref.get("range", {})),
                                        reference_type="reference",  # Default type
                                    )
                                )
                except (ValueError, TypeError, AttributeError, KeyError) as err:
                    logger.debug("Error getting references: %s", err)

                # Get definition
                definition_uri = None
                def_result = lsp.request_definition(relative_path, line, column)
                is_definition = True
                if def_result and isinstance(def_result, list) and len(def_result) > 0:
                    definition = def_result[0]
                    if isinstance(definition, dict):
                        definition_uri = definition.get("uri")
                        # Check if this chunk is a definition or a reference
                        is_definition = definition_uri == f"file://{file_path}"

                # Get document symbols to infer type information
                type_info = None
                symbols_result = lsp.request_document_symbols(relative_path)
                if symbols_result and isinstance(symbols_result, list):
                    # Try to find a symbol that corresponds to this chunk
                    for symbol in symbols_result:
                        if not isinstance(symbol, dict):
                            continue

                        symbol_range = symbol.get("range", {})
                        if not isinstance(symbol_range, dict):
                            continue

                        symbol_start = symbol_range.get("start", {})
                        if not isinstance(symbol_start, dict):
                            continue

                        symbol_line = symbol_start.get("line", -1)

                        # Check if this symbol corresponds to our chunk
                        if symbol_line == line:
                            kind = symbol.get("kind")
                            symbol_name = symbol.get("name", "")

                            # Map LSP symbol kinds to type information
                            # See https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#symbolKind
                            if kind in [
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                            ]:  # Class, Method, Property, Field, Constructor, Enum, Interface
                                type_info = LSPTypeInfo(type_name=symbol_name, is_built_in=False)
                            break

                # Create LSP metadata
                lsp_metadata = LSPMetadata(
                    symbol_references=references,
                    type_info=type_info,
                    hover_text=hover_text,
                    definition_uri=definition_uri,
                    is_definition=is_definition,
                )

            except Exception:
                logger.exception("Error during LSP analysis of %s", chunk.metadata.name)

        return lsp_metadata

    def enrich_chunks(self, chunks: list[Chunk]) -> dict[str, LSPMetadata]:
        """Enrich a list of chunks with LSP metadata.

        Args:
            chunks: List of chunks to analyze and enrich.

        Returns:
            Dictionary mapping chunk full names to their LSP metadata.
        """
        results: dict[str, LSPMetadata] = {}

        for chunk in chunks:
            # Extract to separate try/except to avoid performance issue in loop
            metadata = self._analyze_chunk_safely(chunk)
            results[chunk.full_name] = metadata

            # Recursively process children
            if chunk.children:
                child_results = self.enrich_chunks(list(chunk.children))
                results.update(child_results)

        return results

    def _analyze_chunk_safely(self, chunk: Chunk) -> LSPMetadata:
        """Safely analyze a chunk, catching any exceptions.

        Args:
            chunk: The chunk to analyze.

        Returns:
            LSP metadata for the chunk or an empty metadata object if analysis fails.
        """
        try:
            return self.analyze_chunk(chunk)
        except Exception:
            logger.exception("Failed to enrich chunk %s with LSP metadata", chunk.full_name)
            return LSPMetadata()

    def close(self) -> None:
        """Close all language server instances."""
        # Just clear the language servers dictionary
        # The servers should be properly closed by their respective context managers
        self.language_servers.clear()
        logger.debug("Cleared all language server references")

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """Exit context manager and ensure all servers are closed."""
        self.close()
