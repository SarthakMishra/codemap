"""Tests for the LSP analyzer implementation."""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from multilspy import SyncLanguageServer

from codemap.processor.analysis.lsp.analyzer import LSPAnalyzer
from codemap.processor.analysis.lsp.models import LSPMetadata, LSPReference, LSPTypeInfo
from codemap.processor.chunking.base import Chunk, ChunkMetadata, EntityType, Location


class TestLSPAnalyzer:
    """Test LSP analyzer functionality."""

    @pytest.fixture
    def sample_repo_path(self) -> Path:
        """Return the path to the sample repo."""
        return Path(__file__).parent / "fixtures" / "sample_repo"

    @pytest.fixture
    def lsp_analyzer(self, sample_repo_path: Path) -> Generator[LSPAnalyzer, None, None]:
        """Create an LSP analyzer instance for testing."""
        with patch("multilspy.SyncLanguageServer.create") as mock_create:
            # Create mock server
            mock_server = MagicMock(spec=SyncLanguageServer)
            mock_server.start_server.return_value.__enter__ = MagicMock(return_value=None)
            mock_server.start_server.return_value.__exit__ = MagicMock(return_value=None)
            mock_create.return_value = mock_server

            analyzer = LSPAnalyzer(sample_repo_path)
            yield analyzer
            analyzer.close()

    @pytest.fixture
    def user_class_chunk(self, sample_repo_path: Path) -> Chunk:
        """Create a sample User class chunk for testing."""
        class_content = """class User(BaseModel):
    \"\"\"User model representing system users.\"\"\"

    def __init__(self, name: str, email: str) -> None:
        \"\"\"Initialize user with name and email.

        Args:
            name: User's full name
            email: User's email address
        \"\"\"
        super().__init__()
        self.name: str = name
        self.email: str = email
        self.orders: list[Order] = []"""

        metadata = ChunkMetadata(
            entity_type=EntityType.CLASS,
            name="User",
            location=Location(
                file_path=sample_repo_path / "models.py",
                start_line=19,
                end_line=32,
            ),
            language="python",
        )

        return Chunk(content=class_content, metadata=metadata)

    def test_init(self, lsp_analyzer: LSPAnalyzer, sample_repo_path: Path) -> None:
        """Test LSP analyzer initialization."""
        assert lsp_analyzer.project_root == sample_repo_path
        assert lsp_analyzer.language_servers == {}

    def test_get_language_server(self, lsp_analyzer: LSPAnalyzer) -> None:
        """Test getting language server instances."""
        # Test getting a supported language
        with patch("multilspy.SyncLanguageServer.create") as mock_create:
            mock_server = MagicMock()
            mock_create.return_value = mock_server

            # For testing purposes, we need to access the private method
            # pylint: disable=protected-access
            server = lsp_analyzer._get_language_server("python")
            assert server is not None

            # Server should be cached
            assert "python" in lsp_analyzer.language_servers

            # Calling again should return the cached server
            mock_create.reset_mock()
            # pylint: disable=protected-access
            server2 = lsp_analyzer._get_language_server("python")
            assert server2 is server
            mock_create.assert_not_called()

        # Test getting an unsupported language
        # pylint: disable=protected-access
        server = lsp_analyzer._get_language_server("unsupported_language")
        assert server is None

    def test_normalize_path(self, lsp_analyzer: LSPAnalyzer, sample_repo_path: Path) -> None:
        """Test path normalization for LSP."""
        # Test relative path
        file_path = sample_repo_path / "models.py"
        # pylint: disable=protected-access
        normalized = lsp_analyzer._normalize_path(file_path)
        assert normalized == "models.py"

        # Test path outside project root
        outside_path = Path("/tmp/some_file.py")
        # pylint: disable=protected-access
        normalized = lsp_analyzer._normalize_path(outside_path)
        assert normalized == str(outside_path)

    def test_position_from_location(self, lsp_analyzer: LSPAnalyzer) -> None:
        """Test converting a Location to LSP position."""
        # Test with column info
        location = Location(
            file_path=Path("test.py"),
            start_line=10,
            end_line=20,
            start_col=5,
            end_col=15,
        )
        # pylint: disable=protected-access
        position = lsp_analyzer._position_from_location(location)
        assert position == (9, 5)  # LSP uses 0-based line numbers

        # Test without column info
        location = Location(
            file_path=Path("test.py"),
            start_line=10,
            end_line=20,
        )
        # pylint: disable=protected-access
        position = lsp_analyzer._position_from_location(location)
        assert position == (9, 0)

    @patch("multilspy.SyncLanguageServer")
    def test_analyze_chunk(
        self, mock_server_class: MagicMock, lsp_analyzer: LSPAnalyzer, user_class_chunk: Chunk
    ) -> None:
        """Test analyzing a code chunk with LSP."""
        # Configure mock responses
        mock_server = mock_server_class.return_value

        # Setup hover response
        mock_server.request_hover.return_value = {"contents": {"value": "User model representing system users."}}

        # Setup references response
        mock_server.request_references.return_value = [
            {
                "uri": "file:///path/to/services.py",
                "range": {"start": {"line": 15, "character": 10}, "end": {"line": 15, "character": 14}},
            }
        ]

        # Setup definition response
        mock_server.request_definition.return_value = [
            {
                "uri": "file:///path/to/models.py",
                "range": {"start": {"line": 18, "character": 6}, "end": {"line": 18, "character": 10}},
            }
        ]

        # Setup document symbols response
        mock_server.request_document_symbols.return_value = [
            {
                "name": "User",
                "kind": 5,  # Class
                "range": {"start": {"line": 18, "character": 0}, "end": {"line": 31, "character": 0}},
                "selectionRange": {"start": {"line": 18, "character": 6}, "end": {"line": 18, "character": 10}},
            }
        ]

        # Create a patched _get_language_server that returns our mock
        with patch.object(lsp_analyzer, "_get_language_server", return_value=mock_server), patch.object(
            mock_server, "start_server"
        ):
            # Run the analysis
            lsp_metadata = lsp_analyzer.analyze_chunk(user_class_chunk)

            # Verify the results
            assert isinstance(lsp_metadata, LSPMetadata)
            assert lsp_metadata.hover_text == "User model representing system users."

            # Verify references were processed
            assert len(lsp_metadata.symbol_references) == 1
            reference = lsp_metadata.symbol_references[0]
            assert reference.target_name == "services"
            assert reference.target_uri == "file:///path/to/services.py"

            # Verify type info was extracted
            assert lsp_metadata.type_info is not None
            assert lsp_metadata.type_info.type_name == "User"
            assert lsp_metadata.type_info.is_built_in is False

    def test_enrich_chunks(self, lsp_analyzer: LSPAnalyzer, user_class_chunk: Chunk) -> None:
        """Test enriching multiple chunks with LSP metadata."""
        # Create a method chunk as a child of the user class
        method_content = """def __init__(self, name: str, email: str) -> None:
        \"\"\"Initialize user with name and email.

        Args:
            name: User's full name
            email: User's email address
        \"\"\"
        super().__init__()
        self.name: str = name
        self.email: str = email
        self.orders: list[Order] = []"""

        method_metadata = ChunkMetadata(
            entity_type=EntityType.FUNCTION,
            name="__init__",
            location=Location(
                file_path=user_class_chunk.metadata.location.file_path,
                start_line=22,
                end_line=32,
            ),
            language="python",
        )

        method_chunk = Chunk(content=method_content, metadata=method_metadata, parent=user_class_chunk)
        user_class_with_method = Chunk(
            content=user_class_chunk.content, metadata=user_class_chunk.metadata, children=[method_chunk]
        )

        # Mock the analyze_chunk method to return test metadata
        with patch.object(lsp_analyzer, "analyze_chunk") as mock_analyze:
            # Define different metadata for each chunk
            class_metadata = LSPMetadata(
                hover_text="Class hover text", type_info=LSPTypeInfo(type_name="User", is_built_in=False)
            )

            method_metadata = LSPMetadata(
                hover_text="Method hover text",
                symbol_references=[
                    LSPReference(
                        target_name="BaseModel",
                        target_uri="file:///path/to/models.py",
                        # Provide an empty dict to satisfy the type requirements
                        target_range={},
                        reference_type="inheritance",
                    )
                ],
            )

            # Configure mock to return different values for different chunks
            mock_analyze.side_effect = lambda chunk: (
                class_metadata if chunk.metadata.name == "User" else method_metadata
            )

            # Enrich the chunks
            result = lsp_analyzer.enrich_chunks([user_class_with_method])

            # Verify the results
            assert len(result) == 2
            assert "User" in result
            assert "User.__init__" in result

            assert result["User"].hover_text == "Class hover text"
            assert result["User.__init__"].hover_text == "Method hover text"
            assert len(result["User.__init__"].symbol_references) == 1
