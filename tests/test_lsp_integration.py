"""Tests for LSP integration with the processing pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from codemap.processor.analysis.git.analyzer import GitMetadataAnalyzer
from codemap.processor.analysis.lsp.analyzer import LSPAnalyzer
from codemap.processor.analysis.lsp.models import LSPMetadata, LSPReference, LSPTypeInfo
from codemap.processor.chunking.base import Chunk
from codemap.processor.pipeline import ProcessingPipeline
from codemap.processor.storage.base import StorageConfig


class TestLSPIntegration:
    """Integration tests for LSP with the processing pipeline."""

    @pytest.fixture
    def temp_repo_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for the test repo."""
        with TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_python_file(self, temp_repo_dir: Path) -> Path:
        """Create a sample Python file in the temporary repo."""
        # Create a simple Python file
        file_path = temp_repo_dir / "sample.py"
        with file_path.open("w") as f:
            f.write("""
class User:
    \"\"\"User class for testing LSP integration.\"\"\"

    def __init__(self, name, email):
        \"\"\"Initialize the user.

        Args:
            name: The user's name
            email: The user's email
        \"\"\"
        self.name = name
        self.email = email

    def get_full_info(self):
        \"\"\"Return the user's full information.\"\"\"
        return f"{self.name} ({self.email})"
""")
        return file_path

    @pytest.fixture
    def mock_lsp_analyzer(self) -> MagicMock:
        """Create a mock LSP analyzer."""
        mock_analyzer = MagicMock(spec=LSPAnalyzer)

        # Configure the mock to return sample LSP metadata
        mock_analyzer.enrich_chunks.return_value = {
            "User": LSPMetadata(
                hover_text="User class for testing LSP integration.",
                type_info=LSPTypeInfo(type_name="User", is_built_in=False),
            ),
            "User.__init__": LSPMetadata(
                hover_text="Initialize the user.",
                symbol_references=[
                    LSPReference(
                        target_name="self", target_uri="file://sample.py", target_range={}, reference_type="reference"
                    )
                ],
            ),
            "User.get_full_info": LSPMetadata(
                hover_text="Return the user's full information.",
                symbol_references=[
                    LSPReference(
                        target_name="self", target_uri="file://sample.py", target_range={}, reference_type="reference"
                    )
                ],
            ),
        }

        return mock_analyzer

    @pytest.fixture
    def mock_git_analyzer(self) -> MagicMock:
        """Create a mock GitMetadataAnalyzer to avoid Git repository validation."""
        mock_git = MagicMock(spec=GitMetadataAnalyzer)
        mock_git.get_current_commit.return_value = "mock-commit-id"
        mock_git.get_current_branch.return_value = "mock-branch"
        return mock_git

    def test_pipeline_with_lsp(
        self, temp_repo_dir: Path, sample_python_file: Path, mock_lsp_analyzer: MagicMock, mock_git_analyzer: MagicMock
    ) -> None:
        """Test the pipeline with LSP integration."""
        # Create a storage config for testing
        storage_config = StorageConfig(uri=str(temp_repo_dir / "storage"))

        # Patch both the LSPAnalyzer and GitMetadataAnalyzer
        with patch("codemap.processor.pipeline.LSPAnalyzer", return_value=mock_lsp_analyzer), patch(
            "codemap.processor.pipeline.GitMetadataAnalyzer", return_value=mock_git_analyzer
        ):
            # Create the pipeline with LSP enabled
            pipeline = ProcessingPipeline(repo_path=temp_repo_dir, storage_config=storage_config, enable_lsp=True)

            # Mock the store_lsp_metadata method to capture the LSP metadata
            with patch.object(pipeline.storage, "store_lsp_metadata") as mock_store_lsp:
                # Process the file
                pipeline.process_file(sample_python_file)

                # Wait a moment for processing to complete
                import time

                time.sleep(0.1)

                # Verify that LSP metadata was stored
                assert mock_store_lsp.call_count == 1

                # Check that the stored LSP metadata contains the expected keys
                args, _ = mock_store_lsp.call_args
                lsp_metadata, chunks, _ = args

                assert "User" in lsp_metadata
                assert "User.__init__" in lsp_metadata
                assert "User.get_full_info" in lsp_metadata

                # Verify that the mock LSP analyzer was used
                assert pipeline.lsp_analyzer == mock_lsp_analyzer
                assert mock_lsp_analyzer.enrich_chunks.call_count == 1

                # Check that chunks were processed
                assert len(chunks) > 0
                assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_pipeline_without_lsp(
        self, temp_repo_dir: Path, sample_python_file: Path, mock_git_analyzer: MagicMock
    ) -> None:
        """Test the pipeline with LSP disabled."""
        # Create a storage config for testing
        storage_config = StorageConfig(uri=str(temp_repo_dir / "storage"))

        # Patch GitMetadataAnalyzer to avoid Git repository validation
        with patch("codemap.processor.pipeline.GitMetadataAnalyzer", return_value=mock_git_analyzer):
            # Create the pipeline with LSP disabled
            pipeline = ProcessingPipeline(repo_path=temp_repo_dir, storage_config=storage_config, enable_lsp=False)

            # Verify that LSP analyzer is not created
            assert pipeline.lsp_analyzer is None

            # Mock the store_lsp_metadata method to verify it's not called
            with patch.object(pipeline.storage, "store_lsp_metadata") as mock_store_lsp:
                # Process the file
                pipeline.process_file(sample_python_file)

                # Wait a moment for processing to complete
                import time

                time.sleep(0.1)

                # Verify that LSP metadata was not stored
                assert mock_store_lsp.call_count == 0
