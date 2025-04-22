"""Tests for the git commit workflow logic."""

import unittest
from typing import cast
from unittest.mock import Mock, patch

from codemap.git.commit.command import CommitCommand
from codemap.git.commit.diff_splitter import DiffChunk
from codemap.git.commit.interactive import ChunkAction, ChunkResult


class TestCommitWorkflow(unittest.TestCase):
    """Test cases for commit workflow logic in CommitCommand."""

    def setUp(self) -> None:
        """Set up test environment with mocks."""
        # Patch get_repo_root to avoid actual Git operations
        self.repo_root_patcher = patch("codemap.git.commit.command.get_repo_root")
        self.mock_get_repo_root = self.repo_root_patcher.start()
        self.mock_get_repo_root.return_value = "/mock/repo/path"

        # Create mock objects for dependencies
        self.mock_ui = Mock()
        self.mock_splitter = Mock()
        self.mock_message_generator = Mock()

        # Create the CommitCommand with patched dependencies
        with patch("codemap.git.commit.command.CommitUI", return_value=self.mock_ui), patch(
            "codemap.git.commit.command.DiffSplitter", return_value=self.mock_splitter
        ), patch("codemap.git.commit.command.MessageGenerator", return_value=self.mock_message_generator):
            self.command = CommitCommand()

        # Create a mock chunk for testing that will be treated as a DiffChunk
        self.mock_chunk = Mock(spec=DiffChunk)
        self.mock_chunk.files = ["file1.py", "file2.py"]
        self.mock_chunk.content = "+new line\n-removed line"
        self.mock_chunk.description = None

    def tearDown(self) -> None:
        """Clean up patches."""
        self.repo_root_patcher.stop()

    @patch("codemap.git.commit.command.generate_message")
    def test_generate_commit_message(self, mock_generate_message: Mock) -> None:
        """Test the _generate_commit_message method."""
        # Set up mock return value
        mock_generate_message.return_value = ("Test commit message", True)

        # Call the method - using private method is necessary for the test
        self.command._generate_commit_message(cast("DiffChunk", self.mock_chunk))  # noqa: SLF001

        # Verify results
        assert self.mock_chunk.description == "Test commit message"
        assert self.mock_chunk.is_llm_generated
        mock_generate_message.assert_called_once_with(self.mock_chunk, self.mock_message_generator)

    @patch("codemap.git.commit.command.run_git_command")
    @patch("codemap.git.commit.command.stage_files")
    @patch("codemap.git.commit.command.unstage_files")
    @patch("codemap.git.commit.command.get_staged_diff")
    @patch("codemap.git.commit.command.commit_only_files")
    def test_perform_commit(
        self, mock_commit: Mock, mock_get_staged: Mock, mock_unstage: Mock, mock_stage: Mock, mock_run_git: Mock
    ) -> None:
        """Test the _perform_commit method."""
        # Set up mocks
        mock_get_staged.return_value = Mock(files=["file1.py", "file2.py", "file3.py"])
        mock_commit.return_value = []

        # Call the method - using private method is necessary for the test
        result = self.command._perform_commit(cast("DiffChunk", self.mock_chunk), "Test commit message")  # noqa: SLF001

        # Verify results
        assert result
        mock_run_git.assert_called()
        mock_unstage.assert_called_with(["file3.py"])
        mock_stage.assert_called_with(self.mock_chunk.files)
        mock_commit.assert_called_with(self.mock_chunk.files, "Test commit message", ignore_hooks=False)
        self.mock_ui.show_success.assert_called()

    @patch.object(CommitCommand, "_generate_commit_message")
    @patch.object(CommitCommand, "_perform_commit")
    def test_process_chunk_accept(self, mock_perform_commit: Mock, mock_generate_message: Mock) -> None:
        """Test _process_chunk with ACCEPT action."""
        # Set up mocks
        mock_perform_commit.return_value = True
        self.mock_ui.process_chunk.return_value = ChunkResult(ChunkAction.ACCEPT, "Test message")

        # Call the method - using private method is necessary for the test
        result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)  # noqa: SLF001

        # Verify results
        assert result
        mock_generate_message.assert_called_once_with(self.mock_chunk)
        self.mock_ui.process_chunk.assert_called_once_with(self.mock_chunk, 0, 1)
        mock_perform_commit.assert_called_once_with(self.mock_chunk, "Test message")

    @patch.object(CommitCommand, "_generate_commit_message")
    def test_process_chunk_skip(self, mock_generate_message: Mock) -> None:
        """Test _process_chunk with SKIP action."""
        # Set up mocks
        self.mock_ui.process_chunk.return_value = ChunkResult(ChunkAction.SKIP)

        # Call the method - using private method is necessary for the test
        result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)  # noqa: SLF001

        # Verify results
        assert result
        mock_generate_message.assert_called_once_with(self.mock_chunk)
        self.mock_ui.process_chunk.assert_called_once_with(self.mock_chunk, 0, 1)
        self.mock_ui.show_skipped.assert_called_once_with(self.mock_chunk.files)

    @patch.object(CommitCommand, "_generate_commit_message")
    def test_process_chunk_abort(self, mock_generate_message: Mock) -> None:
        """Test _process_chunk with ABORT action."""
        # Set up mocks
        self.mock_ui.process_chunk.return_value = ChunkResult(ChunkAction.ABORT)
        self.mock_ui.confirm_abort.return_value = False

        # Call the method - using private method is necessary for the test
        result = self.command._process_chunk(cast("DiffChunk", self.mock_chunk), 0, 1)  # noqa: SLF001

        # Verify results
        assert not result
        mock_generate_message.assert_called_once_with(self.mock_chunk)
        self.mock_ui.process_chunk.assert_called_once_with(self.mock_chunk, 0, 1)
        self.mock_ui.confirm_abort.assert_called_once()

    @patch.object(CommitCommand, "_generate_commit_message")
    @patch.object(CommitCommand, "_perform_commit")
    def test_process_all_chunks_interactive(self, mock_perform_commit: Mock, mock_generate_message: Mock) -> None:  # noqa: ARG002
        """Test process_all_chunks in interactive mode."""
        # Set up mocks
        mock_chunk2 = Mock(spec=DiffChunk)
        mock_chunk2.files = ["file3.py"]
        mock_chunks = [cast("DiffChunk", self.mock_chunk), cast("DiffChunk", mock_chunk2)]

        # Set up patched _process_chunk
        with patch.object(self.command, "_process_chunk", return_value=True) as mock_process_chunk:
            # Call the method
            result = self.command.process_all_chunks(mock_chunks, interactive=True)

            # Verify results
            assert result
            assert mock_process_chunk.call_count == 2
            self.mock_ui.show_all_committed.assert_called_once()

    @patch.object(CommitCommand, "_generate_commit_message")
    @patch.object(CommitCommand, "_perform_commit")
    def test_process_all_chunks_non_interactive(self, mock_perform_commit: Mock, mock_generate_message: Mock) -> None:
        """Test process_all_chunks in non-interactive mode."""
        # Set up mocks
        mock_chunk2 = Mock(spec=DiffChunk)
        mock_chunk2.files = ["file3.py"]
        mock_chunks = [cast("DiffChunk", self.mock_chunk), cast("DiffChunk", mock_chunk2)]
        mock_perform_commit.return_value = True

        # Call the method
        result = self.command.process_all_chunks(mock_chunks, interactive=False)

        # Verify results
        assert result
        assert mock_generate_message.call_count == 2
        assert mock_perform_commit.call_count == 2
        self.mock_ui.show_all_committed.assert_called_once()
