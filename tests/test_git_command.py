"""Tests for the CommitCommand class in git/command.py."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from codemap.git.command import CommitCommand
from codemap.git.diff_splitter import DiffChunk
from codemap.git.message_generator import LLMError
from codemap.utils.git_utils import GitDiff, GitError
from tests.base import GitTestBase


@pytest.mark.unit
@pytest.mark.git
class TestCommitCommandChanges(GitTestBase):
    """Test cases for the _get_changes method in CommitCommand.

    This tests the method that retrieves changes from the Git repository
    and converts them to GitDiff objects for further processing.

    """

    def setup_method(self) -> None:
        """Set up test environment with mocks."""
        # Initialize _patchers list needed by GitTestBase
        self._patchers = []

        # Patch get_repo_root to avoid actual Git operations
        self.mock_repo_path("/mock/repo/path")

        # Create mock objects for dependencies
        self.mock_ui = Mock()
        self.mock_splitter = Mock()
        self.mock_message_generator = Mock()

        # Create the CommitCommand with patched dependencies
        with patch("codemap.git.command.CommitUI", return_value=self.mock_ui), patch(
            "codemap.git.command.DiffSplitter", return_value=self.mock_splitter
        ), patch("codemap.git.command.MessageGenerator", return_value=self.mock_message_generator):
            self.command = CommitCommand()

    def test_get_changes_success(self) -> None:
        """Test successful retrieval of changes from the Git repository."""
        # Arrange: Set up mocks for git operations
        with patch("codemap.git.command.run_git_command") as mock_run_git, patch(
            "codemap.git.command.get_staged_diff"
        ) as mock_staged, patch("codemap.git.command.get_unstaged_diff") as mock_unstaged, patch(
            "codemap.git.command.get_untracked_files"
        ) as mock_untracked:
            # Configure mocks
            mock_run_git.return_value = ""  # Successful git add .

            # Create mock diffs
            staged_diff = GitDiff(
                files=["file1.py", "file2.py"],
                content="diff for file1 and file2",
                is_staged=True,
            )
            unstaged_diff = GitDiff(
                files=["file3.py"],
                content="diff for file3",
                is_staged=False,
            )

            mock_staged.return_value = staged_diff
            mock_unstaged.return_value = unstaged_diff
            mock_untracked.return_value = ["file4.py"]  # Untracked file

            # Act: Call the method
            changes = self.command._get_changes()

            # Assert: Verify results
            assert len(changes) == 3
            assert changes[0] == staged_diff
            assert changes[1] == unstaged_diff
            assert changes[2].files == ["file4.py"]
            assert changes[2].is_staged is False

            # Verify git add . was called
            mock_run_git.assert_called_once_with(["git", "add", "."])

    def test_get_changes_without_staged_or_unstaged(self) -> None:
        """Test getting changes when there are no staged or unstaged files."""
        # Arrange: Set up mocks for git operations
        with patch("codemap.git.command.run_git_command") as mock_run_git, patch(
            "codemap.git.command.get_staged_diff"
        ) as mock_staged, patch("codemap.git.command.get_unstaged_diff") as mock_unstaged, patch(
            "codemap.git.command.get_untracked_files"
        ) as mock_untracked:
            # Configure mocks
            mock_run_git.return_value = ""  # Successful git add .

            # Create mock diffs with empty file lists
            staged_diff = GitDiff(
                files=[],
                content="",
                is_staged=True,
            )
            unstaged_diff = GitDiff(
                files=[],
                content="",
                is_staged=False,
            )

            mock_staged.return_value = staged_diff
            mock_unstaged.return_value = unstaged_diff
            mock_untracked.return_value = ["untracked.py"]  # Only untracked file

            # Act: Call the method
            changes = self.command._get_changes()

            # Assert: Verify results
            assert len(changes) == 1  # Only untracked files
            assert changes[0].files == ["untracked.py"]
            assert changes[0].is_staged is False

    def test_get_changes_with_staging_error(self) -> None:
        """Test error handling when staging files fails."""
        # Arrange: Set up mocks for git operations
        with patch("codemap.git.command.run_git_command") as mock_run_git, patch(
            "codemap.git.command.get_staged_diff"
        ) as mock_staged, patch("codemap.git.command.get_unstaged_diff") as mock_unstaged, patch(
            "codemap.git.command.get_untracked_files"
        ) as mock_untracked, patch("codemap.git.command.logger") as mock_logger:
            # Configure mocks
            mock_run_git.side_effect = GitError("Failed to stage files")

            # Create mock diffs
            staged_diff = GitDiff(
                files=["file1.py"],
                content="diff for file1",
                is_staged=True,
            )
            unstaged_diff = GitDiff(
                files=["file2.py"],
                content="diff for file2",
                is_staged=False,
            )

            mock_staged.return_value = staged_diff
            mock_unstaged.return_value = unstaged_diff
            mock_untracked.return_value = []

            # Act: Call the method
            changes = self.command._get_changes()

            # Assert: Verify results
            assert len(changes) == 2
            assert changes[0] == staged_diff
            assert changes[1] == unstaged_diff

            # Verify error was logged
            mock_logger.warning.assert_called_once()
            assert "Failed to stage all changes" in mock_logger.warning.call_args[0][0]

    def test_get_changes_git_error(self) -> None:
        """Test handling of GitError when getting changes."""
        # Arrange: Set up mocks for git operations
        with patch("codemap.git.command.run_git_command") as mock_run_git, patch(
            "codemap.git.command.get_staged_diff"
        ) as mock_staged:
            # Configure mocks
            mock_run_git.return_value = ""  # Successful git add .
            mock_staged.side_effect = GitError("Failed to get staged diff")

            # Act and Assert: Should raise RuntimeError
            with pytest.raises(RuntimeError) as excinfo:
                self.command._get_changes()

            # Verify error message
            assert "Failed to get changes:" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.git
class TestCommitCommandMessageGeneration(GitTestBase):
    """Test cases for commit message generation in CommitCommand."""

    def setup_method(self) -> None:
        """Set up test environment with mocks."""
        # Initialize _patchers list needed by GitTestBase
        self._patchers = []

        # Patch get_repo_root to avoid actual Git operations
        self.mock_repo_path("/mock/repo/path")

        # Create mock objects for dependencies
        self.mock_ui = Mock()
        self.mock_splitter = Mock()
        self.mock_message_generator = Mock()

        # Create the CommitCommand with patched dependencies
        with patch("codemap.git.command.CommitUI", return_value=self.mock_ui), patch(
            "codemap.git.command.DiffSplitter", return_value=self.mock_splitter
        ), patch("codemap.git.command.MessageGenerator", return_value=self.mock_message_generator):
            self.command = CommitCommand()

        # Set up a mock chunk for testing
        self.mock_chunk = Mock(spec=DiffChunk)
        self.mock_chunk.files = ["file1.py", "file2.py"]
        self.mock_chunk.content = "+new line\n-removed line"
        self.mock_chunk.description = None

    def test_generate_commit_message_success(self) -> None:
        """Test successful commit message generation."""
        # Arrange: Set up mocks
        with patch("codemap.git.command.generate_message") as mock_generate, patch(
            "codemap.git.command.logger"
        ) as mock_logger:
            # Configure the mock to return a message and True for LLM-generated
            mock_generate.return_value = ("feat: Test commit message", True)

            # Act: Call the method
            self.command._generate_commit_message(self.mock_chunk)

            # Assert: Verify results
            assert self.mock_chunk.description == "feat: Test commit message"
            assert self.mock_chunk.is_llm_generated is True

            # Verify message was logged
            mock_logger.debug.assert_any_call("Starting commit message generation for %s", self.mock_chunk.files)
            mock_logger.debug.assert_any_call("Generated commit message using LLM: %s", "feat: Test commit message")

    def test_generate_commit_message_fallback(self) -> None:
        """Test fallback when LLM message generation fails."""
        # Arrange: Set up mocks
        with patch("codemap.git.command.generate_message") as mock_generate, patch(
            "codemap.git.command.logger"
        ) as mock_logger:
            # Configure the mock to return a fallback message
            mock_generate.return_value = ("chore: Update files", False)

            # Act: Call the method
            self.command._generate_commit_message(self.mock_chunk)

            # Assert: Verify results
            assert self.mock_chunk.description == "chore: Update files"
            assert self.mock_chunk.is_llm_generated is False

            # Verify message was logged as warning
            mock_logger.warning.assert_called_with(
                "Using automatically generated fallback message: %s", "chore: Update files"
            )

    def test_generate_commit_message_llm_error(self) -> None:
        """Test handling of LLM errors."""
        # Arrange: Set up mocks
        with patch("codemap.git.command.generate_message") as mock_generate, patch(
            "codemap.git.command.logger"
        ) as mock_logger:
            # Configure the mock to raise an LLMError
            mock_generate.side_effect = LLMError("Failed to generate message with LLM")

            # Configure fallback generation to return a message
            self.mock_message_generator.fallback_generation.return_value = "chore: Fallback commit message"

            # Act: Call the method
            self.command._generate_commit_message(self.mock_chunk)

            # Assert: Verify results
            assert self.mock_chunk.description == "chore: Fallback commit message"
            assert self.mock_chunk.is_llm_generated is False

            # Verify error was logged
            mock_logger.exception.assert_called_once()
            mock_logger.warning.assert_any_call("LLM error: %s", "Failed to generate message with LLM")
            mock_logger.warning.assert_any_call("Using fallback message: %s", "chore: Fallback commit message")

            # Verify fallback generation was called
            self.mock_message_generator.fallback_generation.assert_called_once()

    def test_generate_commit_message_with_existing_description(self) -> None:
        """Test LLM error handling with an existing description."""
        # Arrange: Set up mocks
        with patch("codemap.git.command.generate_message") as mock_generate, patch("codemap.git.command.logger"):
            # Set an existing description
            self.mock_chunk.description = "Pre-existing description"

            # Configure the mock to raise an LLMError
            mock_generate.side_effect = LLMError("Failed to generate message with LLM")

            # Act: Call the method
            self.command._generate_commit_message(self.mock_chunk)

            # Assert: Verify the description was passed to the fallback generator
            call_args = self.mock_message_generator.fallback_generation.call_args[0][0]
            # Just check that it's a dict-like object with files, content and description keys
            assert "files" in call_args
            assert "content" in call_args
            assert "description" in call_args
            assert call_args["description"] == "Pre-existing description"

    def test_generate_commit_message_other_error(self) -> None:
        """Test handling of other errors (ValueError, RuntimeError)."""
        # Arrange: Set up mocks
        with patch("codemap.git.command.generate_message") as mock_generate, patch(
            "codemap.git.command.logger"
        ) as mock_logger:
            # Configure the mock to raise a ValueError
            mock_generate.side_effect = ValueError("Invalid input for message generation")

            # Act and Assert: Should raise RuntimeError
            with pytest.raises(RuntimeError) as excinfo:
                self.command._generate_commit_message(self.mock_chunk)

            # Verify error message
            assert "Failed to generate commit message:" in str(excinfo.value)
            mock_logger.warning.assert_called_with("Other error: %s", "Invalid input for message generation")


@pytest.mark.unit
@pytest.mark.git
class TestCommitCommandHookBypass(GitTestBase):
    """Test cases for handling Git hook failures and bypass options."""

    def setup_method(self) -> None:
        """Set up test environment with mocks."""
        # Initialize _patchers list needed by GitTestBase
        self._patchers = []

        # Patch get_repo_root to avoid actual Git operations
        self.mock_repo_path("/mock/repo/path")

        # Create mock objects for dependencies
        self.mock_ui = Mock()
        self.mock_splitter = Mock()
        self.mock_message_generator = Mock()

        # Create the CommitCommand with patched dependencies
        with patch("codemap.git.command.CommitUI", return_value=self.mock_ui), patch(
            "codemap.git.command.DiffSplitter", return_value=self.mock_splitter
        ), patch("codemap.git.command.MessageGenerator", return_value=self.mock_message_generator):
            self.command = CommitCommand()

        # Set up a mock chunk for testing
        self.mock_chunk = Mock(spec=DiffChunk)
        self.mock_chunk.files = ["file1.py", "file2.py"]
        self.mock_chunk.content = "+new line\n-removed line"

    def test_perform_commit_with_hook_failure_bypass_accepted(self) -> None:
        """Test commit with hook failure where user accepts bypass."""
        # Arrange: Set up mocks
        with patch("codemap.git.command.run_git_command") as mock_run_git, patch(
            "codemap.git.command.get_staged_diff"
        ) as mock_staged, patch("codemap.git.command.stage_files"), patch("codemap.git.command.unstage_files"), patch(
            "codemap.git.command.commit_only_files"
        ) as mock_commit:
            # Configure mocks
            mock_run_git.return_value = ""
            mock_staged.return_value = GitDiff(files=self.mock_chunk.files, content="", is_staged=True)

            # First call raises hook error, second call succeeds
            mock_commit.side_effect = [GitError("hook failed with exit code 1"), []]

            # Configure UI to accept bypass
            self.mock_ui.confirm_bypass_hooks.return_value = True

            # Act: Call the method
            result = self.command._perform_commit(self.mock_chunk, "Test commit message")

            # Assert: Verify results
            assert result is True
            assert mock_commit.call_count == 2

            # Verify calls were made with correct parameters
            # The keyword arg should be ignore_hooks=False, then ignore_hooks=True
            assert mock_commit.call_args_list[0].kwargs.get("ignore_hooks") is False
            assert mock_commit.call_args_list[1].kwargs.get("ignore_hooks") is True

            # Verify UI confirms
            self.mock_ui.confirm_bypass_hooks.assert_called_once()
            self.mock_ui.show_success.assert_called_once()

    def test_perform_commit_with_hook_failure_bypass_rejected(self) -> None:
        """Test commit with hook failure where user rejects bypass."""
        # Arrange: Set up mocks
        with patch("codemap.git.command.run_git_command") as mock_run_git, patch(
            "codemap.git.command.get_staged_diff"
        ) as mock_staged, patch("codemap.git.command.stage_files"), patch("codemap.git.command.unstage_files"), patch(
            "codemap.git.command.commit_only_files"
        ) as mock_commit:
            # Configure mocks
            mock_run_git.return_value = ""
            mock_staged.return_value = GitDiff(files=self.mock_chunk.files, content="", is_staged=True)

            # Commit raises hook error
            mock_commit.side_effect = GitError("hook failed with exit code 1")

            # Configure UI to reject bypass
            self.mock_ui.confirm_bypass_hooks.return_value = False

            # Act: Call the method
            result = self.command._perform_commit(self.mock_chunk, "Test commit message")

            # Assert: Verify results
            assert result is False
            assert mock_commit.call_count == 1

            # Verify call was made with ignore_hooks=False
            assert mock_commit.call_args.kwargs.get("ignore_hooks") is False

            # Verify UI confirms
            self.mock_ui.confirm_bypass_hooks.assert_called_once()
            self.mock_ui.show_error.assert_called_once()

    def test_perform_commit_with_non_hook_error(self) -> None:
        """Test commit with error that is not hook-related."""
        # Arrange: Set up mocks
        with patch("codemap.git.command.run_git_command") as mock_run_git, patch(
            "codemap.git.command.get_staged_diff"
        ) as mock_staged, patch("codemap.git.command.stage_files"), patch("codemap.git.command.unstage_files"), patch(
            "codemap.git.command.commit_only_files"
        ) as mock_commit:
            # Configure mocks
            mock_run_git.return_value = ""
            mock_staged.return_value = GitDiff(files=self.mock_chunk.files, content="", is_staged=True)

            # Commit raises non-hook error
            mock_commit.side_effect = GitError("failed to commit changes")

            # Act: Call the method
            result = self.command._perform_commit(self.mock_chunk, "Test commit message")

            # Assert: Verify results
            assert result is False
            assert mock_commit.call_count == 1

            # Verify UI was not asked about hooks
            self.mock_ui.confirm_bypass_hooks.assert_not_called()
            self.mock_ui.show_error.assert_called_once()


@pytest.mark.unit
@pytest.mark.git
class TestCommitCommandRun(GitTestBase):
    """Test cases for the run method in CommitCommand.

    This tests the main entry point method that drives the commit
    workflow.

    """

    def setup_method(self) -> None:
        """Set up test environment with mocks."""
        # Initialize _patchers list needed by GitTestBase
        self._patchers = []

        # Patch get_repo_root to avoid actual Git operations
        self.mock_repo_path("/mock/repo/path")

        # Create mock objects for dependencies
        self.mock_ui = Mock()
        # Remove the explicit mocks for methods that aren't actually called
        self.mock_ui.show_error = Mock()
        self.mock_ui.show_all_committed = Mock()

        self.mock_splitter = Mock()
        # Add necessary methods that are called on the splitter
        self.mock_splitter._check_sentence_transformers_availability = Mock()
        self.mock_splitter._sentence_transformers_available = False

        self.mock_message_generator = Mock()

        # Create the CommitCommand with patched dependencies
        with patch("codemap.git.command.CommitUI", return_value=self.mock_ui), patch(
            "codemap.git.command.DiffSplitter", return_value=self.mock_splitter
        ), patch("codemap.git.command.MessageGenerator", return_value=self.mock_message_generator):
            self.command = CommitCommand()

    @patch.object(CommitCommand, "_get_changes")
    @patch.object(CommitCommand, "process_all_chunks")
    def test_run_success(self, mock_process_chunks: Mock, mock_get_changes: Mock) -> None:
        """Test successful run of the commit workflow."""
        # Arrange: Set up mocks
        # Create some mock changes
        mock_diff = GitDiff(files=["file1.py"], content="diff content", is_staged=True)
        mock_get_changes.return_value = [mock_diff]

        # Create a mock chunk
        mock_chunk = Mock(spec=DiffChunk)
        mock_chunk.files = ["file1.py"]
        # The actual method call includes 'semantic' parameter
        self.mock_splitter.split_diff.return_value = [mock_chunk]

        # Configure process_all_chunks to succeed
        mock_process_chunks.return_value = True

        # Act: Call the method
        result = self.command.run()

        # Assert: Verify results
        assert result is True
        mock_get_changes.assert_called_once()

        # Just verify that the method was called without checking parameters
        assert self.mock_splitter.split_diff.call_count == 1
        assert mock_process_chunks.call_count == 1

        # No UI methods should be called for success case except show_all_committed
        # which is called inside process_all_chunks

    @patch.object(CommitCommand, "_get_changes")
    @patch.object(CommitCommand, "process_all_chunks")
    def test_run_no_changes(self, mock_process_chunks: Mock, mock_get_changes: Mock) -> None:
        """Test run when there are no changes to commit."""
        # Arrange: Set up mocks
        # No changes
        mock_get_changes.return_value = []

        # Act: Call the method
        result = self.command.run()

        # Assert: Verify results
        assert result is False
        mock_get_changes.assert_called_once()
        # Verify the right methods were called
        assert self.mock_splitter.split_diff.call_count == 0
        assert mock_process_chunks.call_count == 0

        # The code calls show_error, not show_no_changes
        assert self.mock_ui.show_error.call_count == 1

    @patch.object(CommitCommand, "_get_changes")
    def test_run_with_get_changes_error(self, mock_get_changes: Mock) -> None:
        """Test run when _get_changes raises an error."""
        # Arrange: Set up mocks
        # Simulate error
        mock_get_changes.side_effect = RuntimeError("Failed to get changes")

        # Act: Call the method
        result = self.command.run()

        # Assert: Verify results
        assert result is False
        mock_get_changes.assert_called_once()
        assert self.mock_splitter.split_diff.call_count == 0
        assert self.mock_ui.show_error.call_count == 1

    @patch.object(CommitCommand, "_get_changes")
    @patch.object(CommitCommand, "process_all_chunks")
    def test_run_with_process_chunks_error(self, mock_process_chunks: Mock, mock_get_changes: Mock) -> None:
        """Test run when process_all_chunks returns False."""
        # Arrange: Set up mocks
        # Create some mock changes
        mock_diff = GitDiff(files=["file1.py"], content="diff content", is_staged=True)
        mock_get_changes.return_value = [mock_diff]

        # Create a mock chunk
        mock_chunk = Mock(spec=DiffChunk)
        mock_chunk.files = ["file1.py"]
        self.mock_splitter.split_diff.return_value = [mock_chunk]

        # Configure process_all_chunks to fail
        mock_process_chunks.return_value = False

        # Act: Call the method
        result = self.command.run()

        # Assert: Verify results
        assert result is False
        mock_get_changes.assert_called_once()
        assert self.mock_splitter.split_diff.call_count == 1
        assert mock_process_chunks.call_count == 1

    @patch.object(CommitCommand, "_get_changes")
    def test_run_with_no_content_after_split(self, mock_get_changes: Mock) -> None:
        """Test run when split_diff returns empty list."""
        # Arrange: Set up mocks
        # Create some mock changes
        mock_diff = GitDiff(files=["file1.py"], content="diff content", is_staged=True)
        mock_get_changes.return_value = [mock_diff]

        # Split returns empty list
        self.mock_splitter.split_diff.return_value = []

        # Act: Call the method
        result = self.command.run()

        # Assert: Verify results
        assert result is True  # When there are no chunks after split, it returns True
        mock_get_changes.assert_called_once()
        assert self.mock_splitter.split_diff.call_count == 1

        # No UI methods should be called in this case
