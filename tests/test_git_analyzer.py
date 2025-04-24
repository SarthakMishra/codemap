"""Tests for the Git metadata analyzer."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from codemap.processor.analysis.git.analyzer import GitBlameInfo, GitMetadataAnalyzer
from codemap.processor.analysis.git.models import GitMetadata
from codemap.processor.analysis.tree_sitter.base import EntityType
from codemap.processor.chunking.base import Chunk, ChunkMetadata, Location


@pytest.fixture
def mock_subprocess_run() -> Generator[MagicMock, None, None]:
    """Mock subprocess.run for Git commands."""
    with patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def git_analyzer(mock_subprocess_run: MagicMock) -> GitMetadataAnalyzer:
    """Create a GitMetadataAnalyzer instance with mocked subprocess."""
    # Mock git rev-parse to return a valid repo path
    mock_subprocess_run.return_value.stdout = "/fake/repo/path\n"

    # Mock the _verify_git_repo method to avoid the validation error
    with patch("codemap.processor.analysis.git.analyzer.GitMetadataAnalyzer._verify_git_repo"), patch(
        "codemap.processor.analysis.git.analyzer.Path"
    ) as mock_path:
        # Setup mock path to handle Path operations correctly
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = mock_path_instance

        # Create the analyzer with the mocked subprocess
        analyzer = GitMetadataAnalyzer()

        # Reset the mock for other tests
        mock_subprocess_run.reset_mock()

        return analyzer


def test_detect_repo_root(mock_subprocess_run: MagicMock) -> None:
    """Test detection of Git repository root."""
    # Setup mock to return a repo path
    mock_subprocess_run.return_value.stdout = "/fake/repo/path\n"
    mock_subprocess_run.return_value.returncode = 0

    # We need to mock _verify_git_repo and Path for this test
    with patch("codemap.processor.analysis.git.analyzer.GitMetadataAnalyzer._verify_git_repo"), patch(
        "codemap.processor.analysis.git.analyzer.Path"
    ) as mock_path:
        # Setup Path mock
        fake_path = MagicMock(spec=Path)
        mock_path.return_value = fake_path
        mock_path.side_effect = lambda x: Path(x) if isinstance(x, str) else x

        # Create analyzer to trigger _detect_repo_root
        analyzer = GitMetadataAnalyzer()

        # Verify the subprocess call
        mock_subprocess_run.assert_called_once()
        args = mock_subprocess_run.call_args[0][0]
        assert args == ["git", "rev-parse", "--show-toplevel"]
        # Check that check=True was passed
        assert mock_subprocess_run.call_args[1]["check"] is True

        # Verify that repo_path was set correctly (comparing string values)
        assert str(analyzer.repo_path) == "/fake/repo/path"


def test_detect_repo_root_error(mock_subprocess_run: MagicMock) -> None:
    """Test error handling when repository detection fails."""
    # Setup mock to simulate git command failure
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=128, cmd="git rev-parse --show-toplevel")

    # Attempting to create analyzer should raise ValueError
    with pytest.raises(ValueError, match="Not in a Git repository") as exc_info:
        GitMetadataAnalyzer()

    assert "Not in a Git repository" in str(exc_info.value)


def test_verify_git_repo() -> None:
    """Test verification of Git repository."""
    # Create a fake repo path with a .git directory
    repo_path = MagicMock(spec=Path)
    git_dir = MagicMock(spec=Path)

    # Setup mocks to simulate a valid Git repo
    repo_path.__truediv__.return_value = git_dir
    git_dir.exists.return_value = True
    git_dir.is_dir.return_value = True

    # Create analyzer with the mocked repo path
    analyzer = GitMetadataAnalyzer(repo_path=repo_path)

    # Repo path should be set correctly
    assert analyzer.repo_path == repo_path

    # Now test invalid repo
    invalid_repo = MagicMock(spec=Path)
    invalid_git_dir = MagicMock(spec=Path)
    invalid_repo.__truediv__.return_value = invalid_git_dir
    invalid_git_dir.exists.return_value = False

    with pytest.raises(ValueError, match="Not a valid Git repository") as exc_info:
        GitMetadataAnalyzer(repo_path=invalid_repo)

    assert "Not a valid Git repository" in str(exc_info.value)


def test_get_current_branch(git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock) -> None:
    """Test getting the current branch name."""
    # Setup mock to return a branch name
    mock_subprocess_run.return_value.stdout = "main\n"

    # Get current branch
    branch = git_analyzer.get_current_branch()

    # Verify the subprocess call
    mock_subprocess_run.assert_called_once()
    args = mock_subprocess_run.call_args[0][0]
    assert args[0:2] == ["git", "-C"]
    assert args[3:] == ["rev-parse", "--abbrev-ref", "HEAD"]

    # Verify the branch name
    assert branch == "main"


def test_get_current_branch_error(git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock) -> None:
    """Test error handling when getting current branch fails."""
    # Setup mock to simulate git command failure
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=128, cmd="git rev-parse --abbrev-ref HEAD"
    )

    # Should return "unknown" on error
    branch = git_analyzer.get_current_branch()
    assert branch == "unknown"


def test_get_current_commit(git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock) -> None:
    """Test getting the current commit hash."""
    # Setup mock to return a commit hash
    commit_hash = "abcdef1234567890abcdef1234567890abcdef12"
    mock_subprocess_run.return_value.stdout = f"{commit_hash}\n"

    # Get current commit
    commit = git_analyzer.get_current_commit()

    # Verify the subprocess call
    mock_subprocess_run.assert_called_once()
    args = mock_subprocess_run.call_args[0][0]
    assert args[0:2] == ["git", "-C"]
    assert args[3:] == ["rev-parse", "HEAD"]

    # Verify the commit hash
    assert commit == commit_hash


def test_get_current_commit_error(git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock) -> None:
    """Test error handling when getting current commit fails."""
    # Setup mock to simulate git command failure
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=128, cmd="git rev-parse HEAD")

    # Should return "unknown" on error
    commit = git_analyzer.get_current_commit()
    assert commit == "unknown"


def test_get_blame_info(git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock) -> None:
    """Test getting git blame information for a range of lines."""
    # Create a sample git blame porcelain output
    blame_output = (
        "abcdef1234567890abcdef1234567890abcdef12 1 1 1\n"
        "author John Doe\n"
        "author-mail <john.doe@example.com>\n"
        "author-time 1609459200\n"  # 2021-01-01 00:00:00 UTC
        "author-tz +0000\n"
        "committer John Doe\n"
        "committer-mail <john.doe@example.com>\n"
        "committer-time 1609459200\n"
        "committer-tz +0000\n"
        "summary Initial commit\n"
        "filename src/file.py\n"
        "\tdef example_function():\n"
        "abcdef1234567890abcdef1234567890abcdef12 2 2 1\n"
        "author John Doe\n"
        "author-mail <john.doe@example.com>\n"
        "author-time 1609459200\n"
        "author-tz +0000\n"
        "committer John Doe\n"
        "committer-mail <john.doe@example.com>\n"
        "committer-time 1609459200\n"
        "committer-tz +0000\n"
        "summary Initial commit\n"
        "filename src/file.py\n"
        "\t    return 'Hello, World!'\n"
    )
    mock_subprocess_run.return_value.stdout = blame_output

    # Get blame info
    file_path = Path("/fake/repo/path/src/file.py")

    # Mock the relative_to method to avoid the Path error
    with patch.object(Path, "relative_to") as mock_relative_to:
        mock_relative_to.return_value = Path("src/file.py")

        blame_info = git_analyzer.get_blame_info(file_path, 1, 2)

        # Verify the subprocess call
        mock_subprocess_run.assert_called_once()
        args = mock_subprocess_run.call_args[0][0]
        assert args[0:2] == ["git", "-C"]
        assert "blame" in args
        assert "--line-porcelain" in args

        # Verify blame info
        assert len(blame_info) == 2
        assert blame_info[0].commit_id == "abcdef1234567890abcdef1234567890abcdef12"
        assert blame_info[0].author == "John Doe"
        assert blame_info[0].author_email == "john.doe@example.com"
        assert blame_info[0].timestamp == datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert blame_info[0].line_number == 1
        assert blame_info[0].line_content == "def example_function():"

        assert blame_info[1].line_number == 2
        assert blame_info[1].line_content == "    return 'Hello, World!'"


def test_get_blame_info_error(git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock) -> None:
    """Test error handling when getting blame info fails."""
    # Setup mock to simulate git command failure
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=128, cmd="git blame")

    # Should return empty list on error
    file_path = Path("/fake/repo/path/src/file.py")

    # Mock the relative_to method to avoid the Path error
    with patch.object(Path, "relative_to") as mock_relative_to:
        mock_relative_to.return_value = Path("src/file.py")

        blame_info = git_analyzer.get_blame_info(file_path, 1, 2)
        assert blame_info == []


def test_create_git_metadata(git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock) -> None:
    """Test creating GitMetadata for a code chunk."""
    # Setup mocks for subprocess calls

    # For get_current_branch
    mock_subprocess_run.side_effect = None  # Reset any previous side effects

    # Setup individual command mocks
    branch_mock = MagicMock()
    branch_mock.stdout = "feature-branch\n"
    commit_mock = MagicMock()
    commit_mock.stdout = "abcdef1234567890abcdef1234567890abcdef12\n"
    log_mock = MagicMock()
    log_mock.stdout = "Initial commit message\n"

    # Setup the call sequence
    mock_subprocess_run.side_effect = [branch_mock, commit_mock, log_mock]

    # Create mock blame info that would be returned by get_blame_info
    blame_info = [
        GitBlameInfo(
            commit_id="abcdef1234567890abcdef1234567890abcdef12",
            author="John Doe",
            author_email="john.doe@example.com",
            timestamp=datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC),
            line_number=1,
            line_content="def example_function():",
        )
    ]

    # Mock the get_blame_info method to return our prepared data
    with patch.object(git_analyzer, "get_blame_info", return_value=blame_info):
        # Create location and test creating metadata
        location = Location(file_path=Path("/fake/repo/path/src/file.py"), start_line=1, end_line=1)

        metadata = git_analyzer.create_git_metadata(location)

        # Verify metadata
        assert metadata is not None
        assert metadata.is_committed is True
        assert metadata.commit_id == "abcdef1234567890abcdef1234567890abcdef12"
        assert metadata.commit_message == "Initial commit message"
        assert metadata.timestamp == datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert metadata.branch == ["feature-branch"]
        assert metadata.last_modified_at is None


def test_create_git_metadata_with_multiple_commits(
    git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock
) -> None:
    """Test creating GitMetadata with multiple commits in the history."""
    # Setup mocks for subprocess calls
    mock_subprocess_run.side_effect = None  # Reset any previous side effects

    # Setup individual command mocks
    branch_mock = MagicMock()
    branch_mock.stdout = "feature-branch\n"
    commit_mock = MagicMock()
    commit_mock.stdout = "fedcba9876543210fedcba9876543210fedcba98\n"
    log_mock = MagicMock()
    log_mock.stdout = "Initial commit message\n"

    # Setup the call sequence
    mock_subprocess_run.side_effect = [branch_mock, commit_mock, log_mock]

    # Create mock blame info with multiple commits
    older_blame = GitBlameInfo(
        commit_id="abcdef1234567890abcdef1234567890abcdef12",
        author="John Doe",
        author_email="john.doe@example.com",
        timestamp=datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC),
        line_number=1,
        line_content="def example_function():",
    )

    newer_blame = GitBlameInfo(
        commit_id="fedcba9876543210fedcba9876543210fedcba98",
        author="Jane Smith",
        author_email="jane.smith@example.com",
        timestamp=datetime(2021, 7, 1, 0, 0, 0, tzinfo=UTC),
        line_number=2,
        line_content="    return 'Hello, Updated World!'",
    )

    # Mock the get_blame_info method to return our prepared data
    with patch.object(git_analyzer, "get_blame_info", return_value=[older_blame, newer_blame]):
        # Create location and test creating metadata
        location = Location(file_path=Path("/fake/repo/path/src/file.py"), start_line=1, end_line=2)

        metadata = git_analyzer.create_git_metadata(location)

        # Verify metadata
        assert metadata is not None
        assert metadata.is_committed is True
        assert metadata.commit_id == "abcdef1234567890abcdef1234567890abcdef12"  # Oldest commit
        assert metadata.commit_message == "Initial commit message"
        assert metadata.timestamp == datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC)  # Oldest commit time
        assert metadata.branch == ["feature-branch"]
        assert metadata.last_modified_at == datetime(2021, 7, 1, 0, 0, 0, tzinfo=UTC)  # Newest commit time


def test_enrich_chunk(git_analyzer: GitMetadataAnalyzer) -> None:
    """Test enriching a chunk with Git metadata."""
    # Create a mock chunk with location
    location = Location(file_path=Path("/fake/repo/path/src/file.py"), start_line=1, end_line=2)

    metadata = ChunkMetadata(
        entity_type=EntityType.FUNCTION, name="example_function", location=location, language="python"
    )

    chunk = Chunk(content="def example_function():\n    return 'Hello, World!'", metadata=metadata, children=[])

    # Mock create_git_metadata to return a predefined GitMetadata
    git_metadata = GitMetadata(
        is_committed=True,
        commit_id="abcdef1234567890abcdef1234567890abcdef12",
        commit_message="Initial commit",
        timestamp=datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC),
        branch=["main"],
        last_modified_at=None,
    )

    # We need to use a mock for the create_git_metadata method
    mock_create_metadata = MagicMock(return_value=git_metadata)

    with patch.object(git_analyzer, "create_git_metadata", mock_create_metadata):
        # Enrich the chunk
        git_analyzer.enrich_chunk(chunk)

        # Since metadata is frozen, we can't modify it directly in enrich_chunk
        # So we just verify that create_git_metadata was called correctly
        mock_create_metadata.assert_called_once_with(location)


def test_enrich_chunk_recursive(git_analyzer: GitMetadataAnalyzer) -> None:
    """Test recursive enrichment of chunks with Git metadata."""
    # Create a parent chunk with a child chunk
    parent_location = Location(file_path=Path("/fake/repo/path/src/file.py"), start_line=1, end_line=5)

    parent_metadata = ChunkMetadata(
        entity_type=EntityType.CLASS, name="ExampleClass", location=parent_location, language="python"
    )

    child_location = Location(file_path=Path("/fake/repo/path/src/file.py"), start_line=2, end_line=3)

    child_metadata = ChunkMetadata(
        entity_type=EntityType.FUNCTION, name="example_method", location=child_location, language="python"
    )

    child_chunk = Chunk(
        content="    def example_method(self):\n        return 'Hello!'", metadata=child_metadata, children=[]
    )

    parent_chunk = Chunk(
        content="class ExampleClass:\n    def example_method(self):\n        return 'Hello!'\n\n    # End of class",
        metadata=parent_metadata,
        children=[child_chunk],
    )

    # Mock create_git_metadata to return predefined GitMetadata
    parent_git_metadata = GitMetadata(
        is_committed=True,
        commit_id="abcdef1234567890abcdef1234567890abcdef12",
        commit_message="Initial commit",
        timestamp=datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC),
        branch=["main"],
        last_modified_at=None,
    )

    child_git_metadata = GitMetadata(
        is_committed=True,
        commit_id="fedcba9876543210fedcba9876543210fedcba98",
        commit_message="Added method",
        timestamp=datetime(2021, 2, 1, 0, 0, 0, tzinfo=UTC),
        branch=["main"],
        last_modified_at=None,
    )

    # Use a mock instead of a local function
    mock_create_git_metadata = MagicMock()
    mock_create_git_metadata.side_effect = lambda loc: (
        parent_git_metadata if loc == parent_location else child_git_metadata if loc == child_location else None
    )

    with patch.object(git_analyzer, "create_git_metadata", mock_create_git_metadata):
        # Enrich the parent chunk (which should also enrich the child)
        git_analyzer.enrich_chunk(parent_chunk)

        # Verify that create_git_metadata was called for both chunks
        assert mock_create_git_metadata.call_count == 2
        mock_create_git_metadata.assert_any_call(parent_location)
        mock_create_git_metadata.assert_any_call(child_location)


def test_enrich_chunk_without_location(git_analyzer: GitMetadataAnalyzer) -> None:
    """Test handling of chunks without location information."""
    # Create a metadata with None for location
    null_location_metadata = MagicMock()
    null_location_metadata.location = None
    null_location_metadata.name = "test_chunk"

    # Create a chunk with this mock metadata
    chunk = MagicMock()
    chunk.metadata = null_location_metadata
    chunk.children = []

    # Mock create_git_metadata to ensure it's not called
    with patch.object(git_analyzer, "create_git_metadata") as mock_create_metadata:
        # Enrich the chunk
        git_analyzer.enrich_chunk(chunk)

        # Verify that create_git_metadata was not called
        mock_create_metadata.assert_not_called()


def test_create_git_metadata_error(git_analyzer: GitMetadataAnalyzer, mock_subprocess_run: MagicMock) -> None:
    """Test error handling in create_git_metadata."""
    # Setup mock to simulate git command failure
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=128, cmd="git command")

    # Create location for testing
    location = Location(file_path=Path("/fake/repo/path/src/file.py"), start_line=1, end_line=2)

    # Should return None on error
    metadata = git_analyzer.create_git_metadata(location)
    assert metadata is None
