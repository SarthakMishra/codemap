"""Global test fixtures and configuration."""

from __future__ import annotations

import os
import shutil
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console

from codemap.git.commit_generator import CommitMessageGenerator
from codemap.git.diff_splitter import DiffChunk, DiffSplitter
from codemap.git.utils import GitDiff

if TYPE_CHECKING:
	from collections.abc import Generator


# Skip git-dependent tests when SKIP_GIT_TESTS environment variable is set
skip_git_tests = pytest.mark.skipif(
	os.environ.get("SKIP_GIT_TESTS") == "1", reason="Git-dependent tests are skipped in CI environment"
)


@pytest.fixture(autouse=True)
def cleanup_docs_dir() -> Generator[None, None, None]:
	"""
	Auto-use fixture to clean up any docs directories created by tests.

	This ensures we don't leave behind test files in the actual project
	directory.

	"""
	# Setup - nothing to do
	yield

	# Cleanup
	project_root = Path.cwd()
	docs_paths = [project_root / "docs", project_root / "documentation", project_root / "custom_docs_dir"]

	for path in docs_paths:
		if path.exists() and path.is_dir():
			# Don't delete the directory if it's part of the original repo structure
			# Instead, just delete any files that were created during tests
			for item in path.iterdir():
				# Skip .gitkeep and other special files
				if item.name.startswith("."):
					continue

				try:
					if item.is_dir():
						shutil.rmtree(item)
					else:
						item.unlink()
				except (PermissionError, OSError):
					pass


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
	"""Create a temporary directory for testing."""
	yield tmp_path
	# Cleanup
	if tmp_path.exists():
		shutil.rmtree(tmp_path)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
	"""Create a copy of the sample repository for testing."""
	fixtures_path = Path(__file__).parent / "fixtures" / "sample_repo"
	repo_path = tmp_path / "sample_repo"
	shutil.copytree(fixtures_path, repo_path)
	return repo_path


@pytest.fixture
def console() -> Console:
	"""Create a rich console for testing."""
	return Console(highlight=False)


@pytest.fixture
def mock_git_diff() -> GitDiff:
	"""Create a mock GitDiff with sample content."""
	return GitDiff(
		files=["file1.py", "file2.py"],
		content="""diff --git a/file1.py b/file1.py
index 1234567..abcdefg 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@ def existing_function():
     return True

 def new_function():
-    return False
+    return True
diff --git a/file2.py b/file2.py
index 2345678..bcdefgh 100644
--- a/file2.py
+++ b/file2.py
@@ -5,3 +5,6 @@ def old_function():
     # Some code
     pass
+
+def added_function():
+    return "Hello, World!"
""",
		is_staged=False,
	)


@pytest.fixture
def mock_git_utils() -> Generator[dict[str, Mock], None, None]:
	"""Create a standardized mock for git utilities."""
	with (
		patch("codemap.git.utils.get_staged_diff") as mock_staged,
		patch("codemap.git.utils.get_unstaged_diff") as mock_unstaged,
		patch("codemap.git.utils.get_untracked_files") as mock_untracked,
		patch("codemap.git.utils.commit_only_files") as mock_commit,
	):
		# Mock the staged diff
		staged_diff = GitDiff(
			files=["file1.py"],
			content="diff content for file1.py",
			is_staged=True,
		)
		mock_staged.return_value = staged_diff

		# Mock the unstaged diff
		unstaged_diff = GitDiff(
			files=["file2.py"],
			content="diff content for file2.py",
			is_staged=False,
		)
		mock_unstaged.return_value = unstaged_diff

		# Mock untracked files
		mock_untracked.return_value = ["file3.py"]

		# Mock commit
		mock_commit.return_value = []

		yield {
			"get_staged_diff": mock_staged,
			"get_unstaged_diff": mock_unstaged,
			"get_untracked_files": mock_untracked,
			"commit_only_files": mock_commit,
		}


@pytest.fixture
def mock_diff_splitter() -> Mock:
	"""Create a mock DiffSplitter."""
	splitter = Mock(spec=DiffSplitter)
	mock_chunk = Mock(spec=DiffChunk)
	mock_chunk.files = ["file1.py"]
	mock_chunk.content = "+new line\n-removed line"
	mock_chunk.description = None
	splitter.split_diff.return_value = [mock_chunk]
	return splitter


@pytest.fixture
def mock_message_generator() -> MagicMock:
	"""Create a standardized mock MessageGenerator."""
	generator = MagicMock(spec=CommitMessageGenerator)
	# Mock the generate_message method
	generator.generate_message.return_value = ("feat: Test commit message", True)
	# Mock the generate_message_with_linting method
	generator.generate_message_with_linting.return_value = ("feat: Test commit message", True, True)
	# Mock the fallback_generation method
	generator.fallback_generation.return_value = "test: Fallback message"
	# Set resolved_provider
	generator.resolved_provider = "openai"
	# Set up a dict for api_keys
	generator._api_keys = {}
	return generator


@pytest.fixture
def mock_stdin() -> Generator[StringIO, None, None]:
	"""Mock stdin for testing interactive inputs."""
	stdin = StringIO()
	with patch("sys.stdin", stdin):
		yield stdin
