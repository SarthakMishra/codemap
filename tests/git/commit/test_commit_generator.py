"""Tests for the commit generator module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from codemap.cli.commit_cmd import GenerationMode, generate_commit_message
from codemap.git.commit_generator import CommitMessageGenerator
from codemap.git.diff_splitter import DiffChunk
from codemap.git.utils import GitDiff
from tests.base import GitTestBase


@pytest.fixture
def git_diff() -> GitDiff:
	"""Create a mock GitDiff with sample content."""
	return GitDiff(
		files=["file1.py"],
		content="""diff --git a/file1.py b/file1.py
index 1234567..abcdefg 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@ def existing_function():
    return True

def new_function():
-    return False
+    return True
""",
		is_staged=False,
	)


@pytest.fixture
def mock_embeddings() -> Mock:
	"""Create a mock embeddings provider."""
	return Mock()


@pytest.fixture
def mock_process() -> Mock:
	"""Create a mock process."""
	return Mock()


@pytest.fixture
def mock_diff_chunk(git_diff: GitDiff) -> DiffChunk:
	"""Create a mock DiffChunk from the git diff."""
	return DiffChunk(
		files=git_diff.files,
		content=git_diff.content,
		description=None,
	)


@pytest.mark.unit
@pytest.mark.git
class TestCommitMessageGenerator(GitTestBase):
	"""Test the commit message generator functionality."""

	def setup_method(self) -> None:
		"""Set up test environment with mocks."""
		# Initialize _patchers list needed by GitTestBase
		self._patchers = []

		# Patch get_repo_root to avoid actual Git operations
		self.mock_repo_path("/mock/repo/path")

	def test_generate_commit_message(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test the generate_commit_message function."""
		# Create mock generator
		mock_generator = Mock(spec=CommitMessageGenerator)
		mock_generator.generate_message.return_value = ("Generated Commit Message", True)
		mock_generator.fallback_generation.return_value = "fallback: Generated Commit Message"

		# Test with different mode parameters
		with patch("codemap.cli.commit_cmd.generate_message") as mock_gen_message:
			mock_gen_message.return_value = ("Generated Commit Message", True)

			# Call the function under test with the expected parameters
			result = generate_commit_message(mock_diff_chunk, mock_generator, GenerationMode.SMART)

			# Assert
			assert len(result[0]) > 0
			assert "Generated Commit Message" in result[0]
			assert result[1] is True  # Should indicate LLM was used
			mock_gen_message.assert_called_once()
