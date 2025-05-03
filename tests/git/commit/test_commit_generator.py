"""Tests for the commit generator module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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
		"""Test the CommitMessageGenerator.generate_message method."""
		# Create mock for required dependencies
		mock_llm_client = Mock()
		mock_config_loader = Mock()

		# Create an actual generator instance
		generator = CommitMessageGenerator(
			repo_root=Path("/mock/repo/path"),
			llm_client=mock_llm_client,
			prompt_template="test template",
			config_loader=mock_config_loader,
		)

		# Mock the methods we need
		with (
			patch.object(generator, "extract_file_info", return_value={}),
			patch.object(generator, "_call_llm_api", return_value="feat: Generated Commit Message"),
		):
			# Call generate_message directly
			message, used_llm = generator.generate_message(mock_diff_chunk)

			# Assert
			assert message == "feat: Generated Commit Message"
			assert used_llm is True

			# Test fallback as well
			fallback_message = generator.fallback_generation(mock_diff_chunk)
			assert len(fallback_message) > 0
			assert fallback_message.startswith("chore:")
