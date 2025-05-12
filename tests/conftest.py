"""Global test fixtures and configuration."""

from __future__ import annotations

import os
import shutil
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from rich.console import Console

from codemap.config.config_loader import ConfigLoader
from codemap.config.config_schema import AppConfigSchema, LLMSchema
from codemap.git.commit_generator import CommitMessageGenerator
from codemap.git.diff_splitter import DiffChunk, DiffSplitter
from codemap.git.utils import GitDiff

if TYPE_CHECKING:
	from collections.abc import Generator


# Skip database-dependent tests when SKIP_DB_TESTS environment variable is set
skip_db_tests = pytest.mark.skipif(
	os.environ.get("SKIP_DB_TESTS") == "1",
	reason="Database-dependent tests are skipped in environments without PostgreSQL",
)

# Skip git tests when SKIP_GIT_TESTS environment variable is set
skip_git_tests = pytest.mark.skipif(
	os.environ.get("SKIP_GIT_TESTS") == "1",
	reason="Git tests are skipped in environments with SKIP_GIT_TESTS=1",
)


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


@pytest.fixture(autouse=os.environ.get("SKIP_GIT_TESTS") == "1")
def auto_mock_git_utils() -> Generator[dict[str, Mock], None, None]:
	"""Create a standardized mock for all git utilities automatically if SKIP_GIT_TESTS is enabled."""
	if os.environ.get("SKIP_GIT_TESTS") == "1":
		with (
			patch("codemap.git.utils.ExtendedGitRepoContext.get_repo_root") as mock_get_repo_root,
			patch("codemap.git.utils.ExtendedGitRepoContext.validate_repo_path") as mock_validate_repo_path,
			patch("codemap.git.utils.ExtendedGitRepoContext.get_staged_diff") as mock_staged,
			patch("codemap.git.utils.ExtendedGitRepoContext.get_unstaged_diff") as mock_unstaged,
			patch("codemap.git.utils.ExtendedGitRepoContext.stage_files") as mock_stage_files,
			patch("codemap.git.utils.ExtendedGitRepoContext.commit") as mock_commit_command,
			patch("codemap.git.utils.ExtendedGitRepoContext.get_other_staged_files") as mock_get_other_staged,
			patch("codemap.git.utils.ExtendedGitRepoContext.stash_staged_changes") as mock_stash,
			patch("codemap.git.utils.ExtendedGitRepoContext.unstash_changes") as mock_unstash,
			patch("codemap.git.utils.ExtendedGitRepoContext.get_untracked_files") as mock_untracked,
			patch("codemap.git.utils.ExtendedGitRepoContext.commit_only_files") as mock_commit,
			patch("codemap.git.utils.ExtendedGitRepoContext.unstage_files") as mock_unstage,
			patch("codemap.git.utils.ExtendedGitRepoContext.switch_branch") as mock_switch_branch,
			patch("codemap.git.utils.ExtendedGitRepoContext.get_current_branch") as mock_current_branch,
			patch("codemap.git.utils.ExtendedGitRepoContext.is_git_ignored") as mock_is_git_ignored,
			patch("codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_instance") as mock_pr_get_instance,
		):
			# Mock get_repo_root
			mock_get_repo_root.return_value = Path("/mock/repo/root")

			# Mock validate_repo_path
			mock_validate_repo_path.return_value = Path("/mock/repo/root")

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

			# Mock stage_files (void function)
			mock_stage_files.return_value = None

			# Mock commit (void function)
			mock_commit_command.return_value = None

			# Mock get_other_staged_files
			mock_get_other_staged.return_value = []

			# Mock stash_staged_changes
			mock_stash.return_value = False

			# Mock unstash_changes (void function)
			mock_unstash.return_value = None

			# Mock untracked files
			mock_untracked.return_value = ["file3.py"]

			# Mock commit_only_files
			mock_commit.return_value = []

			# Mock unstage_files (void function)
			mock_unstage.return_value = None

			# Mock switch_branch (void function)
			mock_switch_branch.return_value = None

			# Mock get_current_branch
			mock_current_branch.return_value = "main"

			# Mock PR utils
			mock_pr_instance = MagicMock()
			mock_pr_instance.get_current_branch.return_value = "main"
			mock_pr_get_instance.return_value = mock_pr_instance

			# Mock is_git_ignored
			mock_is_git_ignored.return_value = False

			yield {
				"get_repo_root": mock_get_repo_root,
				"validate_repo_path": mock_validate_repo_path,
				"get_staged_diff": mock_staged,
				"get_unstaged_diff": mock_unstaged,
				"stage_files": mock_stage_files,
				"commit": mock_commit_command,
				"get_other_staged_files": mock_get_other_staged,
				"stash_staged_changes": mock_stash,
				"unstash_changes": mock_unstash,
				"get_untracked_files": mock_untracked,
				"commit_only_files": mock_commit,
				"unstage_files": mock_unstage,
				"switch_branch": mock_switch_branch,
				"get_current_branch": mock_current_branch,
				"is_git_ignored": mock_is_git_ignored,
				"pr_get_instance": mock_pr_get_instance,
			}
	else:
		# When SKIP_GIT_TESTS is not enabled, do nothing
		yield {}


@pytest.fixture
def mock_git_utils() -> Generator[dict[str, Mock], None, None]:
	"""Create a standardized mock for all git utilities."""
	with (
		patch("codemap.git.utils.ExtendedGitRepoContext.get_repo_root") as mock_get_repo_root,
		patch("codemap.git.utils.ExtendedGitRepoContext.validate_repo_path") as mock_validate_repo_path,
		patch("codemap.git.utils.ExtendedGitRepoContext.get_staged_diff") as mock_staged,
		patch("codemap.git.utils.ExtendedGitRepoContext.get_unstaged_diff") as mock_unstaged,
		patch("codemap.git.utils.ExtendedGitRepoContext.stage_files") as mock_stage_files,
		patch("codemap.git.utils.ExtendedGitRepoContext.commit") as mock_commit_command,
		patch("codemap.git.utils.ExtendedGitRepoContext.get_other_staged_files") as mock_get_other_staged,
		patch("codemap.git.utils.ExtendedGitRepoContext.stash_staged_changes") as mock_stash,
		patch("codemap.git.utils.ExtendedGitRepoContext.unstash_changes") as mock_unstash,
		patch("codemap.git.utils.ExtendedGitRepoContext.get_untracked_files") as mock_untracked,
		patch("codemap.git.utils.ExtendedGitRepoContext.commit_only_files") as mock_commit,
		patch("codemap.git.utils.ExtendedGitRepoContext.unstage_files") as mock_unstage,
		patch("codemap.git.utils.ExtendedGitRepoContext.switch_branch") as mock_switch_branch,
		patch("codemap.git.utils.ExtendedGitRepoContext.get_current_branch") as mock_current_branch,
		patch("codemap.git.utils.ExtendedGitRepoContext.is_git_ignored") as mock_is_git_ignored,
		patch("codemap.git.pr_generator.pr_git_utils.PRGitUtils.get_instance") as mock_pr_get_instance,
	):
		# Mock get_repo_root
		mock_get_repo_root.return_value = Path("/mock/repo/root")

		# Mock validate_repo_path
		mock_validate_repo_path.return_value = Path("/mock/repo/root")

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

		# Mock stage_files (void function)
		mock_stage_files.return_value = None

		# Mock commit (void function)
		mock_commit_command.return_value = None

		# Mock get_other_staged_files
		mock_get_other_staged.return_value = []

		# Mock stash_staged_changes
		mock_stash.return_value = False

		# Mock unstash_changes (void function)
		mock_unstash.return_value = None

		# Mock untracked files
		mock_untracked.return_value = ["file3.py"]

		# Mock commit_only_files
		mock_commit.return_value = []

		# Mock unstage_files (void function)
		mock_unstage.return_value = None

		# Mock switch_branch (void function)
		mock_switch_branch.return_value = None

		# Mock get_current_branch
		mock_current_branch.return_value = "main"

		# Mock PR utils
		mock_pr_instance = MagicMock()
		mock_pr_instance.get_current_branch.return_value = "main"
		mock_pr_get_instance.return_value = mock_pr_instance

		# Mock is_git_ignored
		mock_is_git_ignored.return_value = False

		yield {
			"get_repo_root": mock_get_repo_root,
			"validate_repo_path": mock_validate_repo_path,
			"get_staged_diff": mock_staged,
			"get_unstaged_diff": mock_unstaged,
			"stage_files": mock_stage_files,
			"commit": mock_commit_command,
			"get_other_staged_files": mock_get_other_staged,
			"stash_staged_changes": mock_stash,
			"unstash_changes": mock_unstash,
			"get_untracked_files": mock_untracked,
			"commit_only_files": mock_commit,
			"unstage_files": mock_unstage,
			"switch_branch": mock_switch_branch,
			"get_current_branch": mock_current_branch,
			"is_git_ignored": mock_is_git_ignored,
			"pr_get_instance": mock_pr_get_instance,
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


@pytest.fixture
def mock_config_loader():
	"""Create a mock ConfigLoader for tests, reflecting new Pydantic schema."""
	mock_loader = Mock(spec=ConfigLoader)

	# Mock the AppConfigSchema instance that config_loader.get would return
	mock_app_config = Mock(spec=AppConfigSchema)

	# Mock the LLMSchema for the .llm attribute of AppConfigSchema
	mock_llm_settings = Mock(spec=LLMSchema)
	# Preserve test values, adapt to new schema (provider merged into model, max_tokens -> max_output_tokens)
	mock_llm_settings.model = "openai:gpt-4"
	mock_llm_settings.temperature = 0.7
	mock_llm_settings.max_output_tokens = 1000

	# Assign the mocked LLMSchema to the .llm attribute of the mocked AppConfigSchema
	mock_app_config.llm = mock_llm_settings

	# Mock embedding configuration for TreeSitterChunker tests
	from codemap.config.config_schema import EmbeddingChunkingSchema, EmbeddingSchema

	mock_chunking_config = Mock(spec=EmbeddingChunkingSchema)
	mock_chunking_config.max_hierarchy_depth = 2
	mock_chunking_config.max_file_lines = 1000

	mock_embedding_config = Mock(spec=EmbeddingSchema)
	mock_embedding_config.chunking = mock_chunking_config

	mock_app_config.embedding = mock_embedding_config

	# Make the .get property of the mock_loader return the mock_app_config
	# We use type() to mock a property on an instance
	type(mock_loader).get = PropertyMock(return_value=mock_app_config)

	return mock_loader
