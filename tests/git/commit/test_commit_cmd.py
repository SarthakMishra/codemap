"""Tests for commit command module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console

from codemap.cli.commit_cmd import (
	ChunkContext,
	CommitOptions,
	GenerationMode,
	RunConfig,
	_commit_changes,
	_commit_with_message,
	_edit_commit_message,
	_load_prompt_template,
	_perform_commit,
	_raise_command_failed_error,
	validate_and_process_commit,
)
from codemap.git.diff_splitter import DiffChunk
from tests.base import GitTestBase

# Import fixtures
pytest.importorskip("dotenv")

# Constants for testing
FAKE_REPO_PATH = Path("/fake/repo")

# Example test data for chunks
TEST_CHUNK = DiffChunk(
	files=["file1.py", "file2.py"],
	content="diff --git a/file1.py b/file1.py\n@@ -1,3 +1,3 @@\n-def old():\n+def new():\n     pass",
)


@pytest.fixture
def mock_console() -> Console:
	"""Create a mock console for testing."""
	return MagicMock(spec=Console)


@pytest.fixture
def mock_diff_chunk() -> DiffChunk:
	"""Create a mock DiffChunk for testing."""
	chunk = Mock(spec=DiffChunk)
	chunk.files = ["file1.py", "file2.py"]
	chunk.content = """
diff --git a/file1.py b/file1.py
index 1234567..abcdef0 100644
--- a/file1.py
+++ b/file1.py
@@ -1,7 +1,7 @@
-def old_function():
+def new_function():
     return True

+def added_function():
+    return True
"""
	chunk.description = None
	chunk.is_llm_generated = False
	return chunk


@pytest.fixture
def commit_options() -> CommitOptions:
	"""Create CommitOptions for testing."""
	return CommitOptions(
		repo_path=Path("/fake/repo"),
		generation_mode=GenerationMode.SMART,
		model="openai/gpt-4o-mini",
		api_base=None,
		commit=True,
		prompt_template=None,
		api_key=None,
	)


@pytest.mark.unit
@pytest.mark.fs
class TestLoadPromptTemplate:
	"""Test loading prompt templates from files."""

	def test_load_prompt_template_exists(self, tmp_path: Path) -> None:
		"""Test loading a prompt template that exists."""
		template_path = tmp_path / "template.txt"
		template_content = "This is a test template"
		template_path.write_text(template_content)

		assert _load_prompt_template(str(template_path)) == template_content

	def test_load_prompt_template_not_exists(self) -> None:
		"""Test loading a prompt template that doesn't exist."""
		with patch("codemap.cli.commit_cmd.show_warning") as mock_show_warning:
			assert _load_prompt_template("/nonexistent/path.txt") is None
			mock_show_warning.assert_called_once()

	def test_load_prompt_template_none(self) -> None:
		"""Test loading with None path."""
		assert _load_prompt_template(None) is None


@pytest.mark.unit
@pytest.mark.git
class TestCommitChanges(GitTestBase):
	"""Test committing changes."""

	def test_commit_changes_success(self) -> None:
		"""Test successful commit."""
		with (
			patch("codemap.cli.commit_cmd.Path") as mock_path,
			patch("codemap.cli.commit_cmd.commit_only_files") as mock_commit_only,
		):
			# Configure mocks for success
			mock_path.return_value.exists.return_value = True
			mock_commit_only.return_value = []

			result = _commit_changes("feat: Test commit", ["file1.py", "file2.py"])

			# Verify result and calls
			assert result is True
			# commit_only_files should be called with our files and message
			mock_commit_only.assert_called_once()
			assert "feat: Test commit" in mock_commit_only.call_args[0]
			assert ["file1.py", "file2.py"] in mock_commit_only.call_args[0]

	def test_commit_changes_with_hooks_bypass(self) -> None:
		"""Test commit with hooks bypass."""
		with (
			patch("codemap.cli.commit_cmd.Path") as mock_path,
			patch("codemap.cli.commit_cmd.commit_only_files") as mock_commit_only,
		):
			# Configure mocks for success
			mock_path.return_value.exists.return_value = True
			mock_commit_only.return_value = []

			result = _commit_changes("feat: Test commit", ["file1.py"], ignore_hooks=True)

			# Verify result and calls
			assert result is True
			# commit_only_files should be called with ignore_hooks=True
			assert mock_commit_only.call_args.kwargs.get("ignore_hooks") is True

	def test_commit_changes_failure(self) -> None:
		"""Test failed commit."""
		with (
			patch("codemap.cli.commit_cmd.Path") as mock_path,
			patch("codemap.cli.commit_cmd.commit_only_files", side_effect=Exception("Commit failed")),
			patch("codemap.cli.commit_cmd.console") as mock_console,
			patch("codemap.cli.commit_cmd.logger") as mock_logger,
		):
			# Configure Path mock
			mock_path.return_value.exists.return_value = True

			# Configure console mock to ensure it's properly tracked
			mock_console.print = MagicMock()

			result = _commit_changes("feat: Test commit", ["file1.py"])

			# Verify result and calls
			assert result is False
			# Should log the error
			assert mock_logger.exception.called
			# Should print error message
			assert mock_console.print.called


@pytest.mark.unit
@pytest.mark.git
class TestPerformCommit:
	"""Test performing commit operations."""

	def test_perform_commit_with_file_checks(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit with file checks."""
		# Patch the _commit_changes function directly called by _perform_commit
		with patch("codemap.cli.commit_cmd._commit_changes") as mock_commit_changes:
			# Configure the mock for success
			mock_commit_changes.return_value = True

			# Ensure mock_diff_chunk has files attribute
			mock_diff_chunk.files = ["file1.py", "file2.py"]

			# Call the function under test
			result = _perform_commit(mock_diff_chunk, "feat: Test commit")

			# Verify result and calls
			assert result is True
			mock_commit_changes.assert_called_once_with("feat: Test commit", mock_diff_chunk.files)

	def test_perform_commit_with_other_files(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit when there are other files."""
		# Patch the _commit_changes function directly
		with patch("codemap.cli.commit_cmd._commit_changes") as mock_commit_changes:
			# Configure the mock for success
			mock_commit_changes.return_value = True

			# Ensure mock_diff_chunk has files attribute
			mock_diff_chunk.files = ["file1.py", "file2.py"]

			# Call the function under test
			result = _perform_commit(mock_diff_chunk, "feat: Test commit")

			# Verify result and calls
			assert result is True
			mock_commit_changes.assert_called_once_with("feat: Test commit", mock_diff_chunk.files)

	def test_perform_commit_failure(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit failure."""
		# Patch the _commit_changes function directly
		with (
			patch("codemap.cli.commit_cmd._commit_changes") as mock_commit_changes,
			patch("codemap.cli.commit_cmd.console.print"),
		):  # Keep console patch if needed for error msg assertion
			# Configure the mock for failure
			mock_commit_changes.return_value = False

			# Ensure mock_diff_chunk has files attribute
			mock_diff_chunk.files = ["file1.py", "file2.py"]

			# Call the function under test
			result = _perform_commit(mock_diff_chunk, "feat: Test commit")

			# Verify result and calls
			assert result is False
			mock_commit_changes.assert_called_once_with("feat: Test commit", mock_diff_chunk.files)
			# Optionally, assert that console.print was called with an error message if _perform_commit handles it


@pytest.mark.unit
@pytest.mark.git
class TestEditCommitMessage:
	"""Test editing commit message."""

	def test_edit_commit_message(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test editing commit message with user input."""
		original_message = "feat: Original message"
		edited_message = "feat: Edited message"

		with patch("codemap.cli.commit_cmd.console"), patch("questionary.text") as mock_text:
			# Set return value of text input
			mock_text.ask.return_value = edited_message
			mock_text.return_value.unsafe_ask.return_value = edited_message

			result = _edit_commit_message(original_message, mock_diff_chunk)

			assert result == edited_message
			# Either ask or unsafe_ask should be called
			assert mock_text.ask.called or mock_text.return_value.unsafe_ask.called


@pytest.mark.unit
@pytest.mark.git
class TestCommitWithMessage:
	"""Test commit with message."""

	def test_commit_with_message(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit with message."""
		with (
			patch("codemap.cli.commit_cmd._perform_commit") as mock_perform,
			patch("codemap.cli.commit_cmd.console") as mock_console,
		):
			mock_perform.return_value = True

			_commit_with_message(mock_diff_chunk, "feat: Test commit")

			mock_perform.assert_called_once_with(mock_diff_chunk, "feat: Test commit")
			# Should print success message
			assert mock_console.print.call_count > 0


@pytest.mark.unit
def test_load_prompt_template_success(tmp_path: Path) -> None:
	"""Test loading prompt template successfully."""
	# Create test template file
	template_path = tmp_path / "template.txt"
	template_content = "This is a test template"
	template_path.write_text(template_content)

	# Load the template
	result = _load_prompt_template(str(template_path))
	assert result == template_content


@pytest.mark.unit
def test_load_prompt_template_nonexistent() -> None:
	"""Test loading prompt template with nonexistent file."""
	with patch("codemap.cli.commit_cmd.show_warning") as mock_show_warning:
		result = _load_prompt_template("/nonexistent/path.txt")
		assert result is None
		mock_show_warning.assert_called_once()
		assert "Could not load prompt template" in mock_show_warning.call_args[0][0]


@pytest.mark.unit
def test_load_prompt_template_none() -> None:
	"""Test loading prompt template with None path."""
	result = _load_prompt_template(None)
	assert result is None


@pytest.mark.unit
def test_commit_options_dataclass() -> None:
	"""Test CommitOptions dataclass initialization and defaults."""
	options = CommitOptions(repo_path=Path("/test/repo"))

	# Check default values
	assert options.repo_path == Path("/test/repo")
	assert options.generation_mode == GenerationMode.SMART
	assert options.model == "openai/gpt-4o-mini"
	assert options.api_base is None
	assert options.commit is True
	assert options.prompt_template is None
	assert options.api_key is None

	# Check custom values
	custom_options = CommitOptions(
		repo_path=Path("/test/repo"),
		generation_mode=GenerationMode.SIMPLE,
		model="custom-model",
		api_base="http://custom-api.example.com",
		commit=False,
		prompt_template="/path/to/template.txt",
		api_key="test-api-key",
	)

	assert custom_options.repo_path == Path("/test/repo")
	assert custom_options.generation_mode == GenerationMode.SIMPLE
	assert custom_options.model == "custom-model"
	assert custom_options.api_base == "http://custom-api.example.com"
	assert custom_options.commit is False
	assert custom_options.prompt_template == "/path/to/template.txt"
	assert custom_options.api_key == "test-api-key"


@pytest.mark.unit
def test_run_config_dataclass() -> None:
	"""Test RunConfig dataclass initialization and defaults."""
	config = RunConfig()

	# Check default values
	assert config.repo_path is None
	assert config.force_simple is False
	assert config.api_key is None
	assert config.model == "openai/gpt-4o-mini"
	assert config.api_base is None
	assert config.commit is True
	assert config.prompt_template is None
	assert config.staged_only is False

	# Check custom values
	custom_config = RunConfig(
		repo_path=Path("/test/repo"),
		force_simple=True,
		api_key="test-key",
		model="custom-model",
		api_base="http://custom-api.example.com",
		commit=False,
		prompt_template="/path/to/template.txt",
		staged_only=True,
	)

	assert custom_config.repo_path == Path("/test/repo")
	assert custom_config.force_simple is True
	assert custom_config.api_key == "test-key"
	assert custom_config.model == "custom-model"
	assert custom_config.api_base == "http://custom-api.example.com"
	assert custom_config.commit is False
	assert custom_config.prompt_template == "/path/to/template.txt"
	assert custom_config.staged_only is True


@pytest.mark.unit
def test_chunk_context_dataclass() -> None:
	"""Test ChunkContext dataclass initialization."""
	# Create mock generator
	mock_generator = MagicMock()

	# Create context
	context = ChunkContext(chunk=TEST_CHUNK, index=2, total=5, generator=mock_generator, mode=GenerationMode.SMART)

	# Check values
	assert context.chunk is TEST_CHUNK
	assert context.index == 2
	assert context.total == 5
	assert context.generator is mock_generator
	assert context.mode == GenerationMode.SMART


@pytest.mark.unit
def test_raise_command_failed_error() -> None:
	"""Test raising command failed error."""
	with pytest.raises(RuntimeError) as excinfo:
		_raise_command_failed_error()

	assert "Command failed to run successfully" in str(excinfo.value)


@pytest.mark.unit
def test_commit_changes_no_valid_files() -> None:
	"""Test commit changes with no valid files."""
	with (
		patch("codemap.cli.commit_cmd.run_git_command") as mock_run_git,
		patch("codemap.cli.commit_cmd.Path.exists", return_value=False),
		patch("codemap.cli.commit_cmd.console") as mock_console,
		patch("codemap.cli.commit_cmd.logger") as mock_logger,
	):
		# Mock tracked files (empty list)
		mock_run_git.return_value = ""

		# Call function with nonexistent files
		result = _commit_changes("Test commit", ["nonexistent.py"], ignore_hooks=False)

		# Should fail because no valid files
		assert result is False
		mock_console.print.assert_called_once()
		assert "Error" in mock_console.print.call_args[0][0]
		assert mock_logger.error.called


@pytest.mark.unit
def test_commit_changes_exception() -> None:
	"""Test commit changes with exception."""
	with (
		patch("codemap.cli.commit_cmd.run_git_command") as mock_run_git,
		patch("codemap.cli.commit_cmd.Path.exists", return_value=True),
		patch("codemap.cli.commit_cmd.commit_only_files") as mock_commit,
		patch("codemap.cli.commit_cmd.console") as mock_console,
		patch("codemap.cli.commit_cmd.logger") as mock_logger,
	):
		# Mock tracked files
		mock_run_git.return_value = "file1.py\nfile2.py"

		# Make commit raise exception
		mock_commit.side_effect = Exception("Test error")

		# Call function
		result = _commit_changes("Test commit", ["file1.py"], ignore_hooks=False)

		# Should fail because of exception
		assert result is False
		mock_console.print.assert_called_once()
		assert "Error" in mock_console.print.call_args[0][0]
		assert mock_logger.exception.called


@pytest.mark.unit
def test_perform_commit_success() -> None:
	"""Test perform commit with success."""
	with (
		patch("codemap.cli.commit_cmd._commit_changes", return_value=True) as mock_commit,
		patch("codemap.cli.commit_cmd.console") as mock_console,
	):
		# Call function
		result = _perform_commit(TEST_CHUNK, "Test commit")

		# Should succeed
		assert result is True
		mock_commit.assert_called_once_with("Test commit", TEST_CHUNK.files)
		mock_console.print.assert_called_once()
		assert "✓" in mock_console.print.call_args[0][0]


@pytest.mark.unit
def test_perform_commit_failure() -> None:
	"""Test perform commit with failure."""
	with patch("codemap.cli.commit_cmd._commit_changes", return_value=False) as mock_commit:
		# Call function
		result = _perform_commit(TEST_CHUNK, "Test commit")

		# Should fail
		assert result is False
		mock_commit.assert_called_once_with("Test commit", TEST_CHUNK.files)


@pytest.mark.unit
@pytest.mark.git
class TestBypassHooksIntegration:
	"""Test cases for bypass_hooks integration in the commit command."""

	def test_bypass_hooks_from_config(self, tmp_path: Path) -> None:
		"""Test that bypass_hooks is correctly loaded from config."""
		# Create a test repository
		repo_path = tmp_path / "test_repo"
		repo_path.mkdir()

		# Create a config file with bypass_hooks enabled
		config_file = repo_path / ".codemap.yml"
		config_content = """
commit:
  bypass_hooks: true
"""
		config_file.write_text(config_content)

		# Mock ConfigLoader to return our test config
		config_loader_mock = Mock()
		config_loader_mock.get_commit_hooks.return_value = True

		# Mock CommitCommand to capture the bypass_hooks param
		commit_command_mock = Mock()

		with (
			patch("codemap.cli.commit_cmd.validate_repo_path", return_value=repo_path),
			patch("codemap.cli.commit_cmd.ConfigLoader", return_value=config_loader_mock),
			patch(
				"codemap.cli.commit_cmd.CommitCommand", return_value=commit_command_mock
			) as mock_commit_command_class,
		):
			# Call the validate_and_process_commit function
			validate_and_process_commit(path=repo_path, all_files=False, model="test-model")

			# Verify that CommitCommand was instantiated
			mock_commit_command_class.assert_called_once()
			_, kwargs = mock_commit_command_class.call_args

			# Check if bypass_hooks is truthy (should be True from config)
			assert bool(kwargs.get("bypass_hooks")) is True

	def test_bypass_hooks_cli_override(self, tmp_path: Path) -> None:
		"""Test that bypass_hooks from CLI overrides config file."""
		# Create a test repository
		repo_path = tmp_path / "test_repo"
		repo_path.mkdir()

		# Create a config file with bypass_hooks disabled
		config_file = repo_path / ".codemap.yml"
		config_content = """
commit:
  bypass_hooks: false
"""
		config_file.write_text(config_content)

		# Mock ConfigLoader to return our test config
		config_loader_mock = Mock()
		config_loader_mock.get_commit_hooks.return_value = False

		# Create a special bypass_hooks object with the _set_explicitly attribute
		# Use MagicMock instead of primitive boolean so we can set attributes
		bypass_hooks_cli = Mock()
		# Define __bool__ as a proper method that returns True
		bypass_hooks_cli.__bool__ = Mock(return_value=True)
		bypass_hooks_cli._set_explicitly = True

		# Mock CommitCommand to capture the bypass_hooks param
		commit_command_mock = Mock()

		with (
			patch("codemap.cli.commit_cmd.validate_repo_path", return_value=repo_path),
			patch("codemap.cli.commit_cmd.ConfigLoader", return_value=config_loader_mock),
			patch(
				"codemap.cli.commit_cmd.CommitCommand", return_value=commit_command_mock
			) as mock_commit_command_class,
		):
			# Call the validate_and_process_commit function with CLI bypass_hooks=True
			validate_and_process_commit(
				path=repo_path, all_files=False, model="test-model", bypass_hooks=bypass_hooks_cli
			)

			# Verify that CommitCommand was instantiated with bypass_hooks=True (from CLI, not config)
			mock_commit_command_class.assert_called_once()
			_, kwargs = mock_commit_command_class.call_args
			# Verify that bypass_hooks was passed correctly
			assert kwargs.get("bypass_hooks") is bypass_hooks_cli
