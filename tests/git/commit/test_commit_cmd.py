"""Tests for commit command module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console

from codemap.cli.commit_cmd import (
	ChunkContext,
	CommitOptions,
	GenerationMode,
	RunConfig,
	_check_other_files,
	_commit_changes,
	_commit_with_message,
	_commit_with_user_input,
	_edit_commit_message,
	_extract_provider_from_model,
	_get_api_key_for_provider,
	_handle_other_files,
	_load_prompt_template,
	_perform_commit,
	_raise_command_failed_error,
	display_suggested_messages,
	generate_commit_message,
	print_chunk_summary,
	process_all_chunks,
	process_chunk_interactively,
	setup_message_generator,
	validate_and_process_commit,
)
from codemap.git.commit_generator import CommitMessageGenerator
from codemap.git.diff_splitter import DiffChunk
from codemap.llm.errors import LLMError
from tests.base import CLITestBase, GitTestBase

if TYPE_CHECKING:
	from collections.abc import Sequence

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
@pytest.mark.git
class TestExtractProviderFromModel:
	"""Test extracting provider from model names."""

	def test_extract_provider_with_prefix(self) -> None:
		"""Test extracting provider when model has a provider prefix."""
		assert _extract_provider_from_model("openai/gpt-4o") == "openai"
		assert _extract_provider_from_model("anthropic/claude-3-sonnet") == "anthropic"
		assert _extract_provider_from_model("groq/llama-3-8b") == "groq"

	def test_extract_provider_with_org(self) -> None:
		"""Test extracting provider when model has organization in path."""
		assert _extract_provider_from_model("openai/org/custom-model") == "openai"

	def test_extract_provider_normalized(self) -> None:
		"""Test that provider names are normalized to lowercase."""
		assert _extract_provider_from_model("OpenAI/gpt-4") == "openai"

	def test_extract_provider_no_prefix(self) -> None:
		"""Test extracting provider when model has no prefix."""
		assert _extract_provider_from_model("gpt-4") is None


@pytest.mark.unit
@pytest.mark.cli
class TestGetApiKeyForProvider:
	"""Test getting API keys for different providers."""

	def test_get_api_key_openai(self) -> None:
		"""Test getting OpenAI API key."""
		with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
			assert _get_api_key_for_provider("openai") == "test-key"

	def test_get_api_key_anthropic(self) -> None:
		"""Test getting Anthropic API key."""
		with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
			assert _get_api_key_for_provider("anthropic") == "test-key"

	def test_get_api_key_openai_fallback(self) -> None:
		"""Test fallback to OpenAI key when primary key is missing."""
		with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
			assert _get_api_key_for_provider("anthropic") == "test-key"

	def test_get_api_key_unknown_provider(self) -> None:
		"""Test behavior with unknown provider."""
		assert _get_api_key_for_provider("unknown_provider") is None

	def test_get_api_key_missing(self) -> None:
		"""Test behavior when API key is missing."""
		with patch.dict(os.environ, {}, clear=True):
			assert _get_api_key_for_provider("openai") is None


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
		with patch("codemap.cli.commit_cmd.console") as mock_console:
			assert _load_prompt_template("/nonexistent/path.txt") is None
			mock_console.print.assert_called_once()

	def test_load_prompt_template_none(self) -> None:
		"""Test loading with None path."""
		assert _load_prompt_template(None) is None


@pytest.mark.unit
@pytest.mark.git
class TestSetupMessageGenerator:
	"""Test setting up message generator."""

	def test_setup_message_generator(self, commit_options: CommitOptions) -> None:
		"""Test setup of message generator with default options."""
		with patch("codemap.cli.commit_cmd.create_universal_generator") as mock_create:
			mock_create.return_value = MagicMock(spec=CommitMessageGenerator)

			setup_message_generator(commit_options)

			# Verify generator was created with correct options
			mock_create.assert_called_once_with(
				repo_path=commit_options.repo_path,
				model=commit_options.model,
				api_key=commit_options.api_key,
				api_base=commit_options.api_base,
				prompt_template=None,
			)

	def test_setup_message_generator_with_template(self, commit_options: CommitOptions, tmp_path: Path) -> None:
		"""Test setup of message generator with prompt template."""
		template_path = tmp_path / "template.txt"
		template_content = "Test template"
		template_path.write_text(template_content)

		commit_options.prompt_template = str(template_path)

		with patch("codemap.cli.commit_cmd.create_universal_generator") as mock_create:
			mock_create.return_value = MagicMock(spec=CommitMessageGenerator)

			setup_message_generator(commit_options)

			# Verify generator was created with template
			mock_create.assert_called_once()
			# Check that the template was loaded
			assert mock_create.call_args[1]["prompt_template"] == template_content


@pytest.mark.unit
@pytest.mark.git
class TestGenerateCommitMessage:
	"""Test commit message generation."""

	def test_generate_commit_message_smart_mode(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test generating message in smart mode."""
		mock_generator = MagicMock(spec=CommitMessageGenerator)

		# Set up mock_diff_chunk with required properties
		mock_diff_chunk.files = ["file1.py"]
		mock_diff_chunk.content = "test diff content"

		with (
			patch("codemap.cli.commit_cmd.logger"),
			patch("codemap.cli.commit_cmd.generate_message") as mock_gen_message,
		):
			# Configure mock to return an LLM-generated message
			mock_gen_message.return_value = ("feat: implement new feature", True)

			# Call function with smart mode
			result, is_llm = generate_commit_message(
				mock_diff_chunk,
				mock_generator,
				mode=GenerationMode.SMART,
			)

			# Verify results
			assert result == "feat: implement new feature"
			assert is_llm is True
			assert mock_gen_message.call_count == 1

	def test_generate_commit_message_simple_mode(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test generating message in simple mode."""
		mock_generator = MagicMock(spec=CommitMessageGenerator)

		# Set up mock_diff_chunk with required properties
		mock_diff_chunk.files = ["file1.py"]
		mock_diff_chunk.content = "test diff content"

		# Setup mock for fallback_generation
		mock_generator.fallback_generation.return_value = "chore: update file1.py"

		with patch("codemap.cli.commit_cmd.logger"):
			# Call function with simple mode
			result, is_llm = generate_commit_message(
				mock_diff_chunk,
				mock_generator,
				mode=GenerationMode.SIMPLE,
			)

			# Verify results
			assert result == "chore: update file1.py"
			assert is_llm is False
			# In simple mode, fallback_generation should be called
			mock_generator.fallback_generation.assert_called_once()

	def test_generate_commit_message_error(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test handling of errors in commit message generation."""
		mock_generator = MagicMock(spec=CommitMessageGenerator)

		# Set up mock_diff_chunk with required properties
		mock_diff_chunk.files = ["file1.py"]
		mock_diff_chunk.content = "test diff content"

		# Setup mock for fallback_generation
		mock_generator.fallback_generation.return_value = "update: file1.py"

		with (
			patch("codemap.cli.commit_cmd.logger"),
			patch("codemap.cli.commit_cmd.generate_message") as mock_gen_message,
		):
			# Configure mock to raise an error
			mock_gen_message.side_effect = LLMError("API error")

			# Call function with smart mode
			result, is_llm = generate_commit_message(
				mock_diff_chunk,
				mock_generator,
				mode=GenerationMode.SMART,
			)

			# Verify fallback to simple mode was used
			assert "update" in result.lower()  # Simple generation format
			assert "file1.py" in result.lower()
			assert is_llm is False  # Should not be LLM-generated due to error
			assert mock_gen_message.call_count == 1  # Should attempt LLM once
			# Fallback generation should be called due to error
			mock_generator.fallback_generation.assert_called_once()


@pytest.mark.unit
@pytest.mark.git
class TestPrintChunkSummary:
	"""Test printing chunk summary."""

	def test_print_chunk_summary(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test that chunk summary is printed correctly."""
		with patch("codemap.cli.commit_cmd.console") as mock_console:
			print_chunk_summary(mock_diff_chunk, 0)

			# Verify console.print was called at least twice
			assert mock_console.print.call_count >= 2


@pytest.mark.unit
@pytest.mark.git
class TestCheckOtherFiles:
	"""Test checking for other files."""

	def test_check_other_files_none(self) -> None:
		"""Test when there are no other files."""
		with (
			patch("codemap.cli.commit_cmd.get_other_staged_files", return_value=[]),
			patch("codemap.cli.commit_cmd.get_untracked_files", return_value=[]),
		):
			chunk_files = ["file1.py", "file2.py"]
			other_staged, other_untracked, has_other = _check_other_files(chunk_files)

			assert other_staged == []
			assert other_untracked == []
			assert has_other is False

	def test_check_other_files_staged(self) -> None:
		"""Test when there are other staged files."""
		with (
			patch("codemap.cli.commit_cmd.get_other_staged_files", return_value=["file3.py"]),
			patch("codemap.cli.commit_cmd.get_untracked_files", return_value=[]),
		):
			chunk_files = ["file1.py", "file2.py"]
			other_staged, other_untracked, has_other = _check_other_files(chunk_files)

			assert other_staged == ["file3.py"]
			assert other_untracked == []
			assert has_other is True

	def test_check_other_files_untracked(self) -> None:
		"""Test when there are untracked files."""
		with (
			patch("codemap.cli.commit_cmd.get_other_staged_files", return_value=[]),
			patch("codemap.cli.commit_cmd.get_untracked_files", return_value=["file3.py"]),
		):
			chunk_files = ["file1.py", "file2.py"]
			other_staged, other_untracked, has_other = _check_other_files(chunk_files)

			assert other_staged == []
			assert other_untracked == ["file3.py"]
			assert has_other is True

	def test_check_other_files_both(self) -> None:
		"""Test when there are both staged and untracked files."""
		with (
			patch("codemap.cli.commit_cmd.get_other_staged_files", return_value=["file3.py"]),
			patch("codemap.cli.commit_cmd.get_untracked_files", return_value=["file4.py"]),
		):
			chunk_files = ["file1.py", "file2.py"]
			other_staged, other_untracked, has_other = _check_other_files(chunk_files)

			assert other_staged == ["file3.py"]
			assert other_untracked == ["file4.py"]
			assert has_other is True


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.parametrize(
	("user_response", "expected_result"),
	[
		("y", True),  # Include other files
		("n", True),  # Continue with current files - still returns True
	],
)
class TestHandleOtherFiles:
	"""Test handling other files."""

	def test_handle_other_files(self, mock_diff_chunk: DiffChunk, user_response: str, expected_result: bool) -> None:
		"""Test handling other files based on user response."""
		with patch("codemap.cli.commit_cmd.console"), patch("questionary.select") as mock_select:
			# Map the 'y' and 'n' to the appropriate action values
			action = "all_staged" if user_response == "y" else "continue"
			mock_select.return_value.ask.return_value = action

			# Create copies of files to check later
			mock_diff_chunk.files = ["file1.py", "file2.py"]  # Ensure consistent starting point
			original_files = mock_diff_chunk.files.copy()
			other_staged = ["file3.py"]
			other_untracked = ["file4.py"]

			result = _handle_other_files(mock_diff_chunk, other_staged, other_untracked)

			assert result == expected_result
			if action == "all_staged":  # Only for "y" response
				# Files should be added to the chunk
				assert set(mock_diff_chunk.files) > set(original_files)
				assert "file3.py" in mock_diff_chunk.files
			else:
				# Files should remain unchanged if user chose continue
				assert set(mock_diff_chunk.files) == set(original_files)


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
		# Directly mock _commit_changes which is actually called
		with patch("codemap.cli.commit_cmd._commit_changes") as mock_commit:
			# Configure mock for success
			mock_commit.return_value = True

			# Ensure mock_diff_chunk has files attribute
			mock_diff_chunk.files = ["file1.py", "file2.py"]

			# Call the function under test
			result = _perform_commit(mock_diff_chunk, "feat: Test commit")

			# Verify result and calls
			assert result is True
			mock_commit.assert_called_once_with("feat: Test commit", mock_diff_chunk.files)

	def test_perform_commit_with_other_files(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit when there are other files."""
		# Modern implementation may not call _check_other_files or handle_other_files
		# Directly mock _commit_changes which is actually called
		with patch("codemap.cli.commit_cmd._commit_changes") as mock_commit:
			# Configure the mock for success
			mock_commit.return_value = True

			# Ensure mock_diff_chunk has files attribute
			mock_diff_chunk.files = ["file1.py", "file2.py"]

			# Call the function under test
			result = _perform_commit(mock_diff_chunk, "feat: Test commit")

			# Verify result and calls
			assert result is True
			mock_commit.assert_called_once_with("feat: Test commit", mock_diff_chunk.files)

	def test_perform_commit_failure(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit failure."""
		# Directly mock _commit_changes which is actually called
		with patch("codemap.cli.commit_cmd._commit_changes") as mock_commit:
			# Configure the mock for failure
			mock_commit.return_value = False

			# Ensure mock_diff_chunk has files attribute
			mock_diff_chunk.files = ["file1.py", "file2.py"]

			# Call the function under test
			result = _perform_commit(mock_diff_chunk, "feat: Test commit")

			# Verify result and calls
			assert result is False
			mock_commit.assert_called_once_with("feat: Test commit", mock_diff_chunk.files)


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
@pytest.mark.git
class TestCommitWithUserInput:
	"""Test commit with user input."""

	def test_commit_with_user_input_accept(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit with user accepting suggested message."""
		with (
			patch("codemap.cli.commit_cmd.console"),
			patch("questionary.select") as mock_select,
			patch("codemap.cli.commit_cmd._perform_commit") as mock_perform,
			patch("codemap.cli.commit_cmd._edit_commit_message") as mock_edit,
		):
			# Configure the mocks
			mock_select.return_value.ask.return_value = "commit"
			# Mock the edit function to avoid EOFError
			mock_edit.return_value = "feat: Test commit"
			mock_perform.return_value = True

			_commit_with_user_input(mock_diff_chunk, "feat: Test commit")

			# _commit_with_message will directly or indirectly call _perform_commit
			mock_perform.assert_called_once_with(mock_diff_chunk, "feat: Test commit")

	def test_commit_with_user_input_edit(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit with user choosing to edit message."""
		with (
			patch("codemap.cli.commit_cmd.console"),
			patch("questionary.select") as mock_select,
			patch("codemap.cli.commit_cmd._edit_commit_message") as mock_edit,
			patch("codemap.cli.commit_cmd._perform_commit") as mock_perform,
		):
			# Configure mocks with return values
			mock_select.return_value.ask.return_value = "edit"
			mock_edit.return_value = "feat: Edited message"
			mock_perform.return_value = True

			_commit_with_user_input(mock_diff_chunk, "feat: Test commit")

			# Verify the edit message was called
			mock_edit.assert_called_once_with("feat: Test commit", mock_diff_chunk)

			# Check that commit is performed with edited message
			mock_perform.assert_called_once_with(mock_diff_chunk, "feat: Edited message")

	def test_commit_with_user_input_skip(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test commit with user choosing to skip."""
		# For this test, we'll directly test process_chunk_interactively instead,
		# which is what actually prints "Skipped commit."
		from codemap.cli.commit_cmd import ChunkContext, GenerationMode

		generator = MagicMock(spec=CommitMessageGenerator)
		context = ChunkContext(chunk=mock_diff_chunk, index=0, total=1, generator=generator, mode=GenerationMode.SMART)

		with (
			patch("codemap.cli.commit_cmd.console") as mock_console,
			patch("codemap.cli.commit_cmd.print_chunk_summary"),
			patch("codemap.cli.commit_cmd.generate_commit_message") as mock_generate,
			patch("questionary.select") as mock_select,
		):
			# Configure console mock to ensure it's properly tracked
			mock_console.print = MagicMock()

			# Configure mocks with return values
			mock_generate.return_value = ("feat: Test commit", True)
			mock_select.return_value.ask.return_value = "skip"

			from codemap.cli.commit_cmd import process_chunk_interactively

			result = process_chunk_interactively(context)

			# Verify the result and skip message
			assert result == "continue"
			mock_console.print.assert_any_call("[yellow]Skipped commit.[/yellow]")


@pytest.mark.unit
@pytest.mark.git
class TestProcessChunkInteractively:
	"""Test processing chunk interactively."""

	def test_process_chunk_interactive_with_llm(self, mock_diff_chunk: DiffChunk) -> None:
		"""Test interactive processing with LLM generation."""
		from codemap.cli.commit_cmd import ChunkContext

		generator = MagicMock(spec=CommitMessageGenerator)

		context = ChunkContext(chunk=mock_diff_chunk, index=0, total=1, generator=generator, mode=GenerationMode.SMART)

		with (
			patch("codemap.cli.commit_cmd.print_chunk_summary"),
			patch("codemap.cli.commit_cmd.generate_commit_message") as mock_generate,
			patch("codemap.cli.commit_cmd.loading_spinner"),
			patch("questionary.select") as mock_select,
			patch("codemap.cli.commit_cmd._commit_with_message") as mock_commit_with_message,
		):
			# Configure mock to return a valid action choice
			mock_select.return_value.ask.return_value = "commit"
			mock_generate.return_value = ("feat: Test commit", True)

			result = process_chunk_interactively(context)

			assert result == "continue"
			mock_generate.assert_called_once()
			# The commit_with_message function is called for the "commit" action
			mock_commit_with_message.assert_called_once_with(mock_diff_chunk, "feat: Test commit")


@pytest.mark.unit
@pytest.mark.git
class TestDisplaySuggestedMessages:
	"""Test displaying suggested messages."""

	def test_display_suggested_messages(self, commit_options: CommitOptions) -> None:
		"""Test displaying suggested messages."""
		# Create mocks with the required attributes
		chunks = []
		for i in range(3):
			mock_chunk = MagicMock(spec=DiffChunk)
			mock_chunk.files = [f"file{i}.py"]
			mock_chunk.content = f"diff content for chunk {i}"
			chunks.append(mock_chunk)

		generator = MagicMock(spec=CommitMessageGenerator)

		with (
			patch("codemap.cli.commit_cmd.console") as mock_console,
			patch("codemap.cli.commit_cmd.generate_commit_message") as mock_generate,
			patch("codemap.cli.commit_cmd.print_chunk_summary"),
		):
			# Return different messages for each chunk
			mock_generate.side_effect = [
				("feat: First commit", True),
				("fix: Second commit", True),
				("docs: Third commit", True),
			]

			display_suggested_messages(commit_options, chunks, generator)

			# Should have called generate_commit_message for each chunk
			assert mock_generate.call_count == 3
			# Should have printed messages
			assert mock_console.print.call_count > 0


@pytest.mark.unit
@pytest.mark.git
class TestProcessAllChunks:
	"""Test processing all chunks."""

	def test_process_all_chunks(self, commit_options: CommitOptions) -> None:
		"""Test processing all chunks."""
		chunks: Sequence[DiffChunk] = [MagicMock(spec=DiffChunk) for _ in range(2)]
		generator = MagicMock(spec=CommitMessageGenerator)

		with patch("codemap.cli.commit_cmd.process_chunk_interactively") as mock_process:
			# Return different messages for each chunk
			mock_process.side_effect = ["feat: First commit", "fix: Second commit"]

			result = process_all_chunks(commit_options, chunks, generator)

			# Should return 0 (success)
			assert result == 0
			# Should have called process_chunk_interactively for each chunk
			assert mock_process.call_count == 2


@pytest.mark.unit
@pytest.mark.parametrize(
	("model", "expected_provider"),
	[
		("openai/gpt-4", "openai"),
		("anthropic/claude-3-opus", "anthropic"),
		("google/gemini-pro", "google"),
		("llama2-70b", None),  # No provider prefix
		("custom-model", None),  # No provider prefix
		("mistral/mistral-large", "mistral"),
		("azure/openai/gpt-4", "azure"),  # Multiple slashes
	],
)
def test_extract_provider_from_model(model: str, expected_provider: str | None) -> None:
	"""Test extracting provider from model name."""
	provider = _extract_provider_from_model(model)
	assert provider == expected_provider


@pytest.mark.unit
@pytest.mark.parametrize(
	("provider", "env_vars", "expected_key"),
	[
		("openai", {"OPENAI_API_KEY": "test-key"}, "test-key"),
		("anthropic", {"ANTHROPIC_API_KEY": "test-key"}, "test-key"),
		("unknown", {"OPENAI_API_KEY": "test-key"}, None),
		("mistral", {"MISTRAL_API_KEY": "test-key"}, "test-key"),
		# Fallback to OpenAI test
		("mistral", {"OPENAI_API_KEY": "fallback-key"}, "fallback-key"),
		# No matching key
		("mistral", {}, None),
	],
)
def test_get_api_key_for_provider(provider: str, env_vars: dict[str, str], expected_key: str | None) -> None:
	"""Test getting API key for provider from environment."""
	with patch.dict(os.environ, env_vars, clear=True):
		key = _get_api_key_for_provider(provider)
		assert key == expected_key


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
	with patch("codemap.cli.commit_cmd.console") as mock_console:
		result = _load_prompt_template("/nonexistent/path.txt")
		assert result is None
		mock_console.print.assert_called_once()
		assert "Warning" in mock_console.print.call_args[0][0]


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


@pytest.mark.integration
@pytest.mark.git
class TestCommitCommand(CLITestBase, GitTestBase):
	"""Integration tests for commit command."""

	def test_commit_command_initialization(self) -> None:
		"""Test that commit command can be initialized."""
		from codemap.cli.commit_cmd import RunConfig, _run_commit_command

		config = RunConfig(repo_path=Path("/fake/repo"))

		# Mock dependencies to avoid actual Git operations
		with (
			patch("codemap.cli.commit_cmd.validate_repo_path", return_value=Path("/fake/repo")),
			patch("codemap.cli.commit_cmd.setup_message_generator"),
			patch("codemap.cli.commit_cmd.console"),
			patch("codemap.cli.commit_cmd.get_staged_diff") as mock_staged,
			patch("codemap.cli.commit_cmd.get_unstaged_diff") as mock_unstaged,
			patch("codemap.cli.commit_cmd.get_untracked_files") as mock_untracked,
		):
			# Set up no changes to test early return
			mock_staged.return_value.files = []
			mock_staged.return_value.content = ""
			mock_unstaged.return_value.files = []
			mock_untracked.return_value = []

			# Should exit with code 0 (no changes to commit)
			result = _run_commit_command(config)
			assert result == 0


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
