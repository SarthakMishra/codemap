"""Tests for llm_utils.py."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest
from litellm.exceptions import APIError

from codemap.git.message_generator import LLMError, MessageGenerator
from codemap.utils.llm_utils import (
	create_universal_generator,
	extract_content_from_response,
	generate_message,
	generate_text_with_llm,
	load_prompt_template,
	setup_message_generator,
)

if TYPE_CHECKING:
	from collections.abc import Generator


class MockDiffChunk:
	"""Mock implementation of DiffChunkLike for testing."""

	def __init__(self, files: list[str], content: str, description: str | None = None) -> None:
		"""
		Initialize a mock diff chunk.

		Args:
		    files: List of files in the chunk
		    content: Content of the chunk
		    description: Optional description of the chunk

		"""
		self.files = files
		self.content = content
		self.description = description


@pytest.fixture
def mock_message_generator() -> MagicMock:
	"""Create a mock MessageGenerator."""
	generator = MagicMock(spec=MessageGenerator)
	# Mock the generate_message method
	generator.generate_message.return_value = ("feat: Test commit message", True)
	# Mock the fallback_generation method
	generator.fallback_generation.return_value = "test: Fallback message"
	# Set resolved_provider
	generator.resolved_provider = "openai"
	# Set up a dict for api_keys
	generator._api_keys = {}
	return generator


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
	"""
	Save and restore environment variables between tests.

	Yields:
	    None: This fixture doesn't yield a value, it just sets up and tears down.

	"""
	orig_env = os.environ.copy()
	yield
	os.environ.clear()
	os.environ.update(orig_env)


class TestPromptTemplates:
	"""Tests for prompt template loading functions."""

	def test_load_prompt_template_empty_path(self) -> None:
		"""Test loading a prompt template with an empty path."""
		assert load_prompt_template(None) is None

	def test_load_prompt_template_valid_path(self, tmp_path: Path) -> None:
		"""Test loading a prompt template with a valid path."""
		# Create a temporary template file
		template_file = tmp_path / "template.txt"
		template_content = "This is a test template"
		template_file.write_text(template_content)

		# Load the template
		loaded_template = load_prompt_template(str(template_file))
		assert loaded_template == template_content

	def test_load_prompt_template_invalid_path(self) -> None:
		"""Test loading a prompt template with an invalid path."""
		# Use a path that doesn't exist
		loaded_template = load_prompt_template("/path/does/not/exist.txt")
		assert loaded_template is None


class TestMessageGenerator:
	"""Tests for message generator setup and configuration."""

	@patch("codemap.utils.llm_utils.MessageGenerator")
	def test_setup_message_generator_basic(self, mock_generator_class: MagicMock) -> None:
		"""Test setting up a message generator with basic options."""
		# Set up the mock to return a mock instance
		mock_instance = MagicMock()
		mock_generator_class.return_value = mock_instance

		# Call the function
		repo_path = Path("/fake/repo")
		result = setup_message_generator(repo_path)

		# Verify the mock was called with expected arguments
		mock_generator_class.assert_called_once()
		call_args = mock_generator_class.call_args[0]
		assert call_args[0] == repo_path

		# Verify the result is the mock instance
		assert result == mock_instance

	@patch("codemap.utils.llm_utils.MessageGenerator")
	@patch("codemap.utils.llm_utils.load_prompt_template")
	def test_setup_message_generator_with_template_path(
		self, mock_load_template: MagicMock, mock_generator_class: MagicMock
	) -> None:
		"""Test setting up a message generator with a template path."""
		# Set up mocks
		mock_load_template.return_value = "Test template content"
		mock_instance = MagicMock()
		mock_generator_class.return_value = mock_instance

		# Call the function with a template path
		repo_path = Path("/fake/repo")
		result = setup_message_generator(repo_path, prompt_template_path="template.txt")

		# Verify template loading was called
		mock_load_template.assert_called_once_with("template.txt")

		# Verify generator was created with the template
		mock_generator_class.assert_called_once()
		call_kwargs = mock_generator_class.call_args[1]
		assert call_kwargs["prompt_template"] == "Test template content"

		# Verify the result is the mock instance
		assert result == mock_instance

	@patch("codemap.utils.llm_utils.MessageGenerator")
	def test_setup_message_generator_with_direct_template(self, mock_generator_class: MagicMock) -> None:
		"""Test setting up a message generator with a direct template."""
		# Set up mock
		mock_instance = MagicMock()
		mock_generator_class.return_value = mock_instance

		# Call the function with a direct template
		repo_path = Path("/fake/repo")
		template = "Direct template content"
		result = setup_message_generator(repo_path, prompt_template=template)

		# Verify generator was created with the template
		mock_generator_class.assert_called_once()
		call_kwargs = mock_generator_class.call_args[1]
		assert call_kwargs["prompt_template"] == template

		# Verify the result is the mock instance
		assert result == mock_instance

	@patch("codemap.utils.llm_utils.MessageGenerator")
	def test_setup_message_generator_with_api_key(self, mock_generator_class: MagicMock) -> None:
		"""Test setting up a message generator with an API key."""
		# Save original environment variables
		orig_env = os.environ.copy()
		try:
			# Set up mock
			mock_instance = MagicMock()
			mock_generator_class.return_value = mock_instance

			# Call the function with an API key
			repo_path = Path("/fake/repo")
			model = "openai/gpt-4"
			api_key = "test-api-key"
			result = setup_message_generator(repo_path, model=model, api_key=api_key)

			# Verify the API key was set in the environment
			assert os.environ.get("OPENAI_API_KEY") == api_key

			# Verify the result is the mock instance
			assert result == mock_instance
		finally:
			# Restore original environment variables
			os.environ.clear()
			os.environ.update(orig_env)

	@patch("codemap.utils.llm_utils.MessageGenerator")
	def test_setup_message_generator_with_api_key_and_provider(self, mock_generator_class: MagicMock) -> None:
		"""Test setting up a message generator with an API key and specific provider."""
		# Save original environment variables
		orig_env = os.environ.copy()
		try:
			# Set up mock
			mock_instance = MagicMock()
			mock_generator_class.return_value = mock_instance

			# Call the function with an API key and anthropic model
			repo_path = Path("/fake/repo")
			model = "anthropic/claude-3-opus"
			api_key = "test-anthropic-key"
			result = setup_message_generator(repo_path, model=model, api_key=api_key)

			# Verify the API key was set in the environment for the correct provider
			assert os.environ.get("ANTHROPIC_API_KEY") == api_key

			# Verify the result is the mock instance
			assert result == mock_instance
		finally:
			# Restore original environment variables
			os.environ.clear()
			os.environ.update(orig_env)


class TestMessageGeneration:
	"""Tests for message generation with LLMs."""

	def test_generate_message_with_llm(self, mock_message_generator: MagicMock) -> None:
		"""Test generating a message with an LLM."""
		# Create a mock chunk
		chunk = MockDiffChunk(files=["test.py"], content="def test(): pass")

		# Generate a message
		message, used_llm = generate_message(chunk, mock_message_generator)

		# Verify the message generator was called correctly
		mock_message_generator.generate_message.assert_called_once()
		# Check the passed argument has the expected structure
		args, _ = mock_message_generator.generate_message.call_args
		arg = args[0]
		# Use dict access instead of checking type directly
		assert arg["files"] == chunk.files
		assert arg["content"] == chunk.content

		# Verify the returned values
		assert message == "feat: Test commit message"
		assert used_llm is True

	def test_generate_message_with_description(self, mock_message_generator: MagicMock) -> None:
		"""Test generating a message with a description."""
		# Create a mock chunk with a description
		chunk = MockDiffChunk(files=["test.py"], content="def test(): pass", description="Test description")

		# Generate a message
		message, used_llm = generate_message(chunk, mock_message_generator)

		# Verify the message generator was called correctly
		mock_message_generator.generate_message.assert_called_once()
		# Check the passed argument contains the description
		args, _ = mock_message_generator.generate_message.call_args
		arg = args[0]
		assert arg["description"] == "Test description"

		# Verify the returned values
		assert message == "feat: Test commit message"
		assert used_llm is True

	def test_generate_message_simple_mode(self, mock_message_generator: MagicMock) -> None:
		"""Test generating a message in simple mode."""
		# Create a mock chunk
		chunk = MockDiffChunk(files=["test.py"], content="def test(): pass")

		# Generate a message in simple mode
		message, used_llm = generate_message(chunk, mock_message_generator, use_simple_mode=True)

		# Verify the fallback generation was used
		mock_message_generator.fallback_generation.assert_called_once()
		mock_message_generator.generate_message.assert_not_called()

		# Verify the returned values
		assert message == "test: Fallback message"
		assert used_llm is False

	def test_generate_message_with_llm_error(self, mock_message_generator: MagicMock) -> None:
		"""Test generating a message with an LLM error."""
		# Create a mock chunk
		chunk = MockDiffChunk(files=["test.py"], content="def test(): pass")

		# Make the generate_message method raise an LLMError
		mock_message_generator.generate_message.side_effect = LLMError("Test LLM error")

		# Generate a message
		message, used_llm = generate_message(chunk, mock_message_generator)

		# Verify the fallback generation was used
		mock_message_generator.fallback_generation.assert_called_once()

		# Verify the returned values
		assert message == "test: Fallback message"
		assert used_llm is False


class TestUniversalGenerator:
	"""Tests for universal generator creation and configuration."""

	@patch("codemap.utils.config_loader.ConfigLoader")
	@patch("codemap.git.message_generator.MessageGenerator")
	def test_create_universal_generator(
		self, mock_generator_class: MagicMock, mock_config_loader_class: MagicMock
	) -> None:
		"""Test creating a universal generator."""
		# Set up mocks
		mock_config_loader = MagicMock()
		mock_config_loader_class.return_value = mock_config_loader
		mock_config_loader.get_llm_config.return_value = {"model": "gpt-4-turbo", "provider": "openai"}

		mock_generator = MagicMock()
		mock_generator_class.return_value = mock_generator
		mock_generator.resolved_provider = "openai"
		mock_generator._api_keys = {}

		# Call the function
		repo_path = Path("/fake/repo")
		result = create_universal_generator(repo_path)

		# Verify the mocks were called correctly
		mock_config_loader_class.assert_called_once_with(repo_root=repo_path)
		mock_generator_class.assert_called_once()

		# Verify the result is the mock generator
		assert result == mock_generator

	@patch("codemap.utils.config_loader.ConfigLoader")
	@patch("codemap.git.message_generator.MessageGenerator")
	def test_create_universal_generator_with_api_key(
		self, mock_generator_class: MagicMock, mock_config_loader_class: MagicMock
	) -> None:
		"""Test creating a universal generator with an API key."""
		# Set up mocks
		mock_config_loader = MagicMock()
		mock_config_loader_class.return_value = mock_config_loader
		mock_config_loader.get_llm_config.return_value = {"model": "gpt-4-turbo", "provider": "openai"}

		mock_generator = MagicMock()
		mock_generator_class.return_value = mock_generator
		mock_generator.resolved_provider = "openai"
		mock_generator._api_keys = {}

		# Call the function with an API key
		repo_path = Path("/fake/repo")
		api_key = "test-api-key"
		result = create_universal_generator(repo_path, api_key=api_key)

		# Verify the mocks were called correctly
		mock_config_loader_class.assert_called_once_with(repo_root=repo_path)
		mock_generator_class.assert_called_once()

		# Verify the API key was set in the generator
		assert mock_generator._api_keys["openai"] == api_key

		# Verify the result is the mock generator
		assert result == mock_generator

	@patch("codemap.utils.config_loader.ConfigLoader")
	@patch("codemap.git.message_generator.MessageGenerator")
	def test_create_universal_generator_with_model_and_prompt(
		self, mock_generator_class: MagicMock, mock_config_loader_class: MagicMock
	) -> None:
		"""Test creating a universal generator with a model and prompt template."""
		# Set up mocks
		mock_config_loader = MagicMock()
		mock_config_loader_class.return_value = mock_config_loader

		mock_generator = MagicMock()
		mock_generator_class.return_value = mock_generator
		mock_generator.resolved_provider = "anthropic"
		mock_generator._api_keys = {}

		# Call the function with a model and prompt template
		repo_path = Path("/fake/repo")
		model = "anthropic/claude-3-haiku"
		prompt_template = "Test prompt template"
		result = create_universal_generator(repo_path, model=model, prompt_template=prompt_template)

		# Verify the mocks were called correctly
		mock_config_loader_class.assert_called_once_with(repo_root=repo_path)
		mock_generator_class.assert_called_once()
		call_kwargs = mock_generator_class.call_args[1]
		assert call_kwargs["model"] == model
		assert call_kwargs["prompt_template"] == prompt_template

		# Verify the result is the mock generator
		assert result == mock_generator

	@patch("codemap.utils.config_loader.ConfigLoader")
	@patch("codemap.git.message_generator.MessageGenerator")
	def test_create_universal_generator_error(
		self, mock_generator_class: MagicMock, mock_config_loader_class: MagicMock
	) -> None:
		"""Test error handling when creating a universal generator."""
		# Set up mock config loader
		mock_config_loader = MagicMock()
		mock_config_loader_class.return_value = mock_config_loader
		mock_config_loader.get_llm_config.return_value = {"model": "gpt-4", "provider": "openai"}

		# Make the MessageGenerator constructor raise an exception
		mock_generator_class.side_effect = ValueError("Test error")

		# Call the function
		repo_path = Path("/fake/repo")
		with pytest.raises(RuntimeError, match="Failed to create message generator: Test error"):
			create_universal_generator(repo_path)


class TestLLMInteractions:
	"""Tests for direct LLM interactions."""

	@patch("litellm.completion")
	def test_generate_text_with_llm(self, mock_completion: MagicMock) -> None:
		"""Test generating text with an LLM."""
		# Set up mock response
		mock_response = MagicMock()
		mock_response.choices = [MagicMock()]
		mock_response.choices[0].message.content = "Generated text"
		mock_completion.return_value = mock_response

		# Save original environment variables
		orig_env = os.environ.copy()
		try:
			# Set an API key in the environment
			os.environ["OPENAI_API_KEY"] = "test-api-key"

			# Call the function
			prompt = "Test prompt"
			result = generate_text_with_llm(prompt)

			# Verify completion was called correctly
			mock_completion.assert_called_once()
			call_kwargs = mock_completion.call_args[1]
			assert call_kwargs["model"] == "gpt-4o-mini"
			assert call_kwargs["messages"] == [{"role": "user", "content": prompt}]
			assert call_kwargs["api_key"] == "test-api-key"

			# Verify the result
			assert result == "Generated text"
		finally:
			# Restore original environment variables
			os.environ.clear()
			os.environ.update(orig_env)

	@patch("litellm.completion")
	def test_generate_text_with_llm_error(self, mock_completion: MagicMock) -> None:
		"""Test error handling when generating text with an LLM."""
		# Make the completion function raise an exception
		mock_completion.side_effect = APIError(
			status_code=401,  # Unauthorized status code
			message="Test API error",
			llm_provider="openai",
			model="gpt-4",
		)

		# Call the function
		prompt = "Test prompt"
		with pytest.raises(RuntimeError, match="Failed to generate text with LLM: .*Test API error"):
			generate_text_with_llm(prompt)


class TestResponseHandling:
	"""Tests for LLM response handling utilities."""

	def test_extract_content_from_response_standard_format(self) -> None:
		"""Test extracting content from a standard OpenAI response format."""
		# Create a mock response with the standard format
		response = Mock()
		response.choices = [Mock()]
		response.choices[0].message.content = "Standard format content"

		# Extract the content
		result = extract_content_from_response(response)

		# Verify the result
		assert result == "Standard format content"

	def test_extract_content_from_response_dict_format(self) -> None:
		"""Test extracting content from a dictionary response format."""
		# Create a mock response as a dictionary
		response = {"choices": [{"message": {"content": "Dict format content"}}]}

		# Extract the content
		result = extract_content_from_response(response)

		# Verify the result
		assert result == "Dict format content"

	def test_extract_content_from_response_text_completion(self) -> None:
		"""Test extracting content from a text completion response format."""
		# Create a mock response for text completion
		response = {"choices": [{"text": "Text completion content"}]}

		# Extract the content
		result = extract_content_from_response(response)

		# Verify the result
		assert result == "Text completion content"

	def test_extract_content_from_response_string(self) -> None:
		"""Test extracting content from a string response."""
		# Create a string response
		response = "Direct string content"

		# Extract the content
		result = extract_content_from_response(response)

		# Verify the result
		assert result == "Direct string content"

	def test_extract_content_from_response_fallback(self) -> None:
		"""Test fallback handling when extraction fails."""

		# Create a response format that doesn't match any expected patterns
		class WeirdResponse:
			"""A class that doesn't match expected response patterns."""

			def __str__(self) -> str:
				return "Weird response string representation"

		# Extract the content
		result = extract_content_from_response(WeirdResponse())

		# Verify we get the string representation
		assert result == "Weird response string representation"
