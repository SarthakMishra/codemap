"""Tests for the LLM utils module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from codemap.llm.client import LLMClient
from codemap.llm.config import LLMConfig
from codemap.llm.errors import LLMError
from codemap.llm.utils import (
	create_client,
	extract_content_from_response,
	generate_text,
	get_llm_client,
	load_prompt_template,
)
from codemap.utils.config_loader import ConfigLoader


@pytest.fixture
def mock_config_loader():
	"""Fixture to create a mock ConfigLoader."""
	mock_loader = Mock(spec=ConfigLoader)
	mock_loader.get_llm_config.return_value = {
		"model": "openai/gpt-4",
		"provider": "openai",
		"temperature": 0.7,
		"max_tokens": 1000,
	}
	return mock_loader


@pytest.mark.unit
def test_load_prompt_template_none():
	"""Test load_prompt_template with None path."""
	assert load_prompt_template(None) is None


@pytest.mark.unit
def test_load_prompt_template_exists():
	"""Test load_prompt_template with existing file."""
	template_content = "This is a test template for {param}"
	mock_file = MagicMock()
	mock_file.__enter__.return_value.read.return_value = template_content

	with patch("pathlib.Path.open", return_value=mock_file), patch("pathlib.Path.exists", return_value=True):
		result = load_prompt_template("test_template.txt")
		assert result == template_content


@pytest.mark.unit
def test_load_prompt_template_not_exists():
	"""Test load_prompt_template with non-existent file."""
	with patch("pathlib.Path.exists", return_value=False):
		result = load_prompt_template("nonexistent.txt")
		assert result is None


@pytest.mark.unit
def test_get_llm_client_success(mock_config_loader):
	"""Test successful LLM client creation."""
	with patch("codemap.llm.utils.LLMClient") as mock_client_cls:
		mock_client = Mock(spec=LLMClient)
		mock_client_cls.return_value = mock_client

		client = get_llm_client(mock_config_loader)

		# Verify client was created with correct config
		mock_client_cls.assert_called_once()
		assert client == mock_client


@pytest.mark.unit
def test_get_llm_client_error():
	"""Test LLM client creation error handling."""
	with (
		patch("codemap.llm.utils.get_llm_config", side_effect=LLMError("Config error")),
		pytest.raises(RuntimeError, match="Failed to create LLM client: Config error"),
	):
		get_llm_client()


@pytest.mark.unit
def test_extract_content_from_response_pass_through():
	"""Test that extract_content_from_response passes through to API function."""
	mock_response = {"choices": [{"message": {"content": "Test content"}}]}

	# We need to mock the function directly in the module where it's imported from, not where it's defined
	with patch("codemap.llm.utils.extract_content_from_response", wraps=extract_content_from_response):
		result = extract_content_from_response(mock_response)

		# Verify API function was called with response
		assert result == "Test content"


@pytest.mark.unit
def test_generate_text_success():
	"""Test successful text generation."""
	with patch("codemap.llm.utils.create_client") as mock_create_client:
		mock_client = Mock(spec=LLMClient)
		mock_client.generate_text.return_value = "Generated text"
		mock_create_client.return_value = mock_client

		result = generate_text(prompt="Test prompt", model="openai/gpt-4", api_key="test-key", temperature=0.7)

		# Verify client was created with correct parameters
		mock_create_client.assert_called_once_with(
			model="openai/gpt-4", api_key="test-key", api_base=None, config_loader=None
		)

		# Verify generate_text was called with correct parameters
		mock_client.generate_text.assert_called_once()
		call_args = mock_client.generate_text.call_args
		assert call_args[1]["prompt"] == "Test prompt"
		assert call_args[1]["temperature"] == 0.7

		assert result == "Generated text"


@pytest.mark.unit
def test_generate_text_error():
	"""Test error handling during text generation."""
	with patch("codemap.llm.utils.create_client") as mock_create_client:
		mock_client = Mock(spec=LLMClient)
		mock_client.generate_text.side_effect = LLMError("Generation error")
		mock_create_client.return_value = mock_client

		with pytest.raises(RuntimeError, match="Failed to generate text with LLM: Generation error"):
			generate_text(prompt="Test prompt")


@pytest.mark.unit
def test_create_client_success():
	"""Test successful client creation with parameters."""
	repo_path = Path("/mock/repo")

	with (
		patch("codemap.llm.utils.get_llm_config") as mock_get_config,
		patch("codemap.llm.utils.LLMClient") as mock_client_cls,
	):
		# Set up mocks
		mock_config = Mock(spec=LLMConfig)
		mock_get_config.return_value = mock_config

		mock_client = Mock(spec=LLMClient)
		mock_client_cls.return_value = mock_client

		# Call function
		client = create_client(
			repo_path=repo_path, model="openai/gpt-4", api_key="test-key", api_base="https://api.example.com"
		)

		# Verify get_llm_config was called with correct overrides
		mock_get_config.assert_called_once()
		assert mock_get_config.call_args[1]["overrides"]["model"] == "openai/gpt-4"
		assert mock_get_config.call_args[1]["overrides"]["api_key"] == "test-key"
		assert mock_get_config.call_args[1]["overrides"]["api_base"] == "https://api.example.com"

		# Verify LLMClient was created with correct parameters
		mock_client_cls.assert_called_once_with(config=mock_config, repo_path=repo_path, config_loader=None)

		assert client == mock_client


@pytest.mark.unit
def test_create_client_error():
	"""Test error handling during client creation."""
	with (
		patch("codemap.llm.utils.get_llm_config", side_effect=Exception("Config error")),
		pytest.raises(RuntimeError, match="Failed to create LLM client: Config error"),
	):
		create_client(model="openai/gpt-4")


@pytest.mark.unit
def test_create_client_with_config_loader(mock_config_loader):
	"""Test client creation with ConfigLoader."""
	with (
		patch("codemap.llm.utils.get_llm_config") as mock_get_config,
		patch("codemap.llm.utils.LLMClient") as mock_client_cls,
	):
		# Set up mocks
		mock_config = Mock(spec=LLMConfig)
		mock_get_config.return_value = mock_config

		mock_client = Mock(spec=LLMClient)
		mock_client_cls.return_value = mock_client

		# Call function
		client = create_client(config_loader=mock_config_loader)

		# Verify get_llm_config was called with config_loader
		mock_get_config.assert_called_once()
		assert mock_get_config.call_args[1]["config_loader"] == mock_config_loader

		# Verify LLMClient was created with correct parameters
		mock_client_cls.assert_called_once_with(config=mock_config, repo_path=None, config_loader=mock_config_loader)

		assert client == mock_client
